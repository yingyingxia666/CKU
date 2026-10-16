import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .artifacts import load_artifact, mask_summary
from .data import AnswerCollator, AnswerDataset, PairedDataset, load_records
from .modeling import load_model_and_tokenizer, model_identity, select_parameters, set_seed


def constrained_loss(nll, threshold):
    return torch.clamp(threshold - nll, min=0.0)


def answer_kl(logits, reference_logits, labels):
    valid = labels[:, 1:] != -100
    if not valid.any():
        raise ValueError("KL batch has no answer tokens")
    current = F.log_softmax(logits[:, :-1].float(), dim=-1)
    reference = F.log_softmax(reference_logits[:, :-1].float(), dim=-1)
    per_token = F.kl_div(current, reference, log_target=True, reduction="none").sum(dim=-1)
    return per_token[valid].mean()


class GradientProtection:
    """Fixed bool masks: True means the corresponding weight is protected.

    Use only with a fresh AdamW optimizer with zero weight decay. Reusing momentum
    from another run/mask or enabling weight decay can move zero-gradient weights.
    """

    def __init__(self, model, artifact):
        metadata = artifact["metadata"]
        self.selected = select_parameters(model, metadata["layer_start"], metadata["layer_end"])
        if model_identity(model, self.selected) != metadata["model_identity"]:
            raise ValueError("Mask does not match this model's selected weights/configuration")
        masks = artifact["tensors"]
        if set(masks) != set(self.selected):
            raise ValueError("Mask coverage does not match selected MLP parameters")
        for name, parameter in self.selected.items():
            mask = masks[name]
            if mask.dtype != torch.bool or mask.shape != parameter.shape:
                raise ValueError(f"Invalid gradient mask: {name}")
        self.flags = {name: param.requires_grad for name, param in model.named_parameters()}
        self.model = model
        self.handles = []
        model.requires_grad_(False)
        for name, parameter in self.selected.items():
            parameter.requires_grad_(True)
            mask = masks[name].to(parameter.device)
            self.handles.append(
                parameter.register_hook(lambda grad, mask=mask: grad.masked_fill(mask, 0))
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self.flags[name])


def train_model(
    model, tokenizer, forget_records, artifact, options, reference=None, retain_records=None
):
    output = Path(options["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if options["kl_weight"] > 0 and (reference is None or not retain_records):
        raise ValueError("Positive kl_weight requires retention data and a reference model")
    set_seed(options["seed"])
    device = next(model.parameters()).device
    forget = AnswerDataset(
        forget_records, tokenizer, options["max_length"], options["prompt_format"]
    )
    retain = (
        AnswerDataset(retain_records, tokenizer, options["max_length"], options["prompt_format"])
        if options["kl_weight"] > 0
        else None
    )
    collator = AnswerCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(options["seed"])
    loader = DataLoader(
        PairedDataset(forget, retain),
        batch_size=options["batch_size"],
        shuffle=True,
        collate_fn=collator.paired,
        generator=generator,
    )
    if not len(loader):
        raise ValueError("Training dataset is empty")
    if reference is not None:
        reference.eval().requires_grad_(False)
    if options["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    protection = GradientProtection(model, artifact)
    optimizer = torch.optim.AdamW(
        protection.selected.values(), lr=options["learning_rate"], weight_decay=0.0
    )
    steps, examples, history = 0, 0, []
    accumulation = options["gradient_accumulation_steps"]
    (output / "run_config.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    (output / "mask_summary.json").write_text(
        json.dumps(mask_summary(artifact["tensors"]), indent=2), encoding="utf-8"
    )
    try:
        with (output / "train_log.jsonl").open("w", encoding="utf-8") as log:
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(options["epochs"]):
                window_metrics = {"forget_nll": 0.0, "forget_loss": 0.0, "kl": 0.0, "loss": 0.0}
                for index, paired in enumerate(loader):
                    # The last accumulation window may contain fewer microbatches.
                    start = (index // accumulation) * accumulation
                    window_size = min(accumulation, len(loader) - start)
                    batch = {key: value.to(device) for key, value in paired["forget"].items()}
                    nll = model(**batch).loss
                    forget_loss = constrained_loss(nll, options["harmful_threshold"])
                    kl = nll.new_zeros(())
                    if options["kl_weight"] > 0:
                        retained = {
                            key: value.to(device) for key, value in paired["retain"].items()
                        }
                        inputs = {key: value for key, value in retained.items() if key != "labels"}
                        current_logits = model(**inputs).logits
                        with torch.no_grad():
                            reference_logits = reference(**inputs).logits
                        kl = answer_kl(current_logits, reference_logits, retained["labels"])
                    loss = forget_loss + options["kl_weight"] * kl
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss at epoch {epoch}, microbatch {index}")
                    (loss / window_size).backward()
                    examples += batch["input_ids"].shape[0]
                    for key, value in {
                        "forget_nll": nll,
                        "forget_loss": forget_loss,
                        "kl": kl,
                        "loss": loss,
                    }.items():
                        window_metrics[key] += float(value.detach()) / window_size
                    if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(protection.selected.values()),
                            options["max_grad_norm"],
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
                        entry = {
                            "step": steps,
                            "epoch": epoch + 1,
                            "grad_norm": float(grad_norm),
                            **window_metrics,
                        }
                        history.append(entry)
                        log.write(json.dumps(entry, allow_nan=False) + "\n")
                        log.flush()
                        window_metrics = dict.fromkeys(window_metrics, 0.0)
                        if options["max_steps"] is not None and steps >= options["max_steps"]:
                            break
                if options["max_steps"] is not None and steps >= options["max_steps"]:
                    break
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
    finally:
        protection.close()
    metrics = {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "updates_per_epoch": math.ceil(len(loader) / accumulation),
        "loss_mode": "threshold_unlearning"
        if options["kl_weight"] == 0
        else "threshold_unlearning_plus_kl",
        "final_log": history[-1],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def train_file(options):
    output = Path(options["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be new or empty: {output}")
    artifact = load_artifact(options["masks"], "masks")
    if options["dtype"] != artifact["metadata"]["dtype"]:
        raise ValueError("Train dtype must match score dtype for exact model/mask verification")
    forget = load_records(options["data"])
    retain = load_records(options["retain_data"]) if options["kl_weight"] > 0 else None
    set_seed(options["seed"])
    model, tokenizer, _ = load_model_and_tokenizer(
        options["model"], options["device"], options["dtype"], options["revision"]
    )
    reference = None
    if options["kl_weight"] > 0:
        reference, _, _ = load_model_and_tokenizer(
            options["model"], options["device"], options["dtype"], options["revision"]
        )
    return train_model(model, tokenizer, forget, artifact, options, reference, retain)
