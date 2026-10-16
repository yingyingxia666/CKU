import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .artifacts import save_artifact
from .data import AnswerCollator, AnswerDataset, load_records
from .modeling import load_model_and_tokenizer, model_identity, select_parameters, set_seed


def compute_scores(model, dataset, collator, layer_start, layer_end, device):
    selected = select_parameters(model, layer_start, layer_end)
    identity = model_identity(model, selected)
    original_flags = {name: param.requires_grad for name, param in model.named_parameters()}
    was_training = model.training
    model.requires_grad_(False)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    scores = {
        name: torch.zeros(parameter.shape, dtype=torch.float32)
        for name, parameter in selected.items()
    }
    model.eval()
    try:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
        if not len(loader):
            raise ValueError("Calibration dataset is empty")
        for index, batch in enumerate(loader):
            model.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite calibration loss at sample {index}")
            loss.backward()
            for name, parameter in selected.items():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Missing or nonfinite calibration gradient: {name}")
                scores[name].add_(
                    (parameter.detach().float() * parameter.grad.detach().float()).abs().cpu()
                )
        for score in scores.values():
            score.div_(len(loader))
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])
        model.train(was_training)
    return scores, identity


def score_file(options):
    if Path(options["output"]).exists():
        raise FileExistsError(f"Output already exists: {options['output']}")
    set_seed(options["seed"])
    records = load_records(options["data"])
    indices = random.Random(options["seed"]).sample(
        range(len(records)), min(options["num_samples"], len(records))
    )
    model, tokenizer, device = load_model_and_tokenizer(
        options["model"], options["device"], options["dtype"], options["revision"]
    )
    dataset = AnswerDataset(
        [records[index] for index in indices],
        tokenizer,
        options["max_length"],
        options["prompt_format"],
    )
    scores, identity = compute_scores(
        model,
        dataset,
        AnswerCollator(tokenizer.pad_token_id),
        options["layer_start"],
        options["layer_end"],
        device,
    )
    metadata = {
        "model_identity": identity,
        "model_source": options["model"],
        "revision": options["revision"],
        "dtype": options["dtype"],
        "layer_start": options["layer_start"],
        "layer_end": options["layer_end"] or model.config.num_hidden_layers,
        "seed": options["seed"],
        "sample_indices": indices,
        "data_sha256": hashlib.sha256(Path(options["data"]).read_bytes()).hexdigest(),
        "max_length": options["max_length"],
        "prompt_format": options["prompt_format"],
        "score_definition": "mean_sample_abs_weight_times_answer_nll_gradient",
    }
    save_artifact(options["output"], "scores", scores, metadata)
    return {"samples": len(dataset), "parameters": len(scores), "output": options["output"]}
