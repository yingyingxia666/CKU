import hashlib
import os
import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MLP_WEIGHT = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.(up_proj|gate_proj|down_proj)\.weight$")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_parameters(model, layer_start: int, layer_end: int | None):
    end = model.config.num_hidden_layers if layer_end is None else layer_end
    if not 0 <= layer_start < end <= model.config.num_hidden_layers:
        raise ValueError("Layer range must satisfy 0 <= start < end <= num_hidden_layers")
    selected = {}
    for name, parameter in model.named_parameters():
        match = MLP_WEIGHT.search(name)
        if match and layer_start <= int(match[1]) < end:
            selected[name] = parameter
    if len(selected) != (end - layer_start) * 3:
        raise ValueError("Expected up_proj/gate_proj/down_proj weights in each selected Llama MLP")
    return selected


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def model_identity(model, selected) -> dict:
    fields = [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "attention_bias",
        "mlp_bias",
        "tie_word_embeddings",
    ]
    return {
        "config": {key: getattr(model.config, key, None) for key in fields},
        "parameters": {
            name: {"shape": list(parameter.shape), "sha256": tensor_sha256(parameter)}
            for name, parameter in selected.items()
        },
    }


def load_model_and_tokenizer(model_path, device="auto", dtype="float32", revision=None):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This release supports one process/device; do not launch with torchrun")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("This release supports CPU and CUDA devices")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(resolved_device)
    model.config.use_cache = False
    return model, tokenizer, resolved_device
