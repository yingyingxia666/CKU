import json
import math
from pathlib import Path

COMMON_DEFAULTS = {
    "model": None,
    "data": None,
    "output": None,
    "device": "auto",
    "dtype": "float32",
    "revision": None,
    "max_length": 2048,
    "prompt_format": "chat",
}
SCORE_DEFAULTS = {
    **COMMON_DEFAULTS,
    "seed": 0,
    "layer_start": 8,
    "layer_end": 12,
    "num_samples": 128,
}
TRAIN_DEFAULTS = {
    **COMMON_DEFAULTS,
    "seed": 42,
    "masks": None,
    "retain_data": None,
    "kl_weight": 0.0,
    "learning_rate": 4e-6,
    "harmful_threshold": 1.5,
    "batch_size": 4,
    "gradient_accumulation_steps": 1,
    "epochs": 1,
    "max_steps": None,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": False,
}


def resolve_options(command, overrides, config_path=None):
    defaults = SCORE_DEFAULTS if command == "score" else TRAIN_DEFAULTS
    values = dict(defaults)
    origins = {}
    if config_path:
        path = Path(config_path).expanduser().resolve()
        configured = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(configured, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(configured) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {command} config keys: {sorted(unknown)}")
        values.update(configured)
        origins = dict.fromkeys(configured, path.parent)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
            origins[key] = Path.cwd()
    validate_options(command, values)
    for key in ["data", "output", "masks", "retain_data"]:
        if values.get(key) is not None:
            path = Path(values[key]).expanduser()
            values[key] = str((origins.get(key, Path.cwd()) / path).resolve())
    model = values["model"]
    base = origins.get("model", Path.cwd())
    if model.startswith((".", "/", "~")) or (base / model).exists():
        values["model"] = str((base / Path(model).expanduser()).resolve())
    return values


def validate_options(command, values):
    required = ["model", "data", "output"] + (["masks"] if command == "train" else [])
    for key in required:
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} is required and must be a nonempty string")
    for key in ["revision", "retain_data"]:
        if key in values and values[key] is not None and not isinstance(values[key], str):
            raise ValueError(f"{key} must be a string or null")
    if values["prompt_format"] not in {"plain", "chat"}:
        raise ValueError("prompt_format must be plain or chat")
    if values["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    if not isinstance(values["device"], str):
        raise ValueError("device must be a string")
    ints = ["max_length", "seed"]
    ints += (
        ["num_samples", "layer_start"]
        if command == "score"
        else ["batch_size", "epochs", "gradient_accumulation_steps"]
    )
    for key in ints:
        minimum = 0 if key in {"seed", "layer_start"} else (4 if key == "max_length" else 1)
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if values["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if command == "score":
        end = values["layer_end"]
        if end is not None and (type(end) is not int or end <= values["layer_start"]):
            raise ValueError("layer_end must be null or an integer greater than layer_start")
    else:
        if type(values["gradient_checkpointing"]) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        if values["max_steps"] is not None and (
            type(values["max_steps"]) is not int or values["max_steps"] < 1
        ):
            raise ValueError("max_steps must be null or an integer >= 1")
        for key in ["learning_rate", "max_grad_norm", "harmful_threshold", "kl_weight"]:
            value = values[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
            if value < 0 or (key in {"learning_rate", "max_grad_norm"} and value == 0):
                raise ValueError(f"Invalid {key}: {value}")
        if values["kl_weight"] > 0 and not values["retain_data"]:
            raise ValueError("Positive kl_weight requires retain_data")
