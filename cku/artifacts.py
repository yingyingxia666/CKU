import math
import os
import tempfile
from pathlib import Path

import torch

FORMAT_VERSION = 1


def save_artifact(path, kind, tensors, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "tensors": tensors,
        "metadata": metadata,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".cku-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_artifact(path, kind):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported CKU artifact format")
    if payload.get("kind") != kind:
        raise ValueError(f"Expected {kind} artifact, got {payload.get('kind')}")
    tensors, metadata = payload.get("tensors"), payload.get("metadata")
    if not isinstance(tensors, dict) or not tensors or not isinstance(metadata, dict):
        raise ValueError("Artifact must contain tensors and metadata")
    parameters = metadata.get("model_identity", {}).get("parameters", {})
    if set(parameters) != set(tensors):
        raise ValueError("Artifact parameter names do not match metadata")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != parameters[name]["shape"]:
            raise ValueError(f"Artifact shape mismatch: {name}")
        if kind == "scores":
            if (
                not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
                or (tensor < 0).any()
            ):
                raise ValueError(f"Scores must be finite nonnegative floats: {name}")
        elif tensor.dtype != torch.bool:
            raise ValueError(f"Mask must be boolean: {name}")
    return payload


def top_fraction_mask(score: torch.Tensor, lock_ratio: float):
    if not math.isfinite(lock_ratio) or not 0 <= lock_ratio <= 1:
        raise ValueError("lock_ratio must be finite and in [0, 1]")
    if not score.is_floating_point() or not torch.isfinite(score).all() or (score < 0).any():
        raise ValueError("Scores must be finite nonnegative floats")
    flat = score.flatten()
    count = int(flat.numel() * lock_ratio)
    if count == 0:
        return torch.zeros_like(score, dtype=torch.bool)
    if count == flat.numel():
        return torch.ones_like(score, dtype=torch.bool)
    threshold = torch.kthvalue(flat, flat.numel() - count + 1).values
    mask = flat > threshold
    remaining = count - int(mask.sum())
    # Exactly floor(p * N) weights, even for tied scores. Lower flat indices win ties.
    ties = torch.nonzero(flat == threshold, as_tuple=False).flatten()
    mask[ties[:remaining]] = True
    return mask.reshape(score.shape)


def build_masks(scores_path, output, lock_ratio: float):
    scores = load_artifact(scores_path, "scores")
    masks = {
        name: top_fraction_mask(score, lock_ratio) for name, score in scores["tensors"].items()
    }
    metadata = dict(scores["metadata"])
    metadata.update(lock_ratio=lock_ratio, selection="per_projection_weight_top_fraction")
    save_artifact(output, "masks", masks, metadata)
    return mask_summary(masks)


def mask_summary(masks):
    return {
        name: {"protected": int(mask.sum()), "total": mask.numel()} for name, mask in masks.items()
    }
