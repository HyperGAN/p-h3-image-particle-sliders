"""Inference-only routed artifacts. No critic, optimizer or target cache needed."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .adapter import RoutedAdapter
from .native_cache import file_hash

MODEL_IDENTITY_KEYS = ("model_revision", "diffusers", "packages", "quantization",
    "int8_activation_quantization_cast", "deterministic_algorithms", "tf32", "product_sources",
    "staged_weights_sha256")


def load_export(host, directory, current_identity):
    directory = Path(directory)
    document = json.loads((directory / "manifest.json").read_text())
    if document.get("schema") != "ph3-native-routed-adapter-v1":
        raise ValueError("native routed artifact schema differs")
    trained = document["signature"]["identity"]
    if any(trained.get(k) != current_identity.get(k) for k in MODEL_IDENTITY_KEYS):
        raise ValueError("inference base/source/precision cohort differs")
    if document["signature"]["targets"] != list(host.targets):
        raise ValueError("native adapter target order differs")
    path = directory / "adapter.safetensors"
    if file_hash(path) != document["weights_sha256"]:
        raise ValueError("native adapter bytes changed")
    values = load_file(str(path), device="cpu")
    if host.adapter is not None:
        raise ValueError("close existing owned adapter before inference load")
    adapter = RoutedAdapter(host.model, host.targets, device=host.device)
    try:
        expected = adapter.state_dict()
        if set(values) != set(expected):
            raise ValueError("native adapter leaves differ")
        for name, value in values.items():
            if value.shape != expected[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("invalid native adapter tensor")
        adapter.load_state_dict(values, strict=True)
        host.adapter = adapter
        return document
    except BaseException:
        adapter.close()
        raise
