"""Portable strict inference artifacts for the optional HNDL attachment.

Training checkpoints belong to the owning trainer. These artifacts contain only
adapter leaves; the frozen native model and full teacher cache stay external.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .networks import RoutedAdapter

SCHEMA = "ph3-hndl-routed-inference-v1"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_signature(adapter, identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError("explicit model/source identity required")
    # A JSON round trip freezes only portable identity content and rejects NaN.
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    return dict(schema=SCHEMA, identity=identity, targets=list(adapter.names),
                architecture=deepcopy(adapter.identity),
                branches=[dict(input_width=b.input_width, output_width=b.output_width,
                               graph=deepcopy(b.identity)) for b in adapter.branches],
                precision="float32", strength_range=[0., 1.],
                response_surface="full_packed_image_then_audio_velocity",
                binding_sha256=_hash(Path(__file__).with_name("networks.py")),
                tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                         for k, v in adapter.state_dict().items()})


def _validate_values(expected, values):
    if set(values) != set(expected):
        raise ValueError("HNDL adapter tensor keys differ")
    for name, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"invalid HNDL adapter tensor: {name}")


def export_adapter(adapter, directory, *, identity, selection="raw", completed_steps=0, values=None):
    """Export raw or explicitly supplied EMA leaves without mutating live state."""
    from safetensors.torch import save_file
    if adapter.closed:
        raise RuntimeError("adapter is closed")
    if selection not in ("raw", "ema") or type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("valid raw/EMA selection and nonnegative completed update count required")
    if selection == "ema" and values is None:
        raise ValueError("EMA export requires the owning trainer's selected adapter values")
    expected = adapter.state_dict()
    values = expected if values is None else values
    _validate_values(expected, values)
    signature = adapter_signature(adapter, identity)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
    try:
        save_file({k: v.detach().cpu().clone().contiguous() for k, v in values.items()},
                  str(temporary / "adapter.safetensors"))
        receipt = dict(schema=SCHEMA, signature=signature, selection=selection,
                       completed_steps=completed_steps,
                       weights_sha256=_hash(temporary / "adapter.safetensors"))
        (temporary / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, directory)
    except BaseException:
        import shutil
        shutil.rmtree(temporary)
        raise
    return receipt


def load_adapter(host, targets, directory, *, identity, device="cpu"):
    """Attach validated inference weights; remove owned hooks if admission fails.

    Caller supplies the exact frozen host and native identity. Assign the result
    to ``H3Host.adapter`` and use its existing ``rollout(..., strength=...)``.
    """
    from safetensors.torch import load_file
    directory = Path(directory)
    receipt = json.loads((directory / "manifest.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("signature", {}).get("schema") != SCHEMA:
        raise ValueError("HNDL inference artifact schema differs")
    if _hash(directory / "adapter.safetensors") != receipt.get("weights_sha256"):
        raise ValueError("HNDL inference weights changed")
    if receipt.get("selection") not in ("raw", "ema") or type(receipt.get("completed_steps")) is not int or receipt["completed_steps"] < 0:
        raise ValueError("HNDL inference selection/update count differs")
    # Graph construction initializes unused leaves; preserve caller RNG exactly.
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        adapter = RoutedAdapter(host, targets, device=device)
    try:
        if adapter_signature(adapter, identity) != receipt["signature"]:
            raise ValueError("HNDL model/source/target/precision signature differs")
        values = load_file(str(directory / "adapter.safetensors"), device="cpu")
        _validate_values(adapter.state_dict(), values)
        adapter.load_state_dict(values, strict=True)
    except BaseException:
        adapter.close()
        raise
    adapter.artifact = deepcopy(receipt)
    return adapter
