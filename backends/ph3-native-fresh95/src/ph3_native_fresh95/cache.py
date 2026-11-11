"""Serialize only completed original PreparedResponses, never requests/teachers."""

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import math

import torch

from . import ownership
from ._config import configuration
from .util import Checkpoint, canonical, equal, read_packet, save_packet, tree

SCHEMA = "PH3Image-completed-coupled-native-PreparedResponses-fresh95-v1"


@dataclass(frozen=True)
class CacheBinding:
    checkpoint: Checkpoint
    provenance: dict


def _pin(value):
    return (
        isinstance(value, dict)
        and set(value) == {"path", "bytes", "sha256"}
        and isinstance(value["path"], str)
        and value["path"].startswith("/")
        and type(value["bytes"]) is int
        and value["bytes"] > 0
        and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"])
    )


def validate_identity(identity):
    canonical(identity)  # Finite portable metadata only; never arbitrary objects.
    if not isinstance(identity, dict) or set(identity) != {"kind", "model", "completed_owner_receipt"}:
        raise ValueError("explicit original model/native ownership identity required")
    if identity["kind"] not in ("authentic-native-owner", "NEW-synthetic-host-protocol-fixture"):
        raise ValueError("native ownership and synthetic software scopes stay distinct")
    if not isinstance(identity["model"], dict) or not identity["model"]:
        raise ValueError("actual model/geometry/source identity required")
    if identity["kind"] == "authentic-native-owner" and not _pin(identity["completed_owner_receipt"]):
        raise ValueError("authentic completed native owner receipt required")
    if identity["kind"] != "authentic-native-owner" and identity["completed_owner_receipt"] is not None:
        raise ValueError("synthetic data may not claim completed pretrained ownership")


def host_identity(host):
    from ph3_research095.native_host import H3Host, SIGMAS, attention_targets

    if type(host) is not H3Host or tuple(host.sigmas) != tuple(SIGMAS):
        raise ValueError("the original H3Host and original eight-step native schedule are required")
    if host.targets != attention_targets(host.model):
        raise ValueError("original complete attention target route differs")
    if any(p.requires_grad or p.device != host.device for p in host.model.parameters()):
        raise ValueError("caller-owned original model must already be frozen on its declared device")
    if any(m.training for m in host.model.modules()):
        raise ValueError("caller-owned original model must already be in evaluation mode")
    return {
        "latent_height": host.latent_height,
        "latent_width": host.latent_width,
        "targets": list(host.targets),
        "sigmas": list(host.sigmas),
        "timesteps": host.timesteps.tolist(),
        "model_geometry": {
            key: list(getattr(host.model.config, key)) if key == "patch_size" else getattr(host.model.config, key)
            for key in ("in_channels", "audio_in_channels", "text_dim", "patch_size", "hidden_size",
                        "num_attention_heads", "attention_head_dim", "num_layers", "num_refiner_layers")
        },
    }


def validate_row(row, host):
    from ph3_research095.bridge import NativeRow
    from ph3_research095.native_host import Conditioning, PackedState

    if type(row) is not NativeRow or type(row.state) is not PackedState:
        raise ValueError("original full packed NativeRow/PackedState required")
    if row.path not in ("neutral", "positive") or row.scale != 1.0 or not isinstance(row.scene, str):
        raise ValueError("original unipolar native row identity required")
    if type(row.seed) is not int or not 0 <= row.seed < 2**63 - 100:
        raise ValueError("original seed range required")
    if type(row.position) is not int or not 0 <= row.position < len(host.timesteps):
        raise ValueError("actual native scheduler position required")
    if type(row.time) not in (int, float) or not math.isfinite(row.time) or row.time != float(host.timesteps[row.position]):
        raise ValueError("exact original scheduler time required")
    pt, ph, pw = host.model.config.patch_size
    if pt != 1 or host.latent_height % ph or host.latent_width % pw:
        raise ValueError("original single-frame integral packed-image geometry required")
    image_shape = (host.latent_height * host.latent_width // (ph * pw), host.model.config.in_channels * pt * ph * pw)
    audio_shape = (4, host.model.config.audio_in_channels)
    for value, shape in ((row.state.image, image_shape), (row.state.audio, audio_shape)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("complete native image and coupled-audio FP32 state required")
    for context in (row.neutral, row.positive):
        if type(context) is not Conditioning or not isinstance(context.name, str) or not context.name:
            raise ValueError("original completed conditioning required")
        value = context.embeddings
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != host.model.config.text_dim or not torch.isfinite(value).all():
            raise ValueError("complete native text embedding geometry required")
        # Only regenerate layout metadata for equality; never fill/replace a
        # missing cached value and never tokenize/encode or call a teacher.
        expected = host.conditioning(context.name, value).layout
        if not equal(context.layout, expected):
            raise ValueError("original full packed conditioning layout differs")


def _dump_row(row):
    return {
        "state": {"image": tree(row.state.image, "cpu"), "audio": tree(row.state.audio, "cpu")},
        "neutral": {"name": row.neutral.name, "embeddings": tree(row.neutral.embeddings, "cpu"), "layout": tree(row.neutral.layout, "cpu")},
        "positive": {"name": row.positive.name, "embeddings": tree(row.positive.embeddings, "cpu"), "layout": tree(row.positive.layout, "cpu")},
        **{f.name: getattr(row, f.name) for f in fields(type(row)) if f.name not in ("state", "neutral", "positive")},
    }


def _restore_row(value):
    from ph3_research095.bridge import NativeRow
    from ph3_research095.native_host import Conditioning, PackedState

    return NativeRow(
        PackedState(**value["state"]), Conditioning(**value["neutral"]), Conditioning(**value["positive"]),
        **{k: v for k, v in value.items() if k not in ("state", "neutral", "positive")},
    )


def prepared_digest(item):
    from .policy import tensor_sha

    def digest(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
        if isinstance(value, dict):
            return {k: digest(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {"container": type(value).__name__, "values": [digest(v) for v in value]}
        return value

    return hashlib.sha256(canonical(digest({"row": _dump_row(item.row), "target": item.target, "neutral": item.neutral})).encode()).hexdigest()


def _validate_prepared(prepared, host):
    from particle_sliders.pg09_backends import PreparedResponse

    if not isinstance(prepared, (list, tuple)) or len(prepared) < 2:
        raise ValueError("at least two authentic already-completed response rows required")
    for item in prepared:
        if type(item) is not PreparedResponse:
            raise ValueError("request/packed teacher records are not PreparedResponses")
        validate_row(item.row, host)
        shape = (1, item.row.state.image.numel() + item.row.state.audio.numel())
        if any(not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or tuple(x.shape) != shape or not torch.isfinite(x).all() for x in (item.target, item.neutral)):
            raise ValueError("full image-plus-audio response vectors required; no crop/pool")


def _validate_origins(prepared, teacher_origins, identity):
    if not isinstance(teacher_origins, list) or len(teacher_origins) != len(prepared):
        raise ValueError("one retained completed origin per row required")
    synthetic = identity["kind"] == "NEW-synthetic-host-protocol-fixture"
    for index, (item, origin) in enumerate(zip(prepared, teacher_origins)):
        if not isinstance(origin, dict) or set(origin) != {"row_index", "completed", "kind", "prepared_sha256", "completed_output_receipt"} or type(origin["row_index"]) is not int or origin["row_index"] != index or origin["completed"] is not True or origin["prepared_sha256"] != prepared_digest(item):
            raise ValueError("exact completed original row provenance required")
        if synthetic:
            if origin["kind"] != "NEW-synthetic-software-data" or origin["completed_output_receipt"] is not None:
                raise ValueError("synthetic rows may not claim native teacher completion")
        elif origin["kind"] != "authentic-native-PreparedResponse" or not _pin(origin["completed_output_receipt"]):
            raise ValueError("actual native output receipt required; no automatic teacher fill")


def save_prepared_cache(path, *, host, identity, config, prepared, teacher_origins, cache_id):
    ownership.guard()
    if configuration(**config) != config:
        raise ValueError("exact explicit new configuration required")
    validate_identity(identity)
    _validate_prepared(prepared, host)
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError("completed cache identity required")
    _validate_origins(prepared, teacher_origins, identity)
    provenance = {"cache_id": cache_id, "identity": deepcopy(identity), "host": host_identity(host), "sources": ownership.sources(), "teacher_origins": deepcopy(teacher_origins)}
    packet = {"schema": SCHEMA, "completed": True, "provenance": provenance, "rows": [
        {"row": _dump_row(x.row), "target": tree(x.target, "cpu"), "neutral": tree(x.neutral, "cpu")}
        for x in prepared
    ], "new_teacher_or_prepare_calls": 0}
    return CacheBinding(save_packet(path, packet), deepcopy(provenance))


def load_cache(binding, *, host, identity):
    from particle_sliders.pg09_backends import PreparedResponse

    if not isinstance(binding, CacheBinding):
        raise ValueError("typed whole-file completed cache binding required")
    validate_identity(identity)
    packet = read_packet(binding.checkpoint)
    if not isinstance(packet, dict) or set(packet) != {"schema", "completed", "provenance", "rows", "new_teacher_or_prepare_calls"} or packet["schema"] != SCHEMA or packet["completed"] is not True or packet["new_teacher_or_prepare_calls"] != 0:
        raise ValueError("only this explicit completed PreparedResponse cache schema is accepted")
    provenance = packet["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"cache_id", "identity", "host", "sources", "teacher_origins"} or not isinstance(provenance["cache_id"], str) or not provenance["cache_id"]:
        raise ValueError("complete exact retained provenance schema required")
    if not equal(provenance, binding.provenance) or provenance["identity"] != identity or provenance["host"] != host_identity(host) or provenance["sources"] != ownership.sources():
        raise ValueError("cache whole-file/model/native/Source provenance differs")
    prepared = [PreparedResponse(_restore_row(x["row"]), x["target"].to(host.device), x["neutral"].to(host.device)) for x in packet["rows"]]
    _validate_prepared(prepared, host)
    _validate_origins(prepared, provenance["teacher_origins"], identity)
    return prepared, {"schema": SCHEMA, "bytes": binding.checkpoint.bytes, "sha256": binding.checkpoint.sha256, "provenance": deepcopy(provenance)}
