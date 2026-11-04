from copy import deepcopy
from pathlib import Path

import pytest
import torch

pytest.importorskip("ph3_sliders.native_runtime")
import ph3_sliders.native_runtime as native
from ph3_sliders.native_cache import file_hash
import ph3_hypergan.teacher_identity as identity


def test_forward_binding_rejects_model_package_source_drift_but_retains_classic_metadata(monkeypatch):
    monkeypatch.setattr(native, "verify_diffusers_source", lambda path: dict(revision="verified-clean-tree"))
    monkeypatch.setattr(identity, "version", lambda name: "same-actual-version")
    previous = dict(diffusers=dict(revision="verified-clean-tree"), model_revision=native.MODEL_REVISION,
        packages={name: "same-actual-version" for name in identity.FORWARD_PACKAGES},
        product_sources={name: file_hash(Path(native.__file__).parent / name) for name in identity.FORWARD_SOURCES},
        quantization="int8", int8_activation_quantization_cast="float16",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        staged_weights_sha256="weights", plan_sha256="collection-plan", encoded_conditioning_sha256="prompts")
    previous["packages"].update(particlegan="0.6.0", **{"particle-sliders-core": "0.3.0"})
    previous["product_sources"]["adapter.py"] = "classic-training-only-source"
    args = dict(source_manifest="mocked", quantization="int8", staged_weights_sha256="weights",
                plan_sha256="collection-plan", encoded_conditioning_sha256="prompts")
    result = identity.verify_teacher_identity(previous, **args)
    assert result["teacher_collection"]["packages"]["particlegan"] == "0.6.0"
    assert "particlegan" not in result["verified_forward"]["packages"]
    for section, key in (("packages", "torch"), ("product_sources", "native_host.py")):
        changed = deepcopy(previous)
        changed[section][key] = "drifted"
        with pytest.raises(ValueError, match="forward identity differs"):
            identity.verify_teacher_identity(changed, **args)
    changed = deepcopy(previous); changed["model_revision"] = "another-model"
    with pytest.raises(ValueError, match="model_revision"):
        identity.verify_teacher_identity(changed, **args)
