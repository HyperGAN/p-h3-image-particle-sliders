"""Keep frozen forward compatibility separate from the new GAN generation."""

from importlib.metadata import version
from pathlib import Path

FORWARD_PACKAGES = ("torch", "transformers", "accelerate", "safetensors", "numpy",
                    "bitsandbytes", "huggingface-hub")
FORWARD_SOURCES = ("native_host.py", "native_runtime.py", "native_cache.py", "native_cli.py", "native_plan.py")


def verify_teacher_identity(previous, *, source_manifest, quantization,
                            staged_weights_sha256, plan_sha256, encoded_conditioning_sha256):
    """Verify all forward-relevant fields and retain the complete old identity.

    Classic PG/core/runner distribution metadata and adapter.py did not execute
    the frozen OFF teacher forward. They remain in ``teacher_collection`` rather
    than being falsely required by the modern runtime. No model, forward source,
    precision, plan, staged-byte or conditioning comparison is omitted.
    """
    import torch
    import ph3_sliders.native_runtime as runtime
    from ph3_sliders.native_cache import file_hash

    root = Path(runtime.__file__).parent
    forward = dict(diffusers=runtime.verify_diffusers_source(source_manifest),
        model_revision=runtime.MODEL_REVISION, quantization=quantization,
        int8_activation_quantization_cast="float16",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        packages={name: version(name) for name in FORWARD_PACKAGES},
        product_sources={name: file_hash(root / name) for name in FORWARD_SOURCES},
        staged_weights_sha256=staged_weights_sha256, plan_sha256=plan_sha256,
        encoded_conditioning_sha256=encoded_conditioning_sha256)
    for key, value in forward.items():
        if key in ("packages", "product_sources"):
            if any(previous[key].get(name) != expected for name, expected in value.items()):
                raise ValueError("frozen teacher forward identity differs: " + key)
        elif previous.get(key) != value:
            raise ValueError("frozen teacher forward identity differs: " + key)
    return dict(teacher_collection=previous, verified_forward=forward)
