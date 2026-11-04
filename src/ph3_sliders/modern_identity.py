"""Explicit modern collection identity; never fabricate classic distribution metadata."""
from pathlib import Path

from .native_cache import file_hash


def modern_engine_identity():
    import ph3_hndl
    import ph3_hypergan
    from ph3_hypergan.runtime import require_runtime

    # Both helpers verify imported source bytes against their qualified locks.
    # Record bridge bytes too: neither distribution names nor versions alone
    # establish that the runtime executing collection is the selected runtime.
    return dict(engine=require_runtime(), hndl=ph3_hndl.runtime_identity(),
        bridges={package.__name__: {p.relative_to(Path(package.__file__).parent).as_posix(): file_hash(p)
            for p in sorted(Path(package.__file__).parent.rglob("*.py"))}
            for package in (ph3_hypergan, ph3_hndl)})


def verify_modern_teacher_identity(previous, *, source_manifest, quantization,
                                   staged_weights_sha256, plan_sha256,
                                   encoded_conditioning_sha256):
    from .native_runtime import MODERN_COHORT, runtime_identity

    if previous.get("runtime_cohort") != MODERN_COHORT:
        raise ValueError("modern collection requires its explicit runtime cohort; classic state cannot be rebound")
    expected = dict(runtime_identity(source_manifest, quantization=quantization, cohort=MODERN_COHORT),
        staged_weights_sha256=staged_weights_sha256, plan_sha256=plan_sha256,
        encoded_conditioning_sha256=encoded_conditioning_sha256)
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError("modern frozen teacher identity differs: " + key)
    return dict(teacher_collection=previous, verified_modern_runtime=expected)
