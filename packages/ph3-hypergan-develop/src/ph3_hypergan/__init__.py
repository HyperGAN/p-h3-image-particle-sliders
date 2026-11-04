"""Separate modern engine cohort; never aliases the classic Anima Game."""

from .runtime import require_runtime
from .normalization import fit_normalization
from .teacher_identity import verify_teacher_identity

__all__ = ["PairedResidualRun", "NativeHyperganRun", "require_runtime", "fit_normalization", "verify_teacher_identity"]


def __getattr__(name):
    if name in ("PairedResidualRun", "NativeHyperganRun"):
        from . import training
        return getattr(training, name)
    raise AttributeError(name)
