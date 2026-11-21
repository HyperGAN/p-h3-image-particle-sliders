"""Use verified shared steering equations beside the scoped RAW runtime."""

from dataclasses import asdict
from hashlib import sha256
from sys import modules
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MEMBER = "particle_sliders/adaptive_steering.py"
CORE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"{__name__}._core_{CORE_REVISION}"
_LOCK = Lock()


def _load_core(core_wheel):
    with ZipFile(core_wheel) as archive:
        source = archive.read(CORE_MEMBER)
    if sha256(source).hexdigest() != CORE_SHA256:
        raise ValueError("P-H3 Image requires the exact adaptive steering source from core PR #269")
    with _LOCK:
        if _MODULE_NAME not in modules:
            module = ModuleType(_MODULE_NAME)
            modules[_MODULE_NAME] = module
            try:
                exec(compile(source, CORE_MEMBER, "exec"), module.__dict__)
            except BaseException:
                del modules[_MODULE_NAME]
                raise
        return modules[_MODULE_NAME]


def packed_latent_readout(state):
    """Concatenate the FP32 packed row means, image first and audio second."""
    latents = (state.image, state.audio)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           or value.ndim != 2 or not all(value.shape) for value in latents):
        raise ValueError("Expected native packed image and audio matrices")
    if state.image.device != state.audio.device:
        raise ValueError("Both native packed latents must be on the same device")
    return torch.cat([value.mean(dim=0, dtype=torch.float32) for value in latents])


class PackedGatePlan:
    """Explicit calibration inputs for CFP, CLABS or the matched random control."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(char not in "0123456789abcdef" for char in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the P-H3 Image gate calibration record")
        if type(parameter) not in (int, float):
            raise ValueError("Provide an explicit numeric gate threshold, bias or rate")
        self._core = _load_core(core_wheel)
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._clabs = self._core.CLABSParameters(**clabs) if clabs is not None else None
        self._calibration_sha256 = calibration_sha256
        self._direction = None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random control does not use a direction")
        else:
            if direction is None:
                raise ValueError("CFP and CLABS require a fitted joint image/audio direction")
            direction = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (direction.ndim != 1 or not torch.isfinite(direction).all()
                    or not torch.isclose(direction.norm(), direction.new_tensor(1.0))):
                raise ValueError("The packed direction must be a finite unit vector")
            self._direction = direction
        self._new_gate()

    def _new_gate(self):
        return self._core.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _score(self, state, strength):
        if self._direction is None:
            return None
        return float(self._core.activation_projection(
            packed_latent_readout(state), self._direction, strength=strength,
        ))

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "core_source_sha256": CORE_SHA256,
            "readout": "fp32_mean_packed_image_then_audio_before_adapter",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }
