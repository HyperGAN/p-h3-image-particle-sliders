"""Allocation-free planning from public model-card facts; no model loader."""

from dataclasses import asdict, dataclass
import math

MODEL_ID = "PrunaAI/P-H3-Image"
MODEL_REVISION = "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"
CORE_REVISION = "a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
REFUSAL = (
    "P-H3-Image live operations are disabled: model-license authorization and "
    "native gmix host qualification are unresolved. This revision has "
    "no model download, loading, training, rendering, or publication path."
)


class LiveOperationUnavailable(RuntimeError):
    """A source-only repository cannot execute a native model operation."""


def refuse_live(operation: str) -> None:
    raise LiveOperationUnavailable(f"{operation}: {REFUSAL}")


@dataclass(frozen=True)
class RequestPlan:
    """A proposed native request, not permission or an executable pipeline."""

    prompt: str
    width: int = 2912
    height: int = 2912
    references: tuple[str, ...] = ()
    seed: int = 7
    strength: float = 1.0

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty text")
        for key in ("width", "height"):
            value = getattr(self, key)
            if type(value) is not int or value < 32 or value % 32:
                raise ValueError(f"{key} must be a positive multiple of 32")
        if not 4_000_000 <= self.width * self.height <= 8_500_000:
            raise ValueError("this planning profile supports the documented 4–8 MP canvas range")
        if isinstance(self.references, (str, bytes)):
            raise ValueError("references must be an ordered sequence of local path labels")
        references = tuple(self.references)
        if len(references) > 8 or any(not isinstance(p, str) or not p.strip() for p in references):
            raise ValueError("supply at most eight nonempty reference path labels")
        object.__setattr__(self, "references", references)
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if (type(self.strength) not in (int, float) or not math.isfinite(self.strength)
                or not 0 <= self.strength <= 1):
            raise ValueError("planned strength must be finite and in [0, 1]")

    def as_dict(self):
        return {
            "status": "plan-only; native execution unavailable",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "core_revision": CORE_REVISION,
            "task": "image-edit" if self.references else "text-to-image",
            "sampling": "checkpoint's native eight-step sigma grid; no CFG",
            "prompt_upsampler": "future paired collection must freeze expanded prompts",
            "request": asdict(self),
            "blockers": ["model-license authorization", "native adapter and sampler qualification"],
        }
