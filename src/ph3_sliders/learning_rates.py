"""Per-run rate binding for an owned, unused classic Game.

This uses public Torch optimizer param groups; no shared source or module-global
recipe is patched. Callers bind the returned recipe into their full checkpoint
identity and verify it before and after loading optimizer state.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LearningRates:
    generator: float = 2e-5
    critic: float = 9e-4
    particles: float = 6e-3

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("learning rates must be finite positive numbers")


def _groups(game):
    if len(game.g.param_groups) != 2 or len(game.d.param_groups) != 1:
        raise ValueError("expected classic generator/particle/critic optimizer groups")
    generator, particles = game.g.param_groups
    critic = game.d.param_groups[0]
    expected = [p for n, p in game.adapter.named_parameters() if n != "particles"]
    if ([id(p) for p in generator["params"]] != [id(p) for p in expected]
            or [id(p) for p in particles["params"]] != [id(game.adapter.particles)]
            or [id(p) for p in critic["params"]] != [id(p) for p in game.critic.parameters()]):
        raise ValueError("optimizer group parameter ownership differs")
    return dict(generator=generator, critic=critic, particles=particles)


def assert_learning_rates(game, rates):
    """Read actual optimizer values; reject a mislabeled or changed recipe."""
    if type(rates) is not LearningRates:
        raise ValueError("explicit LearningRates required")
    actual = {name: group["lr"] for name, group in _groups(game).items()}
    if actual != asdict(rates):
        raise ValueError("actual optimizer rates differ from experiment recipe")
    return actual


def bind_learning_rates(game, rates):
    """Bind only before updates, preserving the original objective and streams."""
    if type(rates) is not LearningRates:
        raise ValueError("explicit LearningRates required")
    groups = _groups(game)
    if game.g.state or game.d.state:
        raise ValueError("bind rates before updates; do not retune a resumed run")
    for name, value in asdict(rates).items():
        groups[name]["lr"] = float(value)
    return assert_learning_rates(game, rates)
