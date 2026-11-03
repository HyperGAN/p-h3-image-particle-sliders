"""Synthetic field factory for the actual shared ParticleGAN update law.

This unrelated eight-channel fixture is not P-H3 code, geometry, conditioning,
weights or outputs. The separate ph3-image registration remains disabled.
"""

import hashlib
import math

import torch

from particle_sliders.field_critic import FieldCritic, seed_initialization_rng
from particle_sliders.lifecycle import Backend, Binding, Capabilities, UnsupportedOperation
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow

from .synthetic import SyntheticProjection

NAME = "ph3-synthetic"
FIXTURE = "unrelated-random-field-v1"
PROTOCOL = "ph3-synthetic-paired-field-pg094-v1"


def validate(options):
    allowed = {"fixture", "device", "base_precision", "steps", "batch", "seed", "ema_decay"}
    if set(options) - allowed:
        raise ValueError("unknown synthetic option: " + ", ".join(sorted(set(options) - allowed)))
    if options.get("fixture") != FIXTURE:
        raise ValueError("explicit unrelated-random-field-v1 fixture required")
    if options.get("device", "cpu") != "cpu" or options.get("base_precision", "float32") != "float32":
        raise UnsupportedOperation("the unrelated synthetic fixture supports CPU/float32 only")
    for name, default, minimum, maximum in (("steps", 4, 1, 16), ("batch", 2, 2, 4), ("seed", 7, 0, 2**31 - 1)):
        if type(options.get(name, default)) is not int or not minimum <= options.get(name, default) <= maximum:
            raise ValueError(f"{name} exceeds the bounded CPU fixture budget")
    decay = options.get("ema_decay")
    if decay is not None and (type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay < 1):
        raise ValueError("EMA decay must be finite and in [0, 1)")


def preflight(options, request, staged):
    if staged or request["native"]:
        raise UnsupportedOperation("the synthetic factory accepts no staged model or native request")


class SyntheticFieldResponse(FrozenResponseAdapter):
    """Shared paired-teacher preparation over an unrelated random Linear field."""

    family = "ph3-synthetic"
    time_coordinates = "dimensionless synthetic time; not an H3 scheduler time"

    def __init__(self, fixture):
        self.fixture = fixture
        named = [(name, parameter) for name, parameter in fixture.named_parameters() if parameter.requires_grad]
        network = AdapterParameters(named, fixture.scaled)
        base_bytes = b"".join(value.detach().numpy().tobytes() for value in fixture.host.state_dict().values())
        super().__init__(network, model_identity={
            "fixture": FIXTURE, "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "native_h3": False,
        })

    def _velocity(self, condition, latent, timestep):
        # The four-dimensional field is the fixture's actual forward output.
        # No native H3 response is projected or reshaped into this geometry.
        if tuple(latent.shape) != (1, 8, 2, 2) or tuple(condition.shape) != (1, 8):
            raise ValueError("unrelated fixture requires its own 1x8x2x2 field and 1x8 context")
        if timestep.numel() != 1 or not 0 <= float(timestep) <= 1:
            raise ValueError("synthetic time must be a scalar in [0, 1]")
        response = self.fixture(latent.movedim(1, -1)).movedim(-1, 1)
        return response + condition[:, :, None, None] + timestep * .01

    def metadata(self):
        return {**super().metadata(), "protocol": PROTOCOL, "no_native_model": True,
                "response_domain": "continuous_field", "fixture_shape": [1, 8, 2, 2],
                "checkpoint_scale_capture": "not used; synthetic host has no activation checkpointing"}


def prepare(options, **staged):
    validate(options)
    if staged:
        raise UnsupportedOperation("no model can be staged into the synthetic factory")
    fixture = None
    try:
        seed = options.get("seed", 7)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            seed_initialization_rng(seed, [])
            fixture = SyntheticProjection(seed, rank=8)
            response = SyntheticFieldResponse(fixture)
            generator = torch.Generator(device="cpu").manual_seed(seed + 100)
            rows, conditions = [], []
            for index in range(2):
                neutral = torch.linspace(-.1, .1, 8, device="cpu", dtype=torch.float32)[None] + index * .05
                latent = torch.randn(1, 8, 2, 2, generator=generator, device="cpu", dtype=torch.float32)
                time = torch.tensor(.25 + index * .5, device="cpu", dtype=torch.float32)
                # The shared adapter evaluates both teachers with strength zero
                # at exactly the same state/time; only conditioning differs.
                rows.append(response.prepare(ResponseRow(neutral=neutral, positive=neutral + .2,
                    negative=None, latent=latent, timestep=time, scale=1., strength=1., guidance=1.)))
                conditions.append(torch.cat((neutral.squeeze(0), time[None], time.new_ones(1))))
            critic = FieldCritic(8, 10, width=8).float()
            game = ConditionalPG09(response, rows, critic, conditions=torch.stack(conditions),
                particles=fixture.particles, response_domain="continuous_field",
                preset="bispf_optimized", sampling="balanced", batch_size=options.get("batch", 2),
                seed=seed, total_steps=options.get("steps", 4), response_scale=1., sigma=1.)
        return Binding(game, fixture.close)
    except BaseException:
        if fixture is not None:
            fixture.close()
        raise


BACKEND = Backend(NAME, Capabilities(
    generator_arm="routed_particle_gan", render=False,
    base_precisions=("float32",), render_outputs=(),
    operations=("train", "resume", "evaluate", "export", "response"),
    qualification="Unrelated synthetic CPU field only; no native H3 model or image qualification",
    unavailable_reasons=(("render", "synthetic numerical responses have no image sampler or decoder"),),
), prepare, validate, preflight=preflight)


def check(steps=2):
    """Bounded shared GAN updates; return metrics, never save model artifacts."""
    from particle_sliders.lifecycle import SessionConfig, open_session

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        config = SessionConfig(NAME, "unrelated-synthetic-law-check-v1", fixture=FIXTURE, steps=steps)
        with open_session(config) as session:
            metrics = session.train()
            return {
                "status": "passed", "device": "cpu", "fixture": FIXTURE,
                "engine": type(session.game).__module__ + "." + type(session.game).__name__,
                "objective": session.game._signature["objective"],
                "particle_shape": list(session.game.prior.z.shape),
                "completed_steps": session.game.completed_steps,
                "last_metrics": metrics, "auxiliary_losses": [],
                "native_h3_qualified": False, "model_weights_loaded": False,
                "artifacts_written": False,
            }
    finally:
        torch.set_num_threads(previous_threads)
