"""Exercise the pinned shared routed projection on an unrelated random Linear.

No H3 implementation, model configuration, pretrained weights, GAN objective,
optimizer, or image sampler is constructed by this software check.
"""

import math
from contextlib import contextmanager

import torch
from torch import nn
from particle_sliders.routed_projection import TinyParticleProjection


class SyntheticProjection(nn.Module):
    """Owned, reversible fixture attachment using the actual shared branch."""

    def __init__(self, seed: int = 7, *, rank: int = 4):
        super().__init__()
        self.closed = False
        # Local fixture construction must not consume the caller's CPU RNG.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.host = nn.Linear(8, 8, device="cpu", dtype=torch.float32).requires_grad_(False)
            self.particles = nn.Parameter(torch.randn(128, 4, device="cpu", dtype=torch.float32))
            self._forward_was_local = "forward" in self.host.__dict__
            self._original_forward = self.host.forward
            self.branch = TinyParticleProjection("synthetic_linear", self.host, self.particles,
                                                 rank, float(rank)).float()

    @contextmanager
    def scaled(self, strength):
        if self.closed:
            raise RuntimeError("synthetic fixture is closed")
        if (type(strength) not in (float, int) or not math.isfinite(strength)
                or not 0 <= strength <= 1):
            raise ValueError("fixture strength must be finite and in [0, 1]")
        previous = self.branch.multiplier
        self.branch.multiplier = float(strength)
        try:
            yield self
        finally:
            self.branch.multiplier = previous

    def forward(self, value):
        if self.closed:
            raise RuntimeError("synthetic fixture is closed")
        return self.host(value)

    def close(self):
        if self.closed:
            return
        if self._forward_was_local:
            self.host.forward = self._original_forward
        else:
            del self.host.forward
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def check():
    """CPU forward/backward probe; no optimizer updates or saved adapters."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with SyntheticProjection() as fixture:
            value = torch.linspace(-1, 1, 24, device="cpu", dtype=torch.float32).reshape(1, 3, 8)
            with fixture.scaled(0):
                base = fixture(value).detach()
            with fixture.scaled(1):
                fresh = fixture(value)
            if not torch.equal(base, fresh):
                raise RuntimeError("fresh shared projection changed the synthetic base")
            # Open the zero-initialized up gate with fixed synthetic values so
            # the probe exercises router/particle gradients without training.
            with torch.no_grad():
                fixture.branch.lora_up.weight.fill_(0.01)
            with fixture.scaled(1):
                full = fixture(value)
                full.square().sum().backward()
            with fixture.scaled(0.5):
                half = fixture(value)
            if not torch.allclose(half - base, (full.detach() - base) * 0.5, atol=1e-7):
                raise RuntimeError("fractional projection strength failed")
            gradient = fixture.particles.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise RuntimeError("shared particle bank did not participate in backward")
            return {
                "status": "passed", "device": "cpu", "fixture": "unrelated random 8x8 Linear",
                "shared_projection": type(fixture.branch).__module__ + "." + type(fixture.branch).__name__,
                "particle_shape": list(fixture.particles.shape),
                "base_identity_at_init": True, "fractional_strength": True,
                "finite_particle_gradients": True, "model_weights_loaded": False,
                "optimizer_steps": 0, "native_h3_qualified": False,
            }
    finally:
        torch.set_num_threads(previous_threads)
