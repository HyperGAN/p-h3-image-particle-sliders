"""Unrelated CPU projection fixture using the released-Anima shared router."""

from contextlib import contextmanager
import torch
from torch import nn

from .adapter import RoutedAdapter


class SyntheticProjection(nn.Module):
    def __init__(self, seed=7):
        super().__init__()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.host = nn.Linear(8, 8, device="cpu", dtype=torch.float32).requires_grad_(False)
            self.adapter = RoutedAdapter(self.host, ("",))

    @property
    def particles(self):
        return self.adapter.particles

    @property
    def branch(self):
        return self.adapter.branches[0]

    @property
    def closed(self):
        return self.adapter.closed

    @contextmanager
    def scaled(self, strength):
        with self.adapter.scaled(strength):
            yield self

    def forward(self, value):
        if self.closed:
            raise RuntimeError("synthetic fixture is closed")
        return self.host(value)

    def close(self):
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def check():
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
                raise RuntimeError("fresh routed projection changed the synthetic base")
            with torch.no_grad():
                fixture.branch.up.weight.fill_(.01)
            with fixture.scaled(1):
                full = fixture(value)
                full.square().sum().backward()  # gradient probe, no optimizer update
            with fixture.scaled(.5):
                half = fixture(value)
            if not torch.allclose(half - base, (full.detach() - base) * .5, atol=1e-7):
                raise RuntimeError("fractional projection strength failed")
            gradient = fixture.particles.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise RuntimeError("shared particle bank did not participate in backward")
            return {"status": "passed", "device": "cpu", "fixture": "unrelated random 8x8 Linear",
                "shared_router": type(fixture.branch.routed).__module__ + "." + type(fixture.branch.routed).__name__,
                "particle_shape": list(fixture.particles.shape), "rank": fixture.branch.rank,
                "base_identity_at_init": True, "fractional_strength": True,
                "finite_particle_gradients": True, "optimizer_steps": 0,
                "model_weights_loaded": False, "native_h3_qualified": False}
    finally:
        torch.set_num_threads(previous_threads)
