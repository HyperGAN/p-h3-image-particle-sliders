from contextlib import contextmanager
from copy import deepcopy
import socket

import pytest
import torch
from torch import nn

from ph3_hypergan import PairedResidualRun, require_runtime
from hypergan.training import ReferenceTrainer


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        pytest.fail("engine test attempted network/CUDA")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.linear = nn.Linear(4, 3)

    @contextmanager
    def scaled(self, value):
        yield self


def make_run():
    torch.manual_seed(11)
    adapter = Adapter()
    critic = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1))
    def predict(ids):
        return torch.stack([adapter.linear(adapter.particles.softmax(0).mean(0) + i * .01) - .2 for i in ids])
    return PairedResidualRun(adapter, critic, predict,
        dict(scale=torch.ones(2, 3), edit_rms=torch.ones(2)), [0, 1, 0, 1],
        identity=dict(fixture="unrelated-three-coordinate-cpu", no_pretrained_model=True), steps=8)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_actual_develop_executor_ownership_and_full_state_replay():
    run = make_run()
    assert isinstance(run, ReferenceTrainer)
    assert run._update.__func__ is ReferenceTrainer._update
    assert run.opt_g.__class__.__module__.startswith("particlegan")
    assert {id(p) for p in run.program.prior_parameters} == {id(run.adapter.particles)}
    assert id(run.adapter.particles) not in {id(p) for p in run.program.generator_parameters}
    initial = deepcopy(run.adapter.state_dict())
    first = run.update()
    assert first["grad_norm/prior"] > 0 and first["grad_norm/generator"] > 0
    assert not torch.equal(run.adapter.particles, initial["particles"])
    assert not torch.equal(run.adapter.linear.weight, initial["linear.weight"])
    saved = run.snapshot()
    run.update(); run.update()
    expected = run.snapshot()
    restored = make_run()
    restored.restore(saved)
    restored.update(); restored.update()
    equal(expected, restored.snapshot())
    assert run.step == 3
    weights = run.adapter_weights(selection="ema")
    assert set(weights) == set(run.adapter.state_dict())
    equal(weights["particles"], run.ema_prior.z)


def test_classic_and_mislabeled_cohort_rejected():
    run = make_run()
    packet = run.snapshot()
    packet["signature"]["cohort"] = "classic-anima-gmix"
    with pytest.raises(ValueError, match="cohort differs"):
        run.restore(packet)
    with pytest.raises(ValueError, match="cohort differs"):
        run.restore(dict(schema="ph3-native-full-training-state-v1"))
    assert require_runtime()["revision"] == "5f368a3770cf5bdae0448995c9f76e29af8049f7"


def test_partial_update_poison_and_explicit_ema_shadow():
    run = make_run()
    saved = run.snapshot()
    with pytest.raises(RuntimeError, match="EMA shadow"):
        run.ema_graph.models["generator"](torch.tensor([0]), torch.ones(1, 3), torch.zeros(1, 3))
    run.graph.models["generator"].predict = lambda ids: torch.full((len(ids), 3), float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        run.update()
    with pytest.raises(RuntimeError, match="partial"):
        run.snapshot()
    run.restore(saved)
    assert not run.failed_step
