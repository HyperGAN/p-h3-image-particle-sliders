from copy import deepcopy

import pytest
import torch
from torch import nn

from ph3_sliders.adapter import RoutedAdapter
from ph3_sliders.gmix_fixture import Config, SyntheticGmixRun, check
from ph3_sliders.paired_cache import PairedCache


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


def test_released_source_runner_full_residuals_and_native_bank():
    with SyntheticGmixRun() as run:
        assert type(run.game).__module__ == "particle_sliders_anima_gmix.anima_game"
        assert run.game.adapter is run.adapter
        assert run.game.g.param_groups[1]["params"][0] is run.adapter.particles
        assert run.game.scale.shape == (8, 32)
        assert run.game.critic.mix.in_features == 32  # not projected into rank8
        assert run.game.critic.score_bound == 8
        assert run.game.microbatch == 1
        assert run.game.g.param_groups[0]["lr"] == 2e-5
        assert run.game.g.param_groups[1]["lr"] == .006
        assert run.game.d.param_groups[0]["lr"] == .0009
        base = deepcopy(run.fixture.host.state_dict())
        bank = run.adapter.particles.detach().clone()
        run.train()
        assert not torch.equal(bank, run.adapter.particles)
        assert run.last_metrics["particle_gan_grad_norm"] > 0
        equal(base, run.fixture.host.state_dict())
        assert run.last_metrics["step"] == 4
        off = run.evaluate(strength=0)
        full = run.evaluate(strength=1)
        half = run.evaluate(strength=.5)
        assert torch.equal(off, run.cache[0]["neutral"])
        assert not torch.equal(full, off)
        assert torch.allclose(half - off, .5 * (full - off), atol=1e-7)


def test_exact_full_split_resume_ema_and_export():
    selected = Config(steps=8)
    with SyntheticGmixRun(selected) as full:
        full.train()
        expected = full.snapshot()
    with SyntheticGmixRun(selected) as partial:
        partial.train(stop_after=4)
        packet = partial.snapshot()
    with SyntheticGmixRun(selected) as resumed:
        resumed.restore(packet)
        resumed.train()
        equal(expected, resumed.snapshot())
        before = resumed.snapshot()
        response = resumed.evaluate(selection="ema", strength=.5)
        assert response.shape == (1, 8, 2, 2)
        equal(before, resumed.snapshot())
        exported = resumed.export()
        assert exported["selection"] == "ema"
        equal(exported["adapter"], resumed.ema)
        assert "game" not in exported
        with pytest.raises(ValueError, match="source/cohort"):
            resumed.restore(exported)
        wrong = deepcopy(packet); wrong["signature"]["core_revision"] = "wrong"
        with pytest.raises(ValueError, match="source/cohort"):
            resumed.restore(wrong)
        equal(before, resumed.snapshot())


def test_frozen_cache_ownership_and_position_contract():
    with SyntheticGmixRun(Config(positions=10)) as run:
        records = [run.cache[i] for i in range(len(run.cache))]
        cache = PairedCache(records, positions=10, identity={"split": "train"})
        fingerprint = cache.index["fingerprint"]
        records[0]["positive"].add_(100)
        read = cache[0]; read["positive"].zero_()
        assert cache.index["fingerprint"] == fingerprint
        assert torch.equal(cache[0]["positive"], run.cache[0]["positive"])
        assert cache.normalization()["scale"].shape == (10, 32)
        with pytest.raises(ValueError, match="training"):
            PairedCache(records, positions=10, identity={"split": "dev"})
        records = list(reversed(records))
        with pytest.raises(ValueError, match="source order"):
            PairedCache(records, positions=10, identity={"split": "train"})


def test_projection_attachment_preserves_unrelated_hooks():
    host = nn.Linear(8, 8).requires_grad_(False)
    other = host.register_forward_hook(lambda module, args, value: value)
    original = set(host._forward_hooks)
    adapter = RoutedAdapter(host, ("",))
    assert len(host._forward_hooks) == 2
    assert not {id(p) for p in host.parameters()} & {id(p) for p in adapter.parameters()}
    adapter.close(); adapter.close()
    assert set(host._forward_hooks) == original
    other.remove()


def test_failed_attachment_removes_only_owned_hooks(monkeypatch):
    host = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).requires_grad_(False)
    existing = host[0].register_forward_hook(lambda module, args, value: value)
    before = set(host[0]._forward_hooks)
    def fail(hook):
        raise RuntimeError("registration failed")
    monkeypatch.setattr(host[1], "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        RoutedAdapter(host, ("0", "1"))
    assert set(host[0]._forward_hooks) == before
    existing.remove()


@pytest.mark.parametrize("arguments", [dict(steps=True), dict(steps=17), dict(steps=0),
    dict(positions=1), dict(positions=11), dict(seed=-1), dict(seed=True)])
def test_bounded_fixture_config(arguments):
    with pytest.raises(ValueError):
        Config(**arguments)


def test_no_external_model_or_auxiliary_knobs():
    with pytest.raises(ValueError):
        SyntheticGmixRun({"model_dir": "model"})
    with pytest.raises(TypeError):
        Config(mse_weight=.1)
    with pytest.raises(TypeError):
        Config(perceptual_weight=.1)


def test_check_no_artifacts_and_cpu_rng(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    before = set(tmp_path.rglob("*"))
    report = check()
    assert set(tmp_path.rglob("*")) == before
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    assert report["engine"] == "particle_sliders_anima_gmix.anima_game.Game"
    assert report["completed_steps"] == 4 and report["response_width"] == 32
    assert report["auxiliary_losses"] == [] and report["ema_decay"] == .995
    assert not report["artifacts_written"] and not report["native_h3_qualified"]
    assert not report["published_run_parity"]
