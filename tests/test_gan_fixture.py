from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, backend
from particle_sliders.pg09 import ConditionalPG09

from ph3_sliders.gan_fixture import BACKEND, FIXTURE, NAME, check, validate


def config(**options):
    return SessionConfig(NAME, "synthetic-pg094-tests", fixture=FIXTURE, **options)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def test_registered_shared_game_and_pure_loss_updates():
    assert backend(NAME) is BACKEND
    with BACKEND.open(config(steps=2)) as session:
        game = session.game
        assert isinstance(game, ConditionalPG09)
        assert game._signature["objective"] == "g_adv+public_prior;d_adv+public_penalty"
        assert game.response_domain == "continuous_field"
        assert game.response_shape == (1, 8, 2, 2)
        fixture = game.backend.fixture
        assert fixture.branch.rank == 8
        assert game.prior.z is fixture.particles
        assert game.network.particles is fixture.particles
        assert tuple(game.prior.z.shape) == (128, 4)
        base = deepcopy(fixture.host.state_dict())
        before = deepcopy(game.network.state_dict())
        teacher = [(row.target.clone(), row.neutral.clone()) for row in game.rows]
        stats = session.train()
        assert stats["g_loss"] == pytest.approx(stats["g_adv"] + stats["prior_regularization"], abs=1e-6)
        assert stats["d_loss"] == pytest.approx(stats["d_adv"] + stats["d_pen"], abs=1e-6)
        assert game.completed_steps == 2
        assert any(not torch.equal(before[name], value) for name, value in game.network.state_dict().items())
        assert not torch.equal(before["weights.0"], game.network.state_dict()["weights.0"])
        same(base, fixture.host.state_dict())
        for row, (positive, neutral) in zip(game.rows, teacher):
            assert torch.equal(row.target, positive) and torch.equal(row.neutral, neutral)
            assert not row.target.requires_grad and not row.neutral.requires_grad
            assert torch.allclose(row.target - row.neutral, torch.full_like(row.target, .2), atol=1e-7)
        off = session.evaluate(strength=0)
        full = session.evaluate(strength=1)
        half = session.evaluate(strength=.5)
        assert torch.equal(off, game.rows[0].neutral)
        assert not torch.equal(off, full)
        assert torch.allclose(half - off, .5 * (full - off), atol=1e-7)
    assert fixture.closed
    assert "forward" not in fixture.host.__dict__


def test_exact_split_resume_and_export():
    selected = config(steps=4, ema_decay=.9)
    with BACKEND.open(selected) as full:
        full.train()
        expected_game = full.game.state_dict()
        expected_ema = full.ema.state_dict()
    with BACKEND.open(selected) as partial:
        partial.train(stop_after=2)
        packet = partial.checkpoint()
    with BACKEND.open(selected) as resumed:
        resumed.restore(packet)
        resumed.train()
        same(expected_game, resumed.game.state_dict())
        same(expected_ema, resumed.ema.state_dict())
        exported = resumed.export(selection="ema")
        assert exported["schema"].endswith("-inference")
        assert "game" not in exported["payload"]
        with pytest.raises(CheckpointError):
            resumed.restore(exported)


def test_paired_prompt_changes_teacher_but_not_neutral_student():
    with BACKEND.open(config()) as session:
        response = session.game.backend
        original = session.game.rows[0]
        changed = response.prepare(replace(original.row, positive=original.row.positive + .1))
        assert torch.equal(original.neutral, changed.neutral)
        assert torch.allclose(changed.target - original.target, torch.full_like(changed.target, .1), atol=1e-7)
        assert torch.equal(response.generated(original), response.generated(changed))


@pytest.mark.parametrize("options", [
    {"fixture": "ph3-image"}, {"device": "cuda:0"}, {"base_precision": "bfloat16"},
    {"allow_hub": True}, {"model_dir": "private-model"}, {"mse_weight": 0.1},
    {"perceptual_weight": 0.1}, {"recipe_overrides": {"lr": 1e-4}},
    {"steps": True}, {"steps": 17}, {"batch": 1}, {"seed": -1},
])
def test_fixture_refuses_models_auxiliary_knobs_and_unbounded_work(options):
    with pytest.raises((ValueError, UnsupportedOperation)):
        validate({"fixture": FIXTURE} | options)


def test_no_staged_runtime_or_media():
    with pytest.raises(UnsupportedOperation, match="no staged model"):
        BACKEND.open(config(), runtime=object())
    with pytest.raises(UnsupportedOperation, match="no image sampler"):
        BACKEND.open(config(), operation="render")


def test_failed_observer_closes_owned_fixture():
    run = BACKEND.open(config())
    fixture = run.game.backend.fixture
    def fail(metrics):
        raise RuntimeError("observer failed")
    with pytest.raises(RuntimeError, match="observer failed"):
        run.train(stop_after=1, on_step=fail)
    assert run.closed and fixture.closed


def test_deterministic_creation_and_training_preserve_cpu_rng():
    state = torch.get_rng_state().clone()
    with BACKEND.open(config(steps=2)) as session:
        assert torch.equal(state, torch.get_rng_state())
        session.train()
        assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(state, torch.get_rng_state())


def test_gan_check_leaves_no_artifact_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    report = check(steps=2)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert before == after
    assert report["status"] == "passed"
    assert report["engine"] == "particle_sliders.pg09.ConditionalPG09"
    assert report["completed_steps"] == report["last_metrics"]["step"] == 2
    assert report["objective"] == "g_adv+public_prior;d_adv+public_penalty"
    assert report["particle_shape"] == [128, 4]
    assert report["auxiliary_losses"] == []
    assert not report["artifacts_written"]
    assert not report["model_weights_loaded"]
    assert not report["native_h3_qualified"]
