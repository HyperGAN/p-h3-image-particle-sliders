from copy import deepcopy

import pytest
import torch

from ph3_sliders.gmix_fixture import Config, SyntheticGmixRun
from ph3_sliders.learning_rates import LearningRates, assert_learning_rates, bind_learning_rates


def test_rates_bind_actual_groups_without_global_recipe_or_stream_mutation():
    from particle_sliders_anima_gmix import TRAINING_CONFIG
    original = deepcopy(TRAINING_CONFIG)
    with SyntheticGmixRun(Config(steps=2)) as run:
        streams = deepcopy(run.game.sampler.state_dict())
        rates = LearningRates(generator=6e-5)
        assert bind_learning_rates(run.game, rates) == dict(generator=6e-5, critic=9e-4, particles=6e-3)
        assert all(torch.equal(v, run.game.sampler.state_dict()["generators"][k])
                   for k, v in streams["generators"].items())
        run.step()
        assert_learning_rates(run.game, rates)
        with pytest.raises(ValueError, match="before updates"):
            bind_learning_rates(run.game, LearningRates())
    assert TRAINING_CONFIG == original
    with SyntheticGmixRun() as other:
        assert_learning_rates(other.game, LearningRates())


def test_optimizer_checkpoint_restores_rates_and_recipe_mismatch_is_rejected():
    rates = LearningRates(generator=6e-6)
    with SyntheticGmixRun(Config(steps=2)) as source:
        bind_learning_rates(source.game, rates)
        source.step()
        state = deepcopy(source.game.state_dict())
    with SyntheticGmixRun(Config(steps=2)) as target:
        target.game.load_state_dict(state)
        assert_learning_rates(target.game, rates)
        with pytest.raises(ValueError, match="differ"):
            assert_learning_rates(target.game, LearningRates())


def test_wrong_parameter_groups_are_rejected_before_any_rate_changes():
    with SyntheticGmixRun() as run:
        run.game.g.param_groups[1]["params"] = list(run.game.g.param_groups[0]["params"])
        before = [g["lr"] for g in run.game.g.param_groups]
        with pytest.raises(ValueError, match="ownership"):
            bind_learning_rates(run.game, LearningRates(generator=6e-5))
        assert [g["lr"] for g in run.game.g.param_groups] == before


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "0.001"])
def test_bad_rates(value):
    with pytest.raises(ValueError):
        LearningRates(generator=value)
