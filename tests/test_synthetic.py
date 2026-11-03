import pytest
import torch

from ph3_sliders.synthetic import SyntheticProjection, check


def test_actual_shared_projection_and_backward():
    report = check()
    assert report["shared_router"] == "particle_sliders.reference.RoutedMLP"
    assert report["rank"] == 8
    assert report["finite_particle_gradients"]
    assert report["fractional_strength"]
    assert report["optimizer_steps"] == 0
    assert not report["model_weights_loaded"]


def test_rng_threads_unique_bank_and_ownership():
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    with SyntheticProjection() as fixture:
        assert torch.equal(state, torch.get_rng_state())
        assert [name for name, _ in fixture.named_parameters() if name.endswith("particles")] == ["adapter.particles"]
        assert sum(p is fixture.particles for p in fixture.parameters()) == 1
        assert not any(p.requires_grad for p in fixture.host.parameters())
        assert all(p.dtype == torch.float32 for p in fixture.parameters() if p.requires_grad)
    check()
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == threads


def test_nested_scale_exception_and_cleanup():
    with SyntheticProjection() as fixture:
        host = fixture.host
        original = host.forward
        assert len(host._forward_hooks) == 1
        with fixture.scaled(0.75):
            with pytest.raises(RuntimeError, match="deliberate"):
                with fixture.scaled(0):
                    raise RuntimeError("deliberate")
            assert fixture.adapter.multiplier == 0.75
        assert fixture.adapter.multiplier == 0
    assert "forward" not in host.__dict__
    assert not host._forward_hooks
    assert host.forward == original
    fixture.close()
    with pytest.raises(RuntimeError, match="closed"):
        fixture(torch.zeros(1, 8))


def test_cpu_float32_despite_caller_defaults():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            with SyntheticProjection() as fixture:
                assert all(p.device.type == "cpu" for p in fixture.parameters())
                assert all(p.dtype == torch.float32 for p in fixture.parameters())
            assert check()["device"] == "cpu"
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1, 2, True, "1"])
def test_bad_scale_leaves_fixture_untouched(strength):
    with SyntheticProjection() as fixture:
        with pytest.raises(ValueError):
            with fixture.scaled(strength):
                pytest.fail("invalid scale was accepted")
        assert fixture.adapter.multiplier == 0
