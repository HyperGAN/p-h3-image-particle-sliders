"""Backend parity, autograd and frozen-base ownership checks; CPU only."""
from importlib.util import module_from_spec, spec_from_file_location
import os
import hashlib
from pathlib import Path
import socket

import pytest
import torch
from torch import nn

from ph3_hndl import RoutedAdapter, RoutedProjection, GlobalMixCritic, runtime_identity


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network/GPU access attempted")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    torch.set_num_threads(2)


@pytest.fixture
def reference():
    path = os.getenv("PH3_REFERENCE_SOURCE")
    if not path:
        pytest.skip("set PH3_REFERENCE_SOURCE to the audited Anima reference.py for backend parity")
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == "d35963f072c4cd4acc92d255af5d3ea32535d2deb3d4a3c18625d85f30c42de3"
    spec = spec_from_file_location("anima_reference", Path(path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_linear(destination, source):
    destination.load_state_dict(source.state_dict())


def test_routed_native_graph_matches_actual_reference_values_and_gradients(reference):
    torch.manual_seed(61)
    classic = reference.RoutedMLP(8, 8, z_dim=4, width=48, router_width=16)
    down, up = nn.Linear(5, 8, bias=False), nn.Linear(8, 7, bias=False)
    native = RoutedProjection(nn.Linear(5, 7))
    copy_linear(native.graph["down"], down)
    copy_linear(native.graph["up"], up)  # nonzero for meaningful complete gradients
    for i in range(4):
        copy_linear(native.graph[f"router{i}"], classic.router[i * 2])
        copy_linear(native.graph[f"net{i}"], classic.net[i * 2])
    value = torch.randn(2, 3, 5, requires_grad=True)
    particles = torch.randn(128, 4, requires_grad=True)
    actual = native(value, particles)
    expected = up(classic(down(value), particles))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    actual_grads = torch.autograd.grad(actual.square().sum(), (value, particles))
    expected_grads = torch.autograd.grad(expected.square().sum(), (value, particles))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)


def test_gmix_native_graph_matches_actual_reference_and_second_derivatives(reference):
    torch.manual_seed(67)
    classic = reference.GlobalMixErrorCritic(torch.randn(3, 12), tokens=4, width=8, layers=2, heads=2)
    native = GlobalMixCritic(12, tokens=4, width=8, layers=2, heads=2)
    copy_linear(native.graph["mix"], classic.mix)
    native.graph["pos"].weight.data.copy_(classic.pos[0])
    for i, block in enumerate(classic.blocks):
        for label in ("norm1", "norm2", "qkv", "out"):
            copy_linear(native.graph[f"{label}_{i}"], getattr(block, label))
        copy_linear(native.graph[f"ffn1_{i}"], block.ffn[0])
        copy_linear(native.graph[f"ffn2_{i}"], block.ffn[2])
    copy_linear(native.graph["out_norm"], classic.out_norm)
    copy_linear(native.graph["head"], classic.head)
    value = torch.randn(3, 12, requires_grad=True)
    actual, expected = native(value), classic(value)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    first = [torch.autograd.grad(y.sum(), value, create_graph=True)[0] for y in (actual, expected)]
    torch.testing.assert_close(*first, rtol=3e-5, atol=1e-6)
    second = [torch.autograd.grad(g.square().sum(), value)[0] for g in first]
    torch.testing.assert_close(*second, rtol=8e-5, atol=2e-5)


def test_actual_hndl_hooks_train_all_groups_and_leave_frozen_host_identical():
    torch.manual_seed(71)
    host = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 5)).requires_grad_(False)
    state = {k: v.clone() for k, v in host.state_dict().items()}
    unrelated = host[0].register_forward_hook(lambda module, args, output: output)
    adapter = RoutedAdapter(host, ["0", "2"])
    value = torch.randn(2, 4, 5)
    expected = host(value).detach()
    critic = GlobalMixCritic(40, tokens=2, width=8, heads=2)
    owned = list(adapter.parameters())
    assert len({id(p) for p in owned}) == len(owned)
    assert not {id(p) for p in host.parameters()} & {id(p) for p in owned}
    branches = [p for n, p in adapter.named_parameters() if n != "particles"]
    optimizer = torch.optim.Adam([{"params": branches, "lr": 1e-3},
                                  {"params": [adapter.particles], "lr": 6e-3},
                                  {"params": list(critic.parameters()), "lr": 9e-4}])
    initial = adapter.particles.detach().clone()
    try:
        with adapter.scaled(1):
            assert torch.equal(host(value), expected)  # zero up initialization
            for _ in range(3):
                optimizer.zero_grad()
                result = host(value)
                loss = critic((result - .3).reshape(1, -1)).square().mean()
                loss.backward()
                optimizer.step()
            assert not torch.equal(host(value), expected)
        assert not torch.equal(adapter.particles, initial)
        assert adapter.particles.grad is not None and torch.isfinite(adapter.particles.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branches)
        assert all(p.grad is None for p in host.parameters())
        assert all(torch.equal(v, host.state_dict()[k]) for k, v in state.items())
        with adapter.scaled(0):
            assert torch.equal(host(value), expected)
        identity = runtime_identity()
        assert identity["commit"] == "df5af35d3b4017e1164b011d12264a01546a3ff4"
        assert type(adapter.branches[0].graph).__module__ == "hndl.torch"
        assert type(critic.graph).__module__ == "hndl.torch"
    finally:
        adapter.close()
    assert torch.equal(host(value), expected)
    assert unrelated.id in host[0]._forward_hooks
    unrelated.remove()
    with pytest.raises(RuntimeError, match="closed"):
        with adapter.scaled(1):
            pass


def test_full_response_critic_double_backward_and_geometry_validation():
    critic = GlobalMixCritic(88, tokens=2, width=8, heads=2)
    value = torch.randn(2, 88, requires_grad=True)
    gradient = torch.autograd.grad(critic(value).sum(), value, create_graph=True)[0]
    gradient.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters())
    with pytest.raises(ValueError, match="full normalized"):
        critic(value[:, :-1])
    with pytest.raises(ValueError, match="divisible"):
        GlobalMixCritic(88, width=7, heads=4)


def test_inference_artifact_roundtrip_off_rng_and_failed_load_cleanup(tmp_path):
    from ph3_hndl import export_adapter, load_adapter
    host = nn.Sequential(nn.Linear(5, 7)).requires_grad_(False)
    adapter = RoutedAdapter(host, ["0"])
    with torch.no_grad():
        adapter.branches[0].graph["up"].weight.normal_(std=.1)
    value = torch.randn(2, 3, 5)
    off = host(value)
    with adapter.scaled(1):
        expected = host(value).detach()
    identity = dict(model_revision="reduced-linear-v1", source="native-contract", base_precision="float32")
    export_adapter(adapter, tmp_path / "adapter", identity=identity, completed_steps=3)
    adapter.close()
    rng = torch.get_rng_state().clone()
    loaded = load_adapter(host, ["0"], tmp_path / "adapter", identity=identity)
    try:
        assert torch.equal(torch.get_rng_state(), rng)
        with loaded.scaled(1):
            assert torch.equal(host(value), expected)
        with loaded.scaled(0):
            assert torch.equal(host(value), off)
    finally:
        loaded.close()
    assert not host[0]._forward_hooks
    with pytest.raises(ValueError, match="signature differs"):
        load_adapter(host, ["0"], tmp_path / "adapter", identity=dict(model_revision="different"))
    assert not host[0]._forward_hooks
    path = tmp_path / "adapter" / "adapter.safetensors"
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="weights changed"):
        load_adapter(host, ["0"], tmp_path / "adapter", identity=identity)
    assert not host[0]._forward_hooks
