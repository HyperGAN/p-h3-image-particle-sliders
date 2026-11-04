"""CPU-only full-output/all-adapter-gradient equivalence on actual H3+HNDL.

No pretrained weight, BnB kernel,
network access or CUDA allocation is involved; this verifies the checkpoint
algorithm and actual hook-bound gradient paths rather than full GPU fit.
"""

from copy import deepcopy
import socket

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
RoutedAdapter = pytest.importorskip("ph3_hndl").RoutedAdapter
import ph3_sliders.native_checkpointing as checkpointing
from ph3_sliders.native_host import H3Host
from ph3_sliders.native_cache import tensor_hash


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint equivalence attempted network or CUDA")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()


def execute(enabled):
    torch.manual_seed(47)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.requires_grad_(False).eval()
    if enabled:
        assert checkpointing.enable_native_checkpointing(model) == checkpointing.MODE
    else:
        model.disable_gradient_checkpointing()
    host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    context = host.conditioning("fixed-native-context", torch.randn(1, 5, 8))
    host.adapter = RoutedAdapter(model, host.targets)
    # Production adapter up projections begin at zero, which hides router/down/
    # bank derivatives in an initial-step comparison. Populate EVERY adapter
    # leaf reproducibly, including all up/router weights and biases.
    generator = torch.Generator(device="cpu").manual_seed(173)
    with torch.no_grad():
        for name, parameter in host.adapter.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) *
                            (1. if name == "particles" else .15))
    initial = deepcopy(host.adapter.state_dict())
    base = tensor_hash(model.state_dict())
    state = host.initial_state(29)
    events = {"refiner_first": [], "main_first": []}
    stage = ["forward"]
    def observe(label):
        def hook(module, args):
            events[label].append(dict(stage=stage[0], grad_enabled=torch.is_grad_enabled(),
                                     input_requires_grad=args[0].requires_grad))
        return hook
    handles = [model.token_refiner.refiner_blocks[0].register_forward_pre_hook(observe("refiner_first")),
               model.transformer_blocks[0].register_forward_pre_hook(observe("main_first"))]
    rng = torch.get_rng_state().clone()
    try:
        with host.adapter.scaled(1.):
            velocity = host.velocity(state, context, 1)
            # Both modalities participate with a nonuniform scalar cotangent;
            # no critic/normalization can hide one output or gradient branch.
            image_weights = torch.linspace(-.7, 1.1, velocity.image.numel()).reshape_as(velocity.image)
            audio_weights = torch.linspace(.9, -.4, velocity.audio.numel()).reshape_as(velocity.audio)
            loss = (velocity.image * image_weights).mean() + (velocity.audio * audio_weights).mean()
            loss = loss + .13 * velocity.image.square().mean() + .21 * velocity.audio.square().mean()
            stage[0] = "backward"
            loss.backward()
        assert host.adapter.multiplier == 0.
        assert tensor_hash(model.state_dict()) == base
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        assert torch.equal(torch.get_rng_state(), rng)
        gradients = {}
        for name, parameter in host.adapter.named_parameters():
            assert parameter.grad is not None, name + " gradient missing"
            assert torch.isfinite(parameter.grad).all(), name + " nonfinite gradient"
            assert parameter.grad.abs().max() > 0, name + " entire gradient is zero"
            gradients[name] = parameter.grad.detach().clone()
        assert not torch.cuda.is_initialized()
        return dict(initial=initial, base=base, image=velocity.image.detach().clone(),
                    audio=velocity.audio.detach().clone(), loss=loss.detach().clone(),
                    gradients=gradients, events=events, targets=list(host.targets))
    finally:
        for handle in handles:
            handle.remove()
        host.adapter.close()


def test_actual_h3_hndl_reentrant_outputs_all_gradients_and_recompute_modes():
    plain = execute(False)
    replay = execute(True)
    assert plain["base"] == replay["base"]
    assert plain["targets"] == replay["targets"] and len(plain["targets"]) == 16
    for name in plain["initial"]:
        assert torch.equal(plain["initial"][name], replay["initial"][name]), name
    for output in ("image", "audio", "loss"):
        assert torch.equal(plain[output], replay[output]), output
    assert plain["gradients"].keys() == replay["gradients"].keys()
    differences = {}
    for name, gradient in plain["gradients"].items():
        actual = replay["gradients"][name]
        torch.testing.assert_close(actual, gradient, rtol=2e-5, atol=1e-7, msg=name)
        differences[name] = float((actual - gradient).abs().max())
    for label in ("refiner_first", "main_first"):
        assert plain["events"][label] == [dict(stage="forward", grad_enabled=True,
            input_requires_grad=(label == "main_first"))]
        assert replay["events"][label] == [
            dict(stage="forward", grad_enabled=False, input_requires_grad=True),
            dict(stage="backward", grad_enabled=True, input_requires_grad=True)]
