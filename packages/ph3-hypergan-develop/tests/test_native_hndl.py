"""Reduced real H3/HNDL/develop CPU proof, without pretrained quality claims."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("ph3_hndl")
pytest.importorskip("ph3_sliders.native_host")

from diffusers import MiniMaxH3Transformer3DModel
from ph3_hndl import RoutedAdapter, GlobalMixCritic, runtime_identity
from ph3_hypergan import NativeHyperganRun, fit_normalization
from ph3_sliders.native_cache import NativeCache, tensor_hash
from ph3_sliders.native_host import H3Host

from test_engine import cpu_only, equal


def make_native():
    torch.manual_seed(47)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.requires_grad_(False).eval().enable_gradient_checkpointing()
    host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    neutral = host.conditioning("neutral", torch.randn(1, 4, 8))
    positive = host.conditioning("positive", torch.randn(1, 5, 8))
    host.adapter = RoutedAdapter(model, host.targets)
    rows = host.collect_pair(neutral, positive, scene="cup", seed=17)
    cache = NativeCache(rows, {"neutral": neutral, "positive": positive}, positions=2,
        expected_examples=[("cup", 17, "neutral"), ("cup", 17, "positive")],
        identity=dict(split="train", reduced_random_h3=True))
    width = rows[0]["positive"].numel()
    normalization = fit_normalization(cache, positions=2)
    critic = GlobalMixCritic(width)
    run = NativeHyperganRun(host, cache, normalization, critic, batch_size=1,
        identity=dict(reduced_random_h3=True,
                      hndl=runtime_identity()), steps=8)
    return host, cache, run


def test_native_h3_hndl_develop_full_response_update_resume_and_off(tmp_path):
    host, cache, run = make_native()
    try:
        base = tensor_hash(host.model.state_dict())
        initial = deepcopy(host.adapter.state_dict())
        assert len(host.targets) == 16
        assert run.signature["full_response_width"] == 88  # 64 image + 24 coupled audio
        assert run.signature["cohort"].endswith("batch1")
        assert run.evaluate(cache, selection="raw")["mean_teacher_student_relative_error"] == 1.
        run.update(); run.update()
        assert not torch.equal(initial["particles"], host.adapter.particles)
        assert any(not torch.equal(initial[k], value) for k, value in host.adapter.state_dict().items()
                   if k != "particles")
        saved = run.snapshot()
        torch.save(saved, tmp_path / "state.pt")
        run.update(); run.update()
        expected = run.snapshot()
        fresh_host, fresh_cache, fresh_run = make_native()
        try:
            fresh_run.restore(torch.load(tmp_path / "state.pt", weights_only=True))
            fresh_run.update(); fresh_run.update()
            equal(expected, fresh_run.snapshot())
        finally:
            fresh_host.adapter.close()
        run.restore(expected)
        metrics = run.evaluate(cache, raw_path=tmp_path / "ema-responses.safetensors")
        assert metrics["mean_teacher_student_relative_error"] >= 0
        assert len(metrics["raw_student_responses"]["sha256"]) == 64
        equal(expected, run.snapshot())
        assert run.step == 4
        assert tensor_hash(host.model.state_dict()) == base
        assert all(p.grad is None for p in host.model.parameters())
        assert host.model.gradient_checkpointing
        assert host.adapter.multiplier == 0.
        row = cache[0]
        with torch.no_grad(), host.adapter.scaled(0.):
            off = host.response(host.velocity(row["state"], cache.contexts["neutral"], 0))
        assert torch.equal(off, row["neutral"])
        # EMA exported leaves execute through the original H3-owned hooks.
        raw = deepcopy(host.adapter.state_dict())
        host.adapter.load_state_dict(run.adapter_weights(selection="ema"))
        with torch.no_grad(), host.adapter.scaled(1.):
            student = host.response(host.velocity(row["state"], cache.contexts["neutral"], 0))
        assert student.shape == row["positive"].shape and torch.isfinite(student).all()
        host.adapter.load_state_dict(raw)
    finally:
        host.adapter.close()
