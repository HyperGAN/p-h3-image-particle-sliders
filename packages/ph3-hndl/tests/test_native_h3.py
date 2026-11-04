"""Actual reduced H3 network execution, independent of classic GAN imports.

Run with the product src and pinned native Diffusers src on PYTHONPATH.
This proves attachment/autograd/inference wiring, never pretrained quality.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from ph3_hndl import RoutedAdapter, GlobalMixCritic, export_adapter, load_adapter, runtime_identity


def test_reduced_native_h3_full_image_audio_adapter_and_inference(tmp_path, monkeypatch):
    if not os.getenv("PH3_NATIVE_TEST"):
        pytest.skip("set PH3_NATIVE_TEST=1 with the pinned native Diffusers/product src")
    from diffusers import MiniMaxH3Transformer3DModel
    from ph3_sliders.native_host import H3Host
    from ph3_sliders.native_cache import tensor_hash
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA initialization attempted")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    torch.set_num_threads(2)
    torch.manual_seed(79)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.requires_grad_(False).eval()
    model.enable_gradient_checkpointing()
    base_hash = tensor_hash(model.state_dict())
    host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    neutral = host.conditioning("neutral", torch.randn(1, 4, 8))
    positive = host.conditioning("positive", torch.randn(1, 5, 8))
    adapter = host.adapter = RoutedAdapter(model, host.targets)
    rows = host.collect_pair(neutral, positive, scene="cup", seed=17)
    assert len(host.targets) == 16 and len(rows) == 4
    row = rows[0]
    def residual():
        velocity = host.velocity(row["state"], neutral, 0)
        return host.response(velocity) - row["positive"]
    before = adapter.particles.detach().clone()
    critic = GlobalMixCritic(88, tokens=2, width=8, heads=2)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.003)
    try:
        # Direct differentiability probe, separate from the training objective
        # implemented and tested by the owning HyperGAN runtime package.
        with adapter.scaled(1):
            for _ in range(3):
                optimizer.zero_grad()
                result = residual()
                assert result.shape == (1, 88)  # all64 image + all24 audio
                result.square().mean().backward()
                optimizer.step()
            expected = host.response(host.velocity(row["state"], neutral, 0)).detach()
            scores = critic(residual())
            scores.sum().backward()
        assert not torch.equal(before, adapter.particles)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        assert all(p.grad is None for p in model.parameters())
        assert tensor_hash(model.state_dict()) == base_hash
        assert not torch.equal(expected, row["neutral"])
        assert not torch.equal(expected[:, 64:], row["neutral"][:, 64:])
        with adapter.scaled(0):
            assert torch.equal(host.response(host.velocity(row["state"], neutral, 0)), row["neutral"])
        identity = dict(reduced_random_h3=True, base_sha256=base_hash,
                        diffusers_revision="35087a644038eb531f48581708d4763044cb4852",
                        base_precision="float32")
        receipt = export_adapter(adapter, tmp_path / "hndl", identity=identity, completed_steps=3)
    finally:
        adapter.close()
    host.adapter = load_adapter(model, host.targets, tmp_path / "hndl", identity=identity)
    try:
        with host.adapter.scaled(1):
            assert torch.equal(host.response(host.velocity(row["state"], neutral, 0)), expected)
        off_state = host.rollout(neutral, 17, strength=0)
        # Exact OFF path equals unadapted native model, including coupled audio.
        host.adapter.close(); host.adapter = None
        plain_state = host.rollout(neutral, 17, strength=0)
        assert torch.equal(off_state.image, plain_state.image)
        assert torch.equal(off_state.audio, plain_state.audio)
    finally:
        if host.adapter is not None:
            host.adapter.close()
    if os.getenv("PH3_HNDL_RECEIPT"):
        Path(os.environ["PH3_HNDL_RECEIPT"]).write_text(json.dumps(dict(
            schema="ph3-hndl-native-cpu-verification-v1", runtime=runtime_identity(),
            reduced_random_h3=True, pretrained_quality_claim=False,
            native_targets=len(host.targets), teacher_rows=len(rows), full_response_width=88,
            image_coordinates=64, audio_coordinates=24, differentiability_probe_updates=3,
            actual_hndl_graphs=len(host.targets)+1, frozen_model_unchanged=True,
            particle_bank_changed=True, exact_off=True, raw_inference_roundtrip_exact=True,
            artifact_schema=receipt["schema"], artifact_signature_sha256=hashlib.sha256(
                json.dumps(receipt["signature"], sort_keys=True).encode()).hexdigest(),
            adapter_binding_sha256=receipt["signature"]["binding_sha256"],
            critic_identity=critic.identity), indent=2) + "\n")
