"""Reduced actual H3 architecture checks; no pretrained or image-quality claim."""

from copy import deepcopy

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel

from ph3_sliders.adapter import RoutedAdapter
from ph3_sliders.learning_rates import LearningRates
from ph3_sliders.native_cache import NativeCache, tensor_hash
from ph3_sliders.native_host import H3Host, PackedState
from ph3_sliders.native_training import NativeTrainingRun


def host_and_cache(checkpointing=True):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(47)
        model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
            hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
            in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
            freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
        model.requires_grad_(False).eval()
        if checkpointing:
            model.enable_gradient_checkpointing()
        host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
        generator = torch.Generator().manual_seed(7)
        neutral = host.conditioning("neutral", torch.randn(1, 4, 8, generator=generator))
        positive = host.conditioning("positive", torch.randn(1, 5, 8, generator=generator))
        host.adapter = RoutedAdapter(model, host.targets)
        rows = host.collect_pair(neutral, positive, scene="cup", seed=17)
        cache = NativeCache(rows, {"neutral": neutral, "positive": positive}, positions=2,
                            expected_examples=[("cup", 17, "neutral"), ("cup", 17, "positive")],
                            identity=dict(split="train", reduced_random_h3=True))
    return host, cache


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_native_targets_packing_same_state_teachers_and_audio():
    host, cache = host_and_cache()
    try:
        assert len(host.targets) == 16
        assert cache[0]["positive"].numel() == 4 * 4 * 4 + 4 * 6
        assert torch.equal(cache[0]["state"].image, cache[2]["state"].image)
        assert torch.equal(cache[0]["state"].audio, cache[2]["state"].audio)
        assert not torch.equal(cache[1]["state"].image, cache[3]["state"].image)
        assert not torch.equal(cache[0]["state"].audio, cache[1]["state"].audio)
        image = host.unpack_image(cache[0]["state"])
        generator = torch.Generator().manual_seed(17)
        assert torch.equal(image, torch.randn(1, 4, 1, 4, 4, generator=generator))
        with torch.no_grad(), host.adapter.scaled(0):
            for row in cache.records:
                for name, field in (("neutral", "neutral"), ("positive", "positive")):
                    actual = host.response(host.velocity(row["state"], cache.contexts[name], row["position"]))
                    assert torch.equal(actual, row[field])
    finally:
        host.adapter.close()


def test_native_scheduler_dataward_sign_and_audio_clock():
    host, _ = host_and_cache()
    try:
        state = PackedState(torch.zeros(4, 16), torch.zeros(4, 6))
        velocity = PackedState(torch.ones_like(state.image), torch.ones_like(state.audio) * 2)
        video_scheduler, audio_scheduler = host.scheduler(), host.scheduler()
        state = host.step(state, velocity, 0, video_scheduler, audio_scheduler)
        assert torch.equal(state.image, torch.full_like(state.image, .5))
        assert torch.equal(state.audio, torch.ones_like(state.audio))
        state = host.step(state, velocity, 1, video_scheduler, audio_scheduler)
        assert torch.equal(state.image, torch.ones_like(state.image))
        assert torch.equal(state.audio, torch.full_like(state.audio, 2.))
    finally:
        host.adapter.close()


def test_native_cache_file_roundtrip_and_content_binding(tmp_path):
    host, cache = host_and_cache()
    try:
        cache.save(tmp_path / "cache")
        loaded = NativeCache.load(tmp_path / "cache")
        assert loaded.index == cache.index
        assert_tree_equal(dict(loaded.tensors()), dict(cache.tensors()))
        with pytest.raises(FileExistsError):
            cache.save(tmp_path / "cache")
        path = tmp_path / "cache" / "responses.safetensors"
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with pytest.raises(ValueError, match="bytes changed"):
            NativeCache.load(tmp_path / "cache")
    finally:
        host.adapter.close()


def test_actual_h3_game_update_checkpoint_resume_and_off_identity(tmp_path):
    host, cache = host_and_cache()
    try:
        base = tensor_hash(host.model.state_dict())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(59)
            run = NativeTrainingRun(host, cache, identity=dict(reduced_random_h3=True))
        initial = run.snapshot()
        run.step(); run.step()
        run.save(tmp_path / "step-2")
        run.step(); run.step()
        expected = run.snapshot()
        run.load(tmp_path / "step-2")
        run.step(); run.step()
        assert_tree_equal(run.snapshot(), expected)
        assert run.completed_steps == 4
        assert tensor_hash(host.model.state_dict()) == base
        assert all(p.grad is None for p in host.model.parameters())
        assert not torch.equal(initial["adapter"]["particles"], run.adapter.particles)
        evaluation = run.evaluate(cache, selection="ema")
        assert evaluation["mean_teacher_student_relative_error"] >= 0
        assert_tree_equal(run.snapshot(), expected)
        with torch.no_grad(), host.adapter.scaled(0):
            row = cache[0]
            off = host.response(host.velocity(row["state"], cache.contexts["neutral"], 0))
        assert torch.equal(off, row["neutral"])
        assert host.model.gradient_checkpointing
        assert host.model.token_refiner.gradient_checkpointing
        mismatched = deepcopy(expected)
        mismatched["signature"]["rates"]["generator"] *= 2
        with pytest.raises(ValueError, match="cohort differs"):
            run.restore(mismatched)
    finally:
        host.adapter.close()


def test_checkpointed_and_plain_h3_adapter_gradients_match():
    results = []
    for checkpointing in (False, True):
        host, cache = host_and_cache(checkpointing)
        try:
            with host.adapter.scaled(1):
                row = cache[0]
                response = host.response(host.velocity(row["state"], cache.contexts["neutral"], 0))
                response.square().mean().backward()
            results.append({k: p.grad.clone() for k, p in host.adapter.named_parameters() if p.grad is not None})
            assert host.adapter.multiplier == 0.
        finally:
            host.adapter.close()
    assert_tree_equal(*results)


def test_host_rollout_exactly_matches_upstream_modular_pipeline():
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.p_h3_image.modular_blocks_p_h3_image import PH3ImageCoreDenoiseStep
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=24, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2).requires_grad_(False).eval()
    host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    context = host.conditioning("native", torch.randn(1, 4, 8))
    actual = host.rollout(context, 17)
    pipeline = PH3ImageCoreDenoiseStep().init_pipeline()
    pipeline.update_components(transformer=model,
        scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False),
        audio_scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False))
    pipeline.set_progress_bar_config(disable=True)
    reference = pipeline(prompt_embeds=context.embeddings, text_token_tags=torch.ones(4, dtype=torch.long),
        height=64, width=64, generator=torch.Generator().manual_seed(17), sigmas=[1., .5, 0.])
    assert torch.equal(host.unpack_image(actual), reference.get("latents"))
    assert torch.equal(actual.audio.reshape(2, 2, 6).permute(0, 2, 1), reference.get("audio_latents"))
