"""Bounded GPU1 check of HNDL/HyperGAN with real random int8 H3 layers."""

import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    from ph3_sliders.gpu_lease import GpuLease
    with GpuLease(required_free_mib=4096) as lease:
        qualify(args, lease)


def qualify(args, lease):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from ph3_hndl import RoutedAdapter, GlobalMixCritic, runtime_identity
    from ph3_hypergan import NativeHyperganRun, fit_normalization
    from ph3_hypergan.runtime import require_runtime
    from ph3_sliders.native_cache import NativeCache, file_hash, tensor_hash
    from ph3_sliders.native_host import H3Host
    from ph3_sliders.native_runtime import (configure_numeric_runtime,
        verify_diffusers_source, load_transformer, verify_loaded_transformer)
    from modern_native import equal, save, restore, evaluate_saved, export, write, event, parity

    args.output.mkdir(parents=True, exist_ok=False)
    configure_numeric_runtime(47)
    torch.cuda.set_per_process_memory_fraction(.08, 0)
    identity = dict(engine=require_runtime(), hndl=runtime_identity(),
        diffusers=verify_diffusers_source(args.source_manifest),
        driver_sha256=file_hash(Path(__file__).with_name("modern_native.py")),
        GPU_tiny_qualifier_sha256=file_hash(Path(__file__)),
        model="reduced random actual H3", quantization="int8 threshold6 BF16/FP16 declared conversion")
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.save_pretrained(args.output / "random-model/transformer")
    del model
    model = load_transformer(args.output / "random-model")
    binding = verify_loaded_transformer(model, args.output / "random-model")
    host = H3Host(model, device="cuda:0", latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    neutral = host.conditioning("neutral", torch.randn(1, 4, 8))
    positive = host.conditioning("positive", torch.randn(1, 5, 8))
    rows = host.collect_pair(neutral, positive, scene="cup", seed=17)
    cache = NativeCache(rows, {"neutral": neutral, "positive": positive}, positions=2,
        expected_examples=[("cup", 17, "neutral"), ("cup", 17, "positive")],
        identity=dict(split="train", actual_random_int8_h3=True, pretrained=False))
    cache.save(args.output / "paired-cache")
    upstream_parity = parity(host, cache)
    base = tensor_hash(model.state_dict())
    host.adapter = RoutedAdapter(model, host.targets, device=host.device)
    try:
        run = NativeHyperganRun(host, cache, fit_normalization(cache, positions=cache.positions),
            GlobalMixCritic(rows[0]["positive"].numel(), device=host.device),
            batch_size=1, identity=identity, steps=1600)
        started = time.monotonic()
        for _ in range(2):
            run.update(); lease.check()
        save(run, args.output / "step-000002")
        run.update(); run.update()
        expected = run.snapshot()
        restore(run, args.output / "step-000002")
        run.update(); run.update()
        equal(expected, run.snapshot())
        if tensor_hash(model.state_dict()) != base:
            raise ValueError("random quantized frozen base changed")
        with torch.no_grad(), host.adapter.scaled(0.):
            off = host.response(host.velocity(cache[0]["state"], neutral, 0))
        if not torch.equal(off.cpu(), cache[0]["neutral"]):
            raise ValueError("trained HNDL int8 OFF differs")
        evaluation = evaluate_saved(run, cache, args.output / "evaluation")
        export(run, args.output / "step-000004-ema", identity)
        equal(expected, run.snapshot())
        result = dict(schema="ph3-modern-random-native-gpu-qualification-v1", status="passed",
            identity=identity, weight_binding=binding, full_state_exact_replay=True,
            upstream_sampler_parity=upstream_parity,
            frozen_base_exact=True, OFF_exact=True, evaluation_and_export_preserve_state=True,
            actual_random_updates=run.step, replay_calls=2, pretrained_training_updates=0,
            native_targets=len(host.targets), response_elements=cache[0]["positive"].numel(),
            evaluation=evaluation, elapsed_seconds=time.monotonic()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=lease.check())
        write(args.output / "result.json", result)
        event(phase="modern_random_int8_gpu_passed", completed_steps=run.step,
            peak_bytes=result["gpu_peak_allocated_bytes"])
    finally:
        host.adapter.close()


if __name__ == "__main__":
    main()
