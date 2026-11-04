"""Bounded actual random H3 architecture qualification; not pretrained fitting."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .gpu_lease import GpuLease


def same(left, right, path="state"):
    import torch
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError("replay tensor differs: " + path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("replay keys differ: " + path)
        for name in left:
            same(left[name], right[name], f"{path}.{name}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError("replay length differs: " + path)
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f"{path}.{index}")
    elif left != right:
        raise AssertionError("replay value differs: " + path)


def qualify(output, source_manifest):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from .adapter import RoutedAdapter
    from .native_cache import NativeCache, tensor_hash
    from .native_host import H3Host
    from .native_runtime import configure_numeric_runtime, load_transformer, runtime_identity, verify_loaded_transformer
    from .native_training import NativeTrainingRun

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    configure_numeric_runtime(47)
    torch.cuda.set_per_process_memory_fraction(0.08, 0)
    torch.random.default_generator.manual_seed(47)
    torch.cuda.default_generators[0].manual_seed(47)
    identity = runtime_identity(source_manifest, quantization="bitsandbytes_0.50.2_int8_threshold6")
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.save_pretrained(output / "random-model" / "transformer")
    del model
    model = load_transformer(output / "random-model", quantization="int8")
    weight_binding = verify_loaded_transformer(model, output / "random-model")
    host = H3Host(model, latent_height=4, latent_width=4, device="cuda:0", sigmas=(1., .5, 0.))
    generator = torch.Generator().manual_seed(7)
    neutral = host.conditioning("neutral", torch.randn(1, 4, 8, generator=generator))
    positive = host.conditioning("positive", torch.randn(1, 5, 8, generator=generator))
    host.adapter = RoutedAdapter(model, host.targets, device="cuda:0")
    try:
        rows = host.collect_pair(neutral, positive, scene="cup", seed=17)
        cache = NativeCache(rows, {"neutral": neutral, "positive": positive}, positions=2,
            expected_examples=[("cup", 17, "neutral"), ("cup", 17, "positive")],
            identity=dict(split="train", actual_random_h3=True, pretrained=False))
        cache.save(output / "paired-cache")
        base = tensor_hash(model.state_dict())
        run = NativeTrainingRun(host, cache, seed=17, identity=identity)
        run.step(); run.step()
        run.save(output / "step-2")
        run.step(); run.step()
        expected = run.snapshot()
        run.load(output / "step-2")
        run.step(); run.step()
        same(run.snapshot(), expected)
        if tensor_hash(model.state_dict()) != base:
            raise AssertionError("frozen quantized base changed")
        with torch.no_grad(), host.adapter.scaled(0):
            off = host.response(host.velocity(cache[0]["state"], neutral, 0))
        if not torch.equal(off.cpu(), cache[0]["neutral"]):
            raise AssertionError("native OFF identity changed")
        score = run.evaluate(cache)
        same(run.snapshot(), expected)
        result = dict(schema="ph3-random-native-gpu-qualification-v1", status="passed",
            source_identity=identity, model_class=type(model).__name__,
            weight_binding=weight_binding,
            quantized_linear_modules=sum(type(m).__name__ == "Linear8bitLt" for m in model.modules()),
            adapter_targets=len(host.targets), adapter_dtype="float32",
            main_and_refiner_checkpointing=model.gradient_checkpointing and model.token_refiner.gradient_checkpointing,
            coupled_image_audio=True, full_response_width=cache[0]["positive"].numel(),
            completed_random_updates=4, replayed_updates=2, exact_full_state_replay=True,
            frozen_quantized_base=True, OFF_exact=True, evaluation_state_preserved=True,
            score=score, optimizer_rates=asdict(run.rates), pretrained_training_updates=0,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started)
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        host.adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    with GpuLease(required_free_mib=4096) as lease:
        result = qualify(args.output, args.source_manifest)
        result["final_gpu"] = lease.check()
        print(json.dumps({k: result[k] for k in ("status", "quantized_linear_modules", "exact_full_state_replay",
            "OFF_exact", "gpu_peak_allocated_bytes", "pretrained_training_updates")}))


if __name__ == "__main__":
    main()
