"""Offline CPU interruption/replay check of the actual native run driver."""

import json
import os
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run this verification with CUDA_VISIBLE_DEVICES='' before importing Torch")
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from ph3_hndl import RoutedAdapter, GlobalMixCritic, load_adapter
    from ph3_hypergan import NativeHyperganRun, fit_normalization
    from ph3_sliders.native_cache import NativeCache, file_hash
    from ph3_sliders.native_host import H3Host
    import modern_native as driver

    torch.set_num_threads(1)
    torch.manual_seed(47)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
        hidden_size=24, num_layers=2, num_refiner_layers=2, ffn_dim=32,
        in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2)
    model.requires_grad_(False).eval().enable_gradient_checkpointing()
    host = H3Host(model, latent_height=4, latent_width=4, sigmas=(1., .5, 0.))
    neutral = host.conditioning("neutral", torch.randn(1, 4, 8))
    positive = host.conditioning("positive", torch.randn(1, 5, 8))
    records = host.collect_pair(neutral, positive, scene="cup", seed=17)
    cache = NativeCache(records, {"neutral": neutral, "positive": positive}, positions=2,
        expected_examples=[("cup", 17, "neutral"), ("cup", 17, "positive")],
        identity=dict(split="train", reduced_random_h3=True))
    host.adapter = RoutedAdapter(model, host.targets)
    run = NativeHyperganRun(host, cache, fit_normalization(cache, positions=cache.positions),
        GlobalMixCritic(records[0]["positive"].numel()), batch_size=1,
        identity=dict(probe="random reduced native H3; no pretrained quality claim"), steps=8)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU verification attempted network or CUDA")

    try:
        with patch.object(socket, "create_connection", forbidden), patch.object(torch.cuda, "_lazy_init", forbidden), tempfile.TemporaryDirectory(prefix="ph3-modern-io-") as temporary:
            root = Path(temporary)
            run.update(); run.update()
            original_write = driver.write
            def fail_manifest(path, value):
                if path.name == "manifest.json":
                    raise RuntimeError("injected interruption before checkpoint publication")
                return original_write(path, value)
            with patch.object(driver, "write", fail_manifest):
                try:
                    driver.save(run, root / "checkpoint")
                except RuntimeError as error:
                    assert "injected interruption" in str(error)
                else:
                    raise AssertionError("checkpoint fault was not injected")
            assert not (root / "checkpoint").exists()
            driver.save(run, root / "checkpoint")
            # A legacy interrupted final slot is retained and safely replaced.
            (root / "legacy").mkdir()
            driver.save(run, root / "legacy")
            assert list(root.glob("legacy.incomplete-*"))
            run.update(); run.update()
            expected = run.snapshot()
            driver.restore(run, root / "checkpoint")
            run.update(); run.update()
            driver.equal(expected, run.snapshot())
            evaluation = root / "evaluation"
            original_rename = os.rename
            def fail_evaluation(source, destination):
                if Path(destination) == evaluation:
                    raise RuntimeError("injected interruption before evaluation publication")
                return original_rename(source, destination)
            with patch.object(driver.os, "rename", fail_evaluation):
                try:
                    driver.evaluate_saved(run, cache, evaluation)
                except RuntimeError as error:
                    assert "injected interruption" in str(error)
                else:
                    raise AssertionError("evaluation fault was not injected")
            assert not evaluation.exists()
            driver.equal(expected, run.snapshot())
            result = driver.evaluate_saved(run, cache, evaluation)
            driver.equal(expected, run.snapshot())
            assert driver.evaluate_saved(run, cache, evaluation) == result
            off = driver.evaluate_off(run, cache, root / "OFF-evaluation")
            assert off["mean_teacher_student_relative_error"] == 1.
            driver.equal(expected, run.snapshot())
            assert file_hash(evaluation / "student.safetensors") == result["raw_student_responses"]["sha256"]
            driver.export(run, root / "ema", {"probe": "reduced random H3"})
            driver.equal(expected, run.snapshot())
            ema = run.adapter_weights(selection="ema")
            identity = json.loads((root / "ema/manifest.json").read_text())["signature"]["identity"]
            host.adapter.close()
            host.adapter = load_adapter(model, host.targets, root / "ema", identity=identity)
            for name, value in host.adapter.state_dict().items():
                assert torch.equal(value, ema[name])
            assert not torch.cuda.is_initialized()
            receipt = dict(schema="ph3-modern-gpu-driver-cpu-io-v2",
                full_checkpoint_exact_replay=True, completed_updates=4, replay_calls=2,
                checkpoint_interruption_recovery=True, legacy_incomplete_slot_preserved=True,
                raw_and_metric_atomic_publication=True, evaluation_interruption_recovery=True,
                evaluation_idempotent=True, export_and_evaluation_preserve_full_state=True,
                full_trained_attachment_OFF_control_exact=True,
                HNDL_EMA_export_load_exact=True, GPU_initialized=False,
                pretrained_training_updates=0, driver_sha256=file_hash(Path(driver.__file__)))
            output = Path(__file__).resolve().parents[1] / "validation/modern-driver-cpu-io.json"
            driver.write(output, receipt)
            print(json.dumps(receipt))
    finally:
        host.adapter.close()


if __name__ == "__main__":
    main()
