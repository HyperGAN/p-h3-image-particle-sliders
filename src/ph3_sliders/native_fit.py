"""Loaded-model qualification and equal-budget native LR trials."""

from dataclasses import asdict
import gc
import json
from pathlib import Path
import signal
import time

import torch

from .adapter import RoutedAdapter
from .learning_rates import LearningRates, assert_learning_rates
from .native_cache import NativeCache, file_hash, tensor_hash
from .native_cli import event, load_encoded, verify_staged, write_json
from .native_host import H3Host
from .native_qualify import same
from .native_runtime import configure_numeric_runtime, load_transformer, runtime_identity, verify_loaded_transformer
from .native_training import NativeTrainingRun


def prepare(args):
    plan = json.loads(args.plan.read_text())
    staged = verify_staged(args.model_directory, args.inventory)
    stored = json.loads((args.cache_root / "STAGED_WEIGHTS.json").read_text())
    if staged != stored:
        raise ValueError("staged source files changed after teacher collection")
    configure_numeric_runtime()
    torch.cuda.set_per_process_memory_fraction(.88, 0)
    identity = dict(runtime_identity(args.source_manifest, quantization=plan["numerical_cohort"]),
        staged_weights_sha256=file_hash(args.cache_root / "STAGED_WEIGHTS.json"),
        plan_sha256=file_hash(args.plan), encoded_conditioning_sha256=file_hash(args.conditioning / "manifest.json"))
    train = NativeCache.load(args.cache_root / "train")
    development = NativeCache.load(args.cache_root / "development")
    for cache in (train, development):
        previous = cache.identity
        for key in identity:
            if key == "product_sources":
                # Target caches depend on the frozen forward/collector surface;
                # training/checkpoint-only additions do not change their values.
                for name in ("native_host.py", "native_runtime.py", "native_cache.py", "adapter.py"):
                    if identity[key][name] != previous[key][name]:
                        raise ValueError("native cache forward source differs: " + name)
            elif identity[key] != previous[key]:
                raise ValueError("native cache numerical/model cohort differs: " + key)
    event(phase="fit_transformer_loading")
    model = load_transformer(args.model_directory)
    binding = verify_loaded_transformer(model, args.model_directory)
    host = H3Host(model, device="cuda:0", latent_height=plan["resolution"]["height"] // 16,
        latent_width=plan["resolution"]["width"] // 16, sigmas=plan["sigmas"])
    event(phase="fit_transformer_bound", keys=binding["keys"])
    return plan, host, train, development, identity, binding


def make_run(host, train, identity, *, rate, seed):
    if host.adapter is not None:
        raise RuntimeError("one owned adapter/game at a time")
    devices = [host.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.default_generators[devices[0]].manual_seed(seed)
        host.adapter = RoutedAdapter(host.model, host.targets, device=host.device)
        return NativeTrainingRun(host, train, identity=identity, seed=seed,
            rates=LearningRates(generator=rate, critic=9e-4, particles=6e-3))


def release_run(host, run):
    host.adapter.close()
    host.adapter = None
    # The caller deletes its final run reference before another large critic.
    gc.collect()


@torch.no_grad()
def upstream_sampler_parity(host, conditioning, seed):
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.p_h3_image.modular_blocks_p_h3_image import PH3ImageCoreDenoiseStep
    actual = host.rollout(conditioning, seed, strength=0.)
    pipeline = PH3ImageCoreDenoiseStep().init_pipeline()
    pipeline.update_components(transformer=host.model,
        scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False),
        audio_scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False))
    pipeline.set_progress_bar_config(disable=True)
    reference = pipeline(prompt_embeds=conditioning.embeddings.to(host.device),
        text_token_tags=torch.ones(conditioning.embeddings.shape[1], dtype=torch.long, device=host.device),
        height=host.latent_height * 16, width=host.latent_width * 16,
        generator=torch.Generator().manual_seed(seed), sigmas=list(host.sigmas))
    actual_image = host.unpack_image(actual)
    actual_audio = actual.audio.reshape(2, 2, host.model.config.audio_in_channels).permute(0, 2, 1)
    if not torch.equal(actual_image, reference.get("latents")) or not torch.equal(actual_audio, reference.get("audio_latents")):
        raise ValueError("full loaded native sampler differs from upstream pipeline")
    return dict(image_exact=True, coupled_audio_exact=True, seed=seed, context=conditioning.name)


def qualify_loaded(args, lease, prepared=None):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, host, train, development, identity, binding = prepared or prepare(args)
    write_json(args.output / "LOADED_WEIGHT_BINDING.json", binding)
    row = development[0]
    parity = upstream_sampler_parity(host, development.contexts[row["neutral_context"]], row["seed"])
    event(phase="loaded_upstream_sampler_parity", **parity)
    base = tensor_hash(host.model.state_dict())
    run = make_run(host, train, identity, rate=2e-5, seed=17)
    try:
        initial = run.evaluate(development, selection="raw", raw_path=args.output / "OFF-responses.safetensors")
        if initial["mean_teacher_student_relative_error"] != 1.:
            raise ValueError("full development OFF control must score exactly1")
        run.step(); lease.check(); run.step(); lease.check()
        run.save(args.output / "step-000002")
        run.step(); lease.check(); run.step(); lease.check()
        expected = run.snapshot()
        run.load(args.output / "step-000002")
        run.step(); lease.check(); run.step(); lease.check()
        same(run.snapshot(), expected)
        if tensor_hash(host.model.state_dict()) != base:
            raise ValueError("full frozen native base changed during qualification")
        with torch.no_grad(), host.adapter.scaled(0.):
            off = host.response(host.velocity(row["state"], development.contexts[row["neutral_context"]], row["position"]))
        if not torch.equal(off.cpu(), row["neutral"]):
            raise ValueError("loaded trained-attachment OFF differs from frozen cache")
        run.save(args.output / "step-000004")
        result = dict(schema="ph3-full-pretrained-native-qualification-v1", status="passed",
            identity=identity, model_keys_bound=binding["keys"], sampler_parity=parity,
            train_cache=train.index["fingerprint"], development_cache=development.index["fingerprint"],
            response_elements=train[0]["positive"].numel(), positions=train.positions,
            target_count=len(host.targets), full_image_audio_response=True,
            actual_pretrained_qualification_updates=4, replay_calls=2, exact_full_state_replay=True,
            full_frozen_base_unchanged=True, OFF_exact=True, OFF_metric=initial,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=lease.check())
        write_json(args.output / "QUALIFICATION.json", result)
        event(phase="loaded_qualification_passed", steps=4, replay_calls=2,
              peak_bytes=result["gpu_peak_allocated_bytes"])
        return result
    finally:
        release_run(host, run)
        del run
        torch.cuda.empty_cache()


def eligible(result):
    return (result["mean_teacher_student_relative_error"] < 1
        and result["mean_signed_gain"] > 0
        and all(value < 1 for value in result["scene_relative_error"].values()))


def run_trials(args, lease):
    args.output.mkdir(parents=True, exist_ok=True)
    plan, host, train, development, identity, binding = prepare(args)
    qualification = json.loads((args.qualification / "QUALIFICATION.json").read_text())
    if (qualification.get("status") != "passed" or qualification["identity"] != identity
            or qualification["train_cache"] != train.index["fingerprint"]
            or qualification["development_cache"] != development.index["fingerprint"]):
        raise ValueError("native qualification does not bind this exact source/model/cache cohort")
    identity = dict(identity, native_qualification_sha256=file_hash(args.qualification / "QUALIFICATION.json"))
    interrupted = []
    previous_handlers = {sig: signal.signal(sig, lambda number, frame: interrupted.append(number))
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    results = []
    reference_file = args.output / f"INITIAL-STATE-seed-{args.seed}.json"
    try:
        for rate in args.rates:
            directory = args.output / f"g-{rate:.8g}-seed-{args.seed}"
            directory.mkdir(parents=True, exist_ok=True)
            run = make_run(host, train, identity, rate=rate, seed=args.seed)
            try:
                initial = dict(adapter=tensor_hash(run.adapter.state_dict()),
                    critic=tensor_hash(run.game.critic.state_dict()),
                    streams=tensor_hash(run.game.sampler.state_dict()["generators"]),
                    empty_optimizer_state=not run.game.g.state and not run.game.d.state,
                    cache=train.index["fingerprint"], horizon=1600, seed=args.seed)
                if reference_file.exists():
                    if json.loads(reference_file.read_text()) != initial:
                        raise ValueError("LR trial initial tensors/streams differ")
                else:
                    write_json(reference_file, initial)
                trial = dict(schema="ph3-native-lr-trial-v1", rate=rate, seed=args.seed,
                    signature=run.signature, initial_state=initial)
                trial_path = directory / "TRIAL.json"
                if trial_path.exists() and json.loads(trial_path.read_text()) != trial:
                    raise ValueError("existing LR trial belongs to a different cohort")
                write_json(trial_path, trial)
                checkpoints = sorted((directory / "checkpoints").glob("step-*"))
                if checkpoints:
                    run.load(checkpoints[-1])
                    event(phase="trial_resumed", lr=rate, step=run.completed_steps)
                if run.completed_steps > args.steps:
                    raise ValueError("requested budget predates the saved continuation")
                while run.completed_steps < args.steps:
                    started = time.monotonic()
                    metrics = run.step()
                    lease.check()
                    record = dict(phase="training_step", lr=rate, seed=args.seed,
                        completed_steps=run.completed_steps, seconds=time.monotonic()-started,
                        metrics=metrics, optimizer_rates=assert_learning_rates(run.game, run.rates))
                    with open(directory / "training.jsonl", "a") as log:
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                    event(**{k: v for k, v in record.items() if k != "metrics"})
                    step = run.completed_steps
                    if step in plan["training"]["metric_checkpoints"] or step % 64 == 0 or step == args.steps or interrupted:
                        checkpoint = directory / "checkpoints" / f"step-{step:06d}"
                        if not checkpoint.exists():
                            run.save(checkpoint)
                        export = directory / "exports" / f"step-{step:06d}-ema"
                        if not export.exists():
                            run.export(export)
                    if step in plan["training"]["metric_checkpoints"] or step == args.steps:
                        path = directory / f"eval-{step:06d}-ema.json"
                        if not path.exists():
                            evaluation = run.evaluate(development, selection="ema",
                                raw_path=directory / f"eval-{step:06d}-ema.safetensors")
                            write_json(path, evaluation)
                            event(phase="trial_evaluated", lr=rate, step=step,
                                  score=evaluation["mean_teacher_student_relative_error"])
                    write_json(directory / "STATE.json", dict(completed_steps=step, interrupted=bool(interrupted)))
                    if interrupted:
                        event(phase="owned_run_stopped_cleanly", lr=rate, step=step)
                        return
                evaluation_path = directory / f"eval-{args.steps:06d}-ema.json"
                if not evaluation_path.exists():
                    write_json(evaluation_path, run.evaluate(development,
                        raw_path=directory / f"eval-{args.steps:06d}-ema.safetensors"))
                evaluation = json.loads(evaluation_path.read_text())
                results.append(dict(lr=rate, seed=args.seed, completed_steps=args.steps,
                    evaluation=evaluation, metric_feasible=eligible(evaluation), trial=str(directory),
                    checkpoint=str(directory / "checkpoints" / f"step-{args.steps:06d}"),
                    export=str(directory / "exports" / f"step-{args.steps:06d}-ema")))
                write_json(directory / "STATE.json", dict(status="budget_complete", completed_steps=run.completed_steps))
            except BaseException as error:
                write_json(directory / "FAILURE.json", dict(error=repr(error),
                    completed_steps=run.completed_steps, partial_game_update=run.failed_step))
                raise
            finally:
                release_run(host, run)
                del run
                torch.cuda.empty_cache()
        feasible = sorted((r for r in results if r["metric_feasible"]),
                          key=lambda r: (r["evaluation"]["mean_teacher_student_relative_error"], r["lr"]))
        nomination = dict(status="metric_candidate_requires_matched_media" if feasible else "no_feasible_candidate",
            common_budget=args.steps, seed=args.seed, results=results,
            candidate=feasible[0] if feasible else None, native_qualification=identity["native_qualification_sha256"])
        write_json(args.output / f"NOMINATION-step-{args.steps:06d}-seed-{args.seed}.json", nomination)
        event(phase="common_budget_complete", step=args.steps, candidate_lr=feasible[0]["lr"] if feasible else None)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
