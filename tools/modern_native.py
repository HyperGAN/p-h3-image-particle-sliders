"""Offline, physical-GPU1 H3 qualification and LR trials with real HyperGAN/HNDL."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time


def write(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def event(**values):
    print(json.dumps(dict(utc_epoch=time.time(), **values), allow_nan=False), flush=True)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError("full-state tensor replay differs")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("full-state mapping differs")
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("full-state sequence differs")
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    elif left != right:
        raise ValueError("full-state scalar differs")


def prepare(args):
    import torch
    from ph3_hypergan import verify_teacher_identity, fit_normalization
    from ph3_hypergan.runtime import require_runtime
    from ph3_hndl import runtime_identity as hndl_identity
    from ph3_sliders.native_cache import NativeCache, file_hash
    from ph3_sliders.native_cli import verify_staged
    from ph3_sliders.native_host import H3Host
    from ph3_sliders.native_runtime import configure_numeric_runtime, load_transformer, verify_loaded_transformer

    configure_numeric_runtime()
    torch.cuda.set_per_process_memory_fraction(.88, 0)
    collection_plan = json.loads(args.plan.read_text())
    training_plan = json.loads(args.training_plan.read_text())
    if (training_plan["batch_size"] != 1 or training_plan["schedule_horizon"] < 512
            or training_plan["collection_plan_sha256"] != file_hash(args.plan)):
        raise ValueError("explicit batch1 modern plan must bind the immutable collection plan")
    staged = verify_staged(args.model_directory, args.inventory)
    stored_path = args.cache_root / "STAGED_WEIGHTS.json"
    if staged != json.loads(stored_path.read_text()):
        raise ValueError("staged model bytes differ from collected teachers")
    train, development = (NativeCache.load(args.cache_root / split) for split in ("train", "development"))
    teacher_identities = {}
    for name, cache in (("train", train), ("development", development)):
        teacher_identities[name] = verify_teacher_identity(cache.identity,
            source_manifest=args.source_manifest, quantization=collection_plan["numerical_cohort"],
            staged_weights_sha256=file_hash(stored_path), plan_sha256=file_hash(args.plan),
            encoded_conditioning_sha256=file_hash(args.conditioning / "manifest.json"))
    identity = dict(schema="ph3-modern-native-qualified-cohort-v1", engine=require_runtime(),
        hndl=hndl_identity(), teachers=teacher_identities,
        training_plan=training_plan, training_plan_sha256=file_hash(args.training_plan),
        driver_sha256=file_hash(Path(__file__)),
        caches={"train": train.index["fingerprint"], "development": development.index["fingerprint"]})
    event(phase="modern_full_transformer_loading")
    model = load_transformer(args.model_directory)
    binding = verify_loaded_transformer(model, args.model_directory)
    event(phase="modern_full_transformer_bound", keys=binding["keys"])
    host = H3Host(model, device="cuda:0", latent_height=collection_plan["resolution"]["height"] // 16,
        latent_width=collection_plan["resolution"]["width"] // 16, sigmas=collection_plan["sigmas"])
    normalization = fit_normalization(train, positions=train.positions)
    return host, train, development, normalization, identity, binding


def make_run(host, train, normalization, identity, rate, seed):
    import torch
    from ph3_hndl import RoutedAdapter, GlobalMixCritic
    from ph3_hypergan import NativeHyperganRun

    if host.adapter is not None:
        raise RuntimeError("one owned native adapter at a time")
    plan = identity["training_plan"]
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.default_generators[0].manual_seed(seed)
        host.adapter = RoutedAdapter(host.model, host.targets, device=host.device)
        try:
            critic = GlobalMixCritic(train[0]["positive"].numel(), device=host.device)
            return NativeHyperganRun(host, train, normalization, critic, identity=identity,
                batch_size=plan["batch_size"], seed=seed, generator_lr=rate,
                critic_lr=plan["critic_rate"], particle_lr=plan["particle_rate"],
                steps=plan["schedule_horizon"])
        except BaseException:
            host.adapter.close()
            host.adapter = None
            raise


def close(host):
    if host.adapter is not None:
        host.adapter.close()
        host.adapter = None


def save(run, directory):
    import torch
    from ph3_sliders.native_cache import file_hash
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        if (directory / "manifest.json").is_file():
            raise FileExistsError(directory)
        # Preserve an interrupted legacy slot; it is never a completed state.
        os.rename(directory, directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
    torch.save(run.snapshot(), temporary / "state.pt")
    write(temporary / "manifest.json", dict(schema="ph3-hypergan-hndl-full-checkpoint-v1",
        signature=run.signature, completed_steps=run.step,
        weights_sha256=file_hash(temporary / "state.pt")))
    os.rename(temporary, directory)


def restore(run, directory):
    import torch
    from ph3_sliders.native_cache import file_hash
    document = json.loads((directory / "manifest.json").read_text())
    if (document["signature"] != canonical(run.signature) or document["weights_sha256"] != file_hash(directory / "state.pt")):
        raise ValueError("full checkpoint identity/bytes differ")
    run.restore(torch.load(directory / "state.pt", map_location="cpu", weights_only=True))
    if run.step != document["completed_steps"]:
        raise ValueError("checkpoint completed counter differs")


def export(run, directory, identity):
    from ph3_hndl import export_adapter
    export_adapter(run.adapter, directory,
        identity=canonical(dict(identity, training_signature=run.signature)),
        selection="ema", completed_steps=run.step, values=run.adapter_weights(selection="ema"))


def evaluate_saved(run, cache, directory, *, selection="ema", qualification_sha256=None):
    """Publish raw responses and their metric together, using an atomic directory."""
    from ph3_sliders.native_cache import file_hash, tensor_hash
    signature = canonical(dict(training=run.signature, cache=cache.index["fingerprint"],
        completed_steps=run.step, selection=selection,
        selected_weights_sha256=tensor_hash(run.adapter_weights(selection=selection)),
        qualification_sha256=qualification_sha256))
    if directory.exists():
        document = json.loads((directory / "result.json").read_text())
        if (document.get("evaluation_signature") != signature
                or document["raw_student_responses"]["sha256"] != file_hash(directory / "student.safetensors")):
            raise ValueError("saved evaluation cohort or raw bytes differ")
        return document
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
    document = run.evaluate(cache, selection=selection, raw_path=temporary / "student.safetensors")
    document.update(evaluation_signature=signature, pretrained_qualified=qualification_sha256 is not None,
        native_h3_qualified=qualification_sha256 is not None)
    if qualification_sha256 is not None:
        document["qualification_sha256"] = qualification_sha256
    document["raw_student_responses"]["path"] = str(directory / "student.safetensors")
    write(temporary / "result.json", document)
    os.rename(temporary, directory)
    return document


def parity(host, cache):
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.p_h3_image.modular_blocks_p_h3_image import PH3ImageCoreDenoiseStep
    row = cache[0]
    context = cache.contexts[row["neutral_context"]]
    actual = host.rollout(context, row["seed"], strength=0.)
    pipeline = PH3ImageCoreDenoiseStep().init_pipeline()
    pipeline.update_components(transformer=host.model,
        scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False),
        audio_scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False))
    pipeline.set_progress_bar_config(disable=True)
    reference = pipeline(prompt_embeds=context.embeddings.to(host.device),
        text_token_tags=torch.ones(context.embeddings.shape[1], dtype=torch.long, device=host.device),
        height=host.latent_height * 16, width=host.latent_width * 16,
        generator=torch.Generator().manual_seed(row["seed"]), sigmas=list(host.sigmas))
    actual_audio = actual.audio.reshape(2, 2, host.model.config.audio_in_channels).permute(0, 2, 1)
    if not torch.equal(host.unpack_image(actual), reference.get("latents")) or not torch.equal(actual_audio, reference.get("audio_latents")):
        raise ValueError("actual pretrained native sampler differs from upstream image/audio pipeline")
    return dict(image_exact=True, audio_exact=True, seed=row["seed"], context=context.name)


def qualify(args, lease, prepared):
    import torch
    from ph3_sliders.native_cache import tensor_hash
    host, train, development, normalization, identity, binding = prepared
    write(args.output / "LOADED_WEIGHT_BINDING.json", binding)
    native_parity = parity(host, development)
    event(phase="modern_pretrained_upstream_sampler_exact")
    base = tensor_hash(host.model.state_dict())
    run = make_run(host, train, normalization, identity, args.rate, args.seed)
    update_calls = 0
    def qualification_update(*, replay=False):
        nonlocal update_calls
        started = time.monotonic()
        run.update()
        update_calls += 1
        write(args.output / "STATE.json", dict(completed_steps=run.step,
            actual_qualification_update_calls=update_calls, replay=replay))
        event(phase="modern_pretrained_qualification_update", completed_steps=run.step,
            update_calls=update_calls, replay=replay, seconds=time.monotonic()-started)
        lease.check()
    try:
        initial = evaluate_saved(run, development, args.output / "OFF-evaluation", selection="raw")
        if initial["mean_teacher_student_relative_error"] != 1.:
            raise ValueError("complete development OFF control must score exactly1")
        for _ in range(2):
            qualification_update()
        save(run, args.output / "step-000002")
        for _ in range(2):
            qualification_update()
        expected = run.snapshot()
        restore(run, args.output / "step-000002")
        for _ in range(2):
            qualification_update(replay=True)
        equal(expected, run.snapshot())
        if tensor_hash(host.model.state_dict()) != base:
            raise ValueError("frozen pretrained weights changed")
        row = development[0]
        with torch.no_grad(), host.adapter.scaled(0.):
            off = host.response(host.velocity(row["state"], development.contexts[row["neutral_context"]], row["position"]))
        if not torch.equal(off.cpu(), row["neutral"]):
            raise ValueError("trained HNDL attachment OFF differs from recorded frozen base")
        save(run, args.output / "step-000004")
        export(run, args.output / "step-000004-ema", identity)
        write(args.output / "QUALIFICATION.json", dict(schema="ph3-modern-native-qualification-v1",
            status="passed", identity=identity, completed_steps=run.step, replay_calls=2,
            full_state_exact=True, full_pretrained_frozen_base_exact=True, OFF_exact=True,
            OFF_evaluation=initial, native_sampler_parity=native_parity,
            response_elements=train[0]["positive"].numel(), targets=len(host.targets),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=lease.check()))
        event(phase="modern_pretrained_qualification_passed", completed_steps=run.step)
    finally:
        close(host)


def trial(args, lease, prepared):
    import torch
    from ph3_sliders.native_cache import file_hash
    host, train, development, normalization, identity, binding = prepared
    qualification_path = args.qualification / "QUALIFICATION.json"
    qualification = json.loads(qualification_path.read_text())
    if qualification.get("status") != "passed" or qualification["identity"] != identity:
        raise ValueError("qualification does not bind exact modern source/model/cache/recipe cohort")
    identity = dict(identity, native_qualification_sha256=file_hash(qualification_path))
    run = make_run(host, train, normalization, identity, args.rate, args.seed)
    stopped = []
    handlers = {s: signal.signal(s, lambda n, f: stopped.append(n)) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        reference_path = args.output.parent / f"INITIAL-STATE-seed-{args.seed}.json"
        initial = dict(adapter_sha256=run.signature["initial_adapter_sha256"],
            critic_sha256=run.signature["initial_critic_sha256"], identity=identity, seed=args.seed)
        if reference_path.exists() and json.loads(reference_path.read_text()) != canonical(initial):
            raise ValueError("LR candidates do not have identical initial adapter/critic/cohort")
        if not reference_path.exists():
            write(reference_path, initial)
        receipt = canonical(dict(signature=run.signature, qualification_sha256=file_hash(qualification_path)))
        path = args.output / "TRIAL.json"
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError("existing LR directory belongs to another complete training cohort")
        write(path, receipt)
        completed = sorted(p for p in (args.output / "checkpoints").glob("step-*")
                           if re.fullmatch(r"step-[0-9]{6}", p.name) and (p / "manifest.json").is_file())
        if completed:
            restore(run, completed[-1])
            event(phase="modern_native_resumed", completed_steps=run.step)
        if not run.step <= args.steps <= identity["training_plan"]["schedule_horizon"]:
            raise ValueError("requested budget outside saved continuation or fixed schedule horizon")
        while run.step < args.steps:
            started = time.monotonic(); metrics = run.update()
            record = dict(completed_steps=run.step, seconds=time.monotonic()-started,
                metrics=metrics, generator_lr=args.rate, seed=args.seed,
                optimizer_rates={"g": [v["lr"] for v in run.opt_g.param_groups],
                                 "d": [v["lr"] for v in run.opt_d.param_groups]})
            with open(args.output / "training.jsonl", "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            event(phase="modern_native_step", **{k:v for k,v in record.items() if k != "metrics"})
            write(args.output / "STATE.json", dict(completed_steps=run.step, clean_stop=bool(stopped)))
            lease.check()
            if run.step % 64 == 0 or run.step in (32, 64, 128) or run.step == args.steps or stopped:
                destination = args.output / "checkpoints" / f"step-{run.step:06d}"
                if not (destination / "manifest.json").is_file():
                    save(run, destination)
                destination = args.output / "exports" / f"step-{run.step:06d}-ema"
                if not destination.exists():
                    export(run, destination, identity)
            if run.step in (32, 64, 128) or run.step == args.steps:
                destination = args.output / f"eval-{run.step:06d}-ema.json"
                if not destination.exists():
                    result = evaluate_saved(run, development,
                        args.output / "evaluations" / f"step-{run.step:06d}-ema",
                        qualification_sha256=file_hash(qualification_path))
                    write(destination, result)
                    event(phase="modern_native_evaluated", completed_steps=run.step,
                        score=result["mean_teacher_student_relative_error"])
            write(args.output / "STATE.json", dict(completed_steps=run.step, clean_stop=bool(stopped)))
            if stopped:
                event(phase="modern_native_clean_stop", completed_steps=run.step)
                return
        # A process may have stopped after its final checkpoint but before its
        # evaluation/export; resuming that completed budget must finish both.
        destination = args.output / f"eval-{run.step:06d}-ema.json"
        if not destination.exists():
            result = evaluate_saved(run, development,
                args.output / "evaluations" / f"step-{run.step:06d}-ema",
                qualification_sha256=file_hash(qualification_path))
            write(destination, result)
        destination = args.output / "exports" / f"step-{run.step:06d}-ema"
        if not destination.exists():
            export(run, destination, identity)
        write(args.output / "STATE.json", dict(status="budget_complete", completed_steps=run.step))
    finally:
        for s,h in handlers.items():
            signal.signal(s,h)
        close(host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "trial"))
    for name in ("model-directory", "inventory", "source-manifest", "plan", "training-plan", "conditioning", "cache-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.command == "trial" and args.qualification is None:
        parser.error("trial requires --qualification")
    args.output.mkdir(parents=True, exist_ok=args.command == "trial")
    from ph3_sliders.gpu_lease import GpuLease
    try:
        with GpuLease(required_free_mib=40500) as lease:
            prepared = prepare(args)
            (qualify if args.command == "qualify" else trial)(args, lease, prepared)
    except BaseException as error:
        state = json.loads((args.output / "STATE.json").read_text()) if (args.output / "STATE.json").exists() else None
        write(args.output / "FAILURE.json", dict(error=repr(error), status="unscoreable",
            last_recorded_state=state,
            note="A partial engine update is never labeled a completed checkpoint."))
        raise


if __name__ == "__main__":
    main()
