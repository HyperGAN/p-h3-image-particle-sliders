"""Native staged-model operations for the fixed P-H3 experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write_json(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def event(**values):
    print(json.dumps(dict(time=time.time(), **values), allow_nan=False), flush=True)


def verify_staged(model_directory, inventory_path, components=None):
    from .native_cache import file_hash
    from .native_runtime import MODEL_REVISION

    document = json.loads(Path(inventory_path).read_text())
    if document["revision"] != MODEL_REVISION:
        raise ValueError("staged model revision differs")
    roots = set(components or ("transformer", "text_encoder", "text_connector", "vae", "tokenizer", "processor",
                               "scheduler", "audio_scheduler"))
    root_files = {"LICENSE", "NOTICE", "README.md", "modular_model_index.json"}
    verified = {}
    for item in document["files"]:
        name = item["path"]
        if name.split("/")[0] not in roots and name not in root_files:
            continue
        path = Path(model_directory) / name
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("staged component incomplete: " + name)
        sha = file_hash(path)
        if item["lfs"]:
            if sha != item["lfs"]["sha256"]:
                raise ValueError("staged LFS bytes differ: " + name)
        else:
            data = path.read_bytes()
            if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest() != item["blob_id"]:
                raise ValueError("staged metadata Git blob differs: " + name)
        verified[name] = dict(bytes=item["bytes"], sha256=sha)
        event(phase="component_verified", path=name)
    return dict(schema="ph3-staged-components-v1", model_revision=MODEL_REVISION, files=verified,
                bytes=sum(v["bytes"] for v in verified.values()))


def encode(args):
    import torch
    from safetensors.torch import save_file
    from .native_runtime import configure_numeric_runtime, encode_prompts, runtime_identity
    from .native_cache import file_hash
    from .native_plan import prompt_map

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    staged = verify_staged(args.model_directory, args.inventory,
                           ("text_encoder", "text_connector", "tokenizer", "processor"))
    configure_numeric_runtime()
    torch.cuda.set_per_process_memory_fraction(.40, 0)
    prompts = prompt_map(plan)
    event(phase="encoding_start", prompts=len(prompts))
    tensors = encode_prompts(args.model_directory, prompts)
    save_file(tensors, str(args.output / "conditioning.safetensors"))
    document = dict(schema="ph3-frozen-verbatim-conditioning-v1", prompts=prompts,
        plan_sha256=file_hash(args.plan), tensor_sha256=file_hash(args.output / "conditioning.safetensors"),
        staged_components=staged, source_identity=runtime_identity(args.source_manifest, quantization="encoder_bf16"))
    write_json(args.output / "manifest.json", document)
    event(phase="encoding_complete", prompts=len(tensors), peak_bytes=torch.cuda.max_memory_allocated())


def load_encoded(directory, plan):
    from safetensors.torch import load_file
    from .native_cache import file_hash
    from .native_plan import prompt_map

    document = json.loads((directory / "manifest.json").read_text())
    if document["schema"] != "ph3-frozen-verbatim-conditioning-v1" or document["prompts"] != prompt_map(plan):
        raise ValueError("encoded prompt cohort differs")
    path = directory / "conditioning.safetensors"
    if file_hash(path) != document["tensor_sha256"]:
        raise ValueError("encoded prompt bytes changed")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != set(document["prompts"]):
        raise ValueError("encoded prompt names differ")
    return tensors, document


def collect(args, lease):
    import torch
    from .native_cache import NativeCache, file_hash
    from .native_host import H3Host
    from .native_plan import expected_examples
    from .native_runtime import configure_numeric_runtime, load_transformer, runtime_identity, verify_loaded_transformer

    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    staged_path = args.output / "STAGED_WEIGHTS.json"
    if staged_path.exists():
        staged = json.loads(staged_path.read_text())
        if staged != verify_staged(args.model_directory, args.inventory):
            raise ValueError("staged model files changed since the previous attempt")
    else:
        staged = verify_staged(args.model_directory, args.inventory)
        write_json(staged_path, staged)
    configure_numeric_runtime()
    torch.cuda.set_per_process_memory_fraction(.88, 0)
    event(phase="transformer_loading")
    model = load_transformer(args.model_directory)
    binding = verify_loaded_transformer(model, args.model_directory)
    write_json(args.output / "LOADED_WEIGHT_BINDING.json", binding)
    event(phase="transformer_bound", keys=binding["keys"], quantized=binding["quantized_weight_matrices"])
    host = H3Host(model, device="cuda:0", latent_height=plan["resolution"]["height"] // 16,
                  latent_width=plan["resolution"]["width"] // 16, sigmas=plan["sigmas"])
    embeddings, encoded = load_encoded(args.conditioning, plan)
    identity = dict(runtime_identity(args.source_manifest, quantization=plan["numerical_cohort"]),
        staged_weights_sha256=file_hash(staged_path), plan_sha256=file_hash(args.plan),
        encoded_conditioning_sha256=file_hash(args.conditioning / "manifest.json"))
    for split in args.splits:
        destination = args.output / split
        if destination.exists():
            existing = NativeCache.load(destination)
            if existing.identity != dict(identity, split=split):
                raise ValueError("existing cache belongs to a different native cohort")
            event(phase="cache_already_complete", split=split)
            continue
        all_rows, contexts = [], {}
        for scene in plan["splits"][split]["scenes"]:
            neutral = host.conditioning(scene["id"] + ":neutral", embeddings[scene["id"] + ":neutral"])
            positive = host.conditioning(scene["id"] + ":positive", embeddings[scene["id"] + ":positive"])
            contexts.update({neutral.name: neutral, positive.name: positive})
            for seed in plan["splits"][split]["seeds"]:
                partial = args.output / "partials" / split / f"{scene['id']}-{seed}"
                pair_examples = [(scene["id"], seed, path) for path in ("neutral", "positive")]
                if partial.exists():
                    pair = NativeCache.load(partial)
                    if pair.identity != dict(identity, split=split, partial=True):
                        raise ValueError("partial cache source/cohort differs")
                else:
                    from safetensors.torch import save_file
                    final_states = {}
                    rows = host.collect_pair(neutral, positive, scene=scene["id"], seed=seed, final_states=final_states)
                    pair = NativeCache(rows, {neutral.name: neutral, positive.name: positive},
                        positions=len(host.timesteps), identity=dict(identity, split=split, partial=True),
                        expected_examples=pair_examples)
                    pair.save(partial)
                    save_file({f"{path}.{kind}": getattr(state, kind).contiguous()
                               for path, state in final_states.items() for kind in ("image", "audio")},
                              str(partial / "final-states.safetensors"))
                all_rows.extend(pair.records)
                event(phase="pair_collected", split=split, scene=scene["id"], seed=seed,
                      rows=len(pair), gpu=lease.check())
        cache = NativeCache(all_rows, contexts, positions=len(host.timesteps),
            identity=dict(identity, split=split), expected_examples=expected_examples(plan, split))
        cache.save(destination)
        event(phase="cache_complete", split=split, rows=len(cache), fingerprint=cache.index["fingerprint"])
    write_json(args.output / "COLLECTION_RESULT.json", dict(status="complete", splits=args.splits,
        peak_bytes=torch.cuda.max_memory_allocated(), pretrained_training_updates=0, identity=identity))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "collect", "qualify-loaded", "lr-search"):
        p = sub.add_parser(name)
        p.add_argument("--model-directory", type=Path, required=True)
        p.add_argument("--inventory", type=Path, required=True)
        p.add_argument("--source-manifest", type=Path, required=True)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name in ("collect", "qualify-loaded", "lr-search"):
            p.add_argument("--conditioning", type=Path, required=True)
        if name == "collect":
            p.add_argument("--splits", nargs="+", choices=("train", "development", "confirmation"), required=True)
        if name in ("qualify-loaded", "lr-search"):
            p.add_argument("--cache-root", type=Path, required=True)
        if name == "lr-search":
            p.add_argument("--qualification", type=Path, required=True)
            p.add_argument("--rates", nargs="+", type=float, required=True)
            p.add_argument("--seed", type=int, default=17)
            p.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    from .gpu_lease import GpuLease
    with GpuLease(required_free_mib=14000 if args.command == "encode" else 40500) as lease:
        if args.command == "encode":
            encode(args)
        elif args.command == "collect":
            collect(args, lease)
        elif args.command == "qualify-loaded":
            from .native_fit import qualify_loaded
            qualify_loaded(args, lease)
        elif args.command == "lr-search":
            from .native_fit import run_trials
            run_trials(args, lease)


if __name__ == "__main__":
    main()
