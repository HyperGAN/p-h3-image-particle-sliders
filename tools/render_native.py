"""Render recorded native teacher states or a routed EMA export, offline.

Kept outside the imported product package so rendering preparation cannot change
the source identity of an in-flight frozen-teacher collection.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path


def unpack_image(packed, config, resolution):
    """Inverse of upstream video patchification, preserving native row order."""
    pt, ph, pw = config["patch_size"]
    channels = config["in_channels"]
    height, width = resolution["height"] // 16, resolution["width"] // 16
    if pt != 1 or height % ph or width % pw:
        raise ValueError("single native image patch geometry required")
    rows = packed.reshape(1, 1, height // ph, width // pw, channels, pt, ph, pw)
    return rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, channels, 1, height, width).contiguous()


def contact_sheet(images, labels, path):
    from PIL import Image, ImageDraw

    thumb = 384
    sheet = Image.new("RGB", (thumb * len(images), thumb + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (picture, label) in enumerate(zip(images, labels, strict=True)):
        picture = picture.copy()
        picture.thumbnail((thumb, thumb))
        sheet.paste(picture, (index * thumb, 40))
        draw.text((index * thumb + 8, 12), label, fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cached-teachers", "export"))
    for name in ("model-directory", "inventory", "source-manifest", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-pair", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--conditioning", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--split", choices=("train", "development", "confirmation"), default="development")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "export" and (args.device != "cuda:0" or any(x is None for x in
            (args.export, args.conditioning, args.cache_root, args.scene, args.seed))):
        parser.error("export requires GPU1, --export, --conditioning, --cache-root, --scene and --seed")
    if args.mode == "cached-teachers" and args.cache_pair is None:
        parser.error("cached-teachers requires --cache-pair")
    from ph3_sliders.gpu_lease import GpuLease

    with (GpuLease(required_free_mib=40500 if args.mode == "export" else 10000)
          if args.device == "cuda:0" else nullcontext()) as lease:
        render(args, lease)


def render(args, lease):
    import torch
    from safetensors.torch import load_file, save_file
    from ph3_sliders.native_cache import NativeCache, file_hash
    from ph3_sliders.native_cli import event, load_encoded, verify_staged, write_json
    from ph3_sliders.native_runtime import (configure_numeric_runtime, decode_image,
        load_decoder, load_transformer, runtime_identity, verify_diffusers_source,
        verify_loaded_transformer)

    configure_numeric_runtime()
    if args.device == "cuda:0":
        torch.cuda.set_per_process_memory_fraction(.88, 0)
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True)
    document = dict(schema="ph3-native-samples-v1", status="incomplete", mode=args.mode,
        plan_sha256=file_hash(args.plan), tool_sha256=file_hash(Path(__file__)),
        device=args.device, strengths=plan["samples"]["strengths"], files={})
    write_json(args.output / "RENDER.json", document)
    states, labels = {}, []
    host = None
    try:
        if args.mode == "cached-teachers":
            cache = NativeCache.load(args.cache_pair)
            states_path = args.cache_pair / "final-states.safetensors"
            values = load_file(str(states_path), device="cpu")
            if set(values) != {f"{path}.{kind}" for path in ("neutral", "positive") for kind in ("image", "audio")}:
                raise ValueError("complete recorded native teacher final states required")
            if cache.identity["plan_sha256"] != document["plan_sha256"]:
                raise ValueError("cached teacher plan differs")
            current_diffusers = verify_diffusers_source(args.source_manifest)
            if current_diffusers != cache.identity["diffusers"]:
                raise ValueError("cached teacher source differs")
            config = json.loads((args.model_directory / "transformer/config.json").read_text())
            document.update(cache_fingerprint=cache.index["fingerprint"],
                recorded_states_sha256=file_hash(states_path), source_identity=cache.identity,
                scene=cache.records[0]["scene"], seed=cache.records[0]["seed"],
                note="Frozen native teacher references; these are not trained slider samples.")
            for path, label in (("neutral", "OFF neutral teacher"), ("positive", "positive teacher")):
                for kind in ("image", "audio"):
                    value = values[f"{path}.{kind}"]
                    if value.dtype != torch.float32 or not torch.isfinite(value).all():
                        raise ValueError("finite FP32 recorded native final states required")
                states[path] = unpack_image(values[f"{path}.image"], config, plan["resolution"])
                labels.append(label)
        else:
            from ph3_sliders.native_host import H3Host

            staged = verify_staged(args.model_directory, args.inventory)
            staged_path = args.cache_root / "STAGED_WEIGHTS.json"
            if staged != json.loads(staged_path.read_text()):
                raise ValueError("staged inference weights differ from trained base")
            artifact_document = json.loads((args.export / "manifest.json").read_text())
            modern = artifact_document.get("schema") == "ph3-hndl-routed-inference-v1"
            if modern:
                from ph3_hypergan import verify_teacher_identity
                from ph3_hypergan.runtime import require_runtime, source_digest
                from ph3_hndl import runtime_identity as hndl_identity
                import ph3_hypergan.training as training

                identity = artifact_document["signature"]["identity"]
                if (identity["engine"] != require_runtime() or identity["hndl"] != hndl_identity()
                        or identity["training_signature"]["bridge_python_sha256"] != source_digest(Path(training.__file__).parent)
                        or identity["driver_sha256"] != file_hash(Path(__file__).with_name("modern_native.py"))):
                    raise ValueError("modern inference artifact engine/bridge/driver cohort differs")
                verified = verify_teacher_identity(identity["teachers"]["train"]["teacher_collection"],
                    source_manifest=args.source_manifest, quantization=plan["numerical_cohort"],
                    staged_weights_sha256=file_hash(staged_path), plan_sha256=file_hash(args.plan),
                    encoded_conditioning_sha256=file_hash(args.conditioning / "manifest.json"))
                if verified != identity["teachers"]["train"]:
                    raise ValueError("modern inference frozen forward cohort differs")
            else:
                identity = dict(runtime_identity(args.source_manifest, quantization=plan["numerical_cohort"]),
                    staged_weights_sha256=file_hash(staged_path))
            model = load_transformer(args.model_directory)
            binding = verify_loaded_transformer(model, args.model_directory)
            write_json(args.output / "LOADED_WEIGHT_BINDING.json", binding)
            host = H3Host(model, device=args.device, latent_height=plan["resolution"]["height"] // 16,
                latent_width=plan["resolution"]["width"] // 16, sigmas=plan["sigmas"])
            embeddings, encoded = load_encoded(args.conditioning, plan)
            if args.scene not in {x["id"] for x in plan["splits"][args.split]["scenes"]}:
                raise ValueError("sample scene outside declared split")
            neutral = host.conditioning(args.scene + ":neutral", embeddings[args.scene + ":neutral"])
            positive = host.conditioning(args.scene + ":positive", embeddings[args.scene + ":positive"])
            if modern:
                from ph3_hndl import load_adapter
                host.adapter = load_adapter(host.model, host.targets, args.export,
                    identity=identity, device=host.device)
                artifact = host.adapter.artifact
            else:
                from ph3_sliders.native_artifacts import load_export
                artifact = load_export(host, args.export, identity)
            document.update(artifact=artifact, artifact_manifest_sha256=file_hash(args.export / "manifest.json"),
                source_identity=identity, scene=args.scene, seed=args.seed, split=args.split,
                conditioning_manifest_sha256=file_hash(args.conditioning / "manifest.json"))
            recorded = {}
            for strength in plan["samples"]["strengths"]:
                name = "strength-" + str(strength)
                final = host.rollout(neutral, args.seed, strength=strength)
                states[name] = host.unpack_image(final).cpu()
                recorded[name + ".image"], recorded[name + ".audio"] = final.image.cpu(), final.audio.cpu()
                labels.append("slider " + str(strength))
                event(phase="sample_latent_complete", strength=strength, gpu=lease.check())
            final = host.rollout(positive, args.seed, strength=0.)
            states["positive-teacher"] = host.unpack_image(final).cpu()
            recorded["positive-teacher.image"], recorded["positive-teacher.audio"] = final.image.cpu(), final.audio.cpu()
            labels.append("positive teacher")
            save_file(recorded, str(args.output / "final-states.safetensors"))
            document["files"]["final-states.safetensors"] = file_hash(args.output / "final-states.safetensors")
            host.adapter.close()
            host.adapter = None
            del host, model
            host = None
            torch.cuda.empty_cache()
        # Release the full transformer before loading the tiled native decoder.
        document["decoder_components"] = verify_staged(args.model_directory, args.inventory, ("vae",))
        decoder = load_decoder(args.model_directory, device=args.device)
        images = []
        for name, latents in states.items():
            image = decode_image(args.model_directory, latents, device=args.device, decoder=decoder)
            image.save(args.output / (name + ".png"))
            document["files"][name + ".png"] = file_hash(args.output / (name + ".png"))
            images.append(image)
            event(phase="image_decoded", name=name)
        contact_sheet(images, labels, args.output / "contact-sheet.png")
        document["files"]["contact-sheet.png"] = file_hash(args.output / "contact-sheet.png")
        document.update(status="complete", render_process_training_updates=0,
            pretrained_training_updates=artifact["completed_steps"] if args.mode == "export" else 0,
            rendered_selection=artifact["selection"] if args.mode == "export" else "frozen_teacher",
            peak_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda:0" else None)
        write_json(args.output / "RENDER.json", document)
        event(phase="native_samples_complete", output=str(args.output))
    except BaseException as error:
        document.update(status="failed", error=repr(error))
        write_json(args.output / "RENDER.json", document)
        raise
    finally:
        if host is not None and host.adapter is not None:
            host.adapter.close()


if __name__ == "__main__":
    main()
