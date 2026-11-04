"""Pinned native loaders. No implicit downloads; staged local files only."""

import hashlib
from importlib.metadata import version
import json
import logging
import subprocess
from pathlib import Path

import torch

from .native_cache import file_hash

MODEL_REVISION = "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"
DIFFUSERS_REVISION = "35087a644038eb531f48581708d4763044cb4852"


def configure_numeric_runtime(seed=17):
    """Declare deterministic native settings and log the known BnB cast once."""
    torch.random.default_generator.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.default_generators[0].manual_seed(seed)

    class KnownCastOnce(logging.Filter):
        count = 0

        def filter(self, record):
            if record.getMessage().startswith("MatMul8bitLt: inputs will be cast"):
                self.count += 1
                return self.count == 1
            return True

    cast_filter = KnownCastOnce()
    logging.getLogger("bitsandbytes.autograd._functions").addFilter(cast_filter)
    return cast_filter


def verify_diffusers_source(manifest_path):
    import diffusers

    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text())
    root = Path(diffusers.__file__).resolve().parents[2]
    if document["revision"] != DIFFUSERS_REVISION:
        raise ValueError("native Diffusers revision differs")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != DIFFUSERS_REVISION
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()):
        raise ValueError("complete native Diffusers checkout must match the clean pinned revision")
    verified = {}
    for name, value in document["files"].items():
        path = root / name
        if file_hash(path) != value["sha256"]:
            raise ValueError(f"native Diffusers source changed: {name}")
        verified[name] = value["sha256"]
    return dict(revision=DIFFUSERS_REVISION, complete_git_tree_clean=True, selected_files=verified,
                manifest_sha256=file_hash(manifest_path))


MODERN_COHORT = "ph3-modern-hypergan-hndl-v1"


def runtime_identity(source_manifest, *, quantization, cohort="classic"):
    if cohort not in ("classic", MODERN_COHORT):
        raise ValueError("unknown native collection runtime cohort")
    packages = ["torch", "transformers", "accelerate", "safetensors", "numpy",
                "bitsandbytes", "huggingface-hub", "particlegan", "particle-sliders-core"]
    if cohort == "classic":
        packages.append("particle-sliders-anima-gmix")
    else:
        packages.extend(["hypergan", "hndl", "ph3-hypergan-develop", "p-h3-particle-sliders-hndl"])
    identity = dict(diffusers=verify_diffusers_source(source_manifest), model_revision=MODEL_REVISION,
        packages={name: version(name) for name in packages},
        quantization=quantization, int8_activation_quantization_cast="float16",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        product_sources={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    if cohort == MODERN_COHORT:
        from .modern_identity import modern_engine_identity
        identity.update(runtime_cohort=MODERN_COHORT, modern_runtime=modern_engine_identity())
    return identity


def load_transformer(model_directory, *, device="cuda:0", quantization="int8"):
    from diffusers import BitsAndBytesConfig, MiniMaxH3Transformer3DModel

    options = dict(local_files_only=True, torch_dtype=torch.bfloat16,
                   low_cpu_mem_usage=True, device_map={"": device})
    if quantization == "int8":
        options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
            llm_int8_threshold=6., llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=list(MiniMaxH3Transformer3DModel._keep_in_fp32_modules))
    elif quantization != "bf16":
        raise ValueError("explicit int8 or bf16 cohort required")
    model, loading_info = MiniMaxH3Transformer3DModel.from_pretrained(
        str(Path(model_directory) / "transformer"), output_loading_info=True, **options)
    if any(loading_info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"native checkpoint loading mismatch: {loading_info}")
    object.__setattr__(model, "_ph3_loading_info", loading_info)
    model.requires_grad_(False).eval()
    model.enable_gradient_checkpointing()
    return model


@torch.no_grad()
def verify_loaded_transformer(model, model_directory):
    """Compare every loaded leaf to its pinned source, replaying exact int8 conversion."""
    from safetensors import safe_open
    import bitsandbytes as bnb

    root = Path(model_directory) / "transformer"
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(root / filename, framework="pt", device="cpu") as stream:
            mapping = {k: filename for k in stream.keys()}
    actual_state = model.state_dict()
    expected_keys = set(mapping)
    quantized = {name + ".weight": module for name, module in model.named_modules()
                 if type(module).__name__ == "Linear8bitLt"}
    extra_expected = {name[:-len("weight")] + suffix for name in quantized for suffix in ("SCB", "weight_format")}
    if set(actual_state) != expected_keys | extra_expected:
        raise ValueError("loaded model and checkpoint key sets differ")
    if any(actual_state[name].dtype != torch.uint8 or actual_state[name].numel() != 1
           or actual_state[name].item() != 0 for name in extra_expected if name.endswith("weight_format")):
        raise ValueError("int8 row-major weight format differs")
    verified = {}
    for filename in sorted(set(mapping.values())):
        with safe_open(root / filename, framework="pt", device="cpu") as stream:
            for name in sorted(k for k, v in mapping.items() if v == filename):
                original = stream.get_tensor(name)
                actual = actual_state[name]
                if actual.shape != original.shape:
                    raise ValueError("loaded native tensor shape differs: " + name)
                if name in quantized:
                    module = quantized[name]
                    # Match pinned Diffusers BF16 cast, then Int8Params' FP16 row quantization.
                    value = original.to(torch.bfloat16).to(device=actual.device, dtype=torch.float16).contiguous()
                    q, scale, _ = bnb.functional.int8_vectorwise_quant(value)
                    actual_scale = module.weight.SCB if module.weight.SCB is not None else module.state.SCB
                    if not torch.equal(q, actual) or not torch.equal(scale, actual_scale):
                        raise ValueError("int8 source conversion differs: " + name)
                    verified[name] = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                        source_dtype=str(original.dtype), method="exact_replayed_BF16_FP16_int8_row_quantization")
                    del value, q, scale
                else:
                    if not torch.equal(actual.detach().cpu(), original.to(actual.dtype)):
                        raise ValueError("loaded native weight differs: " + name)
                    verified[name] = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                        source_dtype=str(original.dtype), method="exact_source_bytes_after_declared_dtype_cast")
                del original
    return dict(schema="ph3-full-loaded-weight-binding-v1", keys=len(verified),
        quantized_weight_matrices=len(quantized), all_keys_shapes_and_loaded_values_verified=True,
        loading_info=model._ph3_loading_info, leaves=verified)


@torch.no_grad()
def encode_prompts(model_directory, prompts, *, device="cuda:0"):
    """Verbatim native T2I presentation, no upsampler or chat template."""
    from diffusers import PH3ImageTextConnector
    from diffusers.modular_pipelines.p_h3_image.encoders import get_p_h3_image_prompt_embeds
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    root = Path(model_directory)
    tokenizer = Qwen2TokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(root / "processor", local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(root / "text_encoder",
        dtype=torch.bfloat16, local_files_only=True, device_map={"": device}).requires_grad_(False).eval()
    connector = PH3ImageTextConnector.from_pretrained(root / "text_connector",
        torch_dtype=torch.float32, local_files_only=True).to(device).requires_grad_(False).eval()
    result = {}
    try:
        for name, prompt in prompts.items():
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            result[name] = get_p_h3_image_prompt_embeds(encoder, connector, processor, ids,
                device=torch.device(device), dtype=encoder.dtype).detach().cpu().contiguous()
    finally:
        del encoder, connector
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return result


@torch.no_grad()
def load_decoder(model_directory, *, device="cuda:0"):
    from diffusers import AutoencoderKLPH3Image
    vae = AutoencoderKLPH3Image.from_pretrained(Path(model_directory) / "vae",
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).requires_grad_(False).eval()
    vae.enable_tiling()
    return vae


@torch.no_grad()
def decode_image(model_directory, latents, *, device="cpu", decoder=None):
    from PIL import Image
    vae = decoder if decoder is not None else load_decoder(model_directory, device=device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    actual = latents.to(device) * std + mean
    image = vae.decode(actual.to(vae.dtype), return_dict=False)[0][:, :, 0].float()
    pixel_mean = torch.tensor((.485, .456, .406), device=device).view(1, -1, 1, 1)
    pixel_std = torch.tensor((.229, .224, .225), device=device).view(1, -1, 1, 1)
    pixels = ((image * pixel_std + pixel_mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(pixels)
