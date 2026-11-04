"""Bind owned P-H3 response surfaces to HyperGAN's actual D/G executor.

This is a new KA2 cohort. It does not reproduce or resume the Anima Game.
The host and its hooks remain externally owned; only adapter leaves and the
critic enter HyperGAN's graph/checkpoints. EMA graphs are parameter shadows,
never copied pretrained models or independently attached inference hosts.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import inspect
import platform
from importlib.metadata import version

from .runtime import require_runtime, source_digest

ENGINE = require_runtime()

import torch
from torch import nn
from particlegan import ParticlePrior, Recipe
from hypergan.checkpoints import trainer_state, restore_trainer, capture_rng, restore_rng
from hypergan.config import DEFAULT
from hypergan.objective_program import compile_legacy_program
from hypergan.particlegan_integration import particlegan_recipe
from hypergan.recipes import ComponentGraph
from hypergan.training import ReferenceTrainer, _MetricScalarTransfer, _skip_cpu_capture_guard

COHORT = "ph3-hypergan-develop-ka2-paired-v1"


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return deepcopy(value)


def _hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _Residual(nn.Module):
    def __init__(self, leaves, predict):
        super().__init__()
        self.leaves = nn.ParameterList(leaves)
        object.__setattr__(self, "predict", predict)

    def __deepcopy__(self, memo):
        # The callback closes over a host. Never deepcopy that host or its hooks.
        result = type(self)(deepcopy(list(self.leaves), memo), None)
        memo[id(self)] = result
        return result

    def forward(self, indices, scale, noise):
        if self.predict is None:
            raise RuntimeError("EMA shadow requires export into the owned adapter")
        residual = self.predict(indices.tolist())
        if residual.shape != noise.shape or residual.dtype != torch.float32:
            raise ValueError("callback must retain the full FP32 response matrix")
        return noise + residual / scale


class _Rows:
    resume_stateless = True

    def __init__(self, run, scale, edit_rms, positions):
        self.run, self.scale, self.rms = run, scale.cpu(), edit_rms.cpu()
        self.positions = torch.tensor(positions, dtype=torch.long)

    def __call__(self, batch_size, *, generator):
        ids = torch.randint(len(self.positions), (batch_size,), generator=generator)
        positions = self.positions[ids]
        rms = self.rms[positions]
        start = (rms / .28).clamp_min(.03)
        sigma = start * (.03 / start) ** min(self.run.step / 1600., 1.)
        sigma = torch.where(start > rms * 1.3, torch.maximum(sigma, rms * 1.3), sigma)
        noise = torch.randn(batch_size, self.scale.shape[-1], generator=generator) * sigma[:, None]
        return dict(real=noise, indices=ids, scale=self.scale[positions])


class _Critic(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, candidate):
        return self.module(candidate)


class PairedResidualRun(ReferenceTrainer):
    """Real HyperGAN update/optimizers/penalty/EMA/checkpoint execution.

    ``adapter.particles`` is the one prior bank; all other adapter parameters
    belong only to G. ``predict_residual`` returns adapted-neutral minus frozen
    positive in full response coordinates. A frozen host must stay outside both
    modules. Whole-batch execution needs its own measured memory qualification.
    """

    def __init__(self, adapter, critic, predict_residual, normalization, positions,
                 *, identity, seed=17, generator_lr=2e-5, critic_lr=9e-4,
                 particle_lr=6e-3, steps=1600, batch_size=8):
        require_runtime()
        if not identity or not callable(predict_residual):
            raise ValueError("explicit host/cache identity and full residual callback required")
        self.adapter = adapter
        self.device = adapter.particles.device
        if adapter.particles.shape != (128, 4) or adapter.particles.dtype != torch.float32:
            raise ValueError("this cohort requires one FP32 128x4 particle bank")
        self.leaf_names = tuple(n for n, _ in adapter.named_parameters() if n != "particles")
        leaves = [p for n, p in adapter.named_parameters() if n != "particles"]
        owned = list(adapter.parameters()) + list(critic.parameters())
        if (not leaves or len({id(p) for p in owned}) != len(owned)
                or any(not p.requires_grad or p.device != self.device or p.dtype != torch.float32 for p in owned)):
            raise ValueError("disjoint trainable FP32 adapter and critic ownership required")
        scale, rms = (_cpu(normalization[k]) for k in ("scale", "edit_rms"))
        if (scale.dtype != torch.float32 or rms.dtype != torch.float32
                or scale.ndim != 2 or rms.shape != scale.shape[:1] or not positions
                or any(type(p) is not int or not 0 <= p < len(rms) for p in positions)
                or any(not torch.isfinite(v).all() or (v <= 0).any() for v in (scale, rms))):
            raise ValueError("finite positive per-position full-response normalization required")
        # Snapshot the modern recipe completely. These overrides bind the paired
        # data surface; KA2 interventions, penalty and cosine rates remain modern.
        recipe = Recipe(z_dim=4, num_particles=128, standardize=False,
            batch_size=batch_size, total_steps=steps, lr=generator_lr,
            d_lr_mult=critic_lr / generator_lr, prior_lr_mult=particle_lr / generator_lr,
            prior_reg=1., input_noise_std=0., output_noise_std=0.)
        self.config = deepcopy(DEFAULT)
        self.config["recipe"] = asdict(recipe)
        self.config["optimizer"].update(lr=recipe.lr, d_lr_mult=recipe.d_lr_mult,
            prior_lr_mult=recipe.prior_lr_mult, betas=list(recipe.betas),
            prior_betas=list(recipe.prior_betas or recipe.betas),
            d_guard_ratio=recipe.d_guard_ratio, d_guard_min_steps=recipe.d_guard_min_steps,
            latent_damping_max_rate=recipe.latent_damping_max_rate)
        self.config["training"].update(steps=steps, batch_size=batch_size, seed=seed,
            device=str(self.device), ema=recipe.ema_decay, phase_draws="independent",
            network_lr_floor=recipe.network_lr_floor, network_lr_horizon_cap=recipe.network_lr_horizon_cap,
            input_noise_std=0., output_noise_std=0.)
        self.config["prior_regularizer"].update(weight=1., rows="full")
        self.recipe = particlegan_recipe(self.config)
        self.prior = ParticlePrior(num_particles=128, z_dim=4, device=self.device)
        self.prior.z = adapter.particles
        self.recipe.apply_prior_policy(self.prior)
        self.graph = ComponentGraph.__new__(ComponentGraph)
        nn.Module.__init__(self.graph)
        self.graph.specs = {
            "generator": dict(inputs=dict(indices="batch.indices", scale="batch.scale", noise="batch.real")),
            "discriminator": dict(inputs=dict(candidate="candidate"))}
        # This is a bound-object trainer, reconstructed by this constructor.
        # Do not leave unrelated DEFAULT reference networks/prior in its identity.
        self.config["name"] = COHORT + f"-batch{batch_size}"
        self.config["components"] = {
            "generator": dict(factory="ph3_hypergan.training:_Residual", args={}, trainable=True,
                              inputs=deepcopy(self.graph.specs["generator"]["inputs"])),
            "discriminator": dict(factory=type(critic).__module__ + ":" + type(critic).__qualname__,
                args=deepcopy(getattr(critic, "identity", {})), trainable=True,
                inputs=deepcopy(self.graph.specs["discriminator"]["inputs"]))}
        self.config["data"] = dict(factory="ph3_hypergan.training:_Rows", args=dict(
            binding="owned-full-response-callback", count=len(positions), positions=list(positions)))
        self.config["prior"]["args"] = dict(num_particles=128, z_dim=4)
        critic = _Critic(critic)
        self.graph.models = nn.ModuleDict(dict(generator=_Residual(leaves, predict_residual), discriminator=critic))
        self.gan = self.recipe.make_loss()
        self.spread = self.recipe.make_prior_regularizer(prior=self.prior)
        if self.recipe.prior_regularizer_checkpoint(self.spread):
            raise ValueError("stateful regularizer requires another cohort")
        self.objectives = []
        self.opt_d = _skip_cpu_capture_guard(self.recipe.make_critic_optimizer(
            critic, ema_critic=deepcopy(critic).eval()))
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.program = compile_legacy_program(self.graph, self.prior, self.config,
            [], self.gan, self.penalty, self.spread)
        self.opt_g = _skip_cpu_capture_guard(self.recipe.make_generator_optimizer([
            dict(params=list(self.program.generator_parameters), lr=generator_lr, role="generator"),
            dict(params=[adapter.particles], lr=particle_lr, role="prior", betas=tuple(self.config["optimizer"]["prior_betas"]))],
            modules=self.graph, latent_table=adapter.particles))
        self.lr_roles = [["generator", "prior"], ["critic"]]
        self.base_lrs = [[g["lr"] for g in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.ema_graph = deepcopy(self.graph).eval().requires_grad_(False)
        self.ema_prior = deepcopy(self.prior).eval().requires_grad_(False)
        self.streams = {name: torch.Generator(device=device).manual_seed(seed + offset)
            for name, device, offset in (("data", "cpu", 1), ("prior", self.device, 2),
                ("penalty", self.device, 3), ("noise", self.device, 5))}
        self.step, self.failed_step, self.last_batch = 0, False, None
        self.data = _Rows(self, scale, rms, positions)
        self._metric_transfer, self._unscale_scalars = _MetricScalarTransfer(self.device), {}
        self.signature = dict(cohort=COHORT + f"-batch{batch_size}", engine=ENGINE, identity=deepcopy(identity),
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=version("numpy"),
                device=str(self.device), default_dtype=str(torch.get_default_dtype()),
                deterministic=torch.are_deterministic_algorithms_enabled(), tf32=torch.backends.cuda.matmul.allow_tf32),
            bridge_python_sha256=source_digest(Path(__file__).parent),
            component_python_sha256={label: hashlib.sha256(Path(inspect.getfile(type(module))).read_bytes()).hexdigest()
                for label, module in (("adapter", adapter), ("critic", critic.module))},
            config=deepcopy(self.config), resolved_recipe=asdict(self.recipe), positions=list(positions), seed=seed,
            adapter_architecture=deepcopy(getattr(adapter, "identity", {})),
            adapter_graphs=[deepcopy(getattr(branch, "identity", {})) for branch in getattr(adapter, "branches", [])],
            normalization_sha256=_hash(dict(scale=scale, edit_rms=rms)),
            initial_adapter_sha256=_hash(adapter.state_dict()), initial_critic_sha256=_hash(critic.state_dict()),
            adapter_parameter_names=list(self.leaf_names), full_response_width=scale.shape[1],
            paired_noise_horizon=1600, prior_regularization_rows="full128")
        # Freeze tuples as JSON arrays once so manifest roundtrips compare
        # exactly with checkpoint signatures and reject hidden objects/NaN.
        self.signature = json.loads(json.dumps(self.signature, sort_keys=True, allow_nan=False))

    def update(self, *args, **kwargs):
        if self.failed_step or self.step >= self.config["training"]["steps"]:
            raise RuntimeError("restore after interrupted update, or training budget exhausted")
        try:
            with self.adapter.scaled(1.):
                metrics, self.last_batch = super().update(*args, **kwargs)
            return metrics
        except BaseException:
            self.failed_step = True
            raise

    def snapshot(self):
        if self.failed_step:
            raise RuntimeError("partial engine update cannot be checkpointed")
        return _cpu(dict(schema=COHORT + "-checkpoint", signature=self.signature,
                         trainer=trainer_state(self, self.last_batch)))

    def restore(self, packet):
        if packet.get("schema") != COHORT + "-checkpoint" or packet.get("signature") != self.signature:
            raise ValueError("develop engine/host/cache/recipe cohort differs")
        self.last_batch = restore_trainer(self, deepcopy(packet["trainer"]))
        self.failed_step = False

    def adapter_weights(self, *, selection="ema"):
        if selection == "raw":
            return _cpu(self.adapter.state_dict())
        if selection != "ema":
            raise ValueError("select raw or EMA")
        current_names = tuple(n for n, _ in self.adapter.named_parameters() if n != "particles")
        shadow = self.ema_graph.models["generator"].leaves
        if current_names != self.leaf_names or len(shadow) != len(self.leaf_names):
            raise ValueError("adapter/EMA leaf inventory changed")
        values = _cpu(self.adapter.state_dict())
        for name, value in zip(self.leaf_names, shadow, strict=True):
            if values[name].shape != value.shape or values[name].dtype != value.dtype:
                raise ValueError("adapter/EMA leaf shape or dtype changed")
            values[name] = _cpu(value)
        values["particles"] = _cpu(self.ema_prior.z)
        return values


class NativeHyperganRun(PairedResidualRun):
    """End-to-end native packed H3 binding with an alternate modern adapter.

    The caller must attach that adapter explicitly; the classic attachment
    helper's strict runtime check must never be bypassed.
    """
    def __init__(self, host, training_cache, normalization, critic, *, identity, batch_size, **options):
        if (host.adapter is None or training_cache.identity["split"] != "train"
                or training_cache.positions != len(host.timesteps)
                or any(p.requires_grad for p in host.model.parameters())
                or any(row["time"] != float(host.timesteps[row["position"]]) for row in training_cache.records)):
            raise ValueError("frozen native host and matching training cache required")
        if normalization.get("provenance", {}).get("cache_fingerprint") != training_cache.index["fingerprint"]:
            raise ValueError("normalization must bind the same training cache")
        self.host, self.cache = host, training_cache
        def residual(indices):
            values = []
            for index in indices:
                row = training_cache[index]
                velocity = host.velocity(row["state"], training_cache.contexts[row["neutral_context"]], row["position"])
                values.append((host.response(velocity) - row["positive"].to(host.device)).flatten())
            return torch.stack(values)
        super().__init__(host.adapter, critic, residual, normalization,
            [row["position"] for row in training_cache.records],
            identity=dict(identity, cache_fingerprint=training_cache.index["fingerprint"],
                response_surface="full_packed_image_then_audio_velocity", sigmas=list(host.sigmas),
                normalization_provenance=normalization["provenance"],
                targets=list(host.adapter.names)), batch_size=batch_size, **options)

    def evaluate(self, cache, *, selection="ema", raw_path=None):
        from ph3_sliders.evaluation import aggregate_rows, response_statistics
        from ph3_sliders.native_cache import file_hash
        original = _cpu(self.adapter.state_dict())
        modes = [(module, module.training) for module in self.host.model.modules()]
        devices = [self.host.device.index or 0] if self.host.device.type == "cuda" else []
        rng = capture_rng()
        rows, raw = [], {}
        try:
            self.host.model.eval()
            self.adapter.load_state_dict(self.adapter_weights(selection=selection), strict=True)
            with torch.random.fork_rng(devices=devices), torch.no_grad(), self.adapter.scaled(1.):
                for row in cache.records:
                    velocity = self.host.velocity(row["state"], cache.contexts[row["neutral_context"]], row["position"])
                    student = self.host.response(velocity)
                    labels = {key: row[key] for key in ("scene", "seed", "path", "position")}
                    rows.append(dict(labels, **response_statistics(student, row["neutral"], row["positive"])))
                    if raw_path is not None:
                        raw[f"student.{len(rows)-1}"] = student.detach().cpu().contiguous()
            result = dict(aggregate_rows(rows, expected_examples=cache.examples,
                expected_positions=range(cache.positions)), selection=selection, completed_steps=self.step,
                cache_fingerprint=cache.index["fingerprint"], cohort=self.signature["cohort"],
                native_h3_qualified=bool(self.signature["identity"].get("native_qualification_sha256")))
            if raw_path is not None:
                from safetensors.torch import save_file
                raw_path = Path(raw_path)
                if raw_path.exists():
                    raise FileExistsError(raw_path)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(raw, str(raw_path), metadata=dict(cache_fingerprint=cache.index["fingerprint"],
                    selection=selection, completed_steps=str(self.step), cohort=self.signature["cohort"],
                    row_labels=json.dumps([{k: row[k] for k in ("scene", "seed", "path", "position")} for row in rows])))
                result["raw_student_responses"] = dict(path=str(raw_path), sha256=file_hash(raw_path))
            return result
        finally:
            self.adapter.load_state_dict(original, strict=True)
            for module, training in modes:
                module.training = training
            restore_rng(rng)
