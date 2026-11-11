"""Fresh-only public PG095 QR/R2 at the actual p-h3-image host laws."""

import hashlib
import math

import torch
from torch import nn
from particlegan import init as pg_init

from . import PROFILE, ownership


class _DeclaredScope(nn.Module):
    """Alias the existing fresh Parameters; never replace or wrap their math."""

    def __init__(self, named, specifications):
        super().__init__()
        self.specifications = specifications
        for name, parameter in named:
            self.register_parameter(name, parameter)


# Only this new helper-owned declaration class is registered. No original
# module class/registry entry/core function is overridden or monkeypatched.
pg_init.register(_DeclaredScope, lambda module: module.specifications)


def tensor_sha(value):
    source = value.detach().cpu()
    flat = torch.empty((source.numel(),), dtype=source.dtype, device="cpu")
    flat.copy_(source.reshape(-1))
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _seed(seed, name):
    return int.from_bytes(hashlib.sha256((PROFILE + ":" + str(seed) + ":" + name).encode()).digest()[:8], "little") % (2**63 - 1)


def initialize(owner, config, *, optimizers=()):
    if owner.mode != "fresh" or owner._initialization is not None or optimizers or hasattr(owner, "game"):
        raise ValueError("fresh original components only, before every optimizer")
    adapter, critic = owner.adapter, owner.critic
    named = [("adapter." + n, p) for n, p in adapter.named_parameters()]
    named += [("critic." + n, p) for n, p in critic.named_parameters()]
    if len({id(p) for _, p in named}) != len(named) or any(p.grad is not None or not p.requires_grad or p.dtype != torch.float32 or not torch.isfinite(p).all() for _, p in named):
        raise ValueError("unique finite original fresh FP32 Parameters with no gradients required")
    parameters = dict(named)
    if adapter.particles.shape != (128, 4):
        raise ValueError("original complete128x4 bank required")
    declared = {id(p): (pg_init.KEEP, "original-constructor-KEEP", "KEEP") for _, p in named}
    declared[id(adapter.particles)] = (pg_init.R2Normal(0.0, 1.0), "bank-R2Normal-std1", "R2Normal(0,1)")
    zero_up = []
    for branch in adapter.branches:
        down, up = branch.graph["down"].weight, branch.graph["up"].weight
        fan = branch.graph["down"].in_features
        bound = 1.0 / math.sqrt(fan)
        declared[id(down)] = (pg_init.Uniform(-bound, bound), "down-sqrt5-constructor-scale-QR", "Uniform(+/-1/sqrt(fan));variance1/(3fan)")
        if torch.count_nonzero(up).item() != 0:
            raise ValueError("original fresh zero UP required")
        declared[id(up)] = (pg_init.KEEP, "protected-zero-UP", "KEEP-zero")
        zero_up.append(up)
    before = {name: tensor_sha(p) for name, p in named}
    identities = [(name, id(p)) for name, p in named]
    # Base metadata is compared only around initialization. No coarse immutable
    # version-counter claim is made across ordinary native inference/restores.
    base = [(id(p), p._version, p.requires_grad, p.dtype, p.device, tuple(p.shape)) for p in owner.host.model.parameters()]
    aliases = [("p" + str(i), p) for i, (_, p) in enumerate(named)]
    specs = {alias: declared[id(p)][0] for alias, p in aliases}
    seeds = {alias: _seed(config["seed"], name) for (alias, _), (name, _) in zip(aliases, named)}
    scope = _DeclaredScope(aliases, specs)
    pg_init.deterministic_orthogonal_(scope, seed=config["seed"], strict=True, parameter_seeds=seeds)
    if identities != [(name, id(p)) for name, p in list(("adapter." + n, p) for n, p in adapter.named_parameters()) + list(("critic." + n, p) for n, p in critic.named_parameters())]:
        raise RuntimeError("initializer replaced original Parameters")
    if base != [(id(p), p._version, p.requires_grad, p.dtype, p.device, tuple(p.shape)) for p in owner.host.model.parameters()]:
        raise RuntimeError("fresh initializer touched the frozen pretrained host")
    rows = []
    for name, parameter in parameters.items():
        _, role, distribution = declared[id(parameter)]
        after = tensor_sha(parameter)
        if role in ("original-constructor-KEEP", "protected-zero-UP") and before[name] != after:
            raise RuntimeError("constructor-law KEEP/zero-UP changed")
        rows.append({"name": name, "role": role, "distribution": distribution, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "seed": _seed(config["seed"], name), "before_sha256": before[name], "after_sha256": after})
    if any(torch.count_nonzero(p).item() for p in zero_up):
        raise RuntimeError("zero adapter output was not preserved")
    receipt = {
        "schema": "PH3Image-native-explicit-fresh95-before-optimizers-v1",
        "profile": PROFILE,
        "cohort_id": owner.cohort_id,
        "config": config.copy(),
        "sources": ownership.sources(),
        "seed": config["seed"],
        "parameter_policies": rows,
        "existing_Parameter_objects_preserved": True,
        "pretrained_parameters_touched": False,
        "critic_policy": "original GlobalMixCritic constructor KEEP: Xavier/zero-bias then mix.bias Normal(0,.02), norms/position KEEP",
        "new_optimizer_count_at_receipt": 0,
        "weight_EMA": None,
        "old_cohort_migration": False,
    }
    owner._initialization = receipt
    return receipt
