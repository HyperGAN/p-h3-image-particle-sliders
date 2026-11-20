"""Execute the unchanged Anima routed/gmix network math as native HNDL graphs.

No particle-sliders/classic import is needed. The optional package can coexist
with the current HyperGAN/ParticleGAN runtime in a separately installed process.
It supplies networks, never substitutes its own loss or training loop.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=1)
def _verified_identity():
    import hndl
    if version("hndl") != "0.10.0":
        raise RuntimeError("the qualified network graphs require hndl==0.10.0")
    expected = json.loads(Path(__file__).with_name("runtime.json").read_text())
    root = Path(hndl.__file__).resolve().parent
    actual = {str(p.relative_to(root)): _sha(p.read_bytes()) for p in root.rglob("*.py")}
    if actual != expected["python_sources"]:
        raise RuntimeError("HNDL source closure differs from the qualified canonical runtime")
    return dict(repository=expected["repository"], commit=expected["commit"],
                hndl_version=version("hndl"), python_sources_sha256=_sha(
                    json.dumps(actual, sort_keys=True).encode()))


def runtime_identity():
    return deepcopy(_verified_identity())


def _network(source, inputs, output, device):
    from hndl.torch import network
    runtime_identity()
    graph = network(source, input_shape=inputs, output_shape=output, device="cpu")
    graph.to(device=device, dtype=torch.float32)
    identity = dict(runtime_identity(), source_sha256=_sha(source.encode()),
                    plan_sha256=_sha(graph.plan.to_json().encode()))
    return graph, identity


def projection_source(output_width):
    # mlp(inputs,outputs,width) from the Anima reference has three hidden
    # affine+LeakyReLU(.2) stages, then an affine readout.
    lines = ['h = linear(value, 8, bias=False, name="down")']
    previous = "h"
    for index in range(3):
        lines.extend([f'r{index} = linear({previous}, 16, name="router{index}")',
                      f'ra{index} = leaky_relu(r{index}, 0.2)'])
        previous = f"ra{index}"
    lines.extend([f'q = linear({previous}, 4, name="router3")',
                  'pt = transpose(particles, 1, 2)',
                  'energies = matmul(q, pt)',
                  'scaled = scale(energies, 0.5)',
                  'weights = softmax(scaled, -1)',
                  'z = matmul(weights, particles)',
                  'joined = concat(h, z, axis=2)'])
    previous = "joined"
    for index in range(3):
        lines.extend([f'n{index} = linear({previous}, 48, name="net{index}")',
                      f'na{index} = leaky_relu(n{index}, 0.2)'])
        previous = f"na{index}"
    lines.extend([f'routed = linear({previous}, 8, name="net3")',
                  f'linear(routed, {output_width}, bias=False, name="up")'])
    return "\n".join(lines)


class RoutedProjection(nn.Module):
    """Rank-8 projection; the owned bank is supplied through a named HNDL port."""
    def __init__(self, base, *, device="cpu"):
        super().__init__()
        self.input_width, self.output_width = base.in_features, base.out_features
        self.source = projection_source(self.output_width)
        self.graph, self.identity = _network(self.source,
            {"value": ["B", 1, self.input_width], "particles": ["B", 128, 4]},
            ["B", 1, self.output_width], device)
        nn.init.kaiming_uniform_(self.graph["down"].weight, a=5 ** .5)
        nn.init.zeros_(self.graph["up"].weight)

    def forward(self, value, particles):
        shape = value.shape
        # HNDL matmul uses matching explicit batch dimensions, so a view of
        # the single shared bank is broadcast here without creating parameters.
        value = value.float().reshape(-1, 1, self.input_width)
        result = self.graph(value=value, particles=particles.unsqueeze(0).expand(value.shape[0], -1, -1))
        return result.reshape(*shape[:-1], self.output_width)


class RoutedAdapter(nn.Module):
    """Own hooks and exactly one trainable [128,4] bank, never the frozen host."""
    def __init__(self, host, targets, *, device="cpu"):
        super().__init__()
        device = torch.device(device)
        if any(p.requires_grad or p.device != device for p in host.parameters()):
            raise ValueError("attachment requires a frozen host on the requested device")
        self.names = tuple(targets)
        modules = dict(host.named_modules())
        selected = [modules.get(name) for name in self.names]
        if not selected or len(set(self.names)) != len(self.names):
            raise ValueError("explicit unique projection targets required")
        if any(not isinstance(m, nn.Linear) for m in selected):
            raise ValueError("every target must be a native Linear")
        if len({id(m) for m in selected}) != len(selected):
            raise ValueError("aliased projection targets are not allowed")
        self.handles, self.multiplier, self.closed = [], 0., False
        self.identity = dict(runtime_identity(), formulation="anima-routed-gmix-hndl-v1",
                             rank=8, particles=[128, 4], router_width=16, width=48)
        try:
            self.particles = nn.Parameter(torch.randn(128, 4, device=device, dtype=torch.float32))
            self.branches = nn.ModuleList(RoutedProjection(m, device=device) for m in selected)
            for index, module in enumerate(selected):
                def add_branch(module, args, output, index=index):
                    if self.multiplier == 0:
                        return output
                    delta = self.branches[index](args[0], self.particles) * self.multiplier
                    return output + delta.to(dtype=output.dtype)
                self.handles.append(module.register_forward_hook(add_branch))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        if self.closed:
            raise RuntimeError("adapter is closed")
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite in [0,1]")
        previous, self.multiplier = self.multiplier, float(strength)
        try:
            yield self
        finally:
            self.multiplier = previous

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True


def critic_source(input_width, *, tokens, width, layers, heads, score_bound):
    lines = [f'mixed = linear({tokens * width}, name="mix")',
             f't = reshape(mixed, {tokens}, {width})',
             f'h0 = pos_embed(t, {tokens}, name="pos")']
    for index in range(layers):
        # Explicit matmul/softmax keeps the classic cap penalty's second
        # derivative available; native attention uses fused SDPA kernels.
        i = index
        lines.extend([
            f'norm1_{i} = rms_norm(h{i}, eps=0.001, name="norm1_{i}")',
            f'qkv_{i} = linear(norm1_{i}, {3 * width}, name="qkv_{i}")',
            f'q_{i}, kv_{i} = split(qkv_{i}, {width}, dim=2)',
            f'k_{i}, v_{i} = split(kv_{i}, {width}, dim=2)',
        ])
        for label in ("q", "k", "v"):
            lines.extend([
                f'{label}r_{i} = reshape({label}_{i}, {tokens}, {heads}, {width // heads})',
                f'{label}h_{i} = permute({label}r_{i}, 2, 1, 3)',
            ])
        lines.extend([
            f'kt_{i} = transpose(kh_{i}, 2, 3)',
            f'e_{i} = matmul(qh_{i}, kt_{i})',
            f's_{i} = scale(e_{i}, {(width // heads) ** -.5!r})',
            f'a_{i} = softmax(s_{i}, -1)',
            f'av_{i} = matmul(a_{i}, vh_{i})',
            f'ar_{i} = permute(av_{i}, 2, 1, 3)',
            f'af_{i} = reshape(ar_{i}, {tokens}, {width})',
            f'out_{i} = linear(af_{i}, {width}, name="out_{i}")',
            f'res_{i} = add(h{i}, out_{i})',
            f'norm2_{i} = rms_norm(res_{i}, eps=0.001, name="norm2_{i}")',
            f'ffn1_{i} = linear(norm2_{i}, {width * 2}, name="ffn1_{i}")',
            f'act_{i} = silu(ffn1_{i})',
            f'ffn2_{i} = linear(act_{i}, {width}, name="ffn2_{i}")',
            f'h{i+1} = add(res_{i}, ffn2_{i})',
        ])
    lines.extend([f'avg = pool_tokens(h{layers}, "mean")',
                  f'peak = pool_tokens(h{layers}, "max")',
                  'pooled = concat(avg, peak, axis=1)',
                  f'paired = reshape(pooled, -1, {width})',
                  'normed = rms_norm(paired, eps=0.001, name="out_norm")',
                  f'flattened = reshape(normed, {2 * width})',
                  'score = linear(flattened, 1, name="head")'])
    if score_bound > 0:
        lines.extend([f'unscaled = scale(score, {1 / score_bound!r})',
                      'bounded = tanh(unscaled)', f'scale(bounded, {score_bound!r})'])
    return "\n".join(lines)


class GlobalMixCritic(nn.Module):
    """Full-response global mix, manual set attention, mean/max and bounded score.

    Input coordinates have already been normalized by the owning GAN runtime.
    Image and coupled audio coordinates are passed together, with no crop/pool
    before the learned full-response first layer.
    """
    def __init__(self, input_width, *, tokens=8, width=48, layers=1, heads=4, score_bound=8., device="cpu"):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (input_width, tokens, width, layers, heads)) or width % heads:
            raise ValueError("positive dimensions and a width divisible by heads required")
        if type(score_bound) not in (int, float) or not math.isfinite(score_bound) or score_bound < 0:
            raise ValueError("finite nonnegative score bound required")
        self.input_width, self.score_bound = input_width, float(score_bound)
        self.source = critic_source(input_width, tokens=tokens, width=width, layers=layers,
                                    heads=heads, score_bound=self.score_bound)
        self.graph, self.identity = _network(self.source, ["B", input_width], ["B", 1], device)
        self.identity.update(formulation="anima-global-mix-hndl-v1", input_width=input_width,
                             tokens=tokens, width=width, layers=layers, heads=heads,
                             score_bound=self.score_bound)
        for module in self.graph.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.graph["mix"].bias, std=.02)

    def forward(self, candidate):
        if candidate.ndim != 2 or candidate.shape[1] != self.input_width:
            raise ValueError("critic requires full normalized residual coordinates [B,H]")
        return self.graph(candidate).squeeze(-1)
