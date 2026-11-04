"""CPU attachment glue using Anima's pinned shared routed branch mathematics."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from particle_sliders import winning_formulation
from particle_sliders_anima_gmix import require_runtime


class ProjectionBranch(nn.Module):
    def __init__(self, module, *, device="cpu"):
        super().__init__()
        stamp = winning_formulation()
        self.rank = int(stamp.spec["adapter_rank"])
        self.down = nn.Linear(module.in_features, self.rank, bias=False, device=device, dtype=torch.float32)
        self.routed = stamp.bridge().to(device=device, dtype=torch.float32)
        self.up = nn.Linear(self.rank, module.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** .5)
        nn.init.zeros_(self.up.weight)

    def forward(self, value, particles):
        return self.up(self.routed(self.down(value.float()), particles))


class RoutedAdapter(nn.Module):
    """One bank, explicit targets and owned hooks; no model loader or GAN loop.

    Target names are supplied explicitly. Anima's Cosmos suffixes are not used
    to guess unqualified P-H3 attention targets. Only frozen CPU hosts are
    admitted by this source-qualification attachment helper.
    """

    def __init__(self, host, targets, *, device="cpu"):
        super().__init__()
        require_runtime()
        device = torch.device(device)
        if any(p.requires_grad or p.device != device for p in host.parameters()):
            raise ValueError("attachment requires an already frozen host on the requested device")
        targets = tuple(targets)
        modules = dict(host.named_modules())
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("explicit unique projection targets required")
        selected = [modules.get(name) for name in targets]
        if any(not isinstance(module, nn.Linear) for module in selected):
            raise ValueError("every explicit target must be a native Linear")
        if len({id(module) for module in selected}) != len(selected):
            raise ValueError("aliased projection targets are not allowed")
        object.__setattr__(self, "_host", host)  # base never enters adapter parameters/state
        self.names = targets
        self.handles = []
        self.multiplier = 0.
        self.closed = False
        try:
            with torch.device(device):
                stamp = winning_formulation()
                self.particles = nn.Parameter(torch.randn(int(stamp.spec["parts"]),
                    int(stamp.spec["particle_dim"]), device=device, dtype=torch.float32))
                self.branches = nn.ModuleList(ProjectionBranch(module, device=device) for module in selected)
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
            raise ValueError("unipolar source check strength must be finite in [0,1]")
        previous = self.multiplier
        self.multiplier = float(strength)
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
