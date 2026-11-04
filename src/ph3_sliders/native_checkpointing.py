"""Whole-block checkpointing for frozen int8 H3 with trainable owned hooks.

Pinned BnB MatMul8bitLt stores activation references in ``ctx.tensors``, outside
``save_for_backward``. Saved-tensor hooks in nonreentrant checkpointing cannot
release those references. Reentrant checkpointing drops the whole first-pass
autograd graph, retaining just block inputs for backward recomputation.
"""

import torch
from torch.utils.checkpoint import checkpoint

MODE = "reentrant_whole_block_frozen_input_grad_v1"


def checkpoint_block(module, *args):
    if not torch.is_grad_enabled():
        return module(*args)
    if not args or not isinstance(args[0], torch.Tensor) or not args[0].is_floating_point():
        raise ValueError("native checkpoint expects hidden states as first floating input")
    # Reentrant checkpoint requires a differentiable input even when all base
    # weights are frozen. This leaf is an activation, never an optimizer member.
    # Preserve every existing trainable upstream connection when one exists.
    if not args[0].requires_grad:
        args = (args[0].detach().requires_grad_(True), *args[1:])
    return checkpoint(module.__call__, *args, use_reentrant=True, preserve_rng_state=True)


def enable_native_checkpointing(model):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("native base parameters must stay frozen")
    model.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_block)
    if not model.gradient_checkpointing or not model.token_refiner.gradient_checkpointing:
        raise ValueError("both native main and refiner blocks must be checkpointed")
    return MODE
