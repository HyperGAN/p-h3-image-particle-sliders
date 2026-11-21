"""Denoise packed P-H3 Image latents with an already-opened RAW owner."""

from math import isfinite
from sys import float_info

import torch

from ph3_native_scoped.api import Inference, _admit


@torch.no_grad()
def rollout(inference, conditioning, seed, *, strength=1.0, step_gains):
    """Return coupled image/audio latents and the actual native strengths used."""
    if not isinstance(inference, Inference):
        raise TypeError("an opened P-H3 Image RAW inference owner is required")
    inference.require_open()
    host = inference.host
    if type(strength) not in (float, int) or not 0 <= strength <= 1:
        raise ValueError("nominal strength must be finite in [0,1]")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != len(host.timesteps):
        raise ValueError("one gain per native denoising step is required")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("step gains must be finite and nonnegative")
    strengths = tuple(float(strength * gain) for gain in step_gains)
    if any(not isfinite(value) or value > 1 for value in strengths):
        raise ValueError("effective strengths must be finite in [0,1]")

    _admit(
        inference.admission, operation="native-packed-scheduled-rollout",
        cohort=inference.packet["cohort_id"], identity=inference.identity,
        cache_identity=inference.packet["cache_identity"],
        completed_steps=inference.packet["completed_steps"],
    )
    image_scheduler, audio_scheduler = host.scheduler(), host.scheduler()
    state = host.initial_state(seed)
    for position, value in enumerate(strengths):
        with inference.adapter.scaled(value):
            velocity = host.velocity(state, conditioning, position)
        state = host.step(state, velocity, position, image_scheduler, audio_scheduler)
    return state, {
        "native_timesteps": image_scheduler.timesteps.tolist(),
        "effective_strengths": list(strengths),
    }
