"""Denoise packed P-H3 Image latents with an already-opened RAW owner."""

from math import fsum, isfinite
from sys import float_info

import torch

from ph3_native_scoped.api import Inference, _admit

from .activation_gate import PackedGatePlan


def _fixed_strengths(step_gains, strength, count):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != count:
        raise ValueError("one gain per native denoising step is required")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("step gains must be finite and nonnegative")
    values = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) or value > 1 for value in values):
        raise ValueError("effective strengths must be finite in [0,1]")
    return values


def _native_grid(host, scheduler):
    times, sigmas = scheduler.timesteps.tolist(), scheduler.sigmas.tolist()
    if (times != host.timesteps.tolist() or len(sigmas) != len(times) + 1
            or any(not isfinite(value) for value in sigmas)
            or any(left <= right for left, right in zip(sigmas, sigmas[1:]))):
        raise RuntimeError("P-H3 Image requires the original decreasing native sigma grid")
    return sigmas


@torch.no_grad()
def rollout(
    inference, conditioning, seed, *, strength=1.0, step_gains=None,
    gate_plan=None, record_native_dose=False,
):
    """Return coupled image/audio latents and the actual native strengths used."""
    if not isinstance(inference, Inference):
        raise TypeError("an opened P-H3 Image RAW inference owner is required")
    inference.require_open()
    host = inference.host
    if type(strength) not in (float, int) or not 0 <= strength <= 1:
        raise ValueError("nominal strength must be finite in [0,1]")
    if (step_gains is None) == (gate_plan is None):
        raise ValueError("provide either step_gains or a measured gate_plan")
    if gate_plan is None:
        strengths = _fixed_strengths(step_gains, strength, len(host.timesteps))
    else:
        if not isinstance(gate_plan, PackedGatePlan):
            raise TypeError("gate_plan must be a PackedGatePlan")
        strengths, gate_trace = [], []
        gate = gate_plan._new_gate()

    _admit(
        inference.admission, operation="native-packed-scheduled-rollout",
        cohort=inference.packet["cohort_id"], identity=inference.identity,
        cache_identity=inference.packet["cache_identity"],
        completed_steps=inference.packet["completed_steps"],
    )
    image_scheduler, audio_scheduler = host.scheduler(), host.scheduler()
    sigmas = None
    if gate_plan is not None or record_native_dose:
        sigmas = {"image": _native_grid(host, image_scheduler),
                  "audio": _native_grid(host, audio_scheduler)}
    state = host.initial_state(seed)
    for position in range(len(host.timesteps)):
        if sigmas is not None:
            cursor = position if position else None
            if any(scheduler.step_index != cursor or scheduler.begin_index is not None
                   for scheduler in (image_scheduler, audio_scheduler)):
                raise RuntimeError("P-H3 Image lost native solver interval alignment")
        if gate_plan is None:
            value = strengths[position]
        else:
            value = 0.0
            if strength:
                projection = gate_plan._score(state, strength)
                decision = gate.step(projection)
                value = float(strength) if decision.fired else 0.0
                gate_trace.append({"interval": position, "projection": projection,
                                   "probability": decision.probability, "fired": decision.fired,
                                   "effective_strength": value})
            strengths.append(value)
        with inference.adapter.scaled(value):
            velocity = host.velocity(state, conditioning, position)
        state = host.step(state, velocity, position, image_scheduler, audio_scheduler)
    trace = {"native_timesteps": image_scheduler.timesteps.tolist(), "effective_strengths": strengths}
    if gate_plan is not None:
        usage = gate.usage
        trace.update(
            activation_gate=gate_plan._receipt(), activation_gate_trace=gate_trace,
            gate_firings=usage.fired, gate_eligible_intervals=usage.eligible,
            gate_firing_rate=usage.rate if usage.eligible else None,
        )
    if sigmas is not None:
        trace.update(
            native_sigmas=sigmas,
            sigma_weighted_strength={modality: fsum(value * (left - right)
                for value, left, right in zip(strengths, grid[:-1], grid[1:], strict=True))
                for modality, grid in sigmas.items()},
        )
    return state, trace
