"""Train-only per-position paired normalization from the pinned Anima runner.

Ported without importing that runner's incompatible PG0.6 runtime. Arithmetic
matches runner18ff6c18565263a693b4a85ba6680beb8d577a65/normalization.py.
"""

import hashlib
import torch


def fit_normalization(cache, *, positions):
    if type(positions) is not int or positions < 1:
        raise ValueError("positions must be a positive native position count")
    if cache.index["identity"]["split"] != "train":
        raise ValueError("Normalization may only use training examples")
    scales, rms, moments = [], [], {}
    for index in range(len(cache)):
        record = cache[index]
        position = record["position"]
        if position != index % positions:
            raise ValueError("Cache timesteps were reordered")
        edit = (record["positive"].float() - record["neutral"].float()).flatten()
        if position not in moments:
            moments[position] = [0, torch.zeros_like(edit), torch.zeros_like(edit)]
        state = moments[position]
        state[0] += 1
        delta = edit - state[1]
        state[1] += delta / state[0]
        state[2] += delta * (edit - state[1])
    raw_scales = []
    for position in range(positions):
        count, mean, m2 = moments[position]
        if count < 2:
            raise ValueError("Each timestep needs at least two training samples")
        raw_scales.append((m2 / (count - 1)).clamp_min(0).sqrt().clamp_min(1e-4))
    row_values = [[] for _ in range(positions)]
    for index in range(len(cache)):
        record = cache[index]
        position = record["position"]
        normalized = (record["positive"].float() - record["neutral"].float()).flatten() / raw_scales[position]
        row_values[position].append(normalized.square().mean().sqrt())
    for position in range(positions):
        scale = raw_scales[position]
        row_rms = torch.stack(row_values[position])
        median = row_rms.median().clamp_min(1e-4)
        scales.append(scale * median)
        rms.append((row_rms / median).square().mean().sqrt())
    result = dict(scale=torch.stack(scales), edit_rms=torch.stack(rms),
        provenance=dict(method="paired_edit_per_coordinate_sample_std_median_rms_gain_per_timestep",
                        cache_fingerprint=cache.index["fingerprint"], split="train", positions=positions))
    result["provenance"]["scale_sha256"] = hashlib.sha256(result["scale"].numpy().tobytes()).hexdigest()
    return result
