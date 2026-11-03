"""Pooled teacher/student response metric; no model loading or fitting.

The unipolar target is the frozen positive-conditioned velocity. The student
uses neutral conditioning. Score raw native coordinates, independently of the
training-only gmix whitening. A native caller must supply the qualified grid.
"""

from collections import defaultdict
import math

METRIC = "matched_example_teacher_student_relative_l2_v2"
STAT_KEYS = ("student_teacher_sse", "teacher_edit_energy",
             "student_edit_energy", "student_teacher_edit_dot")
EXAMPLE_KEYS = ("scene", "seed", "path")


def response_statistics(student, neutral, teacher):
    """Detached float64 sufficient statistics; never normalize individual rows."""
    import torch

    if student.shape != neutral.shape or teacher.shape != neutral.shape or not student.numel():
        raise ValueError("nonempty matching response shapes required")
    with torch.no_grad():
        s, n, t = (v.detach().to(device="cpu", dtype=torch.float64).flatten()
                   for v in (student, neutral, teacher))
        if any(not torch.isfinite(v).all() for v in (s, n, t)):
            raise ValueError("finite responses required")
        actual, desired = s - n, t - n
        result = dict(element_count=s.numel(),
            student_teacher_sse=float((s - t).square().sum()),
            teacher_edit_energy=float(desired.square().sum()),
            student_edit_energy=float(actual.square().sum()),
            student_teacher_edit_dot=float(torch.dot(actual, desired)))
    _validate_statistics(result)
    return result


def _validate_statistics(row):
    if type(row.get("element_count")) is not int or row["element_count"] < 1:
        raise ValueError("positive integer element_count required")
    if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in STAT_KEYS):
        raise ValueError("finite additive statistics required")
    if any(row[k] < 0 for k in STAT_KEYS[:3]):
        raise ValueError("squared statistics must be nonnegative")
    error, target, actual, dot = (row[k] for k in STAT_KEYS)
    scale = max(actual, target, error, 1e-300)
    # Normalize before products/sums to avoid overflow in consistency checks.
    if abs(actual / scale + target / scale - 2 * (dot / scale) - error / scale) > 2e-10:
        raise ValueError("inconsistent teacher/student statistics")
    if abs(dot / scale) > math.sqrt(actual / scale) * math.sqrt(target / scale) + 2e-10:
        raise ValueError("statistics violate Cauchy-Schwarz")


def _example_key(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or not isinstance(value[0], str) or not value[0]
            or type(value[1]) is not int or value[1] < 0
            or value[2] not in ("neutral", "positive")):
        raise ValueError("example requires scene, nonnegative seed, neutral/positive path")
    return tuple(value)


def aggregate_rows(rows, *, expected_examples, expected_positions):
    """Require the declared complete cohort, then pool before division.

    An example is (scene, seed, frozen rollout path); positions from each path
    are pooled. Path pairs have equal weight. Do not invent bipolar labels for
    the unipolar Anima formulation. A zero-energy whole example is unscoreable.
    """
    declared = [_example_key(k) for k in expected_examples]
    positions = list(expected_positions)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("unique nonempty declared examples required")
    if (not positions or any(type(p) is not int or p < 0 for p in positions)
            or len(set(positions)) != len(positions)):
        raise ValueError("unique nonnegative declared positions required")
    groups, seen = defaultdict(list), set()
    rows = list(rows)
    for row in rows:
        _validate_statistics(row)
        key = _example_key([row.get(k) for k in EXAMPLE_KEYS])
        position = row.get("position")
        if type(position) is not int or position not in positions or key not in declared:
            raise ValueError("response outside declared cohort")
        identity = (*key, position)
        if identity in seen:
            raise ValueError("duplicate matched response")
        seen.add(identity)
        groups[key].append(row)
    if set(groups) != set(declared) or any(len(groups[k]) != len(positions) for k in declared):
        raise ValueError("incomplete declared cohort")
    examples = []
    for key in declared:
        group = groups[key]
        sums = {k: math.fsum(r[k] for r in group) for k in STAT_KEYS}
        if any(not math.isfinite(v) for v in sums.values()):
            raise ValueError("nonfinite pooled statistics")
        energy = sums["teacher_edit_energy"]
        if energy <= 0:
            raise ValueError("zero-energy teacher example is unscoreable")
        error = sums["student_teacher_sse"] / energy
        gain = sums["student_teacher_edit_dot"] / energy
        relative_actual = sums["student_edit_energy"] / energy
        if any(not math.isfinite(v) for v in (error, gain, relative_actual)):
            raise ValueError("nonfinite relative statistics")
        cosine = gain / math.sqrt(relative_actual) if relative_actual else 0.
        examples.append(dict(zip(EXAMPLE_KEYS, key), **sums,
            element_count=sum(r["element_count"] for r in group),
            relative_error=error, signed_gain=gain,
            cosine=max(-1., min(1., cosine)),
            orthogonal_relative_energy=max(0., relative_actual - gain * gain)))
    scene_scores, path_scores = defaultdict(list), defaultdict(list)
    for example in examples:
        scene_scores[example["scene"]].append(example["relative_error"])
        path_scores[example["path"]].append(example["relative_error"])
    mean = lambda values: math.fsum(values) / len(values)
    score = mean([e["relative_error"] for e in examples])
    return dict(metric=METRIC, lower_is_better=True,
        coordinate_system="unwhitened_native_velocity",
        target="frozen_positive_conditioned_velocity_at_identical_state_time",
        student="adapted_neutral_conditioned_velocity_at_strength_1",
        aggregation="pool positions before division; equal matched example mean",
        example_keys=list(EXAMPLE_KEYS), positions=positions, examples=examples,
        mean_teacher_student_relative_error=score, error_reduction_vs_OFF=1-score,
        scene_relative_error={k: mean(v) for k, v in scene_scores.items()},
        path_relative_error={k: mean(v) for k, v in path_scores.items()},
        mean_signed_gain=mean([e["signed_gain"] for e in examples]),
        unchanged_OFF_relative_error=1., exact_teacher_relative_error=0.,
        reversed_teacher_relative_error=4., rows=rows,
        native_h3_qualified=False)
