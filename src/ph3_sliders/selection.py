"""Common-budget EMA nomination with a declared absolute score tolerance."""

import math


def nominate(results, *, tie_tolerance=1e-6):
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("finite nonnegative tie tolerance required")
    if not results:
        return None
    cohorts = set()
    feasible = []
    for row in results:
        evaluation = row["evaluation"]
        if (type(row["completed_steps"]) is not int or row["completed_steps"] < 1
                or type(row["seed"]) is not int or not math.isfinite(row["lr"]) or row["lr"] <= 0
                or evaluation.get("selection") != "ema"
                or not isinstance(evaluation.get("cache_fingerprint"), str)
                or not evaluation["cache_fingerprint"]):
            raise ValueError("complete declared EMA trial identity required")
        cohorts.add((row["completed_steps"], row["seed"], evaluation["cache_fingerprint"]))
        score, gain = evaluation["mean_teacher_student_relative_error"], evaluation["mean_signed_gain"]
        scenes = evaluation["scene_relative_error"]
        if (not scenes or any(not math.isfinite(v) for v in (score, gain, *scenes.values()))
                or score < 0 or any(v < 0 for v in scenes.values())):
            raise ValueError("nonfinite or incomplete trial is unscoreable")
        if row["metric_feasible"] and score < 1 and gain > 0 and all(v < 1 for v in scenes.values()):
            feasible.append(row)
    if len(cohorts) != 1:
        raise ValueError("compare only identical completed budgets, seed and evaluation cache")
    if not feasible:
        return None
    best = min(r["evaluation"]["mean_teacher_student_relative_error"] for r in feasible)
    tied = [r for r in feasible if r["evaluation"]["mean_teacher_student_relative_error"] <= best + tie_tolerance]
    return min(tied, key=lambda row: row["lr"])
