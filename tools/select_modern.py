"""Recompute full teacher/student metrics from raw responses, then nominate LR."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from ph3_sliders.native_cache import NativeCache, file_hash
    from ph3_sliders.evaluation import response_statistics, aggregate_rows
    from ph3_sliders.selection import nominate
    import ph3_sliders.evaluation as metric
    from modern_native import write

    torch.set_num_threads(4)
    cache = NativeCache.load(args.cache)
    if cache.identity["split"] != "development":
        raise ValueError("nominate on the frozen development split")
    results, cohorts, initial_states = [], [], []
    for directory in args.trials:
        trial = json.loads((directory / "TRIAL.json").read_text())
        signature = trial["signature"]
        if signature["identity"]["metric_source_sha256"] != file_hash(Path(metric.__file__)):
            raise ValueError("candidate metric source differs from this recomputation")
        if signature["seed"] != args.seed:
            raise ValueError("candidate training seed differs")
        cohorts.append(signature["identity"])
        initial_states.append((signature["initial_adapter_sha256"], signature["initial_critic_sha256"]))
        evaluation = json.loads((directory / f"eval-{args.steps:06d}-ema.json").read_text())
        if (evaluation["evaluation_signature"]["training"] != signature
                or evaluation["qualification_sha256"] != trial["qualification_sha256"]
                or evaluation["completed_steps"] != args.steps or evaluation["selection"] != "ema"
                or evaluation["cache_fingerprint"] != cache.index["fingerprint"]):
            raise ValueError("recorded evaluation is outside the exact declared trial")
        checkpoint = directory / "checkpoints" / f"step-{args.steps:06d}"
        receipt = json.loads((checkpoint / "manifest.json").read_text())
        if (receipt["signature"] != signature or receipt["completed_steps"] != args.steps
                or receipt["weights_sha256"] != file_hash(checkpoint / "state.pt")):
            raise ValueError("candidate full checkpoint identity/bytes differ")
        raw = directory / "evaluations" / f"step-{args.steps:06d}-ema" / "student.safetensors"
        if file_hash(raw) != evaluation["raw_student_responses"]["sha256"]:
            raise ValueError("raw student response bytes differ")
        with safe_open(raw, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        labels = [{k:r[k] for k in ("scene", "seed", "path", "position")} for r in cache.records]
        if (json.loads(metadata["row_labels"]) != labels
                or metadata["cache_fingerprint"] != cache.index["fingerprint"]
                or metadata["selection"] != "ema" or int(metadata["completed_steps"]) != args.steps):
            raise ValueError("raw response labels/selection differ")
        values = load_file(str(raw), device="cpu")
        if set(values) != {f"student.{i}" for i in range(len(cache))}:
            raise ValueError("raw responses do not cover the complete declared grid")
        rows = [dict(labels[i], **response_statistics(values[f"student.{i}"], r["neutral"], r["positive"]))
                for i,r in enumerate(cache.records)]
        rebuilt = aggregate_rows(rows, expected_examples=cache.examples, expected_positions=range(cache.positions))
        for field in ("mean_teacher_student_relative_error", "mean_signed_gain"):
            if not math.isclose(rebuilt[field], evaluation[field], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("recorded metric disagrees with raw teacher/student responses")
        rebuilt.update(selection="ema", cache_fingerprint=cache.index["fingerprint"])
        feasible = (rebuilt["mean_teacher_student_relative_error"] < 1 and rebuilt["mean_signed_gain"] > 0
                    and all(v < 1 for v in rebuilt["scene_relative_error"].values()))
        results.append(dict(lr=signature["config"]["recipe"]["lr"], seed=args.seed,
            completed_steps=args.steps, evaluation=rebuilt, metric_feasible=feasible,
            trial=str(directory), checkpoint=str(checkpoint),
            export=str(directory / "exports" / f"step-{args.steps:06d}-ema")))
        del values
    if any(identity != cohorts[0] for identity in cohorts):
        raise ValueError("LR candidates differ in source/model/cache/precision/recipe cohort")
    if len(set(initial_states)) != 1:
        raise ValueError("LR candidates were not initialized with identical adapter/critic tensors")
    candidate = nominate(results, tie_tolerance=args.tie_tolerance)
    edge = bool(candidate and candidate["lr"] in (min(r["lr"] for r in results), max(r["lr"] for r in results)))
    document = dict(schema="ph3-modern-raw-audited-lr-nomination-v1", candidate=candidate,
        results=results, common_budget=args.steps, seed=args.seed, tie_tolerance=args.tie_tolerance,
        raw_response_recomputation_verified=True, edge_requires_bracket_extension=edge,
        matched_media_and_finalist_confirmation_required=True,
        status="metric_candidate_requires_media_and_confirmation" if candidate else "no_feasible_candidate",
        selector_sha256=file_hash(Path(__file__)))
    write(args.output, document)
    print(json.dumps(dict(status=document["status"], lr=candidate["lr"] if candidate else None, edge=edge)))


if __name__ == "__main__":
    main()
