# One-command native training

From the common installed CLI:

```sh
particle-slider train --config wild-hair.json --profile training_profile.json
```

The profile uses the locally pinned P-H3 runtime, physical GPU1, and the real
HyperGAN develop/HNDL backend. It waits for the existing cooperative role lock
and sufficient free memory without stopping other work. An available GPU is
not by itself a source or numerical qualification: every new concept collects
its own frozen teacher pairs and passes a fresh full native qualification.

For explicit physical GPU0 placement, the separate example is a complete
execution envelope, so it needs no extra profile argument:

```sh
particle-slider train --config wild-hair-gpu0.json
```

This selects `training_profile.gpu0.json` and the same prompts, rates, resolution,
precision and training budget. GPU0 is a new compute placement requiring its own
collected teacher cache and full native qualification before training. The GPU1
receipt is not reused. The recipe or `PH3_PHYSICAL_GPU=0|1` chooses the physical
device; conflicting explicit selections fail. Each native child is pinned by
the selected UUID and uses logical `cuda:0`, with matching per-device admission
and P-H3 role locks. It never switches to the other GPU automatically.

Training includes prompt encoding, paired native teacher collection, four
qualification updates plus two replay calls, and 128 separate retained training
updates. Full optimizer/critic/EMA/RNG checkpoints are saved periodically, with
a portable EMA adapter at the final update. Samples and LR search are separate.
The `2e-5` generator LR is an explicit exploratory starting rate, not a measured
winner. This 128-step example demonstrates execution; it does not establish
visual quality. The woman's expression, clothes, hair color, framing and light
are shared by both prompts; the intended edit is hair shape and volume.

Direct native invocation is also available:

```sh
/ml2/codex-storage/future-workspaces/2026-10-31/task-14/.venv-qwen-native/bin/python tools/train_concept.py --config wild-hair.json --output /absolute/fresh/run --check
```

Remove `--check` to run. Check is CPU-only and never loads model tensors.
For a retained run, use the same source/config and add
`--resume /absolute/retained/native/artifacts`. A fresh controller output can
reference the original native directory through `resume-ref.json`; it receives
the new status and metrics while checkpoints remain in the original directory.
Stages with completion receipts are reused;
the native trainer restores its last full checkpoint. Interrupted collection
reuses completed paired teacher caches. A partially written qualification or
encoding directory must be inspected rather than silently discarded.

Local dependencies are explicitly separated from the default classic Anima
environment. The profile uses the archived engine, HyperGAN bridge, HNDL bridge,
the clean pinned native Diffusers source and model revision. A Modal image
needs its own portable runtime profile and qualification; these local absolute
paths do not constitute cloud support.

The local CPU tests cover request rejection, GPU queue state, UUID and logical
device binding, independent per-device locks, headroom guards, allocation-free
checks, final hash validation and failure reporting. Its isolated
archived runtime passes the engine closure and pinned Diffusers source checks.
The common CLI's `--check` also passes. These are preparation checks; a new
native run still requires its own collected cohort and GPU qualification.
