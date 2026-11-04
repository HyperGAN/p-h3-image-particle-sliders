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
qualification updates plus two replay calls, and 1,600 separate retained training
updates. Full optimizer/critic/EMA/RNG checkpoints are saved periodically, with
a portable EMA adapter at the final update. Samples and LR search are separate.
The `2e-5` generator LR is an explicit exploratory starting rate, not a measured
winner. The original 128-step pass was an execution smoke test, too short for the
intended full training. The default now completes the existing 1,600-update
schedule; that budget alone does not establish visual quality. The woman's expression, clothes, hair color, framing and light
are shared by both prompts; the intended edit is hair shape and volume.

Direct native invocation is also available:

```sh
/ml2/codex-storage/future-workspaces/2026-10-31/task-14/.venv-qwen-native/bin/python tools/train_concept.py --config wild-hair.json --output /absolute/fresh/run --check
```

Remove `--check` to run. Check is CPU-only and never loads model tensors.
It now verifies the actual isolated modern environment with CUDA hidden,
including the HyperGAN/HNDL source closures and package metadata. Missing
runtime dependencies fail here before any model is loaded.
For the original Wild Hair pass, continue through the same command:

```sh
particle-slider train --config wild-hair-gpu0.json \
  --resume /absolute/original-controller-output/artifacts
```

Use a new output directory. Resume selects and byte-verifies the latest complete
checkpoint, then copies only that full state, the exact trial identity and the
initialization receipt into the new training output. The old run, its 128-step
result, plans, conditioning, cache, qualification and original request stay
unchanged. Training resumes at the retained step and continues to 1,600 using
the original 1,600-step schedule; no teacher collection or qualification is rerun.
New metrics contain only the continuation's updates. The new report records the
original checkpoint as its initial durable frontier.

`continuation.json` records the original target, retained step, new target,
immutable file hashes and the narrowly allowed control-script migration from
commit `3a5120f`. Any change to prompts, learning rates, seed, precision, device,
runtime, numerical source or fixed schedule is rejected. Only `training.steps`
may increase, up to the existing horizon. `--check --resume ...` performs the
same validation without GPU allocation or copying state. The active original
run can keep training while this read-only check inspects a published checkpoint;
launch the continuation after that run has completed and released its GPU.

Local dependencies are explicitly separated from the default classic Anima
environment. The profile uses the archived engine, HyperGAN bridge, HNDL bridge,
the clean pinned native Diffusers source and model revision. A Modal image
needs its own portable runtime profile and qualification; these local absolute
paths do not constitute cloud support.

New concept collections explicitly use `ph3-modern-hypergan-hndl-v1` identities.
They record the actual HyperGAN, HNDL and bridge packages/source bytes and do not
require or fabricate the classic Anima distribution. Encoded conditioning and
teacher caches must match that exact modern identity before qualification.
The classic collection API retains its default identity; its states cannot be
silently rebound to the new modern cohort. A source update requires a fresh run
directory and new native qualification; preserve earlier failed outputs.

The local CPU tests cover request rejection, GPU queue state, UUID and logical
device binding, independent per-device locks, headroom guards, allocation-free
checks, final hash validation and failure reporting. Its isolated
archived runtime passes the engine closure and pinned Diffusers source checks.
The common CLI's `--check` also passes. These are preparation checks; a new
native run still requires its own collected cohort and GPU qualification.
The runtime-identity fix passed 47 focused CPU checks, including failure before
model/CUDA allocation and rejection of changed modern bridge identities. The
actual archive check is recorded in `validation/modern-runtime-identity.json`.
The extension branch passed 59 CPU checks plus the allocation-free continuation
check against the actual original run. Its receipt is
`validation/budget-extension-cpu.json`; the numerical driver and all 23 native
source files remain byte-identical to the original run.
