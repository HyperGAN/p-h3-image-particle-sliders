# Original declared P-H3 RAW32 product method

Core 0.14.0 at `8a696eb33c1a9d34472afad3a3199bd4776a797f` / PG 0.9.5
discovers `ph3-image-fixed-bank32` through `particle_sliders.training_profiles`.
The native profile preserves the original four-arm bank's code, initialization,
native2048 image/coupled-audio data, update math, seed7, batch2, horizon128,
target32, RAW selection and per-update FULL checkpoint durability.

The options use the common names `lr`, `d_lr_mult`, `prior_lr_mult` and
`ema_decay`. `ema_decay` must be JSON `null` / Python `None`; all weight EMA
enablement is refused. Only the original literal tuples are accepted:

| Arm | lr | d_lr_mult | prior_lr_mult | ema_decay |
| --- | ---: | ---: | ---: | --- |
| cell0 | 0.0004 | 45.0 | 74.99999999999999 | null |
| cell1 | 0.004 | 4.5 | 7.499999999999999 | null |
| cell2 | 0.012 | 1.5000000000000002 | 2.4999999999999996 | null |
| cell3 | 0.036 | 0.5000000000000001 | 0.8333333333333333 | null |

These are original JSON literals, not quotients reconstructed from rounded
actual D/P. The interface neither selects a new default nor changes any score,
tie, search, proof or completed update. A selected finished-output winner can
use its exact retained tuple through this method; a completed arm cannot be
trained again. This is not a generic arbitrary-dataset training implementation.

## Allocation-free planning and checking

Install the proposed product and its pinned current core into a **new**
frontdoor environment. No installation into the captured native environment is
needed. The product's local frozen deployment dependencies are
listed in `ph3_sliders/native-authority.json`; installing this package does not
download, relocate or include those assets.

```sh
ph3-slider bank-config --lr 0.0004 --d-lr-mult 45.0 \
  --prior-lr-mult 74.99999999999999 > bank-request.json
ph3-slider bank-check --config bank-request.json
particle-slider profiles
particle-slider plan --profile ph3-image-fixed-bank32 \
  --config bank-request.json --no-reporting
```

`bank-check` uses Python's standard library, checks the exact declaration and
loads no model. It accepts an unbound preparation receipt and reports that fact.
It does not claim admission, completion or a native proof. The common CLI's
`train --check` is available with `--runner local`; it starts only the product's
allocation-free checker in the new frontdoor interpreter (`sys.executable`). Actual
generic-local training refuses because its result/artifact contract differs
from this original bank's genuine durable outputs.

## Actual owned dispatch

Root first performs the existing fresh admission/preparation for one untouched
arm or exact unfinished FULL continuation. Preparation creates the real
one-job receipt, records exact argv/environment, and keeps Root's ownership,
capture, poll/reap and terminal closure protocol. This adapter never creates
grants, leases, GPU/host observations, pending roles or completion receipts.

Root then binds that actual preparation JSON and its exact SHA:

```sh
ph3-slider bank-config --lr 0.0004 --d-lr-mult 45.0 \
  --prior-lr-mult 74.99999999999999 \
  --prepared-entry /absolute/actual-Root-preparation.json \
  --prepared-entry-sha256 ACTUAL_SHA256 > bound-bank-request.json
ph3-slider bank-train --config bound-bank-request.json
```

`bank-train` dispatches through the common CLI with the explicit
`--runner ph3-root-owned`. The runner authenticates the original plan,
matching rate tuple and actual Root-adopted one-job Source. It requires a fresh
same-boot receipt, no existing attempt or completed arm, and the exact latest
FULL continuation if a prefix exists. A new exclusive dispatch-consumption
record prevents reuse through this adapter, including after an exec failure;
there is no automatic retry or successor.

Profile status `source_only` reports this proposal's integration evidence; the
current014 common CLI does not refuse that status before runner dispatch.
Root's required actual first-use evidence remains pending until a missing arm
really uses this frontdoor. Planning/checking is not that evidence.

The runner calls `os.execvpe` with the **exact recorded** original GPU1 one-job
driver or GPU0 guardian argv and environment. The initial common CLI process
becomes that launcher with the same PID; it was not already a guardian. No extra
waiting parent or invented submitted/completed result is inserted. Root must
launch this common CLI as the actual owned tool action and use the unchanged
existing capture/reap/close workflow after exec. Original output paths, scientific
plan, captured workers and durable metadata remain authoritative. The common
request's generic `output.root` is planning metadata; the actual prepared bank
attempt/output path is retained and printed in the consumption event.

The recorded guardian/driver command still uses the exact captured native
interpreter from `native-authority.json`. Thus the new environment hosts only
the product/common API dispatcher and allocation-free checker; exec crosses
into the unchanged existing native environment for actual work.

Both the original GPU1/own48 and the separately adopted GPU0/own32 preparation
formats are supported. Their real execution placements/caps remain distinct;
the adapter does not clone GPU1 inference onto GPU0 or fake an own48 closure.

## Existing RAW inference and portability

`particle_sliders.inference_backends` resolves `ph3-native-response095` to the
**exact old** `INFERENCE_BACKEND` object loaded from its unchanged pinned Source
file. Its original factory Source identity, NativeRow protocol, model/data
identity and strict whole-RAW checks remain intact. It supports native response
evaluation with the original local options, not generic JSON prompts or a
public image render method. The separate finished-output consumer/ordinary
Generate route supplies native rollouts and tiled VAE decode.

This adapter is a local deployment proposal. Original absolute Source paths,
frozen model/conditioning/data, native Python environment, GPU UUID/lock and
admission infrastructure must exist. Those dependencies are not bundled into
this package. Moving them or generalizing the engine requires an explicit
future integration; this proposal does not retag Source identities or RAWs.
