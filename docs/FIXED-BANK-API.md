# Original declared P-H3 RAW32 product method

Core 0.14.0 at `8a696eb33c1a9d34472afad3a3199bd4776a797f`
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

## Finished-output selection policy

**Current eligible saved recipe:** cell1/G0.004 with exact overrides
`(0.004,4.5,7.499999999999999,None)`, original mean
`+0.002527267051239808`. All8 nominal outputs and its exact FULL32 guard pass.
The separate catastrophic-only gate rejects cell3/G0.036; its failed state and
all original metrics remain. This changes no completed update, package default,
native Source identity or dispatch behavior. [Current metadata](https://github.com/HyperGAN/p-h3-image-particle-sliders/blob/cb6e01ec9357661c7de85ce879908e9a7335b164/docs/current-eligible-rate.json)
and [eligibility scope](current-eligibility-20261109.md) identify the new current
recommendation separately from the original comparison history below.

The original four-arm comparison is complete: existing cell3/G0.036 has the
highest frozen warmth-proxy mean,+0.017445931987216074 over eight matched
native2048 cases. **2026-11-09:** the user found its yellow/noisy outputs broken;
this proxy failed the intended illumination meaning and is not a working-control
recommendation. Preserve its exact tuple, saved RAW32 and all original results;
all four arms are complete and must not be retrained. A same-RAW bottle171
diagnostic retained a clear bottle at α0.125 and α0.25, with warm amber
illumination at α0.25. Full strength overdrives this one case; no new LR comparison,
reranking or general strength recommendation is claimed. See
[the dated caveat and images](measurement-caveat-20261109.md).
This changes no package default. Rank only the equal-case arithmetic mean intended
effect in finished ON minus exact matched OFF, retaining negatives, zeros and
exact ties. Missing/failed outputs are not zero. Qualified catastrophic failures
are excluded separately before
effect ranking; weak effect is not failure, and missing/observer error is
unmeasured. Failed candidates must not update, resume or promote. SSE is historical
diagnostic
information, never a selector or tiebreak; preservation, beauty, confidence and
completion do not select. Reuse completed training/proofs, saved RAWs, outputs
and first observations, with no additional fresh confirmation or prefix replay.
See [current scope and options](finished-output-learning-rates.md).

## Allocation-free planning and checking

Install the current product and its pinned shared core into a new environment. The planning and checking commands use the four exact recorded rate tuples. They require no local native assets or historical environment.

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
It does not claim admission, completion or a native proof. The common CLI's `train --check` starts the same allocation-free checker.

## Archived dispatch

[The original checkout](https://github.com/HyperGAN/p-h3-image-particle-sliders/blob/98b99f48fc36b1165b6a29f1ab6999f03d886996/docs/FIXED-BANK-API.md#actual-owned-dispatch) records the complete local launch and restore protocol. For new native sessions, install the [scoped runtime](../backends/ph3-native-scoped/README.md).

## Completed common API first use

On 2026-11-09, actual tool **89012** entered through the installed common CLI. [The recorded command and evidence root](https://github.com/HyperGAN/p-h3-image-particle-sliders/blob/98b99f48fc36b1165b6a29f1ab6999f03d886996/docs/FIXED-BANK-API.md#completed-common-api-first-use) identify the exact inputs.

The original **cell2 / G0.012** arm completed **32 new updates, zero inherited**,
using `d_lr_mult=1.5000000000000002`, `prior_lr_mult=2.4999999999999996`,
RAW current weights and `ema_decay=None`. The common dispatcher exec'd the
exact admitted original launcher; native math, local assets, the captured
native environment and original durability remained unchanged. The actual
outer exit was **0**, reap chunk **88bcd9**, followed by Root closure **60e7e0**.
This records a real fixed-bank training run, not a planning/check result.

The paths below are relative to the archived evidence root.

| Evidence | Relative path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Genuine Root terminal | `attempts/cell2/attempt-0000/root-actual-terminal-reap-resource0-bank-v1.json` | 38459 | `43e0c9bcb2ef50243f25f82609f9b7d0a96d2ae3dd24363b533aec9cce843fc8` |
| Completed RAW32 arm | `bank/cell2/COMPLETE.json` | 28110 | `d735486cce2fe5aeb85c747babb23e94824c3b7bc1d382575ffb607572ca886d` |
| Opaque public RAW artifact | `attempts/cell2/attempt-0000/cell2/public-RAW32.pt` | 91718006 | `07b8b812d58f299cce660d23febbffcdf26a237c21539d2b48e0a095856a9cf3` |

This success covers the original declared local bank and its common product
training interface. It adds no arbitrary prompt/data training or public image
render capability. This section preserves the completed first-use evidence,
not a changing per-arm status tracker. Actual durable metadata determines
which declared work remains missing. This historical first-use event did not
select an LR winner; the subsequent existing four-arm finished-output comparison
selected cell3/G0.036, separately recorded in [the decision](finished-output-decision.json).
All four declared arms are now complete. Their saved artifacts, matched outputs
and first measurements are retained without replay or extra confirmation.

## Existing RAW inference and portability

`particle_sliders.inference_backends` resolves `ph3-native-response095` to the
**exact old** `INFERENCE_BACKEND` object loaded from its unchanged pinned Source
file. Its original factory Source identity, NativeRow protocol, model/data
identity and strict whole-RAW checks remain intact. It supports native response
evaluation with the original local options, not generic JSON prompts or a
public image render method. The separate finished-output consumer/ordinary
Generate route supplies native rollouts and tiled VAE decode.

The original local fixed bottle171/native2048 preset's decoder continuation is
technically complete: eight prior native responses, zero new responses, one VAE
decode, exact reused OFF, HTTP outputs and owned cleanup. That narrow completion
does not establish arbitrary-prompt public rendering, a fresh complete end-to-end
RNG claim or a working illumination control. The response-only public factory
and the meaning caveat above remain unchanged.

The local Generate default now requests the fixed bottle171 two-alpha diagnostic
at α0.125 and α0.25, using the same retained RAW and local runtime. This is not a
general public rendering API or a new strength recommendation; see
[the current deployment scope](measurement-caveat-20261109.md#completed-one-case-lower-alpha-diagnostic).

This adapter is specific to its local deployment. Original absolute Source paths,
frozen model/conditioning/data, native Python environment, GPU UUID/lock and
admission infrastructure must exist. Those dependencies are not bundled into
this package. Moving them or generalizing the engine requires an explicit
future integration; this adapter does not retag Source identities or RAWs.
