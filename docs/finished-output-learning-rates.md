# Original four-arm RAW32 method: effect decision pending

**No P-H3 learning-rate winner is selected.** The only supported training
method is the already declared four-arm WarmGoldenLight bank: G0.0004,
0.004, 0.012 and 0.036, target32/horizon128/batch2/rank8/seed7, full393344
native response, native2048 image/coupled-audio data and128×4 particle bank.
Actual fixed D is0.018000000000000002 and bank rate0.029999999999999995.
Keep the original initialization, native geometry, math and source identities.

The current core0.14.0/PG0.9.5 product registers training profile
`ph3-image-fixed-bank32` and explicit runner `ph3-root-owned`. Its completed
common API first use is static evidence of a real original cell2 G0.012
RAW32 run (32 new, zero inherited), not a winner. Determine genuinely missing
work from actual durable metadata, rather than rerunning a completed arm.

## Existing options and planning

[The fixed-bank API](FIXED-BANK-API.md) lists the exact four accepted
`lr`/`d_lr_mult`/`prior_lr_mult`/`ema_decay` tuples and real commands.
[The portable summary](finished-output-rate.json) retains the same literals.
For example, these existing commands construct/check a declaration and plan;
they do not admit or launch native training:

```sh
ph3-slider bank-config --lr 0.0004 --d-lr-mult 45.0 \
  --prior-lr-mult 74.99999999999999 > bank-request.json
ph3-slider bank-check --config bank-request.json
particle-slider plan --profile ph3-image-fixed-bank32 \
  --config bank-request.json --no-reporting
```

Actual `ph3-slider bank-train --config ...` requires a genuine future owned
preparation for that exact missing arm or latest complete FULL continuation,
with its actual source-bound receipt. No generic arbitrary prompt/data trainer
is supplied. Preserve inherited original proof prefixes, both optimizer states,
particle bank and RNG/stream state; do not retag an old packet for another rate
or replay a committed prefix, teacher preparation or completed native proof.
The exact existing RAW inference factory `ph3-native-response095` is
response-only. This method guide does not claim an ordinary Generate route or
public finished-image renderer is ready.

After existing RAWs and their finished matched ON/OFF outputs are available,
use the single metric below. Reuse existing matched OFF, outputs and first
observations; keep failures missing and ties tied. No extra top-two extension,
new confirmation or teacher-SSE winner rule is part of this method. No
per-arm effect score or unique/default rate is asserted here.

## Method and reuse

For each fixed matched case, subtract intended-effect intensity in OFF from ON,
then take the arithmetic mean with equal case weights. Higher wins. Keep
negative gains, zeros and exact ties. Missing/failed outputs are missing evidence,
not synthetic zero. Teacher/student relative SSE, preservation, beauty,
confidence and completion do not select, veto or break ties.

Use RAW current adapter, critic and particle-bank weights where present, with
weight EMA disabled (`ema_decay=None` / JSON `null`). Official scalar optimizer
statistics are unchanged. Reuse the completed searches, exact source-bound RAW
checkpoints, outputs and first observations. Finish only genuinely missing
declared work or exactly resume its latest complete FULL state; never replay
committed updates. No additional fresh confirmation or worse-arm completion is
required. Older SSE tables and source-cohort reports remain historical diagnostics.

A rate comparison varies G while retaining its declared actual D and bank rate
where present. The option names are `lr` (G), `d_lr_mult` (D/G), and
`prior_lr_mult` (bank/G). Copy the saved literals: rounded quotients can alter
exact recipe identity. These conditional comparisons are not global or joint
optima, and this documentation changes no package defaults or checkpoint pins.

## Availability

The local frozen model/conditioning/data, captured native Python environment,
original Source paths and owned admission infrastructure are not distributed
with this package. Installing/checking/planning alone is not native completion
or portable image reproduction. [The API](FIXED-BANK-API.md) preserves original
completed-first-use evidence and the explicit preparation/dispatch contract;
its historical receipts are not instructions to repeat that work.
