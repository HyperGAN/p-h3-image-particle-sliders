# Best tested existing WarmGoldenLight RAW32 recipe

**Cell3/G`0.036` wins the original four-rate comparison**, with equal-eight-case
intended-effect ON−OFF mean **+`0.017445931987216074`**. Keep the saved RAW32;
no new search or fresh confirmation is required. This is the original native2048,
full-coupled image/audio training cohort, batch2/horizon128/rank8/seed7,
full393344 native response and128×4 particle bank under core0.14.0/PG0.9.5.
Actual fixed D is `0.018000000000000002`; bank is `0.029999999999999995`.
The image comparison uses four original DEV scenes (bottle, bicycle, room, boat)
with seeds171/172, eight equally weighted cases per arm, strength1 ON vs exact
matched strength0 OFF. Native full-coupled rollout and original tiled VAE are retained.

| Arm | G | Mean finished-effect gain |
| --- | ---: | ---: |
| cell0 | 0.0004 | -0.00029275131722291336 |
| cell1 | 0.004 | +0.002527267051239808 |
| cell2 | 0.012 | +0.0009762506621579328 |
| **cell3** | **0.036** | **+0.017445931987216074** |

All32 matched pairs and40 first image intensities are copied in
[the portable decision](finished-output-decision.json), with separate identities
for the original CPU result and Root's accepted selection. This frozen SigLIP2
warm-light instrument reused its original model revision, three positive/neutral
wording pairs and preprocessing. Each image intensity is their mean cosine margin;
each case gain is ON minus OFF, and each arm score is the equal-eight-case mean.
No new instrument calibration, image generation, rating or arithmetic selector
is introduced by these docs. The full negative/zero/tied outcomes remain retained;
missing technical evidence is never converted to zero. This is a scoped
model-instrument best-tested recipe, not a global/joint optimum, unseen holdout,
human validation or changed package default.

## Actual common options

Use the exact saved literals in [the portable summary](finished-output-rate.json):
`lr=0.036`, `d_lr_mult=0.5000000000000001`,
`prior_lr_mult=0.8333333333333333`, `ema_decay=None` / JSON `null`.
`lr` is G; `d_lr_mult` scales G into D; `prior_lr_mult` scales G into the
particle-bank rate. Rounded quotients can change exact recipe identity.
[The fixed-bank API](FIXED-BANK-API.md) preserves all four accepted tuples and
its real preparation/dispatch contract. These existing commands only construct,
check and plan a declaration; they do not retrain the completed selected arm:

```sh
ph3-slider bank-config --lr 0.036 --d-lr-mult 0.5000000000000001 \
  --prior-lr-mult 0.8333333333333333 > bank-request.json
ph3-slider bank-check --config bank-request.json
particle-slider plan --profile ph3-image-fixed-bank32 \
  --config bank-request.json --no-reporting
```

All four declared arms have completed RAW32. Do not invoke `bank-train` to repeat
them. The original common API cell2 first use is preserved as distinct training
interface evidence, not the LR winner. Any genuinely unfinished declared work
would require its actual fresh Root-owned preparation and exact latest FULL
continuation; no prefix, teacher preparation or completed proof is replayed.

## One score and reuse

Only equal-case intended-effect ON−OFF selects. Teacher/student SSE is historical
diagnostic information; preservation, beauty, confidence and completion are not
selectors, vetoes or tiebreaks. Preserve negative gains, native zeros and exact
ties. Use RAW current adapter/critic/particle-bank weights with no weight EMA
(`ema_decay=None`); official scalar optimizer statistics are unchanged.
Reuse completed searches, source-bound RAWs, outputs and first observations.
No extra top-two extension, worse-arm completion, rerating or fresh confirmation
is needed. Earlier planning, CPU fixtures and source-cohort reports remain
historical and are not reinterpreted as this native result.

## Availability

All32ON+8sharedOFF native2048 preview images exist in the original local deployment.
They represent one trained WarmGoldenLight control and zero released weights.
The local model/conditioning/data, captured native Python environment, original
Source identities and owned admission infrastructure are not bundled. Artifact
names/bytes/hashes identify retained evidence; they are not public download links.
Installing/checking/planning alone is not portable native reproduction.
`ph3-native-response095` returns the exact existing response-only factory;
no generic arbitrary prompt/data trainer or public image renderer is supplied.
Ordinary Generate/preset deployment and integration remain separately pending.
