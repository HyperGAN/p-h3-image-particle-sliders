# WarmGoldenLight measurement caveat — 2026-11-09

**Current eligibility update:** the separate qualified catastrophic-only check
rejects cell3/G0.036 and passes all8 nominal cell1/G0.004 saved outputs. Its
exact FULL32 guard also passes; cell1's unchanged mean
`+0.002527267051239808` is the current best remaining saved recipe. See
[current eligibility and settings](current-eligibility-20261109.md). This does
not repair the warmth proxy, rewrite its original numerical decision, clear
cell3's failed state or retarget the fixed cell3 lower-alpha Generate preset.
The sections below retain the earlier proxy/diagnostic evidence and scope.

The original frozen SigLIP2 warmth proxy assigns its largest eight-case mean to
cell3/G0.036: +0.017445931987216074. The user confirmed the output is broken. Review of the saved bottle, room and
bicycle cases found yellow/noisy fields without recognizable subjects or surfaces.
Those high warmth scores therefore do not validate the intended illumination
effect. G0.036 is the highest recorded proxy arm, not a working-control or
strength recommendation. A subsequent lower-alpha diagnostic shows that this
checkpoint can retain a clear bottle in one matched case; the result and its
limits are recorded below. That lower-alpha diagnostic itself introduced no
replacement score or new LR comparison; the later separate eligibility update
above reuses the original effect means after rejecting a catastrophic candidate.

The [portable decision](finished-output-decision.json) and
[portable recipe snapshot](finished-output-rate.json) are preserved byte-for-byte,
including all four rates, means,32 matched pairs and40 first image intensities.
Their original status/readiness wording belongs to that earlier snapshot. This
dated caveat supplies the current interpretation. No training, saved RAW/FULL,
first observation, score, default, Source lock or historical result is rewritten.

## Existing training diagnostics

An already recorded warning is the official scalar `ema_data_scale` at step32:

| G | Saved data scale | Saved generator loss |
| --- | ---: | ---: |
| 0.0004 | 158.7708650792203 | 0.37205770611763 |
| 0.004 | 156.332122558045 | 0.0160676259547472 |
| 0.012 | 218.90051878932127 | 2.851877450942993 |
| 0.036 | 856.7639633019552 | 5.2079243659973145 |

`ema_data_scale` is a running scalar of the square root of the median pairwise
real/fake distance squared in normalized training-response coordinates. It is
an instability warning, not an image/illumination metric or a calibrated collapse
cutoff. These official scalar statistics are not model-weight EMA. GAN loss is
also not a universal failure threshold: G0.004 reached generator loss
7.527050495147705 at step29, above G0.036's final value. Finite training and high
warmth similarity do not establish a working illumination control. No blanket
GAN-loss cutoff or new preservation/beauty selector is introduced.

All original arms genuinely completed RAW32: G0.004 inherited two proof updates
and added30; the other arms added32 each. There is no established duplicate-alpha,
wrong-rate/update-count, target/cache, formula or decoder-normalization defect
from the finite Source/metadata audit. Tensor bodies were not reaudited. Alpha
multiplies each learned projection residual once throughout the nonlinear
denoising trajectory; it is not a final-image blend. The lower-alpha work below
uses the same RAW for diagnosis, not a new LR search.

## Completed one-case lower-alpha diagnostic

On 2026-11-09, the same cell3/G0.036 RAW32 produced two new native2048 outputs
for the original bottle scene, seed171, at α0.125 and α0.25. Review of the actual
PNGs found a clear bottle with subtle warmth at α0.125 and a clear bottle with
warm amber illumination at α0.25. The reused α1 output is yellow/noisy. Full
strength therefore overdrives **this case**; lower strengths retain its subject.
This does not establish a general
strength setting, an eight-case effect result or a new LR ranking. The original
strength1 comparison and every numerical result remain unchanged.

These full URLs require access to the original Tailscale deployment; the images
are not bundled with this repository:

| Strength | Image | Scope |
| --- | --- | --- |
| OFF / 0 | [Original matched OFF](http://100.90.104.57:8770/media/ph3-warm-golden-raw32-native2048-bottle-seed171-shared-off) | Reused original baseline |
| 0.125 | [New native diagnostic](http://100.90.104.57:8770/outputs/16b279ecd69448f5be1d5819c7b1fad0/ph3-alpha0p125.png) | Clear bottle, subtle warmth in this case |
| 0.25 | [New native diagnostic](http://100.90.104.57:8770/outputs/16b279ecd69448f5be1d5819c7b1fad0/ph3-alpha0p25.png) | Clear bottle, warm amber illumination in this case |
| 1 | [Original full-strength ON](http://100.90.104.57:8770/media/ph3-warm-golden-raw32-native2048-bottle-seed171-cell3-on) | Reused yellow/noisy output |

The local [sampler](http://100.90.104.57:8770/#sampler) Generate default now
requests this exact bottle171 two-alpha diagnostic (α0.125 and α0.25 together),
with the existing OFF/α1 references. It uses the already verified diagnostic
route and unbundled local runtime/assets. The original response-only public
factory is unchanged; this fixed local preset does not add arbitrary-prompt
public rendering or choose a generally recommended strength. Changing the UI
default itself made no new model calls.

The diagnostic completed16 new native responses, eight per lower alpha, followed
by two VAE decodes under one VAE load. OFF and α1 were reused, with zero new
training updates, teacher forwards or LR scores and no post-hoc image blending.
The saved receipt records RAW-before/after and caller Torch/Python RNG checks,
both2048 HTTP outputs with whole-byte checks, worker/guardian exit0, reap and
owned cleanup to zero. These are checks of this two-alpha diagnostic, not a
retroactive claim about the earlier failed full-run checks.

Retained terminal: `Root-actual-two-alpha-success-HTTP-resource0-v1.json`,4284
bytes, SHA256
`3e49d2ee201374e26cb9d6c245c1131b9e9cc5bbb16030a0244aa15173a241f5`.
The terminal is private retained evidence, not a public download. No new LR comparison,
measurement instrument, acceptance threshold or package default is introduced.

### The existing proxy on the two new PNGs

The original frozen SigLIP2 instrument measured only the two new PNGs' first
intensities. All40 original first observations, including this case's OFF and
α1, were reused without rerating. The margin is the original mean of three
positive-minus-neutral cosine margins; gain is ON margin minus the cached
matched OFF margin. These are **one bottle171 case**, not eight-case LR means:

| Strength | Frozen proxy margin | Gain vs matched OFF | Observation |
| --- | ---: | ---: | --- |
| OFF / 0 | -0.013993004957834879 | — | Cached original |
| 0.125 | -0.017160538583993912 | -0.0031675336261590328 | New first observation |
| 0.25 | +0.00650530681014061 | +0.020498311767975487 | New first observation |
| 1 | +0.005606179436047872 | +0.01959918439388275 | Cached original, visibly broken |

α0.25 scores slightly above the broken α1 output, but α1 still receives a
positive gain. This proxy therefore supplies no reliable hard failure cutoff
for this known failure. The negative α0.125 result is retained unchanged.
No new metric, LR ranking, general strength recommendation or loss cutoff is
introduced; the original four-arm comparison remains frozen.

The actual CPU6 comparison completed with worker/guardian exit0, outer reap and
only its own6 released. Retained result: `worker-output/result.json`,10226 bytes,
SHA256 `3e61ca66ac1bb6bd575d10c0fc9af4e159e467785f9df6602e63ffccbc83ff7e`.
Retained terminal: `Root-actual-CPU6-terminal-reap-resource0-v1.json`,5782 bytes,
SHA256 `556fec29a14734957f7b734ea63e2f0896708777b28bce421efd6a3c87ab03c0`.
These identify private retained evidence, not bundled public artifacts.

## Narrow local runtime completion

The fixed bottle171/native2048 ordinary-queue preset completed by continuing
the exact saved native latents: eight previously completed ON responses, zero
new native responses, one new VAE decode and exact original OFF reuse. Both2048
PNG outputs were served with whole-byte identity checks, and worker/guardian
exit0, reap and owned release were recorded. The new ON is byte-identical to the
already published/scored cell3 ON; its appearance predates this continuation.

The original full-run post-decode RNG and final RAW checks were not reached in
the preceding failed attempt. The continuation checked a separate fresh decoder
RNG scope. Actual guardian entry evidence is retained; independent live capture
was not obtained for the short continuation. No retrospective capture or fresh
complete end-to-end claim is made. The local completion is distinct from useful
illumination, arbitrary-prompt support or a public image-render API. The public
factory remains response-only and local assets are not bundled.

The retained local technical terminal is
`Root-actual-v4-decoder-success-HTTP-resource0.json`,4914 bytes,
SHA256 `41a2002b318a8da8a42ed5ce5fb9adda4be62cd2365a26a9384f5439e1311fbc`.
This identifies private retained evidence; it is not a public download link.

The single objective remains intended illumination in finished outputs,
expressed as equal-case ON−matchedOFF gain. The known proxy meaning failure
must remain visible; its scores must not silently become validated effect
scores. Reuse completed work and retain every original numerical result.
