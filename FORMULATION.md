# Selected reference: released Anima gmix

The selected generation is the released Anima gmix formulation, not the newer
ConditionalPG09/BISPF proposal. Nova identified the distinction; Mikkel explicitly
selected the released reference. The prior PG0.9.4 fixture and its59-test receipt
are historical and provide no qualification for this selected generation.

## Exact source reuse

- [Anima source](https://github.com/HyperGAN/anima-particle-sliders/tree/c018f32759f49a05c2360a04b4a83924a6851913): `lumen_studio/game.py`, `cache.py`, `particles.py`.
- [Classic shared core](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/packages/particle-sliders-core): `winning_formulation()`, routed MLP, global-mix critic, paired losses, cap, VIC and noise curve.
- [ParticleGAN source](https://github.com/255BITS/ParticleGAN/tree/82b72661c03377f348ccb76de37cd4ca02d45a8d): exact0.6.0 dependency supplying the adversarial and regularizer primitives.

The additive shared package proposal reuses the entire Anima Game/Sampler module
byte for byte, SHA256 `802ba0706a465aaa3d269064f3deb494e275cc89d4d793eb207856c937ac3ab5`.
It also reuses the original EMA helpers. Normalization changes only the fixed ten
positions to an explicit host position count; Welford/sample-standard-deviation
and median-RMS arithmetic are unchanged. The default ten-position helper matches
the original function exactly in CPU qualification. Eight extracted/reference
updates match adapter/critic weights, optimizer/sampler states, EMA and metrics
exactly on the same synthetic inputs. These are source-extraction checks, not
pretrained-run or image-quality parity.

No existing core or Anima file was edited. The new shared package has its own
namespace/environment and exact source-tree admission checks. It is held locally
for package-owner review, then explicit source-only publication clearance. This
product imports that runner; it does not vendor a local Game, loss or optimizer.

The old shared `FormulationGame` is not substituted: its rank8 feature game and
single noise draw differ from Anima's full residual, independent D/G streams and
per-position/microbatch mechanics.

## Objective and training mechanics

The target and student edits share the same frozen neutral response, so the full
normalized paired residual is:

```text
e_t = (v_adapted(neutral, x, t) - v_frozen(positive, x, t)) / S_t
real = sigma_t(k) * epsilon
fake = real + e_t
D loss = relativistic paired D loss + autograd gradient cap
G loss = relativistic paired G loss + particle VIC
```

Only training targets fit S_t: per-coordinate sample standard deviation of the
positive-minus-neutral edit, floor1e-4, then per-position median-RMS gain. No
spatial pooling, bottleneck or random sketch is introduced. The critic mixes the
full residual into eight width48 tokens, one four-head attention layer and an
8*tanh-bounded score. The original cap is applied lazily every fourth update;
VIC uses64 of the actual128 particles. There is no reconstruction, perceptual,
identity, feature-matching or other auxiliary training objective.

The separate `cpu-check` squared-norm backward is only a gradient diagnostic with
zero optimizer updates. It is not executed by `gan-check` and is not a term in
the Anima training objective.

The actual bank is128×4, with rank8 additive branches, width16 router and width48
branch MLP. All adapter/base ownership is preserved: the base is frozen, only
adapter and critic leaves enter Adam, zero up initialization preserves the base,
training strength is+1, and inference strength is separate. Source learning rates
are g2e-5, d9e-4 and particles6e-3; betas(0,.999), no weight decay, effective batch8,
microbatch1, horizon1600 and EMA decay.995. A shortened CPU budget does not shorten
that source noise horizon. Calibrated alpha values and ordinary-LoRA distillation
from released Anima are not copied onto H3.

**Historical noise discrepancy remains explicit:** Anima FORMULATION.md reports
absolute noise hold1.0 for the published weights. Its pinned source Game calls
the shared stamp's hold `edit_rms * 1.3`. We preserve that source exactly. This is
not a P-H3-specific modification and is not evidence of bit-identical reproduction
of the published run. The source or original run evidence must resolve that
historical difference before any such claim.

## Necessary P-H3 differences and unverified parts

| Boundary | Anima reference | P-H3 status |
| --- | --- | --- |
| Attention attachment | Cosmos Q/K/V/output names,224 projections | Actual H3 projection/fusion/packed-token boundaries unverified; explicit targets required, no suffix guess |
| Sampling positions | Ten native positions on both same-noise frozen paths | Public image adaptation reports eight steps; actual native grid/time and velocity direction still require source/CPU parity |
| Normalization | Per-position full-velocity training statistics | Same arithmetic, explicit actual position count; native H3 cache has not been collected |
| Conditioning | Text conditioner and paired prompts | Qwen3-VL, frozen expanded prompts, edit target and ordered references need a native pair contract; no reference-conditioned factory exists |
| Critic geometry | Entire native residual flattened for gmix | Same full-residual rule; actual packed H3 response width/memory fit unmeasured, no rank8 approximation is substituted |
| Scale/checkpoint replay | Scale held through update and recomputation | Owned CPU hooks tested; native activation-checkpoint/offload/scale replay remains unqualified |
| Artifacts/rendering | Selected EMA routed adapter and native Anima sampler | Only synthetic in-memory continuation/export exists; H3 provenance/schema/decoder/sampler/Comfy compatibility unimplemented |

The fixture is an unrelated random Linear, not a tiny H3 architecture or native
pipeline. Its two-to-ten position contract, four examples and width32 response
prove software wiring only. No external model or records can enter its run factory.
Model rights, immutable Diffusers implementation, native host/teacher validation,
actual checkpoint replay and held-out image quality remain separate gates. The
real H3 native command/interface refuses before runtime allocation.
