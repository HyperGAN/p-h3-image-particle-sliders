# Pure ParticleGAN source preparation and the Anima distinction

The proposed CPU law check uses the repository's existing immutable dependency:
core `5adca87d32a159d529a89592fe35d3476dca7da3` (0.12.0), official ParticleGAN
0.9.4 and the public `bispf_optimized` preset. No recipe loss/penalty/prior
coefficients are overridden. **This is a source proposal, not a decision that
the released Anima gmix experiment has migrated to this generation.** Native
P-H3 adoption and the intended generation remain separately qualified work.

Nova's source review identified three different Anima paths. We inspected the
Anima repository at `c018f32759f49a05c2360a04b4a83924a6851913` and its locks:

| Path | Source evidence | Meaning for this proposal |
| --- | --- | --- |
| Released gmix routed sliders | [game.py](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/game.py), [particles.py](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/particles.py), [core.lock.json](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/core.lock.json) | Shared winning-formulation stamp, older Adam/cap/VIC game, bounded global-mix critic, per-position normalization and noise schedule. This is the released reference, not ConditionalPG09. |
| Anima PG0.9 source proposal | [pg09_training.py](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/pg09_training.py), [PG09_TRAINING.md](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/PG09_TRAINING.md), [lifecycle.py](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/lifecycle.py) | Paired frozen positive velocity, neutral student, actual routed bank, native fields and shared public GAN updates; distinct adoption/history gates. |
| Historical textsliders MSE experiments | The separate historical namespace in the core | Not selected, imported or used for this proposal. |

Anima's released lock names core `a119ca1ecd3d5d6c437065839d22739b04f2f4d8`.
We compared its `formulation.py` and `reference.py` with the P-H3 pin. They are
not byte-identical: legacy PG builders and the loss constructor have changed,
and modern recipe access adds an explicit optimizer-profile policy. These
comparisons do not establish cross-generation numerical, optimizer, checkpoint
or media-result parity. Do not silently swap Anima's historical checkpoints,
normalization, calibrated alpha values or noise-hold behavior onto this stack.

## Implemented shared law in the unrelated fixture

The factory calls the pinned [ConditionalPG09](https://github.com/HyperGAN/particle-sliders/blob/5adca87d32a159d529a89592fe35d3476dca7da3/packages/particle-sliders-core/src/particle_sliders/pg09.py).
For the fixture's actual field response `v`, frozen positive target `v+`, scalar
response scale `s=1`, and independently sampled Gaussian reference noise `n`:

```text
real = n
fake = n + (v_adapted(neutral, x, t) - v_frozen(positive, x, t)) / s
D total = public adversarial D loss + public critic penalty
G total = public adversarial G loss + public particle-prior regularization
```

The shared game constructs the public recipe, optimizers, learning-rate policy,
independent D/G row and noise streams, source-bound state and all update math.
The product adds no reconstruction, pixel/velocity MSE, perceptual, contrastive,
identity, distillation or other auxiliary training loss. It has no local GAN
update/optimizer loop. The shared lifecycle owns stepping, optional EMA, strict
resume and synthetic export. EMA here is shared lifecycle behavior; it is not a
claim that Anima's older standalone PG09Trainer computed EMA.

The validation receipt records the complete public recipe, including dormant
preset fields that are not terms in this executed step. The source-bound engine
signature and tests against real D/G metrics establish the executed loss
decomposition; the presence of a recipe field is not evidence that it is used.

Qualification evidence belongs to this exact generation and source cohort. If
native generation selection later chooses the released gmix stamp, none of the
new PG0.9.4 CPU-law, critic, admission, resume or export evidence transfers; rerun
the wiring qualification under the selected generation. The shared lifecycle's
optional EMA does not establish either historical Anima export contract.

The unrelated fixture follows the reusable structural mechanics: a frozen base,
zero-initialized additive up projection, shared rank-8 routed branch and one
actual 128×4 particle bank, frozen paired teachers at identical state/time,
neutral student and scale +1 training. Conditions contain neutral context and
synthetic time/scale, not row IDs. Only FP32 adapter leaves and critic enter
optimizers; base weights remain frozen. The shared `FieldCritic` consumes the
actual `[1,8,2,2]` synthetic field and emits raw logits. This differs from both
released Anima's bounded global-mix critic and its proposed local-patch critic;
no exact Anima critic or result parity is claimed.

The fixture's field dimensions, two rows, sigma, short budget and random Linear
are software-test choices. They are not H3's packed geometry, native attention,
conditioning, image quality or sampling schedule. No external runtime, files,
teacher cache or model can be supplied to the factory.

## Native P-H3 boundaries remain closed

The real `ph3-image` backend still refuses every native operation before model
allocation. The synthetic backend cannot be used to admit a staged H3 model.
The applicable model rights, exact native Diffusers implementation/projections,
reference-conditioned pair definition, frozen expanded prompts, native velocity
sign/time convention, checkpoint sigma grid, actual response shape and native
sampler parity remain unresolved. H3's eight-step schedule must not be replaced
with Anima's ten-position cache or an invented Euler trajectory.

The modern core refuses flat native continuous vectors of width >=170. The
fixture's genuine 4D field is not an exemption for H3: do not flatten, project,
spatially reshape or relabel native H3 outputs to bypass admission. No weights,
model derivatives, H3 responses or media outputs were produced by this work.
