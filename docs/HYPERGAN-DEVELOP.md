# HyperGAN develop engine audit and isolated binding

Verified against GitHub on 2026-11-04. The canonical repository is
[`HyperGAN/HyperGAN`](https://github.com/HyperGAN/HyperGAN/tree/5f368a3770cf5bdae0448995c9f76e29af8049f7),
whose `develop` head was **5f368a3770cf5bdae0448995c9f76e29af8049f7**, version2.0.0.
This is separate from `255BITS/ParticleGAN` (`develop` was
d2636ac3b08e86e5fd04dbbdc4d8c22e613857c2). HyperGAN's `train` extra requires
ParticleGAN>=0.9.4,<0.10 and HNDL>=0.10.0,<1; it does not request the moving
ParticleGAN develop head.

The P-H3 product at6e3782a does not execute HyperGAN. Its path is
`NativeTrainingRun.step -> particle_sliders_anima_gmix.anima_game.Game.update`
from runner18ff6c18565263a693b4a85ba6680beb8d577a65, shared core
a119ca1ecd3d5d6c437065839d22739b04f2f4d8, and ParticleGAN0.6.0
82b72661c03377f348ccb76de37cd4ca02d45a8d. These pins remain unchanged.

The generations are incompatible at execution time: importing that shared core
with ParticleGAN0.9.4 fails because `particlegan.GradientPenalty` no longer
exists. The classic runner additionally verifies exact0.6.0 distribution and
source hashes. Neither a metadata dependency nor calling classic0.6
"HyperGAN develop" constitutes an integration.

`packages/ph3-hypergan-develop` therefore supplies a separately installed
`ph3_hypergan` namespace and cohort. It uses canonical
`hypergan.training.ReferenceTrainer.update -> run_native_program` unchanged.
HyperGAN compiles the objective program, runs D then G/prior, checks gradients,
applies schedules and EMA, and serializes/restores complete trainer state.
ParticleGAN0.9.4 supplies the real KA2 optimizers, loss, penalty and prior VIC.
The bridge owns sample bindings and parameters, not a copied GAN update loop.

The runtime locks HyperGAN's entire Python source closure (SHA256
`9c725516409bfbb80f9740c65a435c0b7219be55d0ca88e2c777dc76aea2cb1e`),
ParticleGAN0.9.4 and HNDL0.10.0 closures and imported distribution origins.
`foldlitedb==0.1.2` is pinned for HyperGAN checkpoint file primitives. The
HyperGAN wheel built from the canonical checkout matches the checked-out Python
closure byte-for-byte. This does not assert a GPU or pretrained qualification.

| Surface | Classic Anima product | Explicit modern cohort |
|---|---|---|
| Engine | Extracted Anima Game | Canonical HyperGAN develop executor |
| GAN primitives | ParticleGAN0.6 | ParticleGAN0.9.4 KA2 |
| Effective batch |8, microbatch1 | Explicit caller choice; batch1 named separately |
| D/G rows and noise | Five classic independent streams | Independent D/G batches; serialized HyperGAN data/prior/penalty/noise streams |
| Noise in paired coordinates | Shared Gaussian real/fake; geometric1600, edit-RMS hold | Same declared paired surface; extra modern input/output noise explicitly0 |
| Particle VIC |64 sampled of128 | Full128 bank, weight1 |
| Optimizers/penalty | Adam + classic b-cap | Modern recipe KA2, its guard/EMA/damping and penalty |
| Learning rates | Constant per-run groups | Modern recipe cosine network/prior schedules; fully resolved config in identity |
| EMA |0.995 |0.995 via HyperGAN graph/prior shadows |

`ph3-hypergan-develop-ka2-paired-v1-batch1` is a new experiment cohort. Classic
learning-rate winners and qualification receipts do not transfer to it. Every
run records full resolved modern recipe, effective batch, initial adapter and
critic hashes, source hashes, cache/normalization/grid, engine and HNDL identity.
The caller seeds adapter and critic construction identically before each LR
candidate; the supplied seed separately fixes engine streams. Save/restore uses
the original configured horizon; continuation must not shorten/rebase schedules.

Single-process `ReferenceTrainer` has no gradient accumulation interface.
Canonical `ReplicatedTrainer` supports replay accumulation but rejects fewer
than2 ranks and rejects independent phase draws when accumulation>1. The bridge
does not claim supported single-GPU microbatch accumulation. GPU memory must be
measured for the explicitly selected batch; this CPU work allocates no GPU.

The HNDL sibling package supplies the runtime-neutral `RoutedAdapter` and
`GlobalMixCritic`. `NativeHyperganRun` binds them to the existing H3 host and
full image-then-audio cached responses. It requires matching native timestep
geometry, frozen base, explicit batch size, train-only normalization provenance
and training-cache fingerprint. A reduced real H3/HNDL CPU test exercises all16
attention targets and88 full response coordinates, an actual D/G/prior update,
fresh-host complete checkpoint replay, OFF identity and unchanged base.

The host closure is never copied into an EMA model. Generator EMA copies are
parameter-only shadows whose forward explicitly raises. The canonical executor
only updates those shadows; its penalty forwards a separate critic EMA from the
critic optimizer. EMA inference exports map checked parameter names/order/shape
back to the original owned adapter hooks. Checkpoints include graph/prior/EMAs,
both optimizer states (including KA2 counters and critic EMA), named streams,
global Python/NumPy/Torch CPU/CUDA RNG, modes, buffers, trainability, step and
last batch. Interrupted partial updates cannot be snapshotted.

Frozen classic-collected teacher caches can be reused only after
`verify_teacher_identity`: it compares actual forward package versions, clean
pinned Diffusers source, model and precision settings, frozen collection plan,
staged-byte identity, encoded conditioning, and native host/runtime/cache/CLI/
plan source hashes. It retains the complete original collection identity.
Classic PG/core/runner and `adapter.py` metadata did not execute OFF teacher
forward and are retained separately instead of being misrepresented as modern
training requirements. The new modern training plan may reference the original
collection plan without rewriting or recollecting it.

The main pretrained collector and its worktree, installed shared packages,
weights and GPU state were untouched. This patch stays isolated; the existing
shared-package publication hold still applies.
