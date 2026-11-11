# P-H3-Image Particle Sliders

**Current eligible saved recipe — 2026-11-09:** **cell1/G`0.004`**, original
eight-case mean **+`0.002527267051239808`**. All eight nominal-strength saved
outputs passed the qualified catastrophic-only check, and its exact FULL32
checkpoint passed the persistent core guard. Exact options are `lr=0.004`,
`d_lr_mult=4.5`, `prior_lr_mult=7.499999999999999`, `ema_decay=None`.
The failed cell3/G0.036 candidate stays blocked; every original score remains.
See [current eligibility and boundaries](docs/current-eligibility-20261109.md)
and [sanitized current metadata](docs/current-eligible-rate.json).

**2026-11-09 measurement caveat:** cell3/G`0.036` has the highest frozen
warmth-proxy mean, **+`0.017445931987216074`**, across eight original matched
native2048 cases. The user confirmed the output is broken; review of saved bottle, room and
bicycle cases found yellow/noisy fields without recognizable subjects. This proxy result does not
establish a working illumination control or a recommended strength. A subsequent
same-RAW bottle171 diagnostic retained a clear bottle at α0.125 and α0.25, with
warm amber illumination at α0.25; full strength overdrives this case. This is
one-case evidence, not a new LR ranking or general strength recommendation.
See [the current caveat and four images](docs/measurement-caveat-20261109.md).

Retain the historical cell3 RAW32 and its exact measured options `lr=0.036`,
`d_lr_mult=0.5000000000000001`, `prior_lr_mult=0.8333333333333333`,
`ema_decay=None`. Actual fixed D/bank are `0.018000000000000002` /
`0.029999999999999995`. All four declared RAW32 arms are complete; no new
search or confirmation is required. The local fixed-preset decoder continuation
completed with HTTP output and owned cleanup; it does not add arbitrary-prompt
public rendering or establish control quality. See [learning-rate guidance](LEARNING-RATES.md),
[historical portable options](docs/finished-output-rate.json) and
[all32 pairs/40 first image intensities](docs/finished-output-decision.json).

This product revision registers the current core **0.14.0** native
training profile `ph3-image-fixed-bank32`, the explicit `ph3-root-owned` runner,
and the unchanged `ph3-native-response095` public RAW inference factory. PG is
**0.9.5**. See [the fixed-bank API](docs/FIXED-BANK-API.md) for exact supported
options, preparation dependencies, checks and execution ownership.

The supported training method is only the already declared four-arm RAW32 bank.
It uses the frozen native data, native2048 geometry, optimizer and FULL durability
code. It does not add arbitrary prompt/data training, a generic image lifecycle,
or a public image renderer. Installing the adapter does not create a trained
result. On 2026-11-09, the installed common API completed the original cell2
G0.012 arm: 32 new updates, zero inherited updates, RAW weights and no EMA,
with a genuine successful outer exit, reap and Root resource closure. This
verifies the fixed original bank's local training frontdoor. This is a static
first-use record; remaining work is determined from actual durable metadata.
The later frozen numerical comparison selected existing cell3/G0.036 without
repeating training. The separate catastrophic check now rejects that candidate
and permits saved cell1/G0.004 as the best remaining original score. The local
Generate preset still uses cell3 at both fixed lower strengths; no route or
weights were retargeted. Exact first-use evidence and scope are recorded in [the fixed-bank API](docs/FIXED-BANK-API.md#completed-common-api-first-use).

The public inference entry point returns the exact existing response-only factory
object from its pinned original file. NativeRow conditioning, local model/runtime
assets and leases remain required. The separate ordinary Generate integration
owns finished-image rollout and VAE decode.

The original planner and CPU fixture remain available. The following description
records the earlier 0.2.0 planning package, not the capabilities of the explicit
fixed-bank profile. Its unresolved historical TODOs are not new proof or search
requirements for the already captured native work.

## Historical planning package

Current planning/synthetic package: core **0.13.0** at `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, published PG **0.9.5**. This changes no native capability or model authorization. Historical validation below belongs to its recorded source.

Source-only integration planning for [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image),
using [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
This initial revision has a request planner, an installed shared-lifecycle
registration that refuses native operations, and an unrelated synthetic CPU
fixture that exercises the shared routed-particle projection. It contains no
model loader, trained slider, pretrained-model results, or model outputs.

**Native download, loading, training, rendering and model publication are disabled.**
The official MiniMax H3 license excludes the US, EU, UK and Republic of Korea
from its community grant. Separate authorization and native qualification remain
unresolved. See [the license review](docs/LICENSE-REVIEW.md).

## Install and check

Use a new Python >=3.11 environment. Install your selected CPU PyTorch >=2.6,<3
build first, then install this project. This installs the shared core at a full
Git commit; ParticleGAN 0.9.5 is its transitive dependency. This product adds no
Diffusers/Transformers native runtime or model weights. The shared package retains
its own historical backend namespaces; none is invoked by this product.

```bash
python -m pip install '.[dev]'
ph3-slider plan --prompt 'A documentary photograph of a ceramic cup beside a window.'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ph3-slider cpu-check
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The planner can also run directly from source using only Python's standard library:

```bash
PYTHONPATH=src python -m ph3_sliders.cli plan --prompt 'Make the cup red.' \
  --reference cup.png --width 2048 --height 2048 --strength 0.5
```

Reference paths are ordered labels; planning does not read images. The first is
the proposed edit target, followed by up to seven references. Plans validate
32-pixel canvas alignment, an approximately 4–8 MP canvas, seed and proposed
unipolar strength. No native request is executed, and strengths have no image
quality qualification. Native prompt expansion must be frozen for paired teacher
collection in a future implementation.

The common interface is registered as `ph3-image`:

```python
from particle_sliders.lifecycle import SessionConfig, backend

config = SessionConfig('ph3-image', 'new-source-cohort', ema_decay=None)
print(backend('ph3-image').describe(config))  # operations=(), no native allocation
```

The common configuration explicitly disables EMA with `ema_decay=None`. Any
future separately admitted training, evaluation, export and inference must use
RAW current weights, without weight EMA, checkpoint averaging or EMA comparison.
This policy does not enable a native operation or alter historical evidence.

All six shared operations refuse before validation, PG admission, factory
preparation, CUDA selection or weight allocation. There is no environment flag,
`allow_hub`, acknowledgment switch or CLI option to unlock them in this revision.

## What the CPU fixture establishes

`cpu-check` constructs a random frozen 8×8 Linear and attaches the pinned core's
`TinyParticleProjection`. It checks exact base preservation at initialization,
fractional strength, finite nonzero gradients to the actual shared 128×4 particle
bank, and reversible attachment. The up projection is filled with fixed synthetic
values for gradient probing; no optimizer updates occur. Tests also cover nested
strength contexts, failure cleanup, one-time bank ownership, CPU RNG preservation
and native refusal before preparation.

This is a test of shared software wiring on unrelated synthetic tensors. It does
not execute an H3 architecture, evaluate its velocity field, train a useful
concept, establish GPU fit, or qualify a sampler. No synthetic adapter is saved.

## Native integration remains separate work

The inspected release is the **image** adaptation, not the existing MiniMax H3
video trainer. It uses `PH3ImageModularPipeline`, Qwen3-VL-4B conditioning,
`MiniMaxH3Transformer3DModel`, the checkpoint's native eight-step sigma grid, and
no CFG or negative prompt. These facts come from Pruna's public card and metadata;
no H3 implementation is copied into this repository.

The shared core owns the lifecycle, routed projection and future GAN machinery.
This product will own native packing, frozen conditioning, teacher collection,
attachment selection, source identities and native image sampling. Native
dependency and projection targets are deliberately unqualified rather than
guessed from another model's attention names. See [the implementation handoff](docs/NEXT-STEPS.md)
and [source pins](backend.lock.json).

Source: Apache-2.0. Shared core: MIT. These source licenses grant no H3 model rights.


## Original research095 fresh/FULL/RAW helper

The separate [cached fresh95 subproject](backends/ph3-native-fresh95/FRESH95.md) exposes the tested `ph3_native_fresh95` Python API for its exact original research095 installation. It preserves the root CLI/core/defaults and existing LR cohorts; portable dependency availability and pretrained/native rendering are not claimed.
