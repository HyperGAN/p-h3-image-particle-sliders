# P-H3-Image Particle Sliders

Source-only integration planning for [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image),
using [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
This source proposal has a request planner, an installed shared-lifecycle
registration that refuses native operations, and unrelated synthetic CPU
fixtures for the shared routed projection and actual ParticleGAN updates. It contains no
model loader, trained slider, pretrained-model results, or model outputs.

**Native download, loading, training, rendering and model publication are disabled.**
The official MiniMax H3 license excludes the US, EU, UK and Republic of Korea
from its community grant. Separate authorization and native qualification remain
unresolved. See [the license review](docs/LICENSE-REVIEW.md).

## Install and check

Use a new Python >=3.11 environment. Install your selected CPU PyTorch >=2.6,<3
build first, then install this project. This installs the shared core at a full
Git commit; ParticleGAN 0.9.4 is its transitive dependency. This product adds no
Diffusers/Transformers native runtime or model weights. The shared package retains
its own historical backend namespaces; none is invoked by this product.

```bash
python -m pip install '.[dev]'
ph3-slider plan --prompt 'A documentary photograph of a ceramic cup beside a window.'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ph3-slider cpu-check
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ph3-slider gan-check --steps 2
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

The native common interface is registered as `ph3-image`:

```python
from particle_sliders.lifecycle import SessionConfig, backend

config = SessionConfig('ph3-image', 'new-source-cohort')
print(backend('ph3-image').describe(config))  # operations=(), no native allocation
```

All six shared operations refuse before validation, PG admission, factory
preparation, CUDA selection or weight allocation. There is no environment flag,
`allow_hub`, acknowledgment switch or CLI option to unlock them in this revision.

The separate `ph3-synthetic` backend runs only a fixed, unrelated random field:

```python
from particle_sliders.lifecycle import SessionConfig, open_session

config = SessionConfig('ph3-synthetic', 'new-synthetic-cohort',
    fixture='unrelated-random-field-v1', steps=4, ema_decay=.9)
with open_session(config) as run:
    run.train(stop_after=2)
    packet = run.checkpoint()  # synthetic in-memory packet; not an H3 adapter
with open_session(config) as run:
    run.restore(packet)
    run.train()
    response = run.evaluate(strength=.5)  # numerical synthetic field, not an image
```

This factory accepts no staged model, native request, teacher cache, Hub download
or auxiliary-loss options. `gan-check` writes no artifacts. The shared lifecycle
owns updates, EMA, strict checkpoint restore and export. The real H3 registration
is unchanged and remains disabled.

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

`gan-check` additionally runs the pinned `ConditionalPG09` with public
`bispf_optimized`, actual rank-8 routed branches and an actual 128×4 bank. Frozen
positive and neutral teachers share the same synthetic state/time; the student
uses neutral conditioning. Generator loss is the public adversarial loss plus
public particle prior, and critic loss is the public adversarial loss plus public
critic penalty. No reconstruction, perceptual or other auxiliary objective is
added. Resume, EMA and export tests exercise the shared implementations.

**This is not a claim of matching the released Anima recipe.** Anima's released
gmix sliders and its separate PG0.9 training proposal are different generations.
The existing P-H3 stack remains the proposed modern ParticleGAN path. See
[FORMULATION.md](FORMULATION.md) for the source comparison and unresolved native
generation decision, and [VALIDATION.md](VALIDATION.md) for bounded CPU evidence.

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
