# P-H3-Image Particle Sliders

Source-only preparation for [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image),
using **the released Anima gmix formulation as the reference**, as requested.
No model loader, H3 attention-target discovery, trained slider, image sampler or
pretrained-model result is implemented. Native operations remain disabled.

The next native experiment's [teacher/student LR protocol](docs/LR-SEARCH.md)
and [proposed manifest](docs/lr-search.proposed.json) follow the common base
metric. Analytical metric controls are implemented; no native LR candidate or
trained H3 slider exists yet.

The selected source is Anima `c018f327`, classic shared core `a119ca1e` (0.3.0)
and its exact ParticleGAN `82b72661` dependency (0.6.0). The complete Anima
`Game`/`Sampler` module is reused byte for byte through the additive shared
`particle-sliders-anima-gmix` proposal; it is not copied into this product.
The modern PG0.9.4 draft is historical and is not the default generation.

**Shared-package publication is held for owner review.** The product pins local
shared proposal `18ff6c18565263a693b4a85ba6680beb8d577a65`. That source is not yet
publicly pushed, so a normal Git dependency install is unavailable until the owner
clears the source proposal. CPU checks use verified local wheels in isolated
paths. Existing core/Anima sources and active environments have not been changed.

## Software checks

After the shared source proposal is reviewed and reachable, use a new compatible
CPU PyTorch environment and install this product with its exact dependencies:

```bash
python -m pip install '.[dev]'
ph3-slider plan --prompt 'A documentary photograph of a ceramic cup beside a window.'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ph3-slider cpu-check
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ph3-slider gan-check --steps 4
python -m pytest -q
```

Planning and native refusal need only Python's standard library:

```bash
PYTHONPATH=src python -m ph3_sliders.cli plan --prompt 'Make the cup red.' \
  --reference cup.png --width 2048 --height 2048
```

The planner reads no images and allocates no runtime. `train`, `render`,
`download` and `publish` refuse. All six native operations also refuse through
`ph3_sliders.lifecycle.BACKEND`; the gate is independent of the modern core
session API, which does not exist in classic core0.3. There is no acknowledgment
flag, staged-model argument or Hub option that unlocks native H3 use.

## What the gmix fixture executes

The unrelated synthetic host is one frozen random 8×8 Linear. Product glue
attaches a rank-8 branch using the pinned shared `RoutedMLP`, with the actual
shared 128×4 bank, width48 branch MLP and width16 router. The up projection starts
at zero. Strength0 bypasses the branch exactly; strengths between0 and1 interpolate
its contribution. Target names are explicit; no Anima/Cosmos names are guessed
for H3. Only owned hooks are removed, preserving other hooks and frozen weights.

Four synthetic examples supply paired positive/neutral frozen velocities at each
of eight ordered synthetic positions. Both teachers use the same latent/time;
the student uses neutral conditioning. Eight is a geometry test, not a native H3
collector or schedule. Shared normalization retains the original training-only
per-position Welford sample deviation and median-RMS gain. Full responses are
flattened into the gmix critic without spatial pooling, sketches or rank8 residual
projection. The toy residual width is32, separately from adapter rank8.

The imported Anima Game owns independent D/G row and noise streams, microbatch1,
Adam updates, relativistic paired losses, lazy autograd cap and particle VIC.
Learning rates are generator2e-5, critic9e-4 and particles6e-3. Every auxiliary
training weight is zero: no reconstruction, perceptual or distillation loss.
The imported Anima EMA helper uses decay0.995. `gan-check` performs bounded CPU
updates, writes no artifacts and returns detached scalar metrics.

```python
from ph3_sliders.gmix_fixture import Config, SyntheticGmixRun

config = Config(steps=8)
with SyntheticGmixRun(config) as run:
    run.train(stop_after=4)
    packet = run.snapshot()  # in-memory synthetic continuation only
with SyntheticGmixRun(config) as run:
    run.restore(packet)
    run.train()
    response = run.evaluate(strength=.5, selection='ema')
    synthetic_adapter = run.export(selection='ema')  # not an H3 artifact format
```

These checks establish synthetic source wiring, not an H3 architecture, useful
concept, GPU fit, native checkpoint format, sampler parity or image quality.
No model weights are loaded. See [the formulation and source differences](FORMULATION.md),
[CPU evidence](VALIDATION.md), [next native steps](docs/NEXT-STEPS.md) and
[license findings](docs/LICENSE-REVIEW.md).

The official H3 community grant excludes the US, EU, UK and Republic of Korea;
separate applicable rights remain unresolved. Original source is Apache-2.0;
shared core, ParticleGAN and Anima source have their own MIT notices. These source
licenses grant no model-weight or output rights.
