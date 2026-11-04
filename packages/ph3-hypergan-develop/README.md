# Explicit HyperGAN develop P-H3 binding

Install this package in a separate runtime from `particle-sliders-anima-gmix`.
It pins canonical HyperGAN develop5f368a3, ParticleGAN0.9.4 and HNDL0.10.0.
The parent product's default classic dependencies remain unchanged. See
`docs/HYPERGAN-DEVELOP.md` in the product checkout for the execution audit and
semantic differences. This is a local integration proposal with CPU evidence;
pretrained GPU fit, learning rates and image quality require their own run.

With the HNDL sibling package and frozen P-H3 host modules available:

```python
from ph3_hndl import RoutedAdapter, GlobalMixCritic, runtime_identity
from ph3_hypergan import NativeHyperganRun, fit_normalization

# First verify frozen source/model/precision identity with
# ph3_hypergan.verify_teacher_identity, load staged weights with the existing
# runtime-neutral native_runtime loader and bind every loaded weight.
# host is H3Host(frozen_model,...); train is the immutable NativeCache.
# Seed adapter AND critic creation identically for each rate candidate.
host.adapter = RoutedAdapter(host.model, host.targets, device=host.device)
normalization = fit_normalization(train, positions=train.positions)
critic = GlobalMixCritic(normalization['scale'].shape[1], device=host.device)
run = NativeHyperganRun(host, train, normalization, critic,
    identity=dict(verified_identity, hndl=runtime_identity()),
    batch_size=1, seed=17, generator_lr=2e-5,
    critic_lr=9e-4, particle_lr=6e-3, steps=1600)
metrics = run.update()                 # inherited HyperGAN update; run.step == 1
packet = run.snapshot()                # full state, not an adapter-only save
run.restore(packet)
scores = run.evaluate(development, selection='ema', raw_path='student.safetensors')
ema = run.adapter_weights(selection='ema')
# ph3_hndl.export_adapter(host.adapter, directory, identity=...,
#     selection='ema', completed_steps=run.step, values=ema)
```

`evaluate` uses the same complete-grid pooled raw teacher/student metric as the
product, preserves adapter state/modes/RNG, and exports full labeled responses.
Its normalization is used only for training. All state packets bind exact engine,
bridge/component sources, initial weights, cache, rates and configured schedule.
Classic state packets are rejected. Use a fresh modern qualifier and LR plan;
batch1/KA2 measurements are not classic batch8/Adam measurements.

The qualification tests deny network and CUDA initialization. From the product
checkout, with the optional HNDL and native host dependencies on `PYTHONPATH`:

```sh
python -m pytest -q packages/ph3-hypergan-develop/tests
```

The actual native proof uses a reduced randomly initialized real H3 model. It
does not load pretrained H3 weights or claim full-model GPU memory qualification.
