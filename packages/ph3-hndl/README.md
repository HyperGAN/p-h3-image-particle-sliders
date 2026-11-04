# P-H3 native HNDL networks

This optional, separately installed package executes the Anima routed adapter
and full-response global-mix critic with **actual native HNDL graphs**. It does
not import `particle_sliders`, install the classic runtime, load a model, or
implement a GAN loss. The owning trainer supplies the objective and normalization.
The current HyperGAN develop bridge consumes these networks in a separate cohort.

The canonical repository is `https://github.com/HyperGAN/HNDL`, inspected at
`df5af35d3b4017e1164b011d12264a01546a3ff4` (HNDL 0.10.0). All 108 Python source
hashes match the released wheel used for verification; construction rejects a
different source closure or version. `src/ph3_hndl/runtime.json` records it.

The adapter retains rank 8, one shared trainable 128×4 particle bank, the width16
router and width48 branch MLP. The router, scaled particle softmax, weighted bank
lookup, concatenation and up/down maps execute through native HNDL operators.
Hooks attach only to the host's explicit native linear targets. The frozen host
never enters adapter parameters. `scaled(0)` returns before any graph executes;
`close()` removes only this adapter's hooks. Strength is finite and in [0,1].

The critic takes the **entire normalized image+audio residual [B,H]**, globally
mixes it into eight width48 tokens, applies pre-RMSNorm manual attention and a
SiLU FFN, and computes the mean/max bounded readout. Native HNDL matmul/softmax
preserves double backward for the existing lazy cap penalty. Backend parity
against the actual pinned Anima reference was checked for outputs, input and
particle gradients, and critic first/second derivatives. This package supplies
no new reconstruction/distillation loss. A modern HyperGAN/ParticleGAN trainer
must identify its own cohort; it is not a byte-identical classic Anima Game.

Install in a fresh, independently selected runtime:

```bash
python -m pip install ./packages/ph3-hndl
```

Do not overlay modern ParticleGAN on the classic environment: classic core
imports an API removed by ParticleGAN 0.9.4. This package depends on HNDL0.10.0, Torch and `safetensors` for inference
artifacts; it has no ParticleGAN or classic-runner dependency.

```python
from ph3_hndl import RoutedAdapter, GlobalMixCritic, export_adapter, load_adapter

# host is the existing native H3Host; all base parameters are frozen.
host.adapter = RoutedAdapter(host.model, host.targets, device=host.device)
critic = GlobalMixCritic(full_image_audio_width, device=host.device)
# Pass these exact owned objects and the full residual callback to the trainer.

identity = {"model_revision": model_revision, "source": source_identity,
            "base_precision": "int8"}
export_adapter(host.adapter, "slider", identity=identity, completed_steps=steps)
host.adapter.close()
host.adapter = load_adapter(host.model, host.targets, "slider",
                            identity=identity, device=host.device)
result = host.rollout(neutral_context, seed, strength=0.5)
host.adapter.close()
```

`ph3-hndl-routed-inference-v1` artifacts contain only adapter leaves, a weight
hash, native target geometry, exact graph/source/plan identities, float32 adapter
precision, the caller's model/source identity and raw/EMA selection. EMA export
requires the owning trainer's adapter state through `values=...`. Loading checks
all identities and tensors, preserves RNG, and removes owned hooks if admission
fails. Artifacts are inference exports, not resumable training checkpoints.

Run offline CPU checks, supplying the existing pinned native sources and exact
Anima `reference.py` if backend parity is desired:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PH3_NATIVE_TEST=1 PH3_REFERENCE_SOURCE="$ANIMA_REFERENCE_PY" \
PYTHONPATH="packages/ph3-hndl/src:src:$NATIVE_DIFFUSERS_SRC:$HNDL_RUNTIME" \
python -m pytest -q packages/ph3-hndl/tests
```

Six checks passed with Python3.12.15/Torch2.14.1. The actual reduced random H3
proof executed 16 HNDL attention branches and the HNDL critic, kept all64 image
and24 audio coordinates, changed the particle bank, preserved frozen weights,
and reproduced inference and OFF rollouts exactly after export/load. Its three
simple differentiability-probe updates verify autograd wiring, **not a trained
GAN or pretrained image quality**. The separate HyperGAN bridge verifies its
own actual trainer updates/checkpoint replay with these networks.
