# Native HyperGAN/HNDL qualification and LR search

Use `tools/modern_native.py` for the modern single-GPU recipe and
`tools/render_native.py` for its actual native images. These execute the real
HyperGAN develop trainer and HNDL networks; see [engine differences](HYPERGAN-DEVELOP.md)
and [HNDL networks](../packages/ph3-hndl/README.md). This is a separate experiment
from the classic Anima batch-eight runner. Its LR must be measured again.

The reusable [particle-sliders evaluation and LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md)
defines selection. The teacher is the frozen positive-conditioned native
velocity; the student uses neutral conditioning with strength one. Preserve the
entire image plus coupled audio response, matched state/time, both frozen paths,
and the complete declared position grid. Pool squared error and edit energy
across positions before division, then average equally over matched examples.
Report per-scene errors and signed gain. Training whitening and GAN losses do
not define the selection score. OFF, exact teacher and reversal controls must
give 1, 0 and 4. Algebraic controls are not learned predictions.

Freeze a collection plan containing prompts, seeds, layouts, native sigmas,
resolution, model/source/precision and split membership. A distinct modern
training plan references that immutable plan's SHA. The driver fixes effective
batch one and the full schedule horizon; each signature records the actual
resolved KA2 recipe, source closures, initial tensors and normalization. Only
training responses fit normalization. A change to the GAN engine, effective
batch, numerical precision, objective or data cohort requires new qualification
and LR measurement. A longer continuation keeps its original schedule horizon.

The root product still declares the incompatible classic runner dependency.
Keep the modern packages in their own runtime; do not install ParticleGAN0.9.4
over the classic environment. For the staged local experiment, the exact built
and installed modern packages are selected explicitly, and the unchanged native
host is loaded from this checkout. The source-closure guards verify imported
runtime bytes rather than trusting distribution names.

## Local staged invocation

These paths describe the current Pop!_OS experiment. No command downloads a
model implicitly or stops another workload. Physical GPU1 is selected by UUID.

```bash
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/ml2/particlesliders-working/ph3-native-deps:/tmp/ph3-hypergan-develop-runtime:/tmp/ph3-hypergan-bridge-release-installed:/tmp/ph3-hndl-installed:/ml2/particlesliders-working/ph3-diffusers/src:/ml2/particlesliders-working/ph3-modern-native/src
ph3_python=/ml2/codex-storage/future-workspaces/2026-10-31/task-14/.venv-qwen-native/bin/python
ph3_run=/ml2/particlesliders-working/ph3-native-runs/warm-lighting-v1
ph3_common=(
  --model-directory /mnt/nvme/sliders/models/PrunaAI--P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df
  --inventory /ml2/particlesliders-working/ph3-native-source/HF_FILE_INVENTORY.json
  --source-manifest /ml2/particlesliders-working/ph3-native-source/35087a644038eb531f48581708d4763044cb4852/SOURCE_MANIFEST.json
  --plan "$ph3_run/PLAN.json"
  --training-plan "$ph3_run/MODERN-TRAINING-PLAN.json"
  --conditioning "$ph3_run/conditioning"
  --cache-root "$ph3_run/caches"
)
"$ph3_python" tools/modern_native.py qualify "${ph3_common[@]}" \
  --output "$ph3_run/modern-qualification" --rate 2e-5 --seed 17
```

Qualification must finish before candidates can run. It verifies staged bytes,
every loaded checkpoint leaf and exact int8 conversion, upstream native image
and audio sampler parity, full-development OFF, four actual pretrained updates,
two replay calls with exact full trainer state, frozen base integrity and trained
OFF bypass. A reduced CPU model test does not satisfy this gate.

Then run each proposed rate to the **same** completed update count:

```bash
for ph3_rate in 6e-6 2e-5 6e-5; do
  "$ph3_python" tools/modern_native.py trial "${ph3_common[@]}" \
    --qualification "$ph3_run/modern-qualification" \
    --output "$ph3_run/modern-search/g-$ph3_rate-seed-17" \
    --rate "$ph3_rate" --seed 17 --steps 128 || break
done
```

The driver compares initial adapter/critic hashes across rates, binds the exact
qualification and cache identity, records actual optimizer rates, saves complete
critic/optimizer/EMA/stream/RNG checkpoints, and writes raw full student responses
with the metric. Full checkpoint and evaluation directories publish atomically.
Interrupted staging directories remain separate; they never become completed
checkpoints. Repeating a trial resumes its last complete checkpoint. Preserve
training logs: an update beyond the latest checkpoint can have completed before
an interruption and must not be silently counted as retained state.

Use `tools/select_modern.py --cache "$ph3_run/caches/development" --steps 128
--seed 17 --trials <the three trial directories> --output <fresh nomination.json>`
to audit the completed comparison. It verifies full checkpoint and raw-response
hashes, exact row labels and source identity, recomputes the teacher/student
scores from the saved full tensors, then applies the tolerance-aware nomination.

Nominate candidates from common-budget EMA evaluation only. Require finite
complete evaluations, positive gain, and every scene better than OFF. Scores
within 1e-6 favor the lower G rate. An edge winner calls for a targeted logarithmic
bracket extension; a failed or incomplete run is unscoreable. Inspect matched
OFF/strength/teacher media before choosing the winner. Confirm finalists at256
updates and with training seed29 before calling the recipe stable. Extend the
selected full state to at least512 updates with the same horizon; report its own
completed count, separately from qualification, replay and discarded pilots.

To continue a selected candidate, repeat its exact `trial` command with
`--steps 256` and then `--steps 512`. Do not change source, recipe or rates while
resuming. Every retained checkpoint also has a portable HNDL EMA inference export.

The renderer's `export` mode takes the same model/inventory/source/collection
plan/conditioning/cache-root arguments, plus `--export`, `--scene`, `--split`,
`--seed` and a fresh `--output`. Omit the training-plan argument. It verifies the
artifact's frozen forward/engine/graph identity, samples strengths
0/0.25/0.5/0.75/1 and the positive teacher with identical noise/sigmas, releases
the transformer, then decodes every image through the same tiled native VAE.
PNG files, full final image/audio states, hashes and a contact sheet are retained.
`cached-teachers` mode decodes a recorded pair's `final-states.safetensors`; its
images are labeled frozen references, never trained slider results.

## CPU evidence and current limits

The combined installed packages passed12 checks, including actual reduced H3
layers, reference output/gradient/second-derivative parity, native HNDL graphs,
real HyperGAN updates, exact fresh-host full-state replay, OFF and EMA export.
`tools/verify_modern_io.py` additionally injects checkpoint and evaluation
publication interruptions and verifies recovery, state preservation and exact
HNDL EMA reload. Run it with `CUDA_VISIBLE_DEVICES=''` using the selected modern
runtime above. These checks allocate no GPU and make no pretrained-quality claim.

Actual run status, candidate scores and retained step counts belong in run
receipts. Neither this recipe nor a proposed LR grid asserts a measured winner.
