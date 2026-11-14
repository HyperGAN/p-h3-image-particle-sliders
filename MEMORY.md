# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Allocation-free planner / bank checker | 0 | 0 | 1 | No native weights; stdlib metadata only |
| Original fixed-bank cached-response training | 1 | 64 | 32 | B2; native2048²+4audio rows; text≤256; int8/BF16; checkpointed |
| Native response / separate finished Generate | 1 | 48 | 32 | B1;2048²; text≤256; int8/BF16; native8steps |
| Separate conditioning / VAE decode | 1 | 24 | 32 | B1; text≤256 or2048² image; BF16 encoder / FP32 VAE |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Actual native profile

Model `PrunaAI/P-H3-Image@7f98acef503cf2bcfba821a78cb1cf1a9928b2df` is the Pruna eight-step image adaptation.
The published [fixed-bank API](docs/FIXED-BANK-API.md) preserves batch2,
2048² geometry, complete393344-coordinate image/coupled-audio response,
rank8 routed adapter and128×4 bank. The [native authority](src/ph3_sliders/native-authority.json)
pins its original external loader and package. The public inference factory is
response-only; the separate Generate integration owns rollout/decode. The
historical planner's refusal is not a zero-memory claim about that native host.

The original `source/native_loader.py:load_transformer` uses bnb **int8**,
threshold6, `llm_int8_has_fp16_weight=False`, FP32 protected modules and native
checkpointing. It does not load a66 GiB BF16 DiT and does not use the later H3 FP4
implementation. All model math in this profile is on one GPU, with conditioning
cached separately. No CPU layer offload or frozen-base optimizer state is assumed.

## Calculation

* Published mixed-precision DiT tensor bytes: 66,280,430,080 (61.728 GiB).
  Use half this as a conservative count/storage bound for its quantized matrices
  (some source tensors areFP32, making this deliberately larger than their true
  element count), then add1 GiB for sourceFP32 islands, unquantized norms/biases,
  per-output-row FP32 SCB scales and alignment. Frozen int8 upper estimate:
  **31.864 GiB**. It has no permanent FP16 master copy. The threshold6
  outlier path and single-layer dequant/matmul have a separate3 GiB allowance.
* Attention Q/K/V/out hosts:208; rank8 LoRA 20,873,216 scalars +6476 per
  routed bridge +512 bank = 22,220,736 adapter scalars.
  The original full-response critic first mixes `(393344+3)→384`, not a pooled
  response. Its8-token/width48/one-layer set block gives 151,065,025 parameters.
  Combined trainable/update allowance: **12.911 GiB**.
* T≤4096image+4audio+256text=4356. A B2 checkpoint-boundary envelope
  `2*T*5376*2*53` is 4.624 GiB. Reserve8 GiB total for
  these boundaries, one full recomputed native block, FP32 routed128-way logits,
  retained penalty/backward graphs and fields. Add3 GiB int8 workspace +5 GiB
  runtime, then round to64 GiB training. This is deliberately a planning estimate,
  not a new requirement to rerun or invalidate the saved completed RAW32 arms.
* Response/sampling: frozen31.864+4 live+3 int8+5 runtime rounds to48.
  Conditioning and VAE are separate phases: BF16 Qwen3-VL4B is8.266 GiB;
  the 5.177 GiB serialized VAE is conservatively allowed twice that
  residency for FP32 conversion. Add8 live decode +5 workspace and round to24.
* Host32: largest native shard9.19 GiB + one source-cast verification/layer
  copy2 GiB + ≤128 complete cached native rows/conditioning2 GiB + FP32
  checkpoint/trainable snapshots4 GiB + runtime/loader margin. Preparation,
  streaming, and native response do not require all original66 GiB shards to be
  copied into RAM simultaneously. Disk download size is a different quantity.

The table binds256 text tokens and128 cache rows for calculation; source-provided
longer conditioning or references require a fresh estimate. Existing scheduler
caps and card sizes are not the origin of these numbers. No arbitrary new LR arm,
training reset, current winner/default change or native availability is implied.


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Allocation-free planner / bank checker | 0.000 → 0 | 1.000 → 1 |
| Original fixed-bank cached-response training | 60.775 → 64 | 29.190 → 32 |
| Native response / separate finished Generate | 43.864 → 48 | 29.190 → 32 |
| Separate conditioning / VAE decode | 23.355 → 24 | 29.190 → 32 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
