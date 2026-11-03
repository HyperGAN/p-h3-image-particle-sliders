# Pinned upstream native contract — static inspection

Inspected official Diffusers commit
[`35087a644038eb531f48581708d4763044cb4852`](https://github.com/huggingface/diffusers/tree/35087a644038eb531f48581708d4763044cb4852),
including the P-H3 modular pipeline, MiniMax-H3 transformer and scheduler.
Twenty-one files (280,800 bytes) were fetched from that commit and their Git
blob IDs and SHA256 hashes verified. No model was instantiated, checkpoint
loaded or native output produced. The selected files carry Apache-2.0 headers;
model-weight rights remain a separate unresolved prerequisite.

Local evidence outside the product Git tree:

- `/ml2/particlesliders-working/ph3-native-source/35087a644038eb531f48581708d4763044cb4852/SOURCE_MANIFEST.json`,
  SHA256 `dcc1283d30bff7dbc818ffda3ea7cc305486f373d443922111fcd0ab6831bacc`.
- `/ml2/particlesliders-working/ph3-native-source/STATIC_INSPECTION.json`,
  SHA256 `3df216dd2ee6b4e82aed8d37f4f6a089e6abe295748097e577845fbc0693d058`.
- Six metadata files from model revision
  `7f98acef503cf2bcfba821a78cb1cf1a9928b2df` match `backend.lock.json` exactly.
  Fresh official LICENSE retrieval also matched its recorded SHA256.

## Time, velocity and precision

[`MiniMaxH3Scheduler`](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/src/diffusers/schedulers/scheduling_minimax_h3.py)
uses **`t = 1 - sigma`**, with clean time1, and a **data-ward velocity**.
It forms `x0 = x + sigma_from_timestep * velocity`, then blends using the ratio
of consecutive sigma-grid values. The source deliberately distinguishes sigma
recovered from the conditioned timestep from sigma read from the grid because
their floating-point rounding can differ. Preserve that arithmetic for parity.
Do not substitute generic Euler or MOSS's time scaling.

Metadata supplies nine sigma points including terminal0, for eight forwards.
`shift_provided_sigmas=false` keeps that explicit grid unshifted:

```text
1, .984753, .967905, .951688, .925372, .881923, .810083, .528718, 0
```

There is one conditional forward with no CFG or negative prompt. Both image and
audio schedulers receive the explicit grid.

## Packed geometry and coupled state

The [`before-denoise blocks`](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py)
pack text, audio and image rows. Text-to-image still includes four audio rows
(stereo2 × two latent positions), each32 scalars. They start from their own FP32
noise and evolve alongside the image. Removing or freezing them changes the
native trajectory.

Image latents have24 channels, one frame and spatial compression16. Patch
`(1,2,2)` packs frame-major then row-major, with96 scalars per image row.
The paired cache must retain **both latent states**, every conditioning layout
and row-timestep plan. Prompt length shifts media rotary time origins and row
indices. Freeze each conditioning's complete layout; an embedding swap cannot
assume another prompt's indices.

The synthetic `PairedCache` cannot represent these fields or be relabeled as
native. Before collection, explicitly choose and qualify the training/evaluation
surface for image and coupled audio responses. Preserve every declared response
coordinate at the gmix boundary; do not silently drop audio, pool or sketch.
The algebraic evaluator does not qualify that native surface.

The decoder unpacks generated image rows, reverses per-channel VAE latent
mean/std normalization, decodes frame0, then reverses ImageNet pixel mean/std
and clamps. Check against actual native decoding.

## Attachment and memory prerequisites

Source and recorded config identify208 available attention projections across
50 main and two text-refiner blocks: Q/K/V are5376→7168, output7168→5376.
Explicit paths are `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}` and
`token_refiner.refiner_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}`. These are source
candidates, not approved targets or loaded-module/key/shape proof. Qualify both
main and refiner checkpoint recomputation, scale lifetime, dtype and hook cleanup.

The card's33.1B transformer implies about66.2GB inBF16, exceeding GPU1's capacity
before activations and the full-residual critic. Native execution needs measured,
reviewed frozen-weight offloading or a separately qualified quantization cohort.
A full `.to('cuda')` cannot fit. Static arithmetic does not demonstrate fit.

This inspection creates no lease, source/model adoption clearance or pretrained
qualification. Continue from the coordinator handoff and [LR prerequisites](LR-SEARCH.md).
