The completed Wild Hair checkpoint is step128, selected by EMA. This is an early exploratory result; the separate investigation of the training recipe remains unresolved. This sampling branch changes only rendering control code and preserves every numerical product source and the modern native driver from commit3a5120f.

On this host, after the owner pauses GPU0 training at a durable checkpoint:

```sh
./sample-wild-hair --check
./sample-wild-hair
```

The preflight validates the actual pinned modern runtime/teacher identity, serialized adapter SHA and encoded conditioning with CUDA hidden. Full rendering also validates staged base weights and loaded projection binding, then uses the strict original HNDL inference loader. Rendering needs40500MiB free on physicalGPU0; GPU1 remains owned by another job.

The fixed command uses the original2048×2048 plan, held-out `woman_dev` conditioning and seed171: a29-year-old woman leaning forward with a delighted toothy grin, denim jacket and hoop earrings. The matched neutral prompt has ordinary dark brown hair. The positive prompt has a lopsided fan of jagged spikes, corkscrew curls and flyaways.

Six PNGs compare the plan's original strengths0,.25,.5,.75,1 and a positive-prompt teacher reference. The latter is a separate prompt reference, not slider output. The pinned adapter contract supports only0..1. Before attachment the helper renders a seventh latent-only OFF rollout, then requires exact equality of both packed image and coupled audio against attached strength0. Failure aborts the strip.

Outputs are `/ml2/particlesliders-working/single-command/runs/wild-hair-step128-samples`: individual full-resolution PNGs, `contact-sheet.png`, `final-states.safetensors`, loaded base binding and `RENDER.json` with byte hashes, identities, selection, step and OFF parity. No training updates occur. The source checkpoint, training run and weights are read-only inputs.
