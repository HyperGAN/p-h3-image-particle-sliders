# Core0.13 / ParticleGAN0.9.5 planning cohort

Package0.2.0 pins shared core `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` and its published ParticleGAN0.9.5 dependency. P-H3 remains planning-only: model operations are still refused before factory/validation. No foreign training process, environment, model weights or checkpoints are changed.

The unrelated random Linear forward/backward, fractional strength and particle-gradient checks pass with the new core. All40 existing tests also pass from an installed package outside the checkout, CUDA hidden. This is software evidence, not native P-H3 qualification or LR calibration.

The import control compares modules introduced by the product against interpreter startup: the new core's installed legacy namespace .pth imports its own namespace before product code. It still verifies that the planning path introduces no Torch or model runtime.

`backend.lock.json` retains the previous core metadata as history. Original `VALIDATION.md` remains historical; no old packet is relabeled as0.9.5.
