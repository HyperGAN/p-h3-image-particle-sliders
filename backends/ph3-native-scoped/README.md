# P-H3 Image scoped native runtime

This separate package exposes fresh initialization, exact FULL restore and
RAW responses for the packed image-and-audio host. It pins Particle Sliders
core 0.15.2, ParticleGAN 0.11.1, HNDL 0.10.0 and the reviewed Diffusers
layout and scheduler. It verifies the installed source inventory and RECORD
ownership. The [earlier fixed-bank results](../../LEARNING-RATES.md) retain
their source provenance.

```sh
python -m pip install './backends/ph3-native-scoped[test]'
python -m pip check
```

## Optional installed-owner software check

The separate fixture is `tests/scoped_owner_check.py` within this subproject. Its name stays outside default pytest file discovery. The root project's existing `testpaths=["tests"]` and `pythonpath=["src"]` remain unchanged; this standalone helper is not folded into the root package or source-test context.

Select the fixture explicitly only in a genuine installed-helper wheel environment with the exact frozen compatible dependencies/provenance required above. Copy this exact fixture alone into an empty directory outside the repository, change into that directory, and use its installed Python3.12:

```bash
/absolute/path/to/compatible-installed-env/bin/python -I -m pytest \
  -c /dev/null --import-mode=importlib /absolute/neutral-directory/scoped_owner_check.py
```

The neutral directory contains no repository configuration, conftest or Source packages. `-I` ignores Source PYTHONPATH; `-c /dev/null` avoids the root configuration that would add `src`. Source/editable imports do not replace genuine owning-distribution RECORD checks. No existing workflow/test is changed, no guard is relaxed, and this command is not a public CI/native-availability claim or a mandate to replay retained software observations.

## Scheduled packed rollout

An opened strict RAW inference owner can supply its native conditioning and
seed to `ph3_native_scoped.scheduling.rollout`. Pass one `step_gains` entry
per native denoising step. The product of nominal strength and each gain
must lie in `[0,1]`. The function uses the original packed image and audio
scheduler, returning native latents and the applied strengths. The caller
can decode the latents through its admitted native decoder. Pretrained
finished-image effect is pending.
