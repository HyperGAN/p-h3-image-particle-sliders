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

Pass `record_native_dose=True` when rendering a fixed control to get the
same `native_sigmas` and `sigma_weighted_strength` fields as a gated
rollout. The grids come from the image and audio schedulers that ran the
request. For the shared P-H3 grid, divide the gated dose by the total
traversed sigma width to obtain the matched constant strength. Render with
that strength, one gain of `1` per interval, and opt-in dose reporting to
check both modalities.

## Calibrated activation gate

Pass `gate_plan=PackedGatePlan(...)` to the same `scheduling.rollout` entry
point to gate a constant nominal strength on the live packed latents. The
readout concatenates the FP32 row means of the image and audio tensors before
the adapter runs. The shared CFP or CLABS equations use that projection to
select OFF or the same nominal strength for both modalities. The matched
random control uses an explicit seed and firing rate. Each request starts a
fresh gate and makes one decision per native solver interval at nonzero
nominal strength. Zero nominal strength skips gate decisions and rolls out OFF.

```python
from ph3_native_scoped.scheduling import PackedGatePlan, rollout

plan = PackedGatePlan(
    core_wheel=shared_core_269_wheel,
    method="cfp",
    parameter=measured_threshold,
    direction=fitted_joint_direction,
    calibration_sha256=calibration_record_sha256,
)
latents, trace = rollout(owner, conditioning, seed, strength=nominal_strength, gate_plan=plan)
```

Build the optional [core #269 wheel](https://github.com/HyperGAN/particle-sliders/commit/b1f780decc516808cfd3019d5e5b674457161f86)
from `packages/particle-sliders-core` and pass its path. The scheduler verifies
and loads the shared gate source from the wheel alongside the captured strict
RAW owner. For CLABS, supply the calibrated bias as `parameter`, a `seed`, and
`clabs` with the `CLABSParameters` fields. For `random`, supply the measured
matched rate and a `seed`. The receipt records the gate inputs, decisions,
realized firing rate, and sigma-weighted strength for image and audio. For a
recorded gate replay at nonzero nominal strength, pass each effective strength
divided by the nominal strength as the corresponding `step_gains` entry.
[Shared equations and calibration procedure](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

Fit the joint direction from matched target and neutral packed trajectories of
the admitted FP4 model and measure the threshold or bias with steering active
on separate calibration cases. Test the result against constant and matched
random controls on held-out, finished native images using the established
intended-effect and independent integrity checks. The archived fixed-schedule
controls remain available for the admitted RAW profile. Adaptive gains are
unmeasured. Set `PH3_ADAPTIVE_CORE_WHEEL` to the same wheel when running the
installed synthetic fixture to include the adaptive checks.
