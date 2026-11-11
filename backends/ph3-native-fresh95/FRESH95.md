# Explicit cached fresh initialization (original p-h3 Image research095)

This standalone `ph3_native_fresh95` subproject adds fresh, strict FULL restore and RAW response APIs for the original research095 image-plus-audio host. It does not replace the root `ph3_sliders` package, `ph3-slider` CLI, registered profiles, existing native backend or [measured learning-rate guide](../../LEARNING-RATES.md).

The helper requires a genuine installed `ph3-image-native-explicit-fresh95==0.1.0` owner, Python3.12 and the exact original research095/core0.13/PG0.9.5/HNDL context captured in `src/ph3_native_fresh95/source_profile.json`. It checks generated RECORD ownership, exact Source inventories and original installation provenance, including the original local research-owner and PG-wheel URLs. Matching package versions or moving source files into `PYTHONPATH` is insufficient. This is exact original-context API/source exposure, not a portable dependency installer or evidence that the original research owner is publicly available. The root package uses a different core boundary and is not upgraded by this subproject.

An owner must supply the original frozen `H3Host`, its authentic model/geometry/completed-owner identity, and already completed original `PreparedResponse` pairs. Each pair includes the complete FP32 packed image **and** coupled-audio response. `save_prepared_cache(...)` can bind those existing rows and their original teacher-output receipts into a typed whole-file cache; it never calls prepare, collect_pair, rollout or a teacher. Do not reconstruct or reinterpret older cached rows to satisfy this API. Native weights, staging and resource admission remain with the caller.

```python
from ph3_native_fresh95 import configuration, open_training, open_inference

# recipe explicitly supplies lr, d_lr_mult and prior_lr_mult.
config = configuration(**recipe, seed=7, total_steps=128, ema_decay=None)
with open_training(host, completed_cache, "new-fresh-run", mode="fresh",
                   cohort_id="a-new-fresh95-cohort", config=config,
                   identity=owner_identity, owner_admission=owner_admission) as trainer:
    metrics = trainer.update()                 # explicit new work
    full = trainer.save()
    raw = trainer.export("new-raw.pt")

with open_training(restore_host, completed_cache, "new-resume-run", mode="resume",
                   cohort_id="a-new-fresh95-cohort", config=config, resume=full,
                   identity=owner_identity, owner_admission=owner_admission) as restored:
    pass                                      # whole state restored; no initializer

with open_inference(inference_host, raw=raw, identity=owner_identity,
                    owner_admission=owner_admission, ema_decay=None) as inference:
    packed_velocity = inference.response(original_native_row, strength=1.0)
```

The three host variables are owner-provided original hosts; each entry borrows an unattached host and restores its exact hooks/settings on close. Identity and completed-cache validation happen before new adapter/critic/game attachment. `owner_admission` receives the explicit operation and raw/no-EMA/known-stop policy. The caller authenticates the actual frozen model and completed teacher provenance. The helper validates and binds that declared evidence; it is not itself a proof of which pretrained weights the owner loaded.

Fresh mode applies the public PG initializer once before optimizers, on the same Parameters: down QR at the original a=√5 variance `1/(3·fan_in)`, R2Normal bank standard deviation1, protected zero UP, and router/bridge/critic constructor KEEP. The initialization receipt enters the game metadata before optimizer construction. It writes FULL0 and RAW0 before returning. Strict FULL restore keeps the original128 horizon and restores all game/optimizer/scalar/RNG/cursor state; RAW inference creates no critic/cache/initializer. `response` returns the original complete packed image-plus-audio velocity. It is not a finished-image render API or a standalone model-loading CLI.

Pass all three rate fields explicitly. The unchanged official law uses D=`lr*d_lr_mult` and bank=`lr*prior_lr_mult`; no new default, optimized rate or selection result is introduced. Existing saved LR results describe their captured cohorts and remain unchanged. This fresh profile cannot resume or retag an older FULL/RAW.

Four distinct tiny synthetic software cases have passed across retained executions, with one original shared-game update in total. Three passing cases and their update were retained from the original run; the repaired helper was genuinely built/installed and only the failed cleanup case was run again and passed. These observations cover policy/identity, complete coupled response, FULL/RAW and retryable foreign-safe cleanup. They do not establish public dependency availability, a portable install, pretrained native/finished rendering, output quality or new LR/effect performance.

## Optional installed-owner software check

The separate fixture is `tests/fresh95_owner_check.py` within this subproject. Its name stays outside default pytest file discovery. The root project's existing `testpaths=["tests"]` and `pythonpath=["src"]` remain unchanged; this standalone helper is not folded into the root package or source-test context.

Select the fixture explicitly only in a genuine installed-helper wheel environment with the exact frozen compatible dependencies/provenance required above. Copy this exact fixture alone into an empty directory outside the repository, change into that directory, and use its installed Python3.12:

```bash
/absolute/path/to/compatible-installed-env/bin/python -I -m pytest \
  -c /dev/null --import-mode=importlib /absolute/neutral-directory/fresh95_owner_check.py
```

The neutral directory contains no repository configuration, conftest or Source packages. `-I` ignores Source PYTHONPATH; `-c /dev/null` avoids the root configuration that would add `src`. Source/editable imports do not replace genuine owning-distribution RECORD checks. No existing workflow/test is changed, no guard is relaxed, and this command is not a public CI/native-availability claim or a mandate to replay retained software observations.
