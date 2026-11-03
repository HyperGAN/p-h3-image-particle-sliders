# Initial CPU validation

**40 tests passed**, including a fresh-process, source-free installed-wheel run.
The product and exact shared-core wheels were built locally and installed under
an isolated `/tmp/ph3-installed` target. Exact ParticleGAN 0.9.4 was installed under
`/tmp/ph3-python-deps`; its exhaustive official Python source manifest passed the
core's `_released_sources()` check. Existing CPU Torch and test dependencies were
read from the Iris task's Python environment; that environment was not modified.

Runtime: Python 3.12.15, Torch 2.14.1+cpu, core 0.12.0, ParticleGAN 0.9.4,
pytest 9.1.1, setuptools 84.2.1. No native model libraries were required.

The test directory was copied outside the source tree, the checkout was absent
from `PYTHONPATH`, and every installed product Python file was byte-compared to
the proposed source before execution. The installed product CLI's `plan` and
`cpu-check` commands and the installed common `particle-slider capabilities`
command also passed. The latter reports no native operations or base precisions.

Checks used empty `CUDA_VISIBLE_DEVICES`, offline Hugging Face/Transformers modes,
and one OpenMP/MKL thread. Tests additionally reject socket connection creation
and CUDA initialization. No optimizer updates or adapter files were produced.

Coverage includes request validation and ordered references, standard-library-only
planning, installed backend discovery, all six shared lifecycle refusals before
preparation, actual shared projection forward/backward, unique particle-bank
ownership, fractional strength, exact fresh-base preservation, cleanup after
exceptions, CPU RNG/thread restoration, and CPU/FP32 behavior under different
caller defaults.

[validation.json](validation.json) records exact source hashes, wheel hashes,
package versions, installed module locations and the CPU probe result. Wheels
remain local validation artifacts; no package registry or model Hub was used for
publication. This does not qualify an H3 architecture, model license, pretrained
convergence, memory fit, native media sampling or a general runtime matrix.
