# P-H3 teacher/student LR experiment

This is preparation for a native experiment, not an executed LR search. The
native loader, packed response geometry, eight-step schedule, derivative rights
and GPU fit remain unresolved. The existing native refusal is retained.

Follow the [base evaluation and LR protocol](https://github.com/HyperGAN/particle-sliders/blob/3e8052250d593a55d9f071cebae503e764342d05/packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md).
The local `ph3_sliders.evaluation` implementation measures the same pooled
relative squared L2 formula and retains additive raw-response statistics. Its
analytical tests do not qualify a native model.

## Measurement binding

For this unipolar formulation, teacher is the frozen positive-conditioned
velocity and student is adapted neutral-conditioned velocity at strength 1.
Neutral/OFF, teacher and student must use identical packed latent, native time,
expanded prompt identity and reference ordering. The initial concept proposal
is **warm golden lighting**, using text-to-image with no reference images.
Neutral and positive descriptions differ only in the lighting instruction.
Do not substitute MOSS's signed positive-minus-negative field.

Keep raw native velocity coordinates for evaluation. Training-only per-position
gmix whitening does not become evaluation weighting. An example is scene, seed
and frozen rollout path (`neutral` or `positive`); pool every declared native
position before division and average examples equally. Require a complete
declared example/position grid, including both paths. Retain per-scene/path
scores, gain, cosine and orthogonal error. OFF=1, teacher=0, reversed edit=4.
Do not use critic loss, EMA lag or image edit magnitude as the selection score.

Before fitting, freeze actual prompt strings and expansions in disjoint splits:
training ceramic cup / stone cottage / wooded trail / fabric still life;
development glass bottle / city bicycle / reading room / fishing boat;
final confirmation bird / garden bench / mountain lake / portrait.
Use seeds17/29 for training,171/172 for development,271/272 for confirmation.
These are proposed experiment inputs; no cache or untouched confirmation set
exists yet. Model geometry, timestep/grid, prompt expansion and response hashes
must be filled from the qualified native collector, never guessed here.

## Native prerequisites

Pin the model/code/dependencies and demonstrate full checkpoint weight binding,
actual projection targets, FP32 adapter leaves, participating particle bank,
neutral/positive same-state teacher correctness, frozen base and OFF identity.
Measure actual full-response critic and activation memory on GPU1. Confirm one
update and exact full-state split/resume replay, including EMA, optimizer,
sampler/noise streams, RNG and schedule index. Freeze raw teacher/neutral
development responses; analytical controls must pass in the actual precision
cohort. Establish native rendering parity before selecting an LR from images.

The source Game fixes its defaults in TRAINING_CONFIG. The new
`ph3_sliders.learning_rates` binds explicit rates to a fresh owned Game's actual
Torch optimizer groups without patching shared source or module globals. It
checks parameter ownership, refuses binding after optimizer updates, and reads
rates back for verification. Native adoption still needs review and a full
checkpoint recipe identity: serialize `LearningRates` alongside model/cache/
stream hashes, check the declared recipe before loading, then call
`assert_learning_rates` after optimizer restore and around updates. The unrelated
fixture's existing snapshot schema does not implement that native identity.
Preserve exact source objective and other knobs. The source noise schedule is 1,600 steps,
with `edit_rms*1.3` hold, despite the documented published-run discrepancy.
The native work must not silently change that to a 128/512-step horizon.

## Proposed bounded search and continuation

After qualification, generator candidates are 6e-6, 2e-5 and 6e-5, with critic
9e-4 and particles6e-3 fixed. These bracket the selected Anima source starting
point; none is a transferred MOSS winner or a prequalified H3 rate. Record the
actual optimizer-group rates. All candidates start from byte-identical adapter,
bank and critic states and identical streams; use seed17 initially. Save full
checkpoints and development measurements at32/64/128, then compare at128.
Use identical 1,600-step schedule horizon, batch8/microbatch1 and EMA0.995.
Predeclare EMA as the inference selection and score raw weights separately as
diagnostics. Failed/nonfinite/incomplete candidates are unscoreable.

Require improved pooled error overall and in every development scene, positive
gain, exact OFF and useful matched outputs. Ties within1e-6 nominate the smaller
generator LR. Compare finalists at256 and confirm with training seed29. An edge
winner or a rank reversal triggers a targeted follow-up bracket/longer common
screen; do not declare a universal recipe from this small experiment.

Render finalists against OFF and the actual positive-conditioned teacher using
the same native noise, grid, decoder and frozen prompt expansion, strengths
0/.25/.5/.75/1. Check lighting direction, content preservation and artifacts.
Keep image judgments separate from velocity scores and record any rejected
candidate. Run the final heldout confirmation only after the choice is frozen.

Continue the selected seed17 **full state** to at least512 completed updates,
retaining each64-step full checkpoint, selected EMA inference exports, raw
responses, hashes and a final matched contact sheet. A checkpoint at512 must
contain512 completed updates, excluding replay checks. Report every candidate,
actual rates/steps, score/version/cohort, selection, sample paths and cleanup to
Ember. Use a fresh bounded GPU1 ownership/admission record, share within measured
headroom and stop only this run's owned processes. The user's parallel-sharing
authorization does not make another workload's memory available.
