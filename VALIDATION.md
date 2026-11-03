# CPU qualification by source cohort

## Selected classic Anima gmix candidate

The current source candidate selects classic shared core0.3.0 at a119ca1e and its
ParticleGAN0.6.0 dependency at82b72661. It imports the isolated shared extraction
proposal18ff6c1. The complete new evidence is recorded in
[validation/anima-gmix-cpu.json](validation/anima-gmix-cpu.json); publication remains
held for shared-package owner review.

Shared qualification: seven installed-wheel/source-free tests passed. Eight
updates of the extracted Game and pinned original Game match exactly for adapter,
critic, optimizers, saved row/noise streams, EMA and metrics. The generalized
normalizer with its default ten positions exactly matches the pinned original
function. Whole-module SHA proves byte-identical Game/Sampler extraction.

Product tests exercise full width32 responses independently of rank8, actual
128×4 bank/optimizer identity, source learning rates and bound8 critic, frozen
base/teachers, exact continuation with original Game state and EMA, selected EMA
export, source/cohort rejection, cache copy/order/train-only invariants, owned
hook cleanup including partial registration failure, zero/fractional strengths,
CPU/RNG/thread preservation, no artifact files, bounded work and native refusal.
The classic gate is standard-library-only and never imports a model runtime.

The source fixtures use only unrelated random CPU tensors. They do not establish
native H3 geometry, conditioning, architecture, pretrained convergence, image
quality, memory fit, secure native checkpoint IO or sampling compatibility. No
pretrained weights are loaded and no H3 outputs are produced. Synthetic snapshots/exports
are in memory only and are not native H3 artifacts.

## Historical modern PG0.9.4 proposal

[validation/anima-pg094-cpu.json](validation/anima-pg094-cpu.json) records59 passing
checks at the superseded modern proposal fef61c1. Nova reviewed runtime45c9078 and
identified the released-versus-proposed Anima distinction. Its critic, objective,
optimizer, admission and export evidence does not transfer to classic gmix. The
historical receipt is retained unchanged; that generation is not selected.

## Historical initial scaffold

[validation.json](validation.json) records40 installed-wheel CPU checks at870f9d3,
which checked a shared projection and guarded modern lifecycle registration.
Those source hashes belong to that immutable cohort and are not current code
identities. No initial or later software receipt claims native H3 qualification.
