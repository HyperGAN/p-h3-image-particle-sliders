"""Fixed declared-bank product profile for the current common training API.

Stdlib-only planning and checking for the recorded fixed-bank cohort.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

PROFILE = 'ph3-image-fixed-bank32'
RUNNER = 'ph3-root-owned'
SCHEMA = 'ph3-original-declared-bank32-product-options-v1'
ARCHIVED_REVISION = '98b99f48fc36b1165b6a29f1ab6999f03d886996'
# Exact declarations from the completed fixed-bank plan, in original arm order.
RATE_TUPLES = (
    (0.0004, 45.0, 74.99999999999999),
    (0.004, 4.5, 7.499999999999999),
    (0.012, 1.5000000000000002, 2.4999999999999996),
    (0.036, 0.5000000000000001, 0.8333333333333333),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    p = Path(path)
    require(p.is_absolute() and not p.is_symlink() and p.suffix == '.json' and '..' not in p.parts,
            'Literal absolute JSON metadata required')
    raw = p.read_bytes()
    require(len(raw) <= 8 * 2**20, 'Bounded JSON metadata required')
    value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))
    require(isinstance(value, dict), 'JSON object required')
    return value, dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def profile():
    return dict(schema='particle-slider-profile-v1', name=PROFILE,
        product='p-h3-image-particle-sliders', config_kind='native', status='source_only',
        reason='Recorded four-arm RAW32 bank. Use the original product revision for dispatch. '
               'The current native scoped API is installed separately.',
        command=['{python}', '-m', 'ph3_sliders.execution'],
        runtime=dict(python=sys.executable, env=dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')),
        target_steps_path='steps', default_checkpoint_every=1,
        capabilities=dict(check=True, resume=False, periodic_checkpoint=True, device=False))


def rate_options(*, lr, d_lr_mult, prior_lr_mult, ema_decay=None):
    require(ema_decay is None, 'Weight EMA is disabled; ema_decay must be None')
    supplied = (lr, d_lr_mult, prior_lr_mult)
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in supplied),
            'Finite positive literal rate scalars required; booleans are unsupported')
    # Match the stored float values directly, without deriving multipliers.
    for index, (g_rate, d_multiplier, p_multiplier) in enumerate(RATE_TUPLES):
        if supplied == (g_rate, d_multiplier, p_multiplier):
            return dict(lr=g_rate, d_lr_mult=d_multiplier, prior_lr_mult=p_multiplier, ema_decay=None), f'cell{index}'
    raise ValueError('Only the four original declared RAW32 rate tuples are supported')


def config(*, lr, d_lr_mult, prior_lr_mult, ema_decay=None, prepared_entry=None):
    rates, phase = rate_options(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult, ema_decay=ema_decay)
    return dict(schema=SCHEMA, steps=32, batch=2, horizon=128, seed=7,
                phase=phase, prepared_entry=prepared_entry, **rates)


def validate_training(value):
    require(isinstance(value, dict) and set(value) == {'schema', 'steps', 'batch', 'horizon', 'seed', 'phase',
            'prepared_entry', 'lr', 'd_lr_mult', 'prior_lr_mult', 'ema_decay'}, 'Fixed bank options only; no prompts/data/model/device/EMA aliases')
    require(value['schema'] == SCHEMA and all(type(value[k]) is int for k in ('steps', 'batch', 'horizon', 'seed'))
        and (value['steps'], value['batch'], value['horizon'], value['seed']) == (32, 2, 128, 7),
        'Original target32/batch2/h128/seed7 only')
    rates, phase = rate_options(**{k: value[k] for k in ('lr', 'd_lr_mult', 'prior_lr_mult', 'ema_decay')})
    require(value['phase'] == phase, 'Rate tuple and original declared phase differ')
    if value['prepared_entry'] is not None:
        pin = value['prepared_entry']
        require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
            and type(pin['bytes']) is int and pin['bytes'] > 0, 'Actual Root prepared-entry FullRecord required')
    return dict(value, **rates)


def training_request(value):
    """Build a native-profile request for allocation-free checking."""
    return dict(schema='particle-slider-train-v1', profile=PROFILE, training=validate_training(value),
                execution=dict(runner=RUNNER), output=dict(checkpoint_every=1))


def main(argv=None):
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--output', type=Path)
    q.add_argument('--checkpoint-every', type=int, default=1)
    q.add_argument('--check', action='store_true')
    args = q.parse_args(argv)
    value = json.loads(args.config.read_text())
    if value.get('schema') == 'particle-slider-train-v1':
        value = value['training']
    value = validate_training(value)
    require(args.checkpoint_every == 1, 'Original per-committed-update FULL durability is fixed')
    require(args.check, f'Archived bank execution requires checkout {ARCHIVED_REVISION}.')
    print(json.dumps(dict(status='checked', profile=PROFILE, phase=value['phase'], rate_overrides={k: value[k]
        for k in ('lr', 'd_lr_mult', 'prior_lr_mult', 'ema_decay')}, target_steps=32, horizon=128,
        model_loaded=False, native_execution_started=False, Root_prepared_receipt_bound=value['prepared_entry'] is not None,
        required_actual_runner=RUNNER, generic_prompt_training_or_media_render_supported=False), indent=2))
    return 0


def product_cli(argv):
    """Add explicit bank commands without changing the historical image planner."""
    operation, *rest = argv
    if operation == 'bank-config':
        q = argparse.ArgumentParser(description='Create an original fixed-bank common request; no work is launched')
        for name in ('lr', 'd-lr-mult', 'prior-lr-mult'):
            q.add_argument('--' + name, type=float, required=True)
        q.add_argument('--prepared-entry', type=Path)
        q.add_argument('--prepared-entry-sha256')
        args = q.parse_args(rest)
        pin = None
        if args.prepared_entry is not None:
            unused, pin = read_json(args.prepared_entry.absolute())
            require(pin['sha256'] == args.prepared_entry_sha256, 'Explicit actual Root receipt SHA required')
        else:
            require(args.prepared_entry_sha256 is None, 'Prepared-entry path and SHA go together')
        print(json.dumps(training_request(config(lr=args.lr, d_lr_mult=args.d_lr_mult,
            prior_lr_mult=args.prior_lr_mult, ema_decay=None, prepared_entry=pin)), indent=2))
        return 0
    q = argparse.ArgumentParser(description='Check or dispatch the original fixed-bank product method')
    q.add_argument('--config', type=Path, required=True)
    args = q.parse_args(rest)
    if operation == 'bank-check':
        return main(['--config', str(args.config), '--checkpoint-every', '1', '--check'])
    require(operation == 'bank-train', 'Unknown fixed bank command')
    raise ValueError(f'Restore the original checkout {ARCHIVED_REVISION} to dispatch archived bank jobs.')


if __name__ == '__main__':
    raise SystemExit(main())
