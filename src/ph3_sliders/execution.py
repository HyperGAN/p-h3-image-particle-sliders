"""Fixed declared-bank product profile for the current common training API.

Stdlib-only planning/checking. Native execution belongs to the already admitted
Root one-job launcher/guardian, never a replacement training loop.
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
AUTHORITY = Path(__file__).with_name('native-authority.json')


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


def pinned(pin):
    require(set(pin) == {'path', 'bytes', 'sha256'}, 'Exact metadata FullRecord required')
    value, actual = read_json(pin['path'])
    require(actual == pin, 'Pinned metadata changed: ' + pin['path'])
    return value


def source(pin):
    p = Path(pin['path'])
    require(p.is_absolute() and not p.is_symlink() and p.suffix == '.py', 'Exact external Source buffer required')
    raw = p.read_bytes()
    require(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'External Source changed')
    return raw


def authority():
    value, unused = read_json(AUTHORITY.absolute())
    require(value['schema'] == 'ph3-product-fixed-native-authority-v1'
            and value['core_version'] == '0.14.0' and value['PG_version'] == '0.9.5', 'Fixed native deployment authority')
    return value


def profile():
    authority()
    return dict(schema='particle-slider-profile-v1', name=PROFILE,
        product='p-h3-image-particle-sliders', config_kind='native', status='source_only',
        reason='Product adapter proposal over the proven original four-arm RAW32 bank. Actual execution requires '
               'an unconsumed Root-prepared one-job receipt and the ph3-root-owned runner; arbitrary prompt training '
               'and generic local execution are unsupported. No new native/LR proof is requested.',
        command=['{python}', '-m', 'ph3_sliders.execution'],
        # Checks live in the newly installed frontdoor environment. Actual
        # dispatch execs the separate exact native Python from Root's receipt.
        runtime=dict(python=sys.executable, env=dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')),
        target_steps_path='steps', default_checkpoint_every=1,
        capabilities=dict(check=True, resume=False, periodic_checkpoint=True, device=False))


def rate_options(*, lr, d_lr_mult, prior_lr_mult, ema_decay=None):
    require(ema_decay is None, 'Weight EMA is disabled; ema_decay must be None')
    supplied = dict(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in supplied.values()),
            'Finite positive literal rate scalars required; booleans are unsupported')
    # Copy the exact declared literals; never derive multipliers from rounded D/P.
    plan = pinned(authority()['scientific_plan'])
    matches = [c for c in plan['cells'] if all(float(supplied[k]).hex() == c[n].hex()
        for k, n in (('lr', 'G'), ('d_lr_mult', 'd_lr_mult'), ('prior_lr_mult', 'prior_lr_mult')))]
    require(len(matches) == 1, 'Only the four original declared RAW32 rate tuples are supported')
    cell = matches[0]
    return dict(lr=cell['G'], d_lr_mult=cell['d_lr_mult'], prior_lr_mult=cell['prior_lr_mult'], ema_decay=None), 'cell' + str(plan['cells'].index(cell))


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
    """Actual current014 native-profile request; planning never submits work."""
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
    require(args.check, 'Actual bank execution requires --runner ph3-root-owned; generic local completion envelopes differ')
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
    from particle_sliders.execution.cli import main as common_main
    return common_main(['train', '--profile', PROFILE, '--runner', RUNNER, '--config', str(args.config), '--no-reporting'])


if __name__ == '__main__':
    raise SystemExit(main())
