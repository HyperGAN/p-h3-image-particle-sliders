"""Modern collection owns its actual runtime; classic identity stays explicit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ph3_sliders import modern_identity, native_cli, native_runtime


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(native_runtime, 'verify_diffusers_source', lambda path: {'pinned': True})
    monkeypatch.setattr(native_runtime, 'version', lambda name: calls.append(name) or 'test-version')
    monkeypatch.setattr(modern_identity, 'modern_engine_identity', lambda: {'engine': 'verified', 'hndl': 'verified', 'bridges': {}})
    return calls


def test_modern_collection_records_actual_engine_without_anima_distribution(runtime):
    identity = native_runtime.runtime_identity('source.json', quantization='encoder_bf16', cohort=native_runtime.MODERN_COHORT)
    assert identity['runtime_cohort'] == native_runtime.MODERN_COHORT
    assert 'particle-sliders-anima-gmix' not in runtime
    assert {'hypergan', 'hndl', 'ph3-hypergan-develop', 'p-h3-particle-sliders-hndl'} <= set(runtime)
    assert identity['modern_runtime']['engine'] == 'verified'


def test_default_classic_collection_preserves_anima_requirement(runtime):
    identity = native_runtime.runtime_identity('source.json', quantization='encoder_bf16')
    assert 'particle-sliders-anima-gmix' in runtime
    assert 'runtime_cohort' not in identity
    assert 'modern_runtime' not in identity


def test_modern_verifier_rejects_classic_cache_without_rebinding():
    with pytest.raises(ValueError, match='classic state cannot be rebound'):
        modern_identity.verify_modern_teacher_identity({}, source_manifest='source', quantization={},
            staged_weights_sha256='weights', plan_sha256='plan', encoded_conditioning_sha256='prompts')


@pytest.mark.parametrize('field', ['modern_runtime', 'packages', 'product_sources', 'quantization',
                                 'staged_weights_sha256', 'plan_sha256', 'encoded_conditioning_sha256'])
def test_modern_verifier_checks_complete_current_identity(runtime, field):
    args = dict(source_manifest='source.json', quantization={'frozen_base': 'int8'},
        staged_weights_sha256='weights', plan_sha256='plan', encoded_conditioning_sha256='prompts')
    previous = dict(native_runtime.runtime_identity(args['source_manifest'], quantization=args['quantization'],
                    cohort=native_runtime.MODERN_COHORT),
                    **{k: args[k] for k in ('staged_weights_sha256', 'plan_sha256', 'encoded_conditioning_sha256')})
    assert modern_identity.verify_modern_teacher_identity(previous, **args)['teacher_collection'] == previous
    changed = deepcopy(previous)
    changed[field] = 'tampered'
    with pytest.raises(ValueError, match=field):
        modern_identity.verify_modern_teacher_identity(changed, **args)


@pytest.mark.parametrize('phase', ['encode', 'collect'])
def test_missing_runtime_metadata_fails_before_model_or_cuda_allocation(tmp_path, monkeypatch, phase):
    import json
    args = SimpleNamespace(output=tmp_path / 'output', plan=tmp_path / 'plan.json',
                           model_directory=tmp_path, inventory=tmp_path / 'inventory.json',
                           source_manifest=tmp_path / 'source.json')
    args.plan.write_text(json.dumps({'runtime_cohort': native_runtime.MODERN_COHORT, 'numerical_cohort': {}}))
    monkeypatch.setattr(native_cli, 'verify_staged', lambda *a, **kw: {})
    monkeypatch.setattr(native_runtime, 'configure_numeric_runtime', lambda: None)
    def unavailable(*a, **kw):
        raise RuntimeError('required runtime metadata absent')
    def forbidden(*a, **kw):
        pytest.fail('model or CUDA allocator reached before runtime identity validation')
    monkeypatch.setattr(native_runtime, 'runtime_identity', unavailable)
    monkeypatch.setattr(native_runtime, 'encode_prompts', forbidden)
    monkeypatch.setattr(native_runtime, 'load_transformer', forbidden)
    monkeypatch.setattr(native_runtime.torch.cuda, 'set_per_process_memory_fraction', forbidden)
    with pytest.raises(RuntimeError, match='required runtime metadata'):
        native_cli.encode(args) if phase == 'encode' else native_cli.collect(args, lease=None)
