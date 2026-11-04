"""Dependency-free public native request checks; no model/GPU substitution."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ph3_train_concept', ROOT / 'tools/train_concept.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_wild_hair_plan_keeps_matched_scenes_and_separate_split():
    request = worker.recipe(ROOT / 'wild-hair.json')
    plan, modern = worker.plans(request)
    assert modern['requested_steps'] == 128
    assert modern['generator_rates'] == [.00002]
    assert plan['conditioning']['positive_suffix'] == request['concept']['positive']
    assert plan['conditioning']['neutral_suffix'] == request['concept']['neutral']
    assert len(plan['splits']['train']['scenes']) == 2
    assert len(plan['splits']['development']['scenes']) == 1
    assert plan['training']['batch'] == 1
    assert plan['training']['requested_steps'] == 128


@pytest.mark.parametrize('change', [
    {'device': 'cuda:0'}, {'training': {'steps': 0}}, {'training': {'steps': True}},
    {'training': {'generator_lr': float('nan')}}, {'training': {'resolution': 1024}},
    {'training': {'critic_lr': .001}}, {'training': {'particle_lr': .005}}])
def test_incompatible_request_rejected_before_gpu(tmp_path, change):
    value = json.loads((ROOT / 'wild-hair.json').read_text())
    for name, field in change.items():
        if isinstance(field, dict):
            value[name].update(field)
        else:
            value[name] = field
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        worker.recipe(path)


def test_check_never_imports_native_or_torch(tmp_path, capsys):
    imported = set(sys.modules)
    worker.main(['--config', str(ROOT / 'wild-hair.json'), '--output', str(tmp_path / 'unused'), '--check'])
    assert not (tmp_path / 'unused').exists()
    assert json.loads(capsys.readouterr().out)['status'] == 'validated'
    assert not any(name.split('.')[0] in ('torch', 'diffusers', 'ph3_sliders') for name in set(sys.modules)-imported)


def test_duplicate_scene_rejected():
    value = worker.recipe(ROOT / 'wild-hair.json')
    repeated = {'seeds': [17], 'scenes': [{'id': 'same', 'text': 'portrait'}]}
    value['splits'] = {'train': repeated, 'development': repeated}
    with pytest.raises(ValueError, match='disjoint'):
        worker.plans(value)


def test_archived_native_runtime_selected_not_classic_environment():
    env = worker.environment()
    assert env['CUDA_VISIBLE_DEVICES'] == worker.GPU
    assert '20261104-pinned-modern/engine-runtime' in env['PYTHONPATH']
    assert '20261104-pinned-modern/hypergan-bridge' in env['PYTHONPATH']
    assert env['HF_HUB_OFFLINE'] == '1'


def test_busy_gpu_records_queue_and_does_not_start_process(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.subprocess, 'check_output', lambda *args, **kwargs: '1000\n')
    monkeypatch.setattr(worker.fcntl, 'flock', lambda *args: None)
    original_open = worker.Path.open
    monkeypatch.setattr(worker.Path, 'open', lambda self, *args, **kwargs:
        open('/dev/null', 'a+') if str(self).endswith('.lock') else original_open(self, *args, **kwargs))
    with pytest.raises(TimeoutError, match='other workloads preserved'):
        worker.wait_gpu(tmp_path, 40500, timeout=0)
    state = json.loads((tmp_path / 'progress.json').read_text())
    assert state['status'] == 'queued'
    assert state['required_free_mib'] == 40500


def test_final_receipt_requires_valid_hashes_and_exact_target(tmp_path):
    worker.write(tmp_path / 'training/STATE.json', {'status': 'budget_complete', 'completed_steps': 128})
    for directory, filename in ((tmp_path / 'training/checkpoints/step-000128', 'state.pt'),
                                (tmp_path / 'training/exports/step-000128-ema', 'adapter.safetensors')):
        directory.mkdir(parents=True)
        path = directory / filename
        path.write_bytes(b'unit-test-artifact')
        worker.write(directory / 'manifest.json', {'completed_steps': 128, 'weights_sha256': worker.sha(path)})
    assert worker.verify_complete(tmp_path, 128)['status'] == 'completed'
    with pytest.raises(RuntimeError, match='requested steps'):
        worker.verify_complete(tmp_path, 64)
    (tmp_path / 'training/checkpoints/step-000128/state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes differ'):
        worker.verify_complete(tmp_path, 128)


def test_unexpected_failure_records_failed_result(tmp_path, monkeypatch):
    def fail(_args):
        raise TimeoutError('owned job did not acquire GPU')
    monkeypatch.setattr(worker, 'run', fail)
    with pytest.raises(TimeoutError):
        worker.main(['--config', str(ROOT / 'wild-hair.json'), '--output', str(tmp_path)])
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'failed'
