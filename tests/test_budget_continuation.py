"""Control-only continuation must preserve original native state and identity."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('extended_worker', ROOT / 'tools/train_concept.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def prior(tmp_path, monkeypatch):
    monkeypatch.delenv('PH3_PHYSICAL_GPU', raising=False)
    root = tmp_path / 'old-native'
    value = worker.recipe(ROOT / 'wild-hair.json')
    value['training']['steps'] = 128
    value['device'] = 'cuda:0'
    runtime = {'runtime_identity': {'packages': {'engine': 'unchanged'}}}
    chosen = worker.placement(value)
    old = worker.request_identity(value, runtime, chosen, 64)
    old['source']['tools/train_concept.py'] = worker.BASE_CONTROL_SHA256
    write(root / 'native-request.json', old)
    plan, modern = worker.plans(value)
    write(root / 'PLAN.json', plan)
    modern['collection_plan_sha256'] = worker.sha(root / 'PLAN.json')
    write(root / 'MODERN-TRAINING-PLAN.json', modern)
    identity = dict(driver_sha256=old['source']['tools/modern_native.py'],
        training_plan_sha256=worker.sha(root / 'MODERN-TRAINING-PLAN.json'), training_plan=modern,
        compute_placement=dict(chosen, cuda_name='NVIDIA RTX A6000', compute_capability=[8, 6]))
    write(root / 'conditioning/manifest.json', {})
    write(root / 'caches/STAGED_WEIGHTS.json', {})
    identity['teachers'], identity['caches'] = {}, {}
    for split in ('train', 'development'):
        path = root / 'caches' / split
        path.mkdir(parents=True)
        (path / 'responses.safetensors').write_bytes(b'cached responses')
        teacher = dict(split=split, staged_weights_sha256=worker.sha(root / 'caches/STAGED_WEIGHTS.json'),
            plan_sha256=worker.sha(root / 'PLAN.json'),
            encoded_conditioning_sha256=worker.sha(root / 'conditioning/manifest.json'))
        fingerprint = 'fixture-' + split
        write(path / 'manifest.json', dict(identity=teacher, fingerprint=fingerprint,
              tensor_file_sha256=worker.sha(path / 'responses.safetensors')))
        identity['teachers'][split] = dict(teacher_collection=teacher)
        identity['caches'][split] = fingerprint
    write(root / 'qualification/QUALIFICATION.json', {'status': 'passed', 'identity': identity})
    signature = dict(identity=identity, paired_noise_horizon=1600,
        config={'training': {'steps': 1600}}, resolved_recipe={'total_steps': 1600},
        initial_adapter_sha256='adapter', initial_critic_sha256='critic')
    write(root / 'training/TRIAL.json', {'signature': signature,
        'qualification_sha256': worker.sha(root / 'qualification/QUALIFICATION.json')})
    checkpoint = root / 'training/checkpoints/step-000128'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'state.pt').write_bytes(b'opaque full-state checkpoint fixture')
    write(checkpoint / 'manifest.json', {'completed_steps': 128, 'signature': signature,
        'weights_sha256': worker.sha(checkpoint / 'state.pt')})
    write(root / 'training/STATE.json', {'status': 'budget_complete', 'completed_steps': 128})
    write(root / 'INITIAL-STATE-seed-17.json', {'adapter_sha256': 'adapter', 'critic_sha256': 'critic',
        'seed': 17, 'identity': dict(identity, native_qualification_sha256=worker.sha(root / 'qualification/QUALIFICATION.json'))})
    current_value = deepcopy(value)
    current_value['training']['steps'] = 1600
    current = worker.request_identity(current_value, runtime, chosen, 64)
    return root, current


def test_extension_copies_verified_state_without_mutating_original(prior, tmp_path):
    root, requested = prior
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    continuation = worker.validate_continuation(root, requested)
    assert continuation['restored_steps'] == 128
    assert continuation['target_steps'] == continuation['schedule_horizon'] == 1600
    delivery = tmp_path / 'new-native'
    worker.seed_continuation(delivery, continuation)
    assert (delivery / 'training/checkpoints/step-000128/state.pt').read_bytes() == before[root / 'training/checkpoints/step-000128/state.pt']
    assert json.loads((delivery / 'native-request.json').read_text())['recipe']['training']['steps'] == 128
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (delivery / 'training/STATE.json').exists()


@pytest.mark.parametrize('change', ['lr', 'prompts', 'device', 'runtime', 'native_source', 'unknown_control'])
def test_immutable_identity_drift_is_rejected(prior, change):
    root, requested = prior
    if change == 'lr': requested['recipe']['training']['generator_lr'] = .00006
    elif change == 'prompts': requested['recipe']['concept']['positive'] = 'changed'
    elif change == 'device': requested['compute_placement']['physical_index'] = 1
    elif change == 'runtime': requested['runtime_identity'] = {'changed': True}
    elif change == 'native_source': requested['source']['tools/modern_native.py'] = '0' * 64
    else:
        old = json.loads((root / 'native-request.json').read_text())
        old['source']['tools/train_concept.py'] = '0' * 64
        write(root / 'native-request.json', old)
    with pytest.raises(ValueError):
        worker.validate_continuation(root, requested)


def test_tampered_checkpoint_and_plan_rejected(prior):
    root, requested = prior
    state = root / 'training/checkpoints/step-000128/state.pt'
    original = state.read_bytes()
    state.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        worker.validate_continuation(root, requested)
    state.write_bytes(original)
    plan = json.loads((root / 'PLAN.json').read_text())
    plan['sigmas'] = [1, 0]
    write(root / 'PLAN.json', plan)
    with pytest.raises(ValueError, match='plan'):
        worker.validate_continuation(root, requested)


def test_old_completed_target_does_not_skip_extended_training(prior):
    root, _ = prior
    marker = root / 'training/STATE.json'
    assert worker.stage_complete('train', marker, 128)
    assert not worker.stage_complete('train', marker, 1600)


def test_target_bound_by_original_schedule(prior):
    root, requested = prior
    requested['recipe']['training']['steps'] = 1601
    with pytest.raises(ValueError, match='schedule horizon'):
        worker.validate_continuation(root, requested)
    requested['recipe']['training']['steps'] = 64
    with pytest.raises(ValueError, match='decrease'):
        worker.validate_continuation(root, requested)


def test_teacher_cache_payload_tamper_rejected(prior):
    root, requested = prior
    (root / 'caches/train/responses.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='teacher cache'):
        worker.validate_continuation(root, requested)


def test_control_runner_uses_old_context_new_trial_and_no_recollection(prior, tmp_path, monkeypatch):
    root, requested = prior
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    config = tmp_path / 'extended.json'
    write(config, requested['recipe'])
    delivery = tmp_path / 'delivery'
    calls = []
    monkeypatch.setattr(worker, 'runtime_preflight', lambda device: {'runtime_identity': requested['runtime_identity']})
    monkeypatch.setattr(worker, 'wait_gpu', lambda *args: None)
    def child(command, **kwargs):
        calls.append(command)
        assert command[2:4] == ['tools/modern_native.py', 'trial']
        assert command[command.index('--plan') + 1] == str(root / 'PLAN.json')
        assert command[command.index('--qualification') + 1] == str(root / 'qualification')
        assert command[command.index('--steps') + 1] == '1600'
        output = Path(command[command.index('--output') + 1])
        assert output == delivery / 'training'
        cp = output / 'checkpoints/step-001600'
        cp.mkdir()
        (cp / 'state.pt').write_bytes(b'new test state')
        write(cp / 'manifest.json', {'completed_steps': 1600, 'weights_sha256': worker.sha(cp / 'state.pt')})
        adapter = output / 'exports/step-001600-ema'
        adapter.mkdir(parents=True)
        (adapter / 'adapter.safetensors').write_bytes(b'new test adapter')
        write(adapter / 'manifest.json', {'completed_steps': 1600, 'weights_sha256': worker.sha(adapter / 'adapter.safetensors')})
        write(output / 'STATE.json', {'status': 'budget_complete', 'completed_steps': 1600})
        (output / 'training.jsonl').write_text('{"completed_steps":1600}\n')
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(worker.subprocess, 'Popen', child)
    worker.run(SimpleNamespace(config=config, resume=root, output=delivery, checkpoint_every=64,
                              check=False, gpu_wait_seconds=0))
    assert len(calls) == 1
    assert json.loads((delivery / 'result.json').read_text())['completed_steps'] == 1600
    assert all(p.read_bytes() == content for p, content in before.items())
