"""Train a native P-H3 concept using the qualified, isolated local runtime.

The outer CLI owns placement; this worker owns native prompts, collection,
qualification, optimizer state and exports. It never samples or searches LRs.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path('/ml2/particlesliders-working')
PYTHON = Path('/ml2/codex-storage/future-workspaces/2026-10-31/task-14/.venv-qwen-native/bin/python')
ARCHIVE = WORKSPACE / 'ph3-native-runtime/20261104-pinned-modern'
GPU = 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'  # Existing GPU1 default
MODEL = Path('/mnt/nvme/sliders/models/PrunaAI--P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df')
INVENTORY = WORKSPACE / 'ph3-native-source/HF_FILE_INVENTORY.json'
SOURCE = WORKSPACE / 'ph3-native-source/35087a644038eb531f48581708d4763044cb4852/SOURCE_MANIFEST.json'
BASE_CONTROL_COMMIT = '3a5120fe7ac724ace3a83c2b7533471f6ff7d240'
BASE_CONTROL_SHA256 = 'b660b97d1b30a4e878afe8555583491c9c2f782c9caf274ddabdb58244fc2bf1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_death_guard(parent_pid):
    """Bind a native child to this exact controller, including forced kill."""
    def configure():
        import ctypes
        if not sys.platform.startswith('linux'):
            raise RuntimeError('this native process-lifecycle profile requires Linux')
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
            raise OSError(ctypes.get_errno(), 'cannot set parent-death SIGKILL')
        if os.getppid() != parent_pid:
            os._exit(125)
    return configure


def gpu_policy():
    # Load only the standard-library placement helper, not native product deps.
    spec = importlib.util.spec_from_file_location('ph3_gpu_placement', ROOT / 'src/ph3_sliders/gpu_lease.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def placement(value=None):
    return gpu_policy().resolve_gpu(None if value is None else value.get('device'))


def environment(device=None):
    roots = [WORKSPACE / 'ph3-native-deps', ARCHIVE / 'engine-runtime',
             ARCHIVE / 'hypergan-bridge', ARCHIVE / 'hndl-bridge',
             WORKSPACE / 'ph3-diffusers/src', ROOT / 'src']
    chosen = gpu_policy().resolve_gpu(device)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)),
                CUDA_VISIBLE_DEVICES=chosen['uuid'], PH3_PHYSICAL_GPU=str(chosen['physical_index']), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')


def recipe(path):
    value = json.loads(path.read_text())
    concept, training = value['concept'], value['training']
    chosen = placement(value)
    value['device'] = f"cuda:{chosen['physical_index']}"
    for name in ('name', 'neutral', 'positive'):
        if not isinstance(concept[name], str) or not concept[name].strip():
            raise ValueError('concept.' + name + ' must be nonempty text')
    if concept['neutral'] == concept['positive']:
        raise ValueError('neutral and positive hair descriptions must differ')
    steps = training['steps']
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1600:
        raise ValueError('training.steps must be an integer in 1..1600')
    rate = float(training['generator_lr'])
    if not 0 < rate < .01:
        raise ValueError('generator_lr must be positive and below .01')
    if training.get('critic_lr', .0009) != .0009 or training.get('particle_lr', .006) != .006:
        raise ValueError('this qualified profile fixes critic_lr=.0009 and particle_lr=.006')
    if training.get('resolution', 2048) != 2048:
        raise ValueError('this profile requires the qualified 2048px native resolution')
    return value


def plans(value):
    # Keep check/allocation-free dispatch independent of Torch and Diffusers.
    plan = json.loads((ROOT / 'native-collection-template.json').read_text())
    concept = value['concept']
    plan['compute_placement'] = placement(value)
    plan['runtime_cohort'] = 'ph3-modern-hypergan-hndl-v1'
    plan['concept'] = concept['name']
    plan['conditioning']['neutral_suffix'] = concept['neutral']
    plan['conditioning']['positive_suffix'] = concept['positive']
    plan['splits'] = value.get('splits') or {
        'train': {'seeds': [17], 'scenes': [
            {'id': 'woman', 'text': 'A studio waist-up portrait of a 29-year-old woman with a crooked grin, one eyebrow raised, a black tank top, and small silver earrings. Her entire head is visible with space above it against a plain grey background. Soft side lighting.'},
            {'id': 'man', 'text': 'A studio waist-up portrait of a 34-year-old man with an amused sideways glance, a plain black T-shirt, and a silver ear stud. His entire head is visible with space above it against a plain grey background. Soft side lighting.'}]},
        'development': {'seeds': [171], 'scenes': [
            {'id': 'woman_dev', 'text': 'A studio waist-up portrait of a 29-year-old woman leaning forward with a delighted, toothy grin, a denim jacket, and small hoop earrings. Her entire head is visible with space above it against a muted blue background. Soft side lighting.'}]}}
    if not all(plan['splits'].get(split, {}).get('scenes') and plan['splits'][split].get('seeds') for split in ('train', 'development')):
        raise ValueError('train and development must contain scenes and seeds')
    names = [scene['id'] for split in plan['splits'].values() for scene in split['scenes']]
    if len(names) != len(set(names)):
        raise ValueError('scene IDs must be disjoint across splits')
    training = value['training']
    modern = dict(schema='ph3-hypergan-develop-hndl-training-plan-v1', concept=concept['name'],
        batch_size=1, schedule_horizon=1600, critic_rate=.0009, particle_rate=.006,
        generator_rates=[training['generator_lr']], initial_seed=training.get('seed', 17),
        requested_steps=training['steps'], rate_status='explicit starting rate; no LR search or quality claim')
    plan['training'] = dict(generator_rates=[training['generator_lr']], critic_rate=.0009,
        particle_rate=.006, initial_seed=training.get('seed', 17), batch=1, schedule_horizon=1600,
        requested_steps=training['steps'], inference_selection='ema', auxiliary_losses=[])
    plan['selection'] = dict(metric='matched_example_teacher_student_relative_l2_v2',
        purpose='diagnostic evaluation of a single fixed-rate exploratory run; no LR selection',
        coordinate_system='full_packed_image_then_audio_velocity_unwhitened', complete_example_position_grid=True)
    return plan, modern


def wait_gpu(output, needed, timeout, device=None):
    chosen = gpu_policy().resolve_gpu(device)
    start = time.monotonic()
    while True:
        text = subprocess.check_output(['nvidia-smi', '-i', chosen['uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        available = int(text.strip())
        locks = []
        admitted = False
        try:
            for path, shared in gpu_policy().lock_paths(device):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open('a+')
                locks.append(handle)
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            admitted = available >= needed
        except BlockingIOError:
            pass
        finally:
            for handle in reversed(locks):
                handle.close()
        if admitted:
            return
        write(output / 'progress.json', dict(status='queued', phase='waiting_for_gpu', gpu_uuid=chosen['uuid'], physical_gpu=chosen['physical_index'],
              free_mib=available, required_free_mib=needed, elapsed_seconds=round(time.monotonic()-start)))
        if time.monotonic()-start >= timeout:
            raise TimeoutError(f"GPU{chosen['physical_index']} unavailable; other workloads preserved, no training started")
        time.sleep(15)


def verify_complete(output, target):
    state = json.loads((output / 'training/STATE.json').read_text())
    if state.get('status') != 'budget_complete' or state['completed_steps'] != target:
        raise RuntimeError('native trainer did not complete requested steps')
    checkpoint = output / f'training/checkpoints/step-{target:06d}'
    exported = output / f'training/exports/step-{target:06d}-ema'
    for directory, filename in ((checkpoint, 'state.pt'), (exported, 'adapter.safetensors')):
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['completed_steps'] != target or sha(directory / filename) != manifest['weights_sha256']:
            raise ValueError('final checkpoint/export step or bytes differ: ' + str(directory))
    return dict(status='completed', completed_steps=target, checkpoint=str(checkpoint),
        adapter=str(exported), lr_search=False, inference=False,
        quality_status='exploratory; no LR selection or visual acceptance claimed')


def runtime_preflight(device):
    env = environment(device)
    env['CUDA_VISIBLE_DEVICES'] = ''
    completed = subprocess.run([str(PYTHON), str(ROOT / 'tools/runtime_preflight.py'),
        '--source-manifest', str(SOURCE)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    if result.get('status') != 'verified' or result.get('cuda_initialized') is not False:
        raise RuntimeError('isolated modern runtime preflight did not verify a CPU-only identity')
    return result


def request_identity(value, runtime, chosen, checkpoint_every):
    return dict(recipe=value, source={str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / 'tools/train_concept.py', ROOT / 'tools/modern_native.py', ROOT / 'tools/runtime_preflight.py',
        ROOT / 'src/ph3_sliders/gpu_lease.py', ROOT / 'native-collection-template.json')},
        runtime_identity=runtime['runtime_identity'], compute_placement=chosen,
        checkpoint_every=checkpoint_every)


def validate_continuation(source, requested):
    """Authorize only a larger stop budget; preserve the original numerical cohort."""
    source = source.resolve()
    previous = json.loads((source / 'native-request.json').read_text())
    context = source
    ref = source / 'resume-ref.json'
    if ref.is_file():
        reference = json.loads(ref.read_text())
        context = Path(reference['context_root']).resolve()
        if sha(context / 'native-request.json') != reference['original_request_sha256']:
            raise ValueError('original continuation context request changed')
    if json.loads((context / 'native-request.json').read_text()) != previous:
        raise ValueError('continuation must retain the immutable original native request')
    old_recipe, new_recipe = json.loads(json.dumps(previous['recipe'])), json.loads(json.dumps(requested['recipe']))
    old_target = old_recipe['training'].pop('steps')
    target = new_recipe['training'].pop('steps')
    if old_recipe != new_recipe:
        raise ValueError('resume may change only training.steps, not the recipe')
    prior_continuation = source / 'continuation.json'
    prior_target = (json.loads(prior_continuation.read_text())['target_steps']
                    if prior_continuation.is_file() else old_target)
    if target < prior_target:
        raise ValueError('continuation cannot decrease the requested training budget')
    for key in ('runtime_identity', 'compute_placement', 'checkpoint_every'):
        if previous[key] != requested[key]:
            raise ValueError('continuation changed immutable ' + key)
    if previous['source'].keys() != requested['source'].keys():
        raise ValueError('continuation source closure differs')
    for name, checksum in previous['source'].items():
        if name == 'tools/train_concept.py':
            if checksum not in (requested['source'][name], BASE_CONTROL_SHA256):
                raise ValueError('unreviewed control-driver migration')
        elif checksum != requested['source'][name]:
            raise ValueError('continuation changed numerical/runtime source: ' + name)
    plan_path, modern_path = context / 'PLAN.json', context / 'MODERN-TRAINING-PLAN.json'
    plan, modern = json.loads(plan_path.read_text()), json.loads(modern_path.read_text())
    expected_plan, expected_modern = plans(previous['recipe'])
    expected_modern['collection_plan_sha256'] = sha(plan_path)
    if plan != expected_plan or modern != expected_modern:
        raise ValueError('original collection/training plan changed')
    horizon = modern['schedule_horizon']
    if target > horizon:
        raise ValueError('continuation exceeds the original fixed schedule horizon')
    qualification = context / 'qualification/QUALIFICATION.json'
    qualified = json.loads(qualification.read_text())
    if qualified.get('status') != 'passed':
        raise ValueError('continuation requires a passed original native qualification')
    identity = qualified['identity']
    qualified_gpu = identity['compute_placement']
    if (identity['driver_sha256'] != requested['source']['tools/modern_native.py']
            or identity['training_plan_sha256'] != sha(modern_path)
            or identity['training_plan'] != modern
            or set(qualified_gpu) - (set(requested['compute_placement']) | {'cuda_name', 'compute_capability'})
            or any(qualified_gpu.get(key) != value for key, value in requested['compute_placement'].items())):
        raise ValueError('qualification no longer binds the exact native driver, plan or GPU')
    for split in ('train', 'development'):
        cached = json.loads((context / 'caches' / split / 'manifest.json').read_text())
        teacher = identity['teachers'][split]['teacher_collection']
        if (cached['identity'] != teacher or cached['fingerprint'] != identity['caches'][split]
                or sha(context / 'caches' / split / 'responses.safetensors') != cached['tensor_file_sha256']
                or teacher['staged_weights_sha256'] != sha(context / 'caches/STAGED_WEIGHTS.json')
                or teacher['plan_sha256'] != sha(plan_path)
                or teacher['encoded_conditioning_sha256'] != sha(context / 'conditioning/manifest.json')):
            raise ValueError('qualified teacher cache or conditioning changed: ' + split)
    trial_path = source / 'training/TRIAL.json'
    trial = json.loads(trial_path.read_text())
    if trial['qualification_sha256'] != sha(qualification):
        raise ValueError('training trial qualification bytes changed')
    signature = trial['signature']
    if (any(signature['identity'].get(key) != value for key, value in identity.items())
            or signature['paired_noise_horizon'] != horizon
            or signature['config']['training']['steps'] != horizon
            or signature['resolved_recipe']['total_steps'] != horizon):
        raise ValueError('trial signature differs from original qualification or schedule')
    checkpoints = sorted(path for path in (source / 'training/checkpoints').glob('step-*')
                         if path.name[5:].isdigit() and (path / 'manifest.json').is_file())
    if not checkpoints:
        raise ValueError('continuation requires a complete full-state checkpoint')
    checkpoint = max(checkpoints, key=lambda path: int(path.name[5:]))
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    restored = manifest['completed_steps']
    if (restored != int(checkpoint.name[5:]) or manifest['signature'] != signature
            or sha(checkpoint / 'state.pt') != manifest['weights_sha256']):
        raise ValueError('full-state checkpoint signature, step or bytes changed')
    if restored > target:
        raise ValueError('checkpoint is beyond the requested target')
    initial_path = source / f'INITIAL-STATE-seed-{previous["recipe"]["training"].get("seed", 17)}.json'
    initial = json.loads(initial_path.read_text())
    if (initial['adapter_sha256'] != signature['initial_adapter_sha256']
            or initial['critic_sha256'] != signature['initial_critic_sha256']
            or initial.get('seed') != previous['recipe']['training'].get('seed', 17)
            or initial.get('identity') != dict(identity, native_qualification_sha256=sha(qualification))):
        raise ValueError('original initialization identity changed')
    return dict(schema='ph3-control-continuation-v1', context_root=str(context),
        source_artifacts=str(source), original_request_sha256=sha(context / 'native-request.json'),
        original_target_steps=old_target, prior_target_steps=prior_target, target_steps=target,
        schedule_horizon=horizon, restored_steps=restored, checkpoint=str(checkpoint),
        checkpoint_sha256=manifest['weights_sha256'], checkpoint_manifest_sha256=sha(checkpoint / 'manifest.json'),
        trial_sha256=sha(trial_path), initial_state_sha256=sha(initial_path),
        control_driver_migration=dict(from_sha256=previous['source']['tools/train_concept.py'],
            to_sha256=requested['source']['tools/train_concept.py'], reviewed_base_commit=BASE_CONTROL_COMMIT,
            scope='control-only budget extension; numerical driver/runtime unchanged'),
        immutable_files={name: sha(context / name) for name in ('PLAN.json', 'MODERN-TRAINING-PLAN.json',
            'qualification/QUALIFICATION.json', 'conditioning/manifest.json',
            'caches/STAGED_WEIGHTS.json', 'caches/train/manifest.json', 'caches/development/manifest.json')})


def stage_complete(name, marker, target):
    if not marker.is_file():
        return False
    if name != 'train':
        return True
    state = json.loads(marker.read_text())
    return state.get('status') == 'budget_complete' and state.get('completed_steps') == target


def seed_continuation(delivery, continuation):
    """Copy only verified native state; never write the prior run or its context."""
    source = Path(continuation['source_artifacts'])
    checkpoint = Path(continuation['checkpoint'])
    delivery.mkdir(parents=True, exist_ok=True)
    trial = delivery / 'training'
    trial.mkdir()
    target = trial / 'checkpoints' / checkpoint.name
    target.parent.mkdir()
    shutil.copytree(checkpoint, target)
    if (sha(target / 'state.pt') != continuation['checkpoint_sha256']
            or sha(target / 'manifest.json') != continuation['checkpoint_manifest_sha256']):
        raise ValueError('copied checkpoint bytes differ')
    shutil.copyfile(source / 'training/TRIAL.json', trial / 'TRIAL.json')
    if sha(trial / 'TRIAL.json') != continuation['trial_sha256']:
        raise ValueError('copied trial identity differs')
    original = json.loads((source / 'native-request.json').read_text())
    initial = f'INITIAL-STATE-seed-{original["recipe"]["training"].get("seed", 17)}.json'
    shutil.copyfile(source / initial, delivery / initial)
    if sha(delivery / initial) != continuation['initial_state_sha256']:
        raise ValueError('copied initialization identity differs')
    shutil.copyfile(source / 'native-request.json', delivery / 'native-request.json')
    if sha(delivery / 'native-request.json') != continuation['original_request_sha256']:
        raise ValueError('copied original request differs')
    write(delivery / 'resume-ref.json', dict(context_root=continuation['context_root'],
        original_request_sha256=continuation['original_request_sha256']))
    write(delivery / 'continuation.json', continuation)


def run(args):
    value = recipe(args.config)
    chosen = placement(value)
    plan, modern = plans(value)
    for path in (PYTHON, MODEL, INVENTORY, SOURCE, ARCHIVE / 'engine-runtime', ARCHIVE / 'hypergan-bridge', ARCHIVE / 'hndl-bridge'):
        if not path.exists():
            raise FileNotFoundError('required native runtime path is absent: ' + str(path))
    runtime = runtime_preflight(value['device'])
    identity = request_identity(value, runtime, chosen, args.checkpoint_every)
    continuation = validate_continuation(args.resume, identity) if args.resume else None
    if args.check:
        print(json.dumps(dict(status='validated', product='p-h3-image-particle-sliders', gpu_uuid=chosen['uuid'],
            physical_gpu=chosen['physical_index'], logical_device='cuda:0',
            qualification='original exact native qualification preserved' if continuation else 'fresh native qualification required for this run',
            steps=value['training']['steps'], phases=['train', 'export'] if continuation else ['encode', 'collect', 'qualify', 'train', 'export'],
            inference=False, lr_search=False, estimated_train_seconds=16 * value['training']['steps'],
            runtime_cohort=runtime['runtime_identity']['runtime_cohort'],
            verified_packages=runtime['runtime_identity']['packages'], cuda_initialized=False,
            continuation=continuation)))
        return
    delivery = args.output.resolve()
    output = delivery
    context = Path(continuation['context_root']) if continuation else output
    if continuation and delivery == args.resume.resolve():
        raise ValueError('continuation requires a new output directory; original run is immutable')
    output.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    write(delivery / 'runtime-preflight.json', runtime)
    if continuation:
        seed_continuation(delivery, continuation)
        write(delivery / 'progress.json', dict(status='resuming', phase='train',
            completed_steps=continuation['restored_steps'], target_steps=value['training']['steps']))
    else:
        record = output / 'native-request.json'
        if record.exists() and json.loads(record.read_text()) != identity:
            raise ValueError('run recipe/source/checkpoint interval changed; use a fresh output')
        write(record, identity)
        write(output / 'PLAN.json', plan)
        modern['collection_plan_sha256'] = sha(output / 'PLAN.json')
        write(output / 'MODERN-TRAINING-PLAN.json', modern)
    common = ['--model-directory', str(MODEL), '--inventory', str(INVENTORY), '--source-manifest', str(SOURCE), '--plan', str(context / 'PLAN.json')]
    env = environment(value['device'])
    def stage(name, command, marker, needed=40500):
        if stage_complete(name, marker, value['training']['steps']):
            return
        wait_gpu(delivery, needed, args.gpu_wait_seconds, value['device'])
        write(delivery / 'progress.json', dict(status='running', phase=name, gpu_uuid=chosen['uuid'], physical_gpu=chosen['physical_index'], logical_device='cuda:0'))
        with (output / (name + '.log')).open('a') as log:
            child = subprocess.Popen([str(PYTHON), '-u', *command], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=parent_death_guard(os.getpid()))
            try:
                while child.poll() is None:
                    metrics = output / 'training/training.jsonl'
                    if name == 'train' and metrics.exists():
                        (delivery / 'metrics.jsonl').write_bytes(metrics.read_bytes())
                    time.sleep(1)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, child.args)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        if not marker.is_file():
            raise RuntimeError(name + ' exited without its complete receipt')
    if not continuation:
        stage('encode', ['-m', 'ph3_sliders.native_cli', 'encode', *common, '--output', str(context / 'conditioning')], context / 'conditioning/manifest.json', 14000)
        stage('collect', ['-m', 'ph3_sliders.native_cli', 'collect', *common, '--conditioning', str(context / 'conditioning'), '--splits', 'train', 'development', '--output', str(context / 'caches')], context / 'caches/COLLECTION_RESULT.json')
    native = [*common, '--training-plan', str(context / 'MODERN-TRAINING-PLAN.json'), '--conditioning', str(context / 'conditioning'), '--cache-root', str(context / 'caches')]
    rates = ['--rate', str(value['training']['generator_lr']), '--seed', str(value['training'].get('seed', 17))]
    if not continuation:
        stage('qualify', ['tools/modern_native.py', 'qualify', *native, *rates, '--output', str(context / 'qualification')], context / 'qualification/QUALIFICATION.json')
    stage('train', ['tools/modern_native.py', 'trial', *native, *rates, '--qualification', str(context / 'qualification'), '--steps', str(value['training']['steps']), '--checkpoint-every', str(args.checkpoint_every), '--output', str(output / 'training')], output / 'training/STATE.json')
    result = verify_complete(output, value['training']['steps'])
    result['compute_placement'] = chosen
    result['native_qualification'] = str(context / 'qualification/QUALIFICATION.json')
    if continuation:
        result['continuation'] = continuation
    (delivery / 'metrics.jsonl').write_bytes((output / 'training/training.jsonl').read_bytes())
    write(delivery / 'result.json', result)
    write(delivery / 'progress.json', dict(status='completed', phase='export', completed_steps=result['completed_steps']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=64)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--gpu-wait-seconds', type=int, default=86400)
    args = parser.parse_args(argv)
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be positive')
    try:
        run(args)
    except BaseException as error:
        if not args.check:
            status = 'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
            write(args.output / 'result.json', dict(status=status, error=repr(error)))
            write(args.output / 'progress.json', dict(status=status, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
