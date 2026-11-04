"""Train a native P-H3 concept using the qualified, isolated local runtime.

The outer CLI owns placement; this worker owns native prompts, collection,
qualification, optimizer state and exports. It never samples or searches LRs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path('/ml2/particlesliders-working')
PYTHON = Path('/ml2/codex-storage/future-workspaces/2026-10-31/task-14/.venv-qwen-native/bin/python')
ARCHIVE = WORKSPACE / 'ph3-native-runtime/20261104-pinned-modern'
GPU = 'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
MODEL = Path('/mnt/nvme/sliders/models/PrunaAI--P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df')
INVENTORY = WORKSPACE / 'ph3-native-source/HF_FILE_INVENTORY.json'
SOURCE = WORKSPACE / 'ph3-native-source/35087a644038eb531f48581708d4763044cb4852/SOURCE_MANIFEST.json'


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


def environment():
    roots = [WORKSPACE / 'ph3-native-deps', ARCHIVE / 'engine-runtime',
             ARCHIVE / 'hypergan-bridge', ARCHIVE / 'hndl-bridge',
             WORKSPACE / 'ph3-diffusers/src', ROOT / 'src']
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)),
                CUDA_VISIBLE_DEVICES=GPU, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')


def recipe(path):
    value = json.loads(path.read_text())
    concept, training = value['concept'], value['training']
    if value.get('device', 'cuda:1') != 'cuda:1':
        raise ValueError('this locally qualified profile supports physical GPU1 (cuda:1) only')
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


def wait_gpu(output, needed, timeout):
    start = time.monotonic()
    while True:
        text = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        available = int(text.strip())
        locks = []
        admitted = False
        try:
            for name, shared in (('qualification/admission-gpu-1.lock', True), ('locks/gpu-1.lock', True), ('locks/gpu-1-ph3-training.lock', False)):
                path = Path('/mnt/nvme/sliders') / name
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
        write(output / 'progress.json', dict(status='queued', phase='waiting_for_gpu', gpu_uuid=GPU,
              free_mib=available, required_free_mib=needed, elapsed_seconds=round(time.monotonic()-start)))
        if time.monotonic()-start >= timeout:
            raise TimeoutError('GPU1 unavailable; other workloads preserved, no training started')
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


def run(args):
    value = recipe(args.config)
    plan, modern = plans(value)
    for path in (PYTHON, MODEL, INVENTORY, SOURCE, ARCHIVE / 'engine-runtime', ARCHIVE / 'hypergan-bridge', ARCHIVE / 'hndl-bridge'):
        if not path.exists():
            raise FileNotFoundError('required native runtime path is absent: ' + str(path))
    if args.check:
        print(json.dumps(dict(status='validated', product='p-h3-image-particle-sliders', gpu_uuid=GPU,
            steps=value['training']['steps'], phases=['encode', 'collect', 'qualify', 'train', 'export'],
            inference=False, lr_search=False, estimated_train_seconds=16 * value['training']['steps'])))
        return
    delivery = args.output.resolve()
    output = args.resume.resolve() if args.resume else delivery
    if args.resume and not (output / 'native-request.json').is_file():
        raise ValueError('resume must reference an existing native run with native-request.json')
    output.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    if delivery != output:
        write(delivery / 'resume-ref.json', dict(native_run=str(output)))
    identity = dict(recipe=value, source={str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / 'tools/train_concept.py', ROOT / 'tools/modern_native.py', ROOT / 'native-collection-template.json')}, checkpoint_every=args.checkpoint_every)
    record = output / 'native-request.json'
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError('run recipe/source/checkpoint interval changed; use a fresh output')
    write(record, identity)
    write(output / 'PLAN.json', plan)
    modern['collection_plan_sha256'] = sha(output / 'PLAN.json')
    write(output / 'MODERN-TRAINING-PLAN.json', modern)
    common = ['--model-directory', str(MODEL), '--inventory', str(INVENTORY), '--source-manifest', str(SOURCE), '--plan', str(output / 'PLAN.json')]
    env = environment()
    def stage(name, command, marker, needed=40500):
        if marker.exists() and (name != 'train' or json.loads(marker.read_text()).get('status') == 'budget_complete'):
            return
        wait_gpu(delivery, needed, args.gpu_wait_seconds)
        write(delivery / 'progress.json', dict(status='running', phase=name))
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
    stage('encode', ['-m', 'ph3_sliders.native_cli', 'encode', *common, '--output', str(output / 'conditioning')], output / 'conditioning/manifest.json', 14000)
    stage('collect', ['-m', 'ph3_sliders.native_cli', 'collect', *common, '--conditioning', str(output / 'conditioning'), '--splits', 'train', 'development', '--output', str(output / 'caches')], output / 'caches/COLLECTION_RESULT.json')
    native = [*common, '--training-plan', str(output / 'MODERN-TRAINING-PLAN.json'), '--conditioning', str(output / 'conditioning'), '--cache-root', str(output / 'caches')]
    rates = ['--rate', str(value['training']['generator_lr']), '--seed', str(value['training'].get('seed', 17))]
    stage('qualify', ['tools/modern_native.py', 'qualify', *native, *rates, '--output', str(output / 'qualification')], output / 'qualification/QUALIFICATION.json')
    stage('train', ['tools/modern_native.py', 'trial', *native, *rates, '--qualification', str(output / 'qualification'), '--steps', str(value['training']['steps']), '--checkpoint-every', str(args.checkpoint_every), '--output', str(output / 'training')], output / 'training/STATE.json')
    result = verify_complete(output, value['training']['steps'])
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
