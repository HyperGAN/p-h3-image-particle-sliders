"""Current014 runner plugin: replace the dispatcher with Root's exact one job.

submit() intentionally does not return a remote submission or generic local
completion. os.execvpe retains this process PID while replacing its argv/code.
Root keeps its actual outer-tool ownership, capture, poll/reap and role closure.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys

from .execution import PROFILE, RUNNER, authority, pinned, profile as product_profile, require, source, validate_training


def option(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), 'One exact prepared CLI option: ' + name)
    return argv[argv.index(name) + 1]


def prepared(training):
    value = validate_training(training)
    require(value['prepared_entry'] is not None, 'Actual execution needs an unconsumed Root-prepared one-job receipt')
    entry = pinned(value['prepared_entry'])
    a = authority()
    plan = pinned(a['scientific_plan'])
    require(entry['accepted'] is True and entry['phase'] == value['phase'], 'Same actual accepted Root phase')
    command = entry['exact_once_argv']
    require(isinstance(command, list) and all(isinstance(x, str) and x for x in command), 'Exact recorded argv required')
    env = entry['environment']
    require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), 'Recorded environment strings')
    if entry['schema'] == 'Root-PH3-declared4-LR32-actual-ENTRY-and-pending-v1':
        source(a['GPU1_one_job_driver'])
        require(command[:5] == [a['native_python'], '-I', '-B', '-u', a['GPU1_one_job_driver']['path']]
            and entry['plan'] == a['scientific_plan'] and entry['proof_binding'] == a['genuine_native_proofs']
            and entry['own_pending48_added_only'] is True and entry['actual_launch_or_tool_poll'] is False,
            'Exact original GPU1 Root one-job launcher')
        require(option(command, '--phase') == value['phase'] and int(option(command, '--attempt')) == entry['attempt']
            and option(command, '--proof-binding') == a['genuine_native_proofs']['path']
            and option(command, '--proof-binding-sha256') == a['genuine_native_proofs']['sha256'], 'Prepared GPU1 phase/attempt/proofs')
        adopted = pinned(entry['source_adoption'])
        require(adopted['accepted'] is True and adopted['source_files'].get(a['GPU1_one_job_driver']['path']) == a['GPU1_one_job_driver'],
            'Actual original GPU1 Source adoption')
        observation = pinned(entry['observation'])
        issued, boot = observation['UTC'], observation['boot_id']
        require(observation['phase'] == value['phase'] and observation['attempt'] == entry['attempt'], 'Same prepared observation')
        resume = entry['resume_latest']
        attempt = entry['attempt']
        expected = [a['native_python'], '-I', '-B', '-u', a['GPU1_one_job_driver']['path'],
            '--phase', value['phase'], '--attempt', str(attempt),
            '--source-peer', entry['source_adoption']['path'], '--source-peer-sha256', entry['source_adoption']['sha256'],
            '--root-adoption', entry['source_adoption']['path'], '--root-adoption-sha256', entry['source_adoption']['sha256'],
            '--peer-observation', entry['observation']['path'], '--peer-observation-sha256', entry['observation']['sha256'],
            '--proof-binding', a['genuine_native_proofs']['path'], '--proof-binding-sha256', a['genuine_native_proofs']['sha256']]
        if resume is not None:
            expected += ['--resume-latest', resume['path'], '--resume-latest-sha256', resume['sha256']]
        require(command == expected, 'Exact complete GPU1 Root argv; no extra command options')
    elif entry['schema'] == 'Root-PH3-GPU0-bank32-actual-preparation-v1':
        descriptor = pinned(a['GPU0_execution'])
        guardian = descriptor['execution_sources']['guardian']
        source(guardian)
        grant = pinned(entry['grant'])
        adopted = pinned(entry['Root_source_adoption'])
        require(entry['execution'] == a['GPU0_execution'] and grant['execution'] == a['GPU0_execution']
            and grant['schema'] == 'Root-PH3-GPU0-bank32-ONE-v1' and grant['accepted'] is True
            and grant['Root_source_adoption'] == entry['Root_source_adoption']
            and grant['scientific_plan'] == a['scientific_plan'] and grant['genuine_native_proofs'] == a['genuine_native_proofs']
            and entry['only_own_pending32_added'] is True and entry['actual_launch_or_poll'] is False
            and adopted['accepted'] is True and adopted['source_files'].get(guardian['path']) == guardian,
            'Exact genuine GPU0 preparation/Source')
        require(command == [a['native_python'], '-I', '-B', '-u', guardian['path'], '--grant', entry['grant']['path'],
            '--grant-sha256', entry['grant']['sha256']] and grant['phase'] == value['phase'], 'Exact prepared GPU0 guardian argv')
        issued, boot = entry['UTC'], grant['boot_id']
        resume, attempt = grant['resume_latest'], grant['attempt']
    else:
        raise ValueError('Unsupported prepared receipt; no inferred or normalized dispatch')
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(issued)).total_seconds()
    require(0 <= age <= 60 and boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Fresh same-boot Root preparation')
    require(type(attempt) is int and attempt >= 0, 'Original actual attempt number')
    output = Path(plan['attempt_root']) / value['phase'] / ('attempt-%04d' % attempt)
    durable = Path(plan['bank_root']) / value['phase'] / 'durable'
    require(not output.exists() and not (durable.parent / 'COMPLETE.json').exists(), 'Used/completed arm or attempt cannot run again')
    if entry['schema'] == 'Root-PH3-declared4-LR32-actual-ENTRY-and-pending-v1':
        launch = Path(plan['attempt_root']).parent / 'launches' / (value['phase'] + '-attempt' + str(attempt))
        require(not launch.exists(), 'Original GPU1 launch prefix was already used')
    if resume is None:
        require(not durable.exists(), 'Committed prefix needs exact latest FULL Root preparation; no restart')
    else:
        require(resume['path'] == str(durable / 'LATEST.json'), 'Original logical latest FULL path only')
        pinned(resume)  # JSON only. The original engine authenticates the complete FULL payload.
    consumed = Path(value['prepared_entry']['path'] + '.ph3-product-dispatch-once-v1.json')
    require(not consumed.exists(), 'This actual Root receipt was already consumed; no automatic retry')
    return value, command, env, consumed, output


def submit(request):
    from particle_sliders.execution.request import validate_profile, validate_resolved
    profile = validate_resolved(request)
    require(profile == validate_profile(product_profile())
        and request['execution']['runner'] == RUNNER, 'Explicit ph3-root-owned fixed native profile required')
    require(request.get('reporting') is None and not request['input_files'], 'No added reporting/input staging over original fixed bank')
    require(request['output']['checkpoint_every'] == 1 and not any(k in request['execution'] for k in ('resume', 'device', 'stop_after')),
        'Root preparation owns exact FULL resume/device/budget; no generic overrides')
    value, command, env, consumed, output = prepared(request['training'])
    event = dict(schema='ph3-product-common-dispatch-exec-consumption-v1', status='dispatch_consumed_before_exec',
        prepared_entry=value['prepared_entry'], phase=value['phase'], original_common_dispatch_argv=list(sys.argv),
        dispatch_process_pid=os.getpid(), exact_next_argv=command, recorded_environment=env,
        original_fixed_bank_output=str(output), numerical_execution_completed=False,
        note='This common CLI process is about to exec the exact Root-prepared launcher/guardian. '
             'It was not already that guardian. No submitted/completed result is synthesized.')
    raw = (json.dumps(event, indent=2, allow_nan=False) + '\n').encode()
    with consumed.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(consumed.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print('PH3_COMMON_DISPATCH_EXEC', json.dumps(dict(phase=value['phase'], exact_next_argv=command,
        receipt=str(consumed), original_common_dispatch_replaced=True)), flush=True)
    actual_env = dict(os.environ)
    actual_env.update(env)
    os.execvpe(command[0], command, actual_env)


RUNNER_API = sys.modules[__name__]
