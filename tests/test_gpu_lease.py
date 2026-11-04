"""Physical placement and ownership tests; CUDA and nvidia-smi are mocked."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ph3_sliders import gpu_lease as gpu


@pytest.fixture(autouse=True)
def isolated_gpu_environment(monkeypatch):
    monkeypatch.delenv('PH3_PHYSICAL_GPU', raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)


@pytest.mark.parametrize('index', [0, 1])
def test_physical_map_and_single_logical_binding(index, monkeypatch):
    chosen = gpu.resolve_gpu(f'cuda:{index}')
    assert chosen == dict(physical_index=index, uuid=gpu.GPU_UUIDS[index], logical_device='cuda:0')
    monkeypatch.setenv('PH3_PHYSICAL_GPU', str(index))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', chosen['uuid'])
    assert gpu.resolve_gpu() == chosen
    lease = gpu.GpuLease(required_free_mib=40500)
    properties = SimpleNamespace(uuid=chosen['uuid'][4:], name='NVIDIA RTX A6000', major=8, minor=6)
    cuda = SimpleNamespace(device_count=lambda:1, current_device=lambda:0,
                           get_device_properties=lambda device:properties)
    assert lease.validate_cuda_binding(SimpleNamespace(cuda=cuda))['physical_index'] == index


@pytest.mark.parametrize('device', ['cuda:2', 'cuda:-1', 'cuda', '0', True, 0, 'cpu'])
def test_unsupported_physical_indices_rejected(device):
    with pytest.raises(ValueError, match='cuda:0 or cuda:1'):
        gpu.resolve_gpu(device)


def test_default_preserved_and_conflicts_refused(monkeypatch):
    assert gpu.resolve_gpu()['physical_index'] == 1
    monkeypatch.setenv('PH3_PHYSICAL_GPU', '1')
    with pytest.raises(ValueError, match='conflicts'):
        gpu.resolve_gpu('cuda:0')
    monkeypatch.setenv('PH3_PHYSICAL_GPU', '2')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        gpu.resolve_gpu()


@pytest.mark.parametrize('count,current,uuid', [(2,0,gpu.GPU_UUIDS[0]), (1,1,gpu.GPU_UUIDS[0]), (1,0,gpu.GPU_UUIDS[1])])
def test_wrong_cuda_binding_rejected(count,current,uuid):
    lease = gpu.GpuLease(required_free_mib=1, device='cuda:0')
    cuda = SimpleNamespace(device_count=lambda:count,current_device=lambda:current,
        get_device_properties=lambda device:SimpleNamespace(uuid=uuid,name='fake',major=8,minor=6))
    with pytest.raises(RuntimeError):
        lease.validate_cuda_binding(SimpleNamespace(cuda=cuda))


def test_uuid_pin_required_before_lock_or_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(gpu,'lock_paths',lambda *a,**k:pytest.fail('acquired lock before UUID validation'))
    with pytest.raises(RuntimeError,match='selected physical GPU UUID'):
        with gpu.GpuLease(required_free_mib=40500,device='cuda:0'):
            pytest.fail('entered')


def setup_locks(monkeypatch,tmp_path,*,free=42000,used=7000):
    original = gpu.lock_paths
    monkeypatch.setattr(gpu,'lock_paths',lambda device:original(device,root=tmp_path))
    monkeypatch.setattr(gpu,'gpu_snapshot',lambda device:dict(gpu.resolve_gpu(device),free_mib=free,used_mib=used,total_mib=49140))
    monkeypatch.setattr(gpu.GpuLease,'validate_cuda_binding',lambda self:dict(self.placement,cuda_name='mock',compute_capability=[8,6]))


def test_correct_independent_locks_and_role_collision(tmp_path,monkeypatch):
    setup_locks(monkeypatch,tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu.GPU_UUIDS[0])
    with gpu.GpuLease(required_free_mib=40500,device='cuda:0') as first:
        assert len(first.handles)==3
        assert all('gpu-0' in handle.name for handle in first.handles)
        receipt=json.loads((tmp_path/'locks/gpu-0-ph3-training.lock').read_text())
        assert receipt['uuid']==gpu.GPU_UUIDS[0] and receipt['physical_index']==0
        duplicate=gpu.GpuLease(required_free_mib=40500,device='cuda:0')
        with pytest.raises(BlockingIOError):
            duplicate.__enter__()
        assert duplicate.handles==[]
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu.GPU_UUIDS[1])
        with gpu.GpuLease(required_free_mib=40500,device='cuda:1') as other:
            assert all('gpu-1' in handle.name for handle in other.handles)
    assert first.handles==[]


def test_insufficient_headroom_releases_every_lock(tmp_path,monkeypatch):
    setup_locks(monkeypatch,tmp_path,free=40000)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu.GPU_UUIDS[0])
    monkeypatch.setattr(gpu.GpuLease,'validate_cuda_binding',lambda self:pytest.fail('created CUDA context with insufficient headroom'))
    lease=gpu.GpuLease(required_free_mib=40500,device='cuda:0')
    with pytest.raises(RuntimeError,match='GPU0 headroom'):
        lease.__enter__()
    assert lease.handles==[]


def test_guard_fails_owned_run_without_signaling_any_process(tmp_path,monkeypatch):
    setup_locks(monkeypatch,tmp_path,used=46001)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu.GPU_UUIDS[0])
    monkeypatch.setattr(gpu.os,'kill',lambda *a:pytest.fail('signalled another process'))
    monkeypatch.setattr(gpu.os,'killpg',lambda *a:pytest.fail('signalled another process group'))
    with gpu.GpuLease(required_free_mib=40500,device='cuda:0') as lease:
        with pytest.raises(RuntimeError,match='GPU0 memory guard'):
            lease.check()


def test_snapshot_queries_selected_uuid_and_checks_identity(monkeypatch):
    calls=[]
    def snapshot(command,**kw):
        calls.append(command)
        return gpu.GPU_UUIDS[0]+', 49140, 7000, 42140\n'
    monkeypatch.setattr(gpu.subprocess,'check_output',snapshot)
    assert gpu.gpu_snapshot('cuda:0')['physical_index']==0
    assert calls[0][calls[0].index('-i')+1]==gpu.GPU_UUIDS[0]
    with pytest.raises(RuntimeError,match='identity changed'):
        gpu.gpu_snapshot('cuda:1')
