"""Bounded physical GPU0/1 placement and cooperative ownership.

The requested cuda index is physical at this outer boundary. A child pinned to
one UUID always uses logical cuda:0. Never signals or modifies another workload.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

GPU_UUIDS = {
    0: "GPU-72c1b506-891d-b8bc-b353-e020585e1c47",
    1: "GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69",
}
GPU_UUID = GPU_UUIDS[1]  # Compatibility default; existing GPU1 commands retain it.


def resolve_gpu(device=None, *, environ=None):
    env = os.environ if environ is None else environ
    requested = env.get("PH3_PHYSICAL_GPU")
    if requested is not None and requested not in ("0", "1"):
        raise ValueError("PH3_PHYSICAL_GPU must be 0 or 1")
    if device is not None and device not in ("cuda:0", "cuda:1"):
        raise ValueError("physical device must be cuda:0 or cuda:1")
    index = int(device[-1]) if device is not None else int(requested) if requested is not None else None
    if requested is not None and index != int(requested):
        raise ValueError("config physical device conflicts with PH3_PHYSICAL_GPU")
    if index is None:
        visible = env.get("CUDA_VISIBLE_DEVICES")
        matches = [i for i, uuid in GPU_UUIDS.items() if uuid == visible]
        index = matches[0] if matches else 1
    return dict(physical_index=index, uuid=GPU_UUIDS[index], logical_device="cuda:0")


def lock_paths(device=None, *, root=Path("/mnt/nvme/sliders")):
    index = resolve_gpu(device)["physical_index"]
    return ((root / f"qualification/admission-gpu-{index}.lock", True),
            (root / f"locks/gpu-{index}.lock", True),
            (root / f"locks/gpu-{index}-ph3-training.lock", False))


def gpu_snapshot(device=None):
    placement = resolve_gpu(device)
    output = subprocess.check_output(["nvidia-smi", "-i", placement["uuid"],
        "--query-gpu=uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    uuid, total, used, free = [v.strip() for v in output.strip().split(",")]
    if uuid != placement["uuid"]:
        raise RuntimeError("physical GPU identity changed")
    return dict(placement, total_mib=int(total), used_mib=int(used), free_mib=int(free))


class GpuLease:
    def __init__(self, *, required_free_mib, max_used_mib=46000, device=None):
        self.required_free_mib, self.max_used_mib = required_free_mib, max_used_mib
        self.handles = []
        self.started = None
        self.placement = resolve_gpu(device)
        self.device = f'cuda:{self.placement["physical_index"]}'
        self.identity = None

    def validate_cuda_binding(self, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        cuda = torch_module.cuda
        if cuda.device_count() != 1 or cuda.current_device() != 0:
            raise RuntimeError("the pinned physical GPU must be the only visible logical cuda:0")
        properties = cuda.get_device_properties(0)
        actual = str(properties.uuid)
        if actual.removeprefix('GPU-').lower() != self.placement["uuid"].removeprefix('GPU-').lower():
            raise RuntimeError("CUDA device UUID differs from selected physical GPU")
        return dict(self.placement, cuda_name=properties.name,
                    compute_capability=[properties.major, properties.minor])

    def __enter__(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self.placement["uuid"]:
            raise RuntimeError("pin CUDA_VISIBLE_DEVICES to the selected physical GPU UUID before importing Torch")
        try:
            for name, shared in lock_paths(self.device):
                Path(name).parent.mkdir(parents=True, exist_ok=True)
                handle = open(name, "a+")
                self.handles.append(handle)
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            snapshot = gpu_snapshot(self.device)
            if snapshot["free_mib"] < self.required_free_mib:
                raise RuntimeError(f'insufficient current GPU{self.placement["physical_index"]} headroom; other workloads preserved')
            self.identity = self.validate_cuda_binding()
            self.started = time.monotonic()
            owned_role = self.handles[-1]
            owned_role.seek(0); owned_role.truncate()
            json.dump(dict(pid=os.getpid(), **self.identity, started=time.time(), purpose="ph3-native"), owned_role)
            owned_role.flush()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        snapshot = gpu_snapshot(self.device)
        if snapshot["used_mib"] > self.max_used_mib:
            raise RuntimeError(f'GPU{self.placement["physical_index"]} memory guard reached; stop only this owned run')
        return snapshot

    def __exit__(self, *_):
        for handle in reversed(self.handles):
            handle.close()
        self.handles.clear()
