"""GPU1-only cooperative lease. Never signals or modifies another workload."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

GPU_UUID = "GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69"


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-gpu=uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    uuid, total, used, free = [v.strip() for v in output.strip().split(",")]
    if uuid != GPU_UUID:
        raise RuntimeError("physical GPU identity changed")
    return dict(uuid=uuid, total_mib=int(total), used_mib=int(used), free_mib=int(free))


class GpuLease:
    def __init__(self, *, required_free_mib, max_used_mib=46000):
        self.required_free_mib, self.max_used_mib = required_free_mib, max_used_mib
        self.handles = []
        self.started = None

    def __enter__(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise RuntimeError("pin CUDA_VISIBLE_DEVICES to physical GPU1 UUID before importing Torch")
        try:
            for name, shared in (("/mnt/nvme/sliders/qualification/admission-gpu-1.lock", True),
                                 ("/mnt/nvme/sliders/locks/gpu-1.lock", True),
                                 ("/mnt/nvme/sliders/locks/gpu-1-ph3-training.lock", False)):
                Path(name).parent.mkdir(parents=True, exist_ok=True)
                handle = open(name, "a+")
                self.handles.append(handle)
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            snapshot = gpu_snapshot()
            if snapshot["free_mib"] < self.required_free_mib:
                raise RuntimeError("insufficient current GPU1 headroom; other workloads preserved")
            self.started = time.monotonic()
            owned_role = self.handles[-1]
            owned_role.seek(0); owned_role.truncate()
            json.dump(dict(pid=os.getpid(), gpu_uuid=GPU_UUID, started=time.time(), purpose="ph3-native"), owned_role)
            owned_role.flush()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        snapshot = gpu_snapshot()
        if snapshot["used_mib"] > self.max_used_mib:
            raise RuntimeError("GPU1 memory guard reached; stop only this owned run")
        return snapshot

    def __exit__(self, *_):
        for handle in reversed(self.handles):
            handle.close()
        self.handles.clear()
