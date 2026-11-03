import subprocess
import sys

import pytest

from ph3_sliders.contract import LiveOperationUnavailable
from ph3_sliders.lifecycle import BACKEND


@pytest.mark.parametrize("operation", ["train", "resume", "evaluate", "export", "response", "render"])
def test_native_operations_refuse_without_parsing_or_allocation(operation):
    assert BACKEND.describe().operations == ()
    with pytest.raises(LiveOperationUnavailable, match="license"):
        BACKEND.open(object(), operation=operation, device="cuda:0", allow_hub=True, runtime=object())


def test_native_gate_import_is_standard_library_only(tmp_path):
    code = '''
import sys
from ph3_sliders.contract import LiveOperationUnavailable
from ph3_sliders.lifecycle import BACKEND
try:
    BACKEND.open(operation='train', device='cuda:0', allow_hub=True)
except LiveOperationUnavailable:
    pass
else:
    raise AssertionError('native operation accepted')
assert not {'torch', 'diffusers', 'transformers', 'particle_sliders', 'particlegan'} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
