import json
import os
import subprocess
import sys

import pytest

from ph3_sliders.cli import main
from ph3_sliders.contract import LiveOperationUnavailable, RequestPlan, refuse_live


def test_edit_order_and_snapshot():
    references = ["subject.png", "hat.png"]
    plan = RequestPlan("give the subject this hat", references=references)
    references.reverse()
    data = plan.as_dict()
    assert data["task"] == "image-edit"
    assert data["request"]["references"] == ("subject.png", "hat.png")
    assert "no CFG" in data["sampling"]
    assert RequestPlan("a bowl", width=2368, height=1760).width == 2368


@pytest.mark.parametrize("options", [
    {"prompt": " "}, {"width": 2913}, {"width": True}, {"height": -32},
    {"width": 32, "height": 32}, {"width": 8192, "height": 8192},
    {"references": ["x"] * 9}, {"references": "x.png"}, {"references": [""]},
    {"strength": float("nan")}, {"strength": float("inf")}, {"strength": -0.1},
    {"strength": 1.1}, {"strength": True}, {"seed": -1}, {"seed": True},
])
def test_invalid_plans(options):
    with pytest.raises(ValueError):
        RequestPlan(**({"prompt": "a bowl"} | options))


def test_plan_uses_no_native_runtime_or_file_access(tmp_path):
    # A fresh subprocess proves this path does not import Torch or model libraries.
    source = """
import sys
from ph3_sliders.contract import RequestPlan
p = RequestPlan('a bowl', references=('nonexistent.png',)).as_dict()
assert p['task'] == 'image-edit'
assert not {'torch', 'diffusers', 'transformers', 'huggingface_hub', 'particle_sliders'} & sys.modules.keys()
"""
    env = dict(os.environ)
    subprocess.run([sys.executable, "-c", source], check=True, env=env, cwd=tmp_path)


def test_plan_cli(capsys):
    assert main(["plan", "--prompt", "a bowl"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_id"] == "PrunaAI/P-H3-Image"


@pytest.mark.parametrize("operation", ["train", "render", "download", "publish"])
def test_live_cli_refuses(operation, capsys):
    with pytest.raises(SystemExit) as error:
        main([operation])
    assert error.value.code == 2
    assert "disabled" in capsys.readouterr().err
    with pytest.raises(LiveOperationUnavailable):
        refuse_live(operation)
