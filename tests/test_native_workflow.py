"""Interruption and tolerance regressions from the exact native source review."""

import json
from types import SimpleNamespace

import pytest

from ph3_sliders.selection import nominate


def result(lr, score):
    return dict(lr=lr, completed_steps=128, seed=17, metric_feasible=True,
        evaluation=dict(selection="ema", cache_fingerprint="same-full-grid",
            mean_teacher_student_relative_error=score, mean_signed_gain=.5,
            scene_relative_error={"room": score}))


def test_nomination_uses_absolute_tolerance_then_lower_rate():
    rows = [result(6e-6, .5000001), result(2e-5, .5)]
    assert nominate(rows, tie_tolerance=1e-6)["lr"] == 6e-6
    assert nominate(rows, tie_tolerance=0)["lr"] == 2e-5
    assert nominate([result(6e-6, .50001), result(2e-5, .5)])["lr"] == 2e-5


def test_nomination_refuses_mismatched_budgets_and_nonfinite_results():
    rows = [result(6e-6, .6), result(2e-5, .5)]
    rows[1]["completed_steps"] = 64
    with pytest.raises(ValueError, match="identical completed budgets"):
        nominate(rows)
    with pytest.raises(ValueError, match="unscoreable"):
        nominate([result(6e-6, float("nan"))])
    assert nominate([result(6e-6, 1.1)]) is None


def test_classic_native_evaluation_recovers_interrupted_raw_publication(tmp_path, monkeypatch):
    import ph3_sliders.native_fit as native
    from ph3_sliders.native_cache import file_hash

    class Run:
        calls = 0
        def evaluate(self, cache, *, selection, raw_path):
            self.calls += 1
            if raw_path.exists():
                raise FileExistsError(raw_path)
            raw_path.write_bytes(b"test full-response payload")
            return dict(completed_steps=128, cache_fingerprint=cache.index["fingerprint"],
                selection=selection, raw_student_responses=dict(path=str(raw_path), sha256=file_hash(raw_path)))

    cache, run = SimpleNamespace(index={"fingerprint": "frozen-dev-grid"}), Run()
    original = native.os.rename
    def interrupt(source, destination):
        raise RuntimeError("injected after raw and JSON writes, before publication")
    monkeypatch.setattr(native.os, "rename", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        native.evaluate_completed(run, cache, tmp_path, 128)
    assert not (tmp_path / "eval-000128-ema.json").exists()
    monkeypatch.setattr(native.os, "rename", original)
    # An old standalone raw payload is retained and does not obstruct recovery.
    orphan = tmp_path / "eval-000128-ema.safetensors"
    orphan.write_bytes(b"orphan from previous implementation")
    document = native.evaluate_completed(run, cache, tmp_path, 128)
    assert native.evaluate_completed(run, cache, tmp_path, 128) == document
    assert run.calls == 2
    assert orphan.read_bytes() == b"orphan from previous implementation"
    assert json.loads((tmp_path / "eval-000128-ema.json").read_text()) == document
