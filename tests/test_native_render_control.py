"""Render control checks; the real pinned-runtime --check supplies cohort coverage."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).parents[1] / "tools/render_native.py"
spec = importlib.util.spec_from_file_location("ph3_render_control", PATH)
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


class NativeRenderControlTests(unittest.TestCase):
    def test_off_requires_image_and_coupled_audio_exact_equality(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(equal=lambda a, b: a == b)}):
            off = SimpleNamespace(image=b"image", audio=b"audio")
            self.assertTrue(render.verify_exact_off(off, off)["exact"])
            for bad in (SimpleNamespace(image=b"different", audio=b"audio"),
                        SimpleNamespace(image=b"image", audio=b"different")):
                with self.assertRaisesRegex(ValueError, "adapter-detached OFF"):
                    render.verify_exact_off(off, bad)

    def test_render_helper_preserves_frozen_numerical_sources(self):
        root = PATH.parents[1]
        # These are the actual cohort hashes recorded by the completed128 run.
        import hashlib
        self.assertEqual(hashlib.sha256((root / "tools/modern_native.py").read_bytes()).hexdigest(),
                         "3274414d06ea9e5ba94071f5e52745c70bb0467ac41536083ed4359cf0a93c66")


if __name__ == "__main__":
    unittest.main()
