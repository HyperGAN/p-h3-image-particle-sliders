"""Verify the canonical develop implementation, not just a package name."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def require_runtime():
    if "particle_sliders_anima_gmix" in sys.modules:
        raise RuntimeError("HyperGAN develop requires a separate process from classic Anima")
    lock = json.loads(Path(__file__).with_name("engine.lock.json").read_text())
    for name, expected in lock["packages"].items():
        distribution = importlib.metadata.distribution(name)
        spec = importlib.util.find_spec(name)
        if distribution.version != expected["version"] or spec is None or spec.origin is None:
            raise RuntimeError("develop engine distribution differs: " + name)
        root = Path(distribution.locate_file(name)).resolve()
        if Path(spec.origin).resolve().parent != root or source_digest(root) != expected["python_source_sha256"]:
            raise RuntimeError("develop engine imported source differs: " + name)
    return lock
