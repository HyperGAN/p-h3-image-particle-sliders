"""Genuine installed Source admission and exact retryable borrowed-host cleanup."""

import base64
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

from .util import source_record

MARKER = "_ph3_image_explicit_fresh95_owner"
MISSING = object()


def _record(distribution_object, relative, path):
    records = {str(x): x for x in distribution_object.files or ()}
    row = records.get(relative)
    raw = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
    if (
        row is None
        or Path(distribution_object.locate_file(relative)).absolute() != path.absolute()
        or row.hash is None
        or row.hash.mode != "sha256"
        or row.hash.value != digest
        or row.size != len(raw)
    ):
        raise RuntimeError("genuine original/helper RECORD ownership required: " + relative)


def guard():
    """Never patch original PG/HNDL guards or substitute current core Sources."""
    own = distribution("ph3-image-native-explicit-fresh95")
    root = Path(__file__).absolute().parent
    if own.version != "0.1.0":
        raise RuntimeError("exact separate installed helper required")
    expected = {"__init__.py", "_config.py", "api.py", "cache.py", "ownership.py", "policy.py", "util.py", "source_profile.json"}
    actual = {p.name for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".json")}
    if actual != expected:
        raise RuntimeError("exact seven modules plus Source profile required")
    for name in expected:
        _record(own, "ph3_native_fresh95/" + name, root / name)
    profile = json.loads((root / "source_profile.json").read_text())
    for key in ("native", "core", "PG", "native_runtime"):
        declared = profile[key]
        original = distribution(declared["distribution"])
        if original.version != declared["version"]:
            raise RuntimeError("exact original distribution required: " + key)
        if json.loads(original.read_text("direct_url.json") or "null") != declared.get("original_installed_provenance"):
            raise RuntimeError("original installed provenance differs: " + key)
        if declared["scope"] == "complete-package-inventory":
            inventory = set()
            for package in declared["roots"]:
                package_root = Path(original.locate_file(package)).absolute()
                inventory.update(
                    str(p.relative_to(package_root.parent))
                    for p in package_root.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                )
            if inventory != set(declared["files"]):
                raise RuntimeError("original Source inventory differs: " + key)
        elif key != "native_runtime" or declared["scope"] != "selected-original-layout-and-scheduler-Source":
            raise RuntimeError("unsupported original Source scope")
        for relative, pin in declared["files"].items():
            path = Path(original.locate_file(relative)).absolute()
            if source_record(path) != pin:
                raise RuntimeError("captured original Source differs: " + relative)
            _record(original, relative, path)
    from ph3_research095 import networks

    networks.runtime_identity()  # Original complete HNDL0.10 Source guard.
    return profile


def sources():
    root = Path(__file__).absolute().parent
    return {p.name: source_record(p) for p in sorted(root.iterdir()) if p.suffix in (".py", ".json")}


class Owner:
    """Borrow the caller's H3Host; never unload or replace its native model.

    Failed operations stay pending. Only captured original-constructor handles
    may be removed. Foreign attributes/callbacks/containers remain untouched.
    The exclusive marker is removed last, after hooks/settings are restored.
    """

    def __init__(self, host, *, mode):
        self.host, self.mode = host, mode
        self.closed = self.closing = self.cleanup_pending = False
        self.adapter = self.backend = None
        self._container = None
        self._pending_handles = []
        self._handle_state = {}
        self._operations = []
        self._settings = []
        self._markers = []
        self._marker_states = []
        self._sealed = False
        self._model = host.model
        self._model_parameters = tuple((name, id(p)) for name, p in host.model.named_parameters())
        self._parameter_metadata = tuple((name, tuple(p.shape), p.dtype, p.device) for name, p in host.model.named_parameters())
        self._buffer_metadata = tuple((name, id(b), tuple(b.shape), b.dtype, b.device) for name, b in host.model.named_buffers())
        self._model_modules = tuple((name, id(m)) for name, m in host.model.named_modules())
        modules = tuple(host.model.modules())
        if host.adapter is not None or any(vars(m).get(MARKER) is not None for m in modules):
            raise ValueError("foreign native adapter/explicit owner retained")
        self._before = [
            (m, key, vars(m).get(key, MISSING))
            for m in modules
            for key in ("training", "gradient_checkpointing", "_gradient_checkpointing_func")
        ]
        self._before += [(p, "requires_grad", p.requires_grad) for p in host.model.parameters()]
        self._before.append((host, "adapter", host.adapter))
        self._marker_states = [(m, vars(m).get(MARKER, MISSING)) for m in modules]
        try:
            for module, before in self._marker_states:
                try:
                    setattr(module, MARKER, self)
                finally:
                    if vars(module).get(MARKER) is self:
                        self._markers.append((module, before))
        except BaseException as primary:  # noqa: BLE001 - recover partial marker installation
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain exact pending cleanup owner
                failure = BaseExceptionGroup("partial p-h3-image ownership and cleanup failed", [primary, cleanup])
                failure.cleanup_owner = self
                raise failure from primary
            raise

    def attach(self):
        from ph3_research095.bridge import NativeResponse
        from ph3_research095.networks import RoutedAdapter

        self.adapter = RoutedAdapter.__new__(RoutedAdapter)
        try:
            RoutedAdapter.__init__(self.adapter, self.host.model, self.host.targets, device=self.host.device)
            self.host.adapter = self.adapter
            self.backend = NativeResponse(self.host, self.adapter, identity=self.identity)
        finally:
            # Capture a partial original constructor before its exception leaves.
            container = getattr(self.adapter, "handles", None)
            if container is not None:
                from torch.utils.hooks import RemovableHandle

                if type(container) is not list or any(type(h) is not RemovableHandle for h in container):
                    raise RuntimeError("original private handle container/type differs")
                pending = []
                for handle in container:
                    dictionaries = [handle.hooks_dict_ref()]
                    dictionaries += [ref() for ref in getattr(handle, "extra_dict_ref", ())]
                    snapshots = [(d, d.get(handle.id, MISSING)) for d in dictionaries if d is not None]
                    if snapshots and snapshots[0][1] is not MISSING:
                        pending.append(handle)
                        self._handle_state[id(handle)] = snapshots
                if len({id(h) for h in pending}) != len(pending):
                    raise RuntimeError("duplicate private native handle")
                container[:] = pending
                self._container, self._pending_handles = container, pending
            self._seal()

    def _seal(self):
        if self._sealed:
            return
        for owner, key, before in self._before:
            after = owner.requires_grad if key == "requires_grad" else vars(owner).get(key, MISSING)
            # Include unchanged borrowed flags/modes: a later foreign change is
            # still a conflict even though our constructor did not alter them.
            self._operations.append([owner, key, before, after])
        self._settings = list(self._operations)
        self._sealed = True

    def _check(self, *, cleaning=False):
        if self.host.model is not self._model or tuple((n, id(p)) for n, p in self.host.model.named_parameters()) != self._model_parameters or tuple((n, id(m)) for n, m in self.host.model.named_modules()) != self._model_modules:
            raise RuntimeError("foreign original model/Module/Parameter replacement retained")
        if tuple((n, tuple(p.shape), p.dtype, p.device) for n, p in self.host.model.named_parameters()) != self._parameter_metadata or tuple((n, id(b), tuple(b.shape), b.dtype, b.device) for n, b in self.host.model.named_buffers()) != self._buffer_metadata:
            raise RuntimeError("foreign native parameter/buffer placement or geometry retained")
        for module, before in self._marker_states:
            pending = any(module is m for m, _ in self._markers)
            expected = self if pending else before
            if vars(module).get(MARKER, MISSING) is not expected:
                raise RuntimeError("foreign owner marker retained")
        if self._container is not None:
            current = getattr(self.adapter, "handles", None)
            if current is not self._container or len(current) != len(self._pending_handles) or any(
                a is not b for a, b in zip(current, self._pending_handles)
            ):
                raise RuntimeError("foreign appended/replaced native handle container retained")
            for handle in self._pending_handles:
                for dictionary, value in self._handle_state[id(handle)]:
                    now = dictionary.get(handle.id, MISSING)
                    if now is not value and now is not MISSING:
                        raise RuntimeError("foreign native hook callback retained")
        for operation in self._settings:
            owner, key, before, installed = operation
            current = owner.requires_grad if key == "requires_grad" else vars(owner).get(key, MISSING)
            pending = any(operation is item for item in self._operations)
            expected = installed if pending else before
            if current is not expected and not (cleaning and pending and current is before):
                raise RuntimeError("foreign native setting retained: " + key)

    def require_open(self):
        if self.closed or self.closing:
            raise RuntimeError("owner closing/closed; only pending cleanup may be retried")
        self._check()
        if self.adapter is None or self.host.adapter is not self.adapter or self.adapter.closed:
            raise RuntimeError("original native adapter ownership differs")

    def close(self):
        if self.closed:
            return
        self.closing = self.cleanup_pending = True
        self._seal()
        # Preflight BEFORE removing any handle or restoring borrowed settings.
        self._check(cleaning=True)
        errors = []
        remaining = []
        for handle in self._pending_handles:
            try:
                handle.remove()
            except BaseException as error:  # noqa: BLE001 - retry every unfinished cleanup operation
                remaining.append(handle)
                errors.append(error)
        self._pending_handles = remaining
        if self._container is not None:
            self._container[:] = remaining
        if not remaining:
            pending = []
            for operation in self._operations:
                owner, key, before, installed = operation
                try:
                    current = owner.requires_grad if key == "requires_grad" else vars(owner).get(key, MISSING)
                    if current is not before:
                        if current is not installed:
                            raise RuntimeError("foreign setting retained during cleanup: " + key)
                        if key == "requires_grad":
                            owner.requires_grad_(before)
                        elif before is MISSING:
                            delattr(owner, key)
                        else:
                            setattr(owner, key, before)
                except BaseException as error:  # noqa: BLE001 - retain the exact failed operation
                    pending.append(operation)  # Preserve the identity shared with _settings for retry preflight.
                    errors.append(error)
            self._operations = pending
        if errors:
            failure = BaseExceptionGroup("p-h3-image cleanup pending", errors)
            failure.cleanup_owner = self
            raise failure
        if self._pending_handles or self._operations:
            raise RuntimeError("native cleanup remains pending")
        # Marker removal itself is retryable; successful removals are pruned.
        pending = []
        for module, before in self._markers:
            try:
                if vars(module).get(MARKER) is not self:
                    raise RuntimeError("foreign marker retained at final release")
                if before is MISSING:
                    delattr(module, MARKER)
                else:
                    setattr(module, MARKER, before)
            except BaseException as error:  # noqa: BLE001 - marker release is pending until successful
                pending.append((module, before))
                errors.append(error)
        self._markers = pending
        if errors:
            failure = BaseExceptionGroup("p-h3-image marker cleanup pending", errors)
            failure.cleanup_owner = self
            raise failure
        if self.adapter is not None:
            self.adapter.closed = True
        self.closed, self.closing, self.cleanup_pending = True, False, False
