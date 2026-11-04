"""Frozen full H3 image/audio responses and conditioning, with content identity."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .native_host import Conditioning, PackedState


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeCache:
    def __init__(self, records, contexts, *, positions, identity, expected_examples):
        if type(positions) is not int or positions < 1 or identity.get("split") not in ("train", "development", "confirmation"):
            raise ValueError("declared native positions and split required")
        if not records or len(records) % positions:
            raise ValueError("complete native example grids required")
        self.records, self.contexts = tuple(deepcopy(records)), {k: v.copy() for k, v in contexts.items()}
        self.positions, self.identity = positions, deepcopy(identity)
        self.expected_examples = [tuple(k) for k in expected_examples]
        if not self.expected_examples or len(set(self.expected_examples)) != len(self.expected_examples):
            raise ValueError("explicit unique complete native example cohort required")
        labels, seen = [], set()
        reference_shape = records[0]["positive"].shape
        for index, record in enumerate(records):
            key = (record["scene"], record["seed"], record["path"])
            if index % positions == 0:
                if key in seen:
                    raise ValueError("duplicate native example")
                seen.add(key)
                labels.append(key)
            if key != labels[-1] or record["position"] != index % positions:
                raise ValueError("native example-major source position order required")
            state = record["state"]
            for value in (state.image, state.audio, record["positive"], record["neutral"]):
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all():
                    raise ValueError("finite frozen CPU/FP32 full native responses required")
            if record["positive"].shape != reference_shape or record["neutral"].shape != reference_shape:
                raise ValueError("native response geometry differs")
            if record["positive"].numel() != state.image.numel() + state.audio.numel():
                raise ValueError("full image and coupled audio response required")
            for name in ("neutral_context", "positive_context"):
                if record[name] not in contexts:
                    raise ValueError("missing native conditioning")
        if labels != self.expected_examples:
            raise ValueError("native examples differ from declared complete cohort")
        used_contexts = {row[k] for row in records for k in ("neutral_context", "positive_context")}
        if set(contexts) != used_contexts:
            raise ValueError("exact native conditioning cohort required")
        for key, context in contexts.items():
            if context.name != key or context.embeddings.device.type != "cpu" or context.embeddings.requires_grad:
                raise ValueError("frozen named CPU conditioning required")
        self.examples = self.expected_examples
        self.index = dict(identity=dict(self.identity, positions=positions), fingerprint=self.fingerprint())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Internally owned frozen tensors. Public callers must not mutate cache.
        return self.records[index]

    def tensors(self):
        for index, record in enumerate(self.records):
            for name, value in (("image", record["state"].image), ("audio", record["state"].audio),
                                ("neutral", record["neutral"]), ("positive", record["positive"])):
                yield f"row.{index}.{name}", value
        for index, (_, context) in enumerate(sorted(self.contexts.items())):
            yield f"context.{index}.embeddings", context.embeddings
            for name, value in context.layout.items():
                yield f"context.{index}.{name}", value

    def metadata(self):
        return dict(schema="ph3-native-paired-cache-v1", positions=self.positions,
            identity=self.identity, contexts=sorted(self.contexts), expected_examples=self.expected_examples,
            rows=[{k: v for k, v in row.items() if k not in ("state", "neutral", "positive")}
                  for row in self.records])

    def fingerprint(self):
        content = hashlib.sha256(json.dumps(self.metadata(), sort_keys=True, allow_nan=False).encode())
        content.update(tensor_hash(dict(self.tensors())).encode())
        return content.hexdigest()

    def save(self, directory):
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
        save_file({k: v.contiguous() for k, v in self.tensors()}, str(temporary / "responses.safetensors"))
        document = dict(self.metadata(), fingerprint=self.index["fingerprint"],
                        tensor_file_sha256=file_hash(temporary / "responses.safetensors"))
        (temporary / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, directory)
        return document

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        document = json.loads((directory / "manifest.json").read_text())
        if document.get("schema") != "ph3-native-paired-cache-v1":
            raise ValueError("native cache schema differs")
        tensor_path = directory / "responses.safetensors"
        if file_hash(tensor_path) != document["tensor_file_sha256"]:
            raise ValueError("native cache bytes changed")
        tensors = load_file(str(tensor_path), device="cpu")
        contexts = {}
        for index, name in enumerate(document["contexts"]):
            prefix = f"context.{index}."
            embeddings = tensors.pop(prefix + "embeddings")
            layout = {k[len(prefix):]: tensors.pop(k) for k in list(tensors) if k.startswith(prefix)}
            contexts[name] = Conditioning(name, embeddings, layout)
        records = []
        for index, row in enumerate(document["rows"]):
            state = PackedState(tensors.pop(f"row.{index}.image"), tensors.pop(f"row.{index}.audio"))
            records.append(dict(row, state=state, neutral=tensors.pop(f"row.{index}.neutral"),
                                positive=tensors.pop(f"row.{index}.positive")))
        if tensors:
            raise ValueError("unexpected native cache tensors")
        cache = cls(records, contexts, positions=document["positions"], identity=document["identity"],
                    expected_examples=document["expected_examples"])
        if cache.index["fingerprint"] != document["fingerprint"]:
            raise ValueError("native cache cohort/content changed")
        return cache
