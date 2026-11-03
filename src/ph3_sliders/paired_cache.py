"""Training-only paired-record contract; no native target generation or IO."""

from copy import deepcopy
import hashlib
import json

import torch
from particle_sliders_anima_gmix import fit_normalization


class PairedCache:
    """Own frozen same-state/time teacher records in example-major position order.

    The fixture uses this contract; native P-H3 packing, ordered references and
    expanded prompt identities remain unqualified and cannot enter its factory.
    """

    def __init__(self, records, *, positions, identity):
        if type(positions) is not int or positions < 1:
            raise ValueError("positive native position count required")
        if not isinstance(identity, dict) or identity.get("split") != "train":
            raise ValueError("only training records may fit normalization")
        self.positions = positions
        self._records = tuple(records)
        if len(self._records) < 2 * positions or len(self._records) % positions:
            raise ValueError("at least two complete paired examples required")
        shape = tuple(self._records[0]["positive"].shape)
        if len(shape) < 2 or shape[0] != 1:
            raise ValueError("single-example unpooled responses required")
        for index, record in enumerate(self._records):
            if record.get("position") != index % positions:
                raise ValueError("paired native positions must retain their source order")
            for name in ("latent", "neutral", "positive", "conditioning"):
                value = record[name]
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all()):
                    raise ValueError("paired records require frozen finite CPU/FP32 tensors")
            if any(tuple(record[name].shape) != shape for name in ("latent", "neutral", "positive")):
                raise ValueError("same-state teacher geometry differs")
            if type(record["time"]) not in (int, float) or not 0 <= record["time"] <= 1:
                raise ValueError("synthetic fixture time must be in [0,1]")
        self._records = deepcopy(self._records)
        identity = deepcopy(identity)
        identity.update(positions=positions, response_shape=list(shape), records=len(self._records))
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
        for record in self._records:
            digest.update(str(record["time"]).encode())
            for name in ("latent", "neutral", "positive", "conditioning"):
                digest.update(record[name].contiguous().numpy().tobytes())
        self._index = {"identity": identity, "fingerprint": digest.hexdigest()}

    @property
    def index(self):
        return deepcopy(self._index)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        return deepcopy(self._records[index])

    def normalization(self):
        return fit_normalization(self, positions=self.positions)
