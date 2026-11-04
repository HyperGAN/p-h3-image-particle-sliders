import math

import pytest
import torch

from ph3_hypergan import fit_normalization


class Cache:
    index = dict(identity=dict(split="train"), fingerprint="analytical-paired-grid")

    def __init__(self):
        self.rows = [dict(position=position, neutral=torch.zeros(2),
            positive=torch.tensor([float(example), 2. * example]) * (position + 1))
            for example in (1, 2, 3) for position in (0, 1)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_train_only_sample_std_and_per_position_median_gain():
    cache = Cache()
    norm = fit_normalization(cache, positions=2)
    assert torch.equal(norm["scale"], torch.tensor([[2., 4.], [4., 8.]]))
    assert torch.allclose(norm["edit_rms"], torch.full((2,), math.sqrt(3.5 / 3)))
    assert norm["provenance"]["cache_fingerprint"] == cache.index["fingerprint"]
    cache.index = dict(identity=dict(split="development"), fingerprint="held-out")
    with pytest.raises(ValueError, match="training examples"):
        fit_normalization(cache, positions=2)
