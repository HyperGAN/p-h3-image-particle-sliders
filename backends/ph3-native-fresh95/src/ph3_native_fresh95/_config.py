"""Explicit new profile; original rank8, batch2, h128 and official rate law."""

import math


def configuration(*, lr, d_lr_mult, prior_lr_mult, seed=7, total_steps=128, ema_decay=None):
    if ema_decay is not None:
        raise ValueError("weight EMA is forbidden for adapter, critic and bank")
    if type(seed) is not int or not 0 <= seed < 2**63 - 100 or total_steps != 128:
        raise ValueError("explicit seed and original128 horizon required")
    rates = (lr, d_lr_mult, prior_lr_mult)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError("finite positive official rate overrides required")
    return {
        "lr": float(lr),
        "d_lr_mult": float(d_lr_mult),
        "prior_lr_mult": float(prior_lr_mult),
        "seed": seed,
        "total_steps": 128,
        "ema_decay": None,
    }
