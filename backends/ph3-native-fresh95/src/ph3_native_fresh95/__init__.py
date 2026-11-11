"""Opt-in fresh p-h3-image research095 APIs; no default registration changes."""

PROFILE = "ph3-image-coupled-audio-cached-explicit-fresh95-v1"


def configuration(**options):
    from ._config import configuration as make

    return make(**options)


def save_prepared_cache(*args, **kwargs):
    from .cache import save_prepared_cache as save

    return save(*args, **kwargs)


def open_training(*args, **kwargs):
    from .api import Training

    return Training(*args, **kwargs)


def open_inference(*args, **kwargs):
    from .api import Inference

    return Inference(*args, **kwargs)
