"""Return the exact proven response-only factory; never clone or retag it."""
import importlib.util
import sys

from .execution import authority, source

_FACTORY = None


def factory_pin():
    """Allocation-free Source metadata, without importing Torch/native modules."""
    return authority()['public_response_factory']


def __getattr__(name):
    global _FACTORY
    if name != 'INFERENCE_BACKEND':
        raise AttributeError(name)
    if _FACTORY is None:
        pin = factory_pin()
        raw = source(pin)
        spec = importlib.util.spec_from_file_location('_ph3_product_original_native_response_factory', pin['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(raw, pin['path'], 'exec'), module.__dict__)
        _FACTORY = module.INFERENCE_BACKEND
    return _FACTORY
