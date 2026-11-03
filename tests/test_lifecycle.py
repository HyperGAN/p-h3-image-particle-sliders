from dataclasses import replace

import pytest

from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation, backend
from ph3_sliders.lifecycle import BACKEND


def test_installed_registration():
    assert backend("ph3-image") is BACKEND


@pytest.mark.parametrize("operation", ["train", "resume", "evaluate", "export", "response", "render"])
def test_shared_lifecycle_refuses_before_factory_or_dependency_checks(operation):
    def unexpected(*args, **kwargs):
        pytest.fail("native factory/validation reached")

    guarded = replace(BACKEND, prepare=unexpected, validate=unexpected)
    config = SessionConfig("ph3-image", "source-only-test", device="cuda:0", allow_hub=True)
    assert guarded.describe(config).operations == ()
    with pytest.raises(UnsupportedOperation, match="license"):
        guarded.open(config, operation=operation)
