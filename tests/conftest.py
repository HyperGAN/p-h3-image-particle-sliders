import socket

import pytest


@pytest.fixture(autouse=True)
def deny_network_and_cuda(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("test attempted network or CUDA initialization")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
