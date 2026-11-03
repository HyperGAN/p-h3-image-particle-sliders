"""Allocation-free native refusal; classic core has no modern session registry."""

from dataclasses import dataclass
from .contract import refuse_live


@dataclass(frozen=True)
class NativeCapabilities:
    operations: tuple = ()
    render: bool = False
    qualification: str = "Source-only classic gmix preparation; native H3 unimplemented and disabled"


class NativeGate:
    name = "ph3-image"

    def describe(self, config=None):
        return NativeCapabilities()

    def open(self, config=None, *, operation="train", **request):
        refuse_live(operation)


BACKEND = NativeGate()
