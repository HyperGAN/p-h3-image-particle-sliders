"""Optional native HNDL networks; importing this package loads no model or GPU."""
from .networks import RoutedAdapter, RoutedProjection, GlobalMixCritic, runtime_identity

__all__ = ["RoutedAdapter", "RoutedProjection", "GlobalMixCritic", "runtime_identity"]

from .artifacts import adapter_signature, export_adapter, load_adapter
__all__ += ["adapter_signature", "export_adapter", "load_adapter"]
