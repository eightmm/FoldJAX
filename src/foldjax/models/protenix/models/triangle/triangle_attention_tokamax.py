"""Compatibility imports for the shared triangle attention kernel."""

from foldjax.models._tokamax_attention import tokamax_attention_core, tokamax_available

__all__ = ["tokamax_attention_core", "tokamax_available"]
