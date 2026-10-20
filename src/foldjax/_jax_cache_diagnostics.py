"""Compatibility for keyword-form tracing diagnostics in pinned JAX 0.11.1."""

from __future__ import annotations

import threading
from functools import wraps

_LOCK = threading.Lock()
_TRACE_KEYWORDS = frozenset(
    {"debug_info", "fun_takes_flat_tree_arg", "fun_returns_flat_tree", "requires_low"}
)


def _diagnostic_key(key):
    context, arguments, keyword_items = key
    if not keyword_items:
        return key
    keywords = dict(keyword_items)
    if not keywords.keys() <= _TRACE_KEYWORDS:
        return key
    if len(arguments) == 1 and "debug_info" in keywords:
        arguments = (*arguments, keywords["debug_info"])
    elif len(arguments) < 2 or "debug_info" in keywords:
        return key
    # JAX's explanation constructs the new key without keyword options, while
    # its stored key retains them. Only the formatter needs this positional
    # view; actual tracing/persistent keys must retain every option.
    return context, arguments, ()


def install() -> None:
    import jax
    from jax._src.interpreters import partial_eval

    if jax.__version__ != "0.11.1":
        return
    with _LOCK:
        original = partial_eval.diff_tracing_cache_keys
        if getattr(original, "_foldjax_keyword_diagnostics", False):
            return

        @wraps(original)
        def diff(new_key, old_key):
            new_view, old_view = _diagnostic_key(new_key), _diagnostic_key(old_key)
            result = original(new_view, old_view)
            if new_view is not new_key or old_view is not old_key:
                severity, count, explanation = result
                return (
                    severity,
                    count,
                    (
                        explanation
                        + "\n  JAX 0.11.1 keyword-form trace keys normalized "
                        "for diagnostics."
                    ),
                )
            return result

        diff._foldjax_keyword_diagnostics = True
        partial_eval.diff_tracing_cache_keys = diff
