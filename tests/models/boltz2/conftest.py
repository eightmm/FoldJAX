"""Pytest configuration for deterministic CPU unit tests.

The platform is the caller's choice (`JAX_PLATFORMS=cpu`, as CI and the docs
set it). This file used to `setdefault` the deprecated `JAX_PLATFORM_NAME` at
import, which was inert whenever another suite had already initialized JAX in
the shared session, and otherwise forced every later suite onto the CPU too.

Preallocation is different: turning it off can only lower what the process
holds, so it stays, for the case where this suite starts the session's JAX.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
