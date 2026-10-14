"""Shared memory-safety settings for the 24GB M3: all three runners wire the
same cap, so it lives in one place to avoid silently drifting out of sync.

A prior run that pushed the wired-limit ceiling higher froze the machine;
14GB stays comfortably below both the oMLX hard threshold (19.9GB) and the
sysctl-raised iogpu wired limit (22GB).
"""

from __future__ import annotations

import mlx.core as mx

GB = 1024**3
WIRED_CAP_BYTES = int(14.0 * GB)
CACHE_LIMIT_BYTES = int(2 * GB)


def install_wired_limit_clamp() -> None:
    orig = mx.set_wired_limit

    def capped(limit):
        return orig(min(int(limit), WIRED_CAP_BYTES))

    mx.set_wired_limit = capped
