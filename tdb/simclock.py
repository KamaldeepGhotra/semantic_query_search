"""Virtual clock: wall time plus simulated model-inference latency.

Patching time.time lets the authors' code (which measures runtime with
time.time) see cached Gemma lookups as if inference ran live.
"""
import time

_real_time = time.time
_offset = 0.0


def advance(seconds):
    global _offset
    _offset += seconds


def install():
    time.time = lambda: _real_time() + _offset
