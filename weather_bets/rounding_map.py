"""NWS °F → °C → °F rounding map for Kalshi bucket edge detection.

KAUS is a 5-minute ASOS station where temps undergo double-conversion:
  actual °F → round → °C → round → °F (displayed)

This creates ±1°F errors. When the error crosses a Kalshi 2°F bucket boundary,
naive traders may bet on the wrong bucket.

This module provides lookup tools to detect these ambiguities.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def get_kalshi_bucket(temp_f: int) -> tuple[int, int]:
    """Return (low, high) for the 2°F Kalshi bucket containing temp_f."""
    low = (temp_f // 2) * 2
    return (low, low + 1)


def get_bucket_label(temp_f: int) -> str:
    """Return string label like '90-91' for the bucket containing temp_f."""
    lo, hi = get_kalshi_bucket(temp_f)
    return f"{lo}-{hi}"


# ── Build static lookup tables ─────────────────────────────────────────

# Celsius → list of possible actual Fahrenheit values
_C_TO_POSSIBLE_F: dict[int, list[int]] = {}
# Fahrenheit → displayed Fahrenheit (after F→C→F roundtrip)
_F_DISPLAYED: dict[int, int] = {}
# Set of °F values where the roundtrip crosses a bucket boundary
_CROSSING_TEMPS: set[int] = set()
# °C values where the two possible °F values are in different buckets
_AMBIGUOUS_C: dict[int, dict] = {}

for _f in range(15, 125):
    _c_raw = _f_to_c(_f)
    _c_rounded = round(_c_raw)
    _f_back = round(_c_to_f(_c_rounded))
    _F_DISPLAYED[_f] = _f_back

    if _c_rounded not in _C_TO_POSSIBLE_F:
        _C_TO_POSSIBLE_F[_c_rounded] = []
    _C_TO_POSSIBLE_F[_c_rounded].append(_f)

    if get_kalshi_bucket(_f) != get_kalshi_bucket(_f_back):
        _CROSSING_TEMPS.add(_f)

for _c, _fs in _C_TO_POSSIBLE_F.items():
    if len(_fs) == 2:
        lo_f, hi_f = _fs
        lo_bkt = get_kalshi_bucket(lo_f)
        hi_bkt = get_kalshi_bucket(hi_f)
        if lo_bkt != hi_bkt:
            _AMBIGUOUS_C[_c] = {
                "lo_f": lo_f,
                "hi_f": hi_f,
                "lo_bucket": lo_bkt,
                "hi_bucket": hi_bkt,
            }

# Clean up module-level loop vars
del _f, _c_raw, _c_rounded, _f_back, _c, _fs


# ── Public API ─────────────────────────────────────────────────────────

def is_crossing_temp(temp_f: int) -> bool:
    """True if this °F value would display in a DIFFERENT Kalshi bucket
    after the NWS F→C→F roundtrip. These temps are dangerous to bet on naively."""
    return temp_f in _CROSSING_TEMPS


def displayed_temp(actual_f: int) -> int:
    """What the NWS 5-min feed would show for a given actual °F."""
    return _F_DISPLAYED.get(actual_f, actual_f)


def is_ambiguous_celsius(celsius: int) -> bool:
    """True if this °C value maps to two °F values in different Kalshi buckets."""
    return celsius in _AMBIGUOUS_C


def get_ambiguous_info(celsius: int) -> dict | None:
    """For an ambiguous °C value, return both possible °F values and their buckets.
    Returns None if this °C is not ambiguous."""
    return _AMBIGUOUS_C.get(celsius)


def possible_actual_f(celsius: int) -> list[int]:
    """Return all °F values that map to a given °C after rounding."""
    return _C_TO_POSSIBLE_F.get(celsius, [])


def get_crossing_temps() -> set[int]:
    """Return the full set of crossing temperatures."""
    return _CROSSING_TEMPS.copy()


def get_all_ambiguous_c() -> dict[int, dict]:
    """Return all ambiguous °C values and their info."""
    return _AMBIGUOUS_C.copy()


def should_skip_bucket(predicted_high: int) -> bool:
    """Returns True if the predicted high is at a rounding crossing
    and we should skip or require extra confidence."""
    return predicted_high in _CROSSING_TEMPS


# ── Convenience summary ────────────────────────────────────────────────

def print_crossing_summary():
    """Print a summary of all crossing temperatures for debugging."""
    print(f"\n{'Actual°F':>10} {'Displayed':>10} {'Error':>6} {'Actual Bkt':>12} {'Displayed Bkt':>14}")
    print("-" * 55)
    for f in sorted(_CROSSING_TEMPS):
        d = _F_DISPLAYED[f]
        err = d - f
        print(f"{f:>8}°F {d:>8}°F {err:>+4}°F  {get_bucket_label(f):>10}  {get_bucket_label(d):>12}")
