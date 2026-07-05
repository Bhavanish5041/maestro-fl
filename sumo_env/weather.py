"""
sumo_env/weather.py
===================
Small weather policy helpers for signal timing experiments.

The factors are deliberately conservative: bad weather extends green holds and
priority TTLs slightly to account for lower speeds and longer stopping distance.
"""

WEATHER_FACTORS = {
    "clear": 1.00,
    "cloudy": 1.05,
    "rain": 1.20,
    "heavy_rain": 1.35,
    "fog": 1.30,
}


def timing_factor(condition: str = "clear") -> float:
    """Return a timing multiplier for a weather condition."""
    return WEATHER_FACTORS.get((condition or "clear").lower(), 1.0)


def adjusted_duration(base_seconds: float, condition: str = "clear") -> float:
    """Scale a duration by the condition's timing factor."""
    return float(base_seconds) * timing_factor(condition)
