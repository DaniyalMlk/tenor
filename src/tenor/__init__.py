"""tenor — a fixed-income curve toolkit, in pure Python with no dependencies."""

from __future__ import annotations

from .calendar import WEEKENDS_ONLY, Calendar, Rolling, calendar_from
from .daycount import (
    BadPeriod,
    Basis,
    day_count,
    days_in_year,
    denominator,
    is_end_of_february,
    is_leap,
    year_fraction,
)
from .schedule import (
    BadSchedule,
    Frequency,
    Period,
    Schedule,
    Stub,
    add_months,
    end_of_month,
    generate,
)

__version__ = "0.1.0"

__all__ = [
    "WEEKENDS_ONLY",
    "BadPeriod",
    "BadSchedule",
    "Basis",
    "Calendar",
    "Frequency",
    "Period",
    "Rolling",
    "Schedule",
    "Stub",
    "__version__",
    "add_months",
    "calendar_from",
    "day_count",
    "days_in_year",
    "denominator",
    "end_of_month",
    "generate",
    "is_end_of_february",
    "is_leap",
    "year_fraction",
]
