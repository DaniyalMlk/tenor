"""tenor — a fixed-income curve toolkit, in pure Python with no dependencies."""

from __future__ import annotations

from .calendar import WEEKENDS_ONLY, Calendar, Rolling, calendar_from
from .curve import (
    BadCurve,
    DiscountCurve,
    Interpolation,
    OffCurve,
    Pillar,
    flat_curve,
)
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
from .monotone import (
    MonotoneConvex,
    NotMonotone,
    Region,
    Segment,
    classify,
    node_forwards,
)
from .rates import (
    BadRate,
    Compounding,
    Rate,
    convert,
    discount_factor,
    forward_rate,
    rate_from_discount,
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
    "BadCurve",
    "BadPeriod",
    "BadRate",
    "BadSchedule",
    "Basis",
    "Calendar",
    "Compounding",
    "DiscountCurve",
    "Frequency",
    "Interpolation",
    "MonotoneConvex",
    "NotMonotone",
    "OffCurve",
    "Period",
    "Pillar",
    "Rate",
    "Region",
    "Rolling",
    "Schedule",
    "Segment",
    "Stub",
    "__version__",
    "add_months",
    "calendar_from",
    "classify",
    "convert",
    "day_count",
    "days_in_year",
    "denominator",
    "discount_factor",
    "end_of_month",
    "flat_curve",
    "forward_rate",
    "generate",
    "is_end_of_february",
    "is_leap",
    "node_forwards",
    "rate_from_discount",
    "year_fraction",
]
