"""tenor — a fixed-income curve toolkit, in pure Python with no dependencies."""

from __future__ import annotations

from .bond import Accrual, BadBond, Bond, Cashflow
from .bootstrap import BootstrapFailed, Bootstrapped, bootstrap
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
from .instruments import (
    BadInstrument,
    Deposit,
    Future,
    Instrument,
    Swap,
    ho_lee_convexity,
    maturity_of,
    start_of,
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
from .solve import NoRoot, Root, bracket_root, brent, solve

__version__ = "0.1.0"

__all__ = [
    "WEEKENDS_ONLY",
    "Accrual",
    "BadBond",
    "BadCurve",
    "BadInstrument",
    "BadPeriod",
    "BadRate",
    "BadSchedule",
    "Basis",
    "Bond",
    "BootstrapFailed",
    "Bootstrapped",
    "Calendar",
    "Cashflow",
    "Compounding",
    "Deposit",
    "DiscountCurve",
    "Frequency",
    "Future",
    "Instrument",
    "Interpolation",
    "MonotoneConvex",
    "NoRoot",
    "NotMonotone",
    "OffCurve",
    "Period",
    "Pillar",
    "Rate",
    "Region",
    "Rolling",
    "Root",
    "Schedule",
    "Segment",
    "Stub",
    "Swap",
    "__version__",
    "add_months",
    "bootstrap",
    "bracket_root",
    "brent",
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
    "ho_lee_convexity",
    "is_end_of_february",
    "is_leap",
    "maturity_of",
    "node_forwards",
    "rate_from_discount",
    "solve",
    "start_of",
    "year_fraction",
]
