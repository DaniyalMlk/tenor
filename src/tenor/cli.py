"""A command line over a file of quotes.

Every convention in this library is an argument with a name rather than a
default, and a command line that quietly picked one would undo six phases of
being careful about exactly that. So ``--basis`` is required wherever a day
count decides the answer, and a quote file line that omits a convention is
rejected rather than filled in.

The quote file is one instrument per line, whitespace separated, ``#`` starting
a comment:

    deposit  2021-07-05  0.0035  ACT/360
    future   2021-09-15  2021-12-15  99.55  ACT/360  [convexity]
    swap     2031-01-06  0.0195    [frequency]  [basis]

Dates are ISO. The start date of a deposit or a swap is the curve's reference
date; a future carries both of its own, since it does not start today.

``multicurve`` takes a second file in the same shape, of quotes on the index
being forecast rather than on the curve being discounted:

    forward  2026-01-15  2026-04-15  0.0330   [basis]
    swap     2036-01-15  0.0390
    basis    2031-01-15  0.0008    [flat frequency]

A ``forward`` is one period of the index. A ``swap`` is a par rate on it,
discounted off the curve the first file built. A ``basis`` is a tenor basis
spread quoted on the index against a longer one paying flat, and it solves the
*longer* tenor's curve, so a file mixing ``swap`` and ``basis`` lines is asking
for two different curves at once and is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from . import __version__
from .bond import Accrual, Bond
from .bootstrap import Bootstrapped, bootstrap
from .calendar import WEEKENDS_ONLY
from .credit import (
    CreditDefaultSwap,
    bootstrap_hazards,
    risky_bond_price,
    triangle_hazard,
)
from .curve import Interpolation
from .daycount import Basis
from .floating import FloatingNote
from .futures import (
    DEFAULT_DIGITS,
    DEFAULT_NOTIONAL_COUPON,
    BondFuture,
    cheapest_to_deliver,
    delivery_switch,
)
from .horizon import horizon_return
from .inflation import (
    LinkedBond,
    ReferenceIndex,
    breakeven_inflation,
    implied_inflation_from_price,
    index_accretion,
)
from .instruments import Deposit, Future, Instrument, Swap
from .lattice import (
    Exercise,
    Lattice,
    lattice_price,
    option_adjusted_spread,
    option_cost,
    steps_between,
)
from .multicurve import (
    BasisSwap,
    DualSwap,
    FlatLegOf,
    ForecastIndex,
    ForecastQuote,
    IndexForward,
    bootstrap_forecast,
    forward_rate,
    split_risk,
)
from .rates import Compounding
from .risk import buckets_from, instrument_risk, key_rates, level, shape_duration
from .schedule import Frequency, add_months
from .spread import i_spread, z_spread


class BadInput(ValueError):
    """Something the command line was given that it cannot use."""


def _basis(text: str) -> Basis:
    """A day count, by its name or by how the market spells it.

    The market's spelling has spaces in it - "30/360 Bond Basis" - which is
    fine in a docstring and hostile on a command line and in a whitespace
    separated file. So the enum's own name is accepted too, and is what the
    examples use. Both resolve to the same convention; neither is a default.
    """
    try:
        return Basis[text.upper().replace("/", "_").replace("-", "_")]
    except KeyError:
        pass
    try:
        return Basis(text)
    except ValueError:
        raise BadInput(
            f"{text!r} is not a day count. Use one of: "
            + ", ".join(one.name for one in Basis)
        ) from None


def _interpolation(text: str) -> Interpolation:
    try:
        return Interpolation[text.upper().replace("-", "_")]
    except KeyError:
        pass
    try:
        return Interpolation(text)
    except ValueError:
        raise BadInput(
            f"{text!r} is not an interpolation. Use one of: "
            + ", ".join(one.name for one in Interpolation)
        ) from None


def parse_quotes(text: str, reference: date) -> list[Instrument]:
    """Read a quote file into instruments, refusing anything ambiguous."""
    quotes: list[Instrument] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        kind = fields[0].lower()
        try:
            quotes.append(_one(kind, fields[1:], reference))
        except (ValueError, IndexError, KeyError) as bad:
            raise BadInput(f"line {number}: {raw.strip()!r}: {bad}") from bad
    if not quotes:
        raise BadInput("the quote file has no instruments in it")
    return quotes


def _one(kind: str, fields: Sequence[str], reference: date) -> Instrument:
    if kind == "deposit":
        maturity, rate, basis = fields[0], fields[1], fields[2]
        return Deposit(
            reference, date.fromisoformat(maturity), float(rate), _basis(basis)
        )
    if kind == "future":
        start, end, price, basis = fields[0], fields[1], fields[2], fields[3]
        convexity = float(fields[4]) if len(fields) > 4 else 0.0
        return Future(
            date.fromisoformat(start),
            date.fromisoformat(end),
            float(price),
            _basis(basis),
            convexity=convexity,
        )
    if kind == "swap":
        maturity, rate = fields[0], fields[1]
        frequency = Frequency[fields[2].upper()] if len(fields) > 2 else Frequency.SEMI_ANNUAL
        basis = _basis(fields[3]) if len(fields) > 3 else Basis.THIRTY_360_BOND
        return Swap(
            reference,
            date.fromisoformat(maturity),
            float(rate),
            frequency,
            basis,
        )
    raise BadInput(
        f"{kind!r} is not an instrument. Known kinds are deposit, future and swap."
    )


def build(arguments: argparse.Namespace) -> Bootstrapped:
    reference = date.fromisoformat(arguments.reference)
    quotes = parse_quotes(Path(arguments.quotes).read_text(), reference)
    return bootstrap(
        reference,
        quotes,
        basis=_basis(arguments.basis),
        interpolation=_interpolation(arguments.interpolation),
    )


def parse_index(text: str) -> ReferenceIndex:
    """Parse a published index series: one ``YYYY-MM level`` per line.

    Errors name the line, for the same reason the quote parser does: a series of
    price levels is the kind of file that gets pasted together by hand, and "line
    14 is not a month" is a different piece of information from a traceback out of
    a dataclass.
    """
    values: dict[tuple[int, int], float] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"line {number} has {len(fields)} fields; an index line is a "
                f"month and a level: 2025-08 132.4"
            )
        month, level = fields
        parts = month.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"line {number}: {month!r} is not a month; write it as 2025-08"
            )
        try:
            year, month_number = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"line {number}: {month!r} is not a month; write it as 2025-08"
            ) from None
        if not 1 <= month_number <= 12:
            raise ValueError(
                f"line {number}: there is no month {month_number}"
            )
        try:
            value = float(level)
        except ValueError:
            raise ValueError(
                f"line {number}: {level!r} is not an index level"
            ) from None
        if (year, month_number) in values:
            raise ValueError(
                f"line {number}: {year}-{month_number:02d} appears twice"
            )
        values[(year, month_number)] = value
    if not values:
        raise ValueError("the index file has no figures in it")
    return ReferenceIndex(values)


def _linker(arguments: argparse.Namespace, reference: date) -> LinkedBond:
    return LinkedBond(
        date.fromisoformat(arguments.issued) if arguments.issued else reference,
        date.fromisoformat(arguments.maturity),
        arguments.coupon,
        arguments.base_index,
        Frequency[arguments.frequency.upper()],
        _basis(arguments.linker_basis),
        deflation_floor=not arguments.no_floor,
    )


def _bond(arguments: argparse.Namespace, reference: date) -> Bond:
    return Bond(
        reference,
        date.fromisoformat(arguments.maturity),
        arguments.coupon,
        Frequency[arguments.frequency.upper()],
        _basis(arguments.bond_basis),
    )


def _note(arguments: argparse.Namespace, reference: date) -> FloatingNote:
    return FloatingNote(
        date.fromisoformat(arguments.issued) if arguments.issued else reference,
        date.fromisoformat(arguments.maturity),
        arguments.quoted_margin,
        Frequency[arguments.frequency.upper()],
        _basis(arguments.note_basis),
    )


# -- the subcommands ----------------------------------------------------------


def run_curve(arguments: argparse.Namespace) -> dict[str, object]:
    built = build(arguments)
    points = []
    for pillar in built.curve.pillars[1:]:
        points.append(
            {
                "date": pillar.day.isoformat(),
                "years": round(pillar.time, 6),
                "discount": pillar.discount,
                "zero": built.curve.zero_rate(pillar.day, Compounding.CONTINUOUS),
            }
        )
    return {
        "reference": built.curve.reference.isoformat(),
        "interpolation": built.curve.interpolation.value,
        "basis": built.curve.basis.value,
        "sweeps": built.sweeps,
        "worst_repricing_error": built.worst_error,
        "reprices": built.reprices(),
        "pillars": points,
    }


def run_price(arguments: argparse.Namespace) -> dict[str, object]:
    built = build(arguments)
    reference = built.curve.reference
    bond = _bond(arguments, reference)
    dirty = bond.price_from_curve(built.curve, reference)
    accrued = bond.accrued(reference)
    equivalent = bond.curve_yield(built.curve, reference)
    result: dict[str, object] = {
        "bond": bond.name,
        "dirty": dirty,
        "accrued": accrued,
        "clean": dirty - accrued,
        "equivalent_yield": equivalent.value,
        "solver_iterations": equivalent.iterations,
        "macaulay_duration": bond.macaulay_duration(equivalent.value, reference),
        "modified_duration": bond.modified_duration(equivalent.value, reference),
        "convexity": bond.convexity(equivalent.value, reference),
        "pv01": bond.pv01(equivalent.value, reference),
        "dv01": bond.dv01(built.curve, reference),
        "effective_duration": bond.effective_duration(built.curve, reference),
    }
    if arguments.quote is not None:
        spread = z_spread(bond, arguments.quote, built.curve, reference)
        result["quoted_clean"] = arguments.quote
        result["z_spread"] = spread.value
        result["i_spread"] = i_spread(bond, arguments.quote, built.curve, reference)
    return result


def run_floating(arguments: argparse.Namespace) -> dict[str, object]:
    """Price a floating rate note and report both of its durations.

    Both, rather than one described as "the duration", because for this
    instrument they differ by two orders of magnitude and at the quoted margin
    the rate one is zero. A report giving a single number would be giving the
    wrong one whichever it chose.
    """
    built = build(arguments)
    reference = built.curve.reference
    note = _note(arguments, reference)
    fixing: float | None = arguments.fixing
    margin = arguments.margin if arguments.margin is not None else note.quoted_margin
    if arguments.quote is not None:
        solved = note.discount_margin(
            built.curve, arguments.quote, reference, current_fixing=fixing
        )
        margin = solved.value
    dirty = note.dirty_price(
        built.curve, reference, margin=margin, current_fixing=fixing
    )
    accrued = note.accrued(reference, curve=built.curve, current_fixing=fixing)
    coupons = note.coupons(
        built.curve, reference, margin=margin, current_fixing=fixing
    )
    result: dict[str, object] = {
        "note": note.name,
        "settlement": reference.isoformat(),
        "on_reset_date": note.is_reset_date(reference),
        "quoted_margin": note.quoted_margin,
        "discount_margin": margin,
        "dirty": dirty,
        "accrued": accrued,
        "clean": dirty - accrued,
        "spread_duration": note.spread_duration(
            built.curve, reference, margin=margin, current_fixing=fixing
        ),
        "rate_duration": note.rate_duration(
            built.curve, reference, margin=margin, current_fixing=fixing
        ),
        "margin_value_of_a_basis_point": note.margin_value_of_a_basis_point(
            built.curve, reference, margin=margin, current_fixing=fixing
        ),
        "current_fixing_projected": coupons[0].projected,
        "coupon_count": len(coupons),
    }
    if arguments.coupons:
        result["coupons"] = [
            {
                "payment": one.payment.isoformat(),
                "accrual": one.accrual,
                "index_rate": one.index_rate,
                "projected": one.projected,
                "coupon": one.coupon,
                "redemption": one.redemption,
                "discount": one.discount,
                "present_value": one.present_value,
            }
            for one in coupons
        ]
    if arguments.quote is not None:
        result["quoted_clean"] = arguments.quote
    return result


def run_linker(arguments: argparse.Namespace) -> dict[str, object]:
    """Price an index-linked bond, in both spaces, and split its accretion.

    The report keeps real and nominal apart throughout, because the quoted price
    and yield are real and the money that changes hands is neither. It also says
    how much of the accretion over the next year is already published: the lag
    means a linker's near-term inflation exposure is largely arithmetic, and how
    much depends on where in the publication cycle the question is asked.
    """
    built = build(arguments)
    reference = built.curve.reference
    index = parse_index(Path(arguments.index).read_text())
    bond = _linker(arguments, reference)

    if arguments.real_quote is not None:
        solved = bond.real_yield_from_clean(arguments.real_quote, reference)
        if not solved.converged:
            raise ValueError(
                f"the real yield did not converge against a clean price of "
                f"{arguments.real_quote}: residual {solved.residual:.3g}"
            )
        real_yield = solved.value
        real_clean = arguments.real_quote
    else:
        real_yield = arguments.real_yield
        real_clean = bond.real_clean_price(real_yield, reference)

    ratio = bond.index_ratio(index, reference)
    redemption_ratio, floored = bond.redemption_ratio(
        index, projection=arguments.projection
    )
    nominal = bond.nominal_price_from_curve(
        built.curve, index, reference, projection=arguments.projection
    )
    invoice = bond.settlement_amount(real_clean, index, reference)
    result: dict[str, object] = {
        "bond": bond.name,
        "settlement": reference.isoformat(),
        "index_published_to": "{}-{:02d}".format(*index.last_month),
        "reference_index_known_to": index.last_known_date().isoformat(),
        "reference_index": index.reference(reference),
        "index_ratio": ratio,
        "real_yield": real_yield,
        "real_clean": real_clean,
        "real_accrued": bond.real_accrued(reference),
        "real_modified_duration": bond.real_modified_duration(real_yield, reference),
        "real_convexity": bond.real_convexity(real_yield, reference),
        "settlement_amount": invoice,
        "projection": arguments.projection,
        "redemption_index_ratio": redemption_ratio,
        "deflation_floor_binds": floored,
        "nominal_present_value": nominal,
    }
    implied = implied_inflation_from_price(
        bond, built.curve, index, reference, nominal_price=invoice
    )
    if implied.converged:
        result["inflation_implied_by_the_invoice"] = implied.value
    if arguments.nominal_yield is not None:
        result["nominal_yield"] = arguments.nominal_yield
        result["breakeven_exact"] = breakeven_inflation(
            arguments.nominal_yield, real_yield
        )
        result["breakeven_quoted"] = breakeven_inflation(
            arguments.nominal_yield, real_yield, exact=False
        )
    horizon = date(reference.year + 1, reference.month, reference.day)
    split = index_accretion(
        index, reference, horizon, bond.base_index, projection=arguments.projection
    )
    result["accretion_one_year"] = {
        "total": split.total,
        "published": split.known,
        "projected": split.projected,
        "published_share": split.known_fraction,
        "published_through": split.known_through.isoformat(),
    }
    if arguments.flows:
        result["flows"] = [
            {
                "payment": flow.day.isoformat(),
                "real": flow.real_amount,
                "index_ratio": flow.index_ratio,
                "nominal": flow.nominal_amount,
                "projected": flow.projected,
                "redemption": flow.redemption,
            }
            for flow in bond.nominal_cashflows(
                index, reference, projection=arguments.projection
            )
        ]
    return result


def run_horizon(arguments: argparse.Namespace) -> dict[str, object]:
    """Carry and roll-down over a holding period.

    Reports both meanings of "carry" side by side because the market uses
    both, and a decomposition whose terms are ambiguous is worse than none.
    """
    built = build(arguments)
    reference = built.curve.reference
    bond = _bond(arguments, reference)
    horizon = date.fromisoformat(arguments.horizon)
    found = horizon_return(bond, built.curve, reference, horizon)
    return {
        "bond": bond.name,
        "settlement": reference.isoformat(),
        "horizon": horizon.isoformat(),
        "period_years": found.period,
        "start_price": found.start_price,
        "forward_price": found.forward_price,
        "rolled_price": found.rolled_price,
        "coupons": [
            {"day": one.day.isoformat(), "amount": one.amount, "at_horizon": one.value_at_horizon}
            for one in found.coupons
        ],
        "coupon_income": found.coupon_income,
        "forward_price_change": found.forward_price_change,
        "financing_rate": found.financing_rate,
        "financing_cost": found.financing_cost,
        "carry": found.carry,
        "income_less_financing": found.income_less_financing,
        "roll_down": found.roll_down,
        "total_return": found.total_return,
        "total_return_bps": found.total_return_bps,
        "excess_over_financing": found.excess_over_financing,
        "arbitrage_free": found.is_arbitrage_free(),
    }


def run_risk(arguments: argparse.Namespace) -> dict[str, object]:
    built = build(arguments)
    reference = built.curve.reference
    bond = _bond(arguments, reference)

    def value(curve: object) -> float:
        return bond.price_from_curve(curve, reference)  # type: ignore[arg-type]

    buckets = buckets_from(built.curve, [float(one) for one in arguments.buckets])
    parts = key_rates(value, built.curve, buckets)
    total = shape_duration(value, built.curve, level())
    return {
        "bond": bond.name,
        "total_duration": total,
        "key_rates": [
            {
                "bucket": one.bucket.name,
                "duration": one.duration,
                "value": one.value,
            }
            for one in parts
        ],
        "key_rate_sum": sum(one.duration for one in parts),
        "instruments": [
            {"quote": one.name, "value": one.value}
            for one in instrument_risk(value, built)
        ],
    }


def run_option(arguments: argparse.Namespace) -> dict[str, object]:
    built = build(arguments)
    reference = built.curve.reference
    bond = _bond(arguments, reference)
    step = 1.0 / Frequency[arguments.frequency.upper()].value
    steps = steps_between(built.curve, reference, bond.maturity, step)
    tree = Lattice.calibrated(
        built.curve,
        reference,
        steps=steps,
        step=step,
        volatility=arguments.volatility,
    )
    coupon = bond.periodic_coupon
    first = steps_between(built.curve, reference, date.fromisoformat(arguments.first_call), step)
    schedule = tuple((index, arguments.call_price) for index in range(first, steps))
    calls = Exercise(schedule, issuer=not arguments.put)

    bullet = lattice_price(tree, coupon=coupon)
    with_option = lattice_price(tree, coupon=coupon, exercise=calls)
    ignoring = option_adjusted_spread(tree, with_option, coupon=coupon)
    modelling = option_adjusted_spread(tree, with_option, coupon=coupon, exercise=calls)
    return {
        "bond": bond.name,
        "volatility": arguments.volatility,
        "steps": steps,
        "lattice_reprices_curve": tree.reprices(built.curve, reference, 1e-12),
        "bullet": bullet,
        "with_option": with_option,
        "option_value": bullet - with_option,
        "z_spread": ignoring.value,
        "oas": modelling.value,
        "option_cost": option_cost(ignoring.value, modelling.value),
    }


# -- the front door -----------------------------------------------------------


def parse_forecast_quotes(
    text: str,
    reference: date,
    index: ForecastIndex,
    *,
    flat_frequency: Frequency = Frequency.SEMI_ANNUAL,
) -> tuple[list[ForecastQuote], list[BasisSwap]]:
    """Parse quotes on the index being forecast, one per line.

    Returns the one-unknown quotes and the basis quotes separately, and exactly
    one of the two lists is ever non-empty, because mixing them is refused
    rather than resolved. A par swap rate is a statement about *this* index's
    curve; a tenor basis spread is a statement about a longer index's curve
    against this one taken as known. Both in one file asks for two curves from
    one solve, and the answer would be neither of them. They are returned apart
    rather than in one list because a basis swap needs three curves to price and
    the others need two, so they are not the same kind of thing.
    """
    quotes: list[ForecastQuote] = []
    spreads: list[BasisSwap] = []
    kinds: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        kind = fields[0].lower()
        try:
            if kind == "forward":
                basis = _basis(fields[4]) if len(fields) > 4 else index.basis
                quotes.append(
                    IndexForward(
                        date.fromisoformat(fields[1]),
                        date.fromisoformat(fields[2]),
                        float(fields[3]),
                        basis,
                    )
                )
            elif kind == "swap":
                quotes.append(
                    DualSwap(
                        effective=reference,
                        maturity=date.fromisoformat(fields[1]),
                        rate=float(fields[2]),
                        index=index,
                    )
                )
            elif kind == "basis":
                frequency = (
                    Frequency[fields[3].upper()] if len(fields) > 3 else flat_frequency
                )
                if frequency is index.tenor:
                    raise BadInput(
                        f"line {number}: the basis leg and the flat leg are both on "
                        f"the {index.tenor.months}-month tenor, so the quote says "
                        "nothing about either curve"
                    )
                spreads.append(
                    BasisSwap(
                        effective=reference,
                        maturity=date.fromisoformat(fields[1]),
                        spread=float(fields[2]),
                        spread_index=index,
                        flat_index=ForecastIndex(tenor=frequency, basis=index.basis),
                    )
                )
            else:
                raise BadInput(
                    f"line {number}: {fields[0]!r} is not a forecast quote. Known "
                    "kinds are forward, swap and basis."
                )
        except (IndexError, ValueError) as bad:
            if isinstance(bad, BadInput):
                raise
            raise BadInput(f"line {number}: {raw.strip()!r} - {bad}") from bad
        kinds.add(kind)
    if not quotes and not spreads:
        raise BadInput("the forecast quote file has no quotes in it")
    if "basis" in kinds and "swap" in kinds:
        raise BadInput(
            "the forecast file mixes par swap rates with tenor basis spreads. A "
            "swap rate pins this index's own curve and a basis spread pins a "
            "longer index's curve against it, so the two together ask one solve "
            "for two curves. Build the index curve from the swaps first, then the "
            "basis curve from the spreads against it."
        )
    return quotes, spreads


def run_multicurve(arguments: argparse.Namespace) -> dict[str, object]:
    reference = date.fromisoformat(arguments.reference)
    discounting = build(arguments)
    index = ForecastIndex(
        tenor=Frequency[arguments.index_tenor.upper()],
        basis=_basis(arguments.index_basis),
        payment_lag=arguments.payment_lag,
    )
    flat_frequency = Frequency[arguments.flat_frequency.upper()]
    quotes, basis_quotes = parse_forecast_quotes(
        Path(arguments.forecast).read_text(),
        reference,
        index,
        flat_frequency=flat_frequency,
    )
    short_leg_on = "itself"
    if basis_quotes:
        # A basis spread solves the *longer* tenor's curve against the short
        # one, which has to be known first -- and the only curve this command
        # has been given is the discount curve, so that is what the short leg
        # is projected off. It is an assumption rather than a quote, so it is
        # reported in the payload instead of being left for the reader to
        # infer from a number that looks like a basis and is partly a proxy.
        solving = ForecastIndex(tenor=flat_frequency, basis=index.basis)
        short_leg_on = "the discount curve, for want of a quote on it"
        # Any forward lines in a basis file are quotes on the curve being
        # solved for, so they go in alongside rather than on the floor: a
        # dropped quote is the one failure mode a repricing check cannot see,
        # because the curve reprices everything it was actually given.
        forecast = bootstrap_forecast(
            reference,
            [*quotes, *[FlatLegOf(one, discounting.curve) for one in basis_quotes]],
            discount=discounting.curve,
            basis=_basis(arguments.basis),
            interpolation=_interpolation(arguments.interpolation),
        )
    else:
        solving = index
        forecast = bootstrap_forecast(
            reference,
            quotes,
            discount=discounting.curve,
            basis=_basis(arguments.basis),
            interpolation=_interpolation(arguments.interpolation),
        )

    payload: dict[str, object] = {
        "reference": reference.isoformat(),
        "index": solving.name,
        "discount_curve": {
            "sweeps": discounting.sweeps,
            "worst_repricing_error": discounting.worst_error,
            "reprices": discounting.reprices(),
        },
        "forecast_curve": {
            "solving_for": f"{solving.tenor.months}m index",
            "quotes_used": len(forecast),
            "short_leg_projected_on": short_leg_on,
            "sweeps": forecast.sweeps,
            "worst_repricing_error": forecast.worst_error,
            "reprices": forecast.reprices(),
            "pillars": [
                {
                    "date": pillar.day.isoformat(),
                    "years": round(pillar.time, 6),
                    "discount": pillar.discount,
                }
                for pillar in forecast.curve.pillars[1:]
            ],
        },
    }

    months = solving.tenor.months
    spreads = []
    for pillar in forecast.curve.pillars[1:]:
        end = add_months(pillar.day, months, keep_end_of_month=True)
        if end > forecast.curve.pillars[-1].day:
            continue
        projected = forward_rate(forecast.curve, pillar.day, end, solving.basis)
        discounted = forward_rate(discounting.curve, pillar.day, end, solving.basis)
        spreads.append(
            {
                "start": pillar.day.isoformat(),
                "end": end.isoformat(),
                "projected": projected,
                "discounting": discounted,
                "basis_points": (projected - discounted) * 1e4,
            }
        )
    payload["forward_basis"] = spreads

    if arguments.maturity:
        maturity = date.fromisoformat(arguments.maturity)
        swap = DualSwap(
            effective=reference,
            maturity=maturity,
            rate=0.0,
            index=solving,
            frequency=Frequency[arguments.fixed_frequency.upper()],
            basis=_basis(arguments.fixed_basis),
        )
        dual = swap.par_rate(discounting.curve, forecast.curve)
        single = swap.par_rate(discounting.curve, discounting.curve)
        struck = DualSwap(
            effective=reference,
            maturity=maturity,
            rate=dual,
            index=solving,
            frequency=Frequency[arguments.fixed_frequency.upper()],
            basis=_basis(arguments.fixed_basis),
        )
        split = split_risk(struck.par_error, discounting.curve, forecast.curve)
        payload["swap"] = {
            "maturity": maturity.isoformat(),
            "par_rate": dual,
            "par_rate_on_one_curve": single,
            "basis_points_from_separating": (dual - single) * 1e4,
            "risk": {
                "from_the_discount_curve": split.discount,
                "from_the_forecast_curve": split.forecast,
                "from_both_together": split.joint,
                "cross_term": split.crossed,
            },
        }
    return payload


def _report(payload: dict[str, object], as_json: bool) -> str:
    if as_json:
        return json.dumps(payload, indent=2, default=str)
    return "\n".join(_lines(payload, ""))


def _lines(payload: dict[str, object], indent: str) -> list[str]:
    """Render a payload as text, recursing into nested mappings.

    Recursive rather than two levels deep by hand. A nested mapping used to fall
    through to ``str``, so the ``credit`` command's bond block printed as a
    Python dict on one line, complete with quotes and braces -- readable, in the
    sense that everything is in there somewhere.
    """
    lines: list[str] = []
    for key, item in payload.items():
        if isinstance(item, dict):
            lines.append(f"{indent}{key}:")
            lines.extend(_lines(item, indent + "  "))
        elif isinstance(item, list):
            lines.append(f"{indent}{key}:")
            for entry in item:
                if isinstance(entry, dict):
                    lines.append(
                        indent
                        + "  "
                        + "  ".join(f"{k}={_show(v)}" for k, v in entry.items())
                    )
                else:  # pragma: no cover - every list here holds dicts
                    lines.append(f"{indent}  {entry}")
        else:
            lines.append(f"{indent}{key}: {_show(item)}")
    return lines


def _show(item: object) -> str:
    if isinstance(item, float):
        return f"{item:.10g}"
    return str(item)


def _credit_quotes(text: str) -> list[tuple[date, float]]:
    """Parse ``maturity,spread`` rows, naming the line of anything malformed.

    Spreads are read in basis points, because that is how they are quoted and
    a file of decimals is one misplaced factor of ten from a curve that looks
    entirely plausible.
    """
    quotes: list[tuple[date, float]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [cell.strip() for cell in line.replace(",", " ").split()]
        if len(parts) != 2:
            raise ValueError(
                f"line {number}: {raw.strip()!r} has {len(parts)} fields, expected "
                "a maturity and a spread in basis points"
            )
        try:
            maturity = date.fromisoformat(parts[0])
        except ValueError as bad:
            raise ValueError(f"line {number}: {parts[0]!r} is not an ISO date") from bad
        try:
            spread = float(parts[1])
        except ValueError as bad:
            raise ValueError(f"line {number}: {parts[1]!r} is not a number") from bad
        quotes.append((maturity, spread / 1e4))
    if not quotes:
        raise ValueError("the spread file holds no quotes")
    return quotes


def run_credit(arguments: argparse.Namespace) -> dict[str, object]:
    """Strip a survival curve from par spreads and say what it implies.

    The credit triangle is printed beside every stripped hazard rate rather
    than instead of it. The rule of thumb is close enough to be useful and
    wrong by an amount that depends on the interest rate and the premium
    frequency rather than on the credit, which is not what most people expect
    of it, so showing both is more informative than showing either.
    """
    built = build(arguments)
    reference = built.curve.reference
    quotes = _credit_quotes(Path(arguments.spreads).read_text())
    frequency = Frequency[arguments.frequency.upper()]
    swaps = [
        CreditDefaultSwap.standard(
            reference,
            maturity,
            0.0,
            calendar=WEEKENDS_ONLY,
            basis=_basis(arguments.premium_basis),
            frequency=frequency,
        )
        for maturity, _ in quotes
    ]
    spreads = [spread for _, spread in quotes]
    curve = bootstrap_hazards(
        swaps,
        spreads,
        built.curve,
        recovery=arguments.recovery,
        basis=_basis(arguments.basis),
    )

    points = []
    worst = 0.0
    for pillar, swap, quoted in zip(curve.pillars, swaps, spreads, strict=True):
        repriced = swap.par_spread(curve, built.curve)
        worst = max(worst, abs(repriced - quoted))
        points.append(
            {
                "date": pillar.day.isoformat(),
                "years": round(pillar.time, 6),
                "quoted_bp": round(quoted * 1e4, 6),
                "repriced_bp": round(repriced * 1e4, 6),
                "forward_hazard": pillar.hazard,
                # The triangle produces a *flat* hazard to the maturity, so
                # the thing to compare it against is the average hazard to
                # that date, not the forward rate over the last bucket. At
                # the long end of an upward-sloping curve those differ by
                # half as much again, and putting the forward beside the
                # triangle invites exactly the wrong reading.
                "average_hazard": pillar.hazard
                if pillar.time <= 0.0
                else curve.integrated_hazard(pillar.time) / pillar.time,
                "triangle_hazard": triangle_hazard(quoted, arguments.recovery),
                "survival": curve.survival(pillar.day),
                "default_probability": curve.default_probability(pillar.day),
                "risky_annuity": swap.risky_annuity(curve, built.curve),
            }
        )

    result: dict[str, object] = {
        "reference": reference.isoformat(),
        "recovery": arguments.recovery,
        "worst_repricing_error_bp": worst * 1e4,
        "pillars": points,
    }
    if arguments.bond_maturity:
        bond = Bond(
            effective=reference,
            maturity=date.fromisoformat(arguments.bond_maturity),
            coupon=arguments.bond_coupon,
            frequency=frequency,
            basis=_basis(arguments.basis),
        )
        risky = risky_bond_price(bond, curve, built.curve)
        riskless = bond.price_from_curve(built.curve, reference)
        result["bond"] = {
            "maturity": bond.maturity.isoformat(),
            "coupon": arguments.bond_coupon,
            "riskless_price": riskless,
            "risky_price": risky,
            "credit_cost": riskless - risky,
        }
    return result


def _basket(text: str, *, bond_basis: Basis, frequency: Frequency) -> list[tuple[Bond, float]]:
    """Parse a deliverable basket: ``maturity, coupon, clean price[, label]``.

    Comma separated if the line has a comma, whitespace otherwise, and a label
    may carry spaces because it is whatever is left of the line. The coupon is a
    decimal and the price is per 100, both stated in the help rather than
    guessed at here: a basket file with coupons in per cent parses perfectly and
    produces conversion factors an order of magnitude out, so the error to avoid
    is the one that does not raise.
    """
    basket: list[tuple[Bond, float]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = (
            [cell.strip() for cell in line.split(",", 3)]
            if "," in line
            else line.split(None, 3)
        )
        if len(fields) < 3:
            raise ValueError(
                f"line {number}: {raw.strip()!r} has {len(fields)} fields, expected a "
                "maturity, a coupon and a clean price"
            )
        try:
            maturity = date.fromisoformat(fields[0])
        except ValueError as bad:
            raise ValueError(f"line {number}: {fields[0]!r} is not an ISO date") from bad
        try:
            coupon = float(fields[1])
            price = float(fields[2])
        except ValueError as bad:
            raise ValueError(
                f"line {number}: {fields[1]!r} and {fields[2]!r} are not both numbers"
            ) from bad
        if coupon > 1.0:
            raise ValueError(
                f"line {number}: a coupon of {coupon} is a percentage, not a decimal; "
                "write 0.0475 rather than 4.75"
            )
        label = fields[3].strip() if len(fields) > 3 else ""
        basket.append(
            (
                Bond(
                    effective=date(maturity.year - 40, maturity.month, maturity.day),
                    maturity=maturity,
                    coupon=coupon,
                    frequency=frequency,
                    basis=bond_basis,
                    accrual=Accrual.PERIOD_FRACTION,
                    label=label,
                ),
                price,
            )
        )
    if not basket:
        raise ValueError("the basket file holds no deliverable bonds")
    return basket


def run_futures(arguments: argparse.Namespace) -> dict[str, object]:
    """Price the basis of a deliverable bond future against a quoted basket.

    This is the one command that takes no curve. The whole calculation is the
    bond's own quoted price, its conversion factor and a money-market financing
    rate, and asking for a file of swap quotes to compute it would be asking for
    something it does not read. The switch walk prices the basket off a flat
    yield instead, which is stated in the output so that nobody reads those
    prices as the quoted ones.
    """
    bond_basis = _basis(arguments.bond_basis)
    frequency = Frequency[arguments.frequency.upper()]
    basket = _basket(
        Path(arguments.basket).read_text(), bond_basis=bond_basis, frequency=frequency
    )
    settlement = date.fromisoformat(arguments.settlement)
    future = BondFuture(
        first_delivery=date.fromisoformat(arguments.first_delivery),
        delivery=date.fromisoformat(arguments.delivery),
        notional_coupon=arguments.notional_coupon,
        digits=arguments.digits,
        money_basis=_basis(arguments.money_basis),
    )
    bonds = [bond for bond, _ in basket]
    prices = [price for _, price in basket]
    ranked = cheapest_to_deliver(
        bonds,
        future,
        clean_prices=prices,
        futures_price=arguments.price,
        settlement=settlement,
        repo=arguments.repo,
    )
    deliverables = [
        {
            "bond": entry.bond.name,
            "maturity": entry.bond.maturity.isoformat(),
            "coupon": entry.bond.coupon,
            "conversion_factor": entry.conversion_factor,
            "clean_price": entry.clean_price,
            "accrued_at_delivery": entry.accrued_at_delivery,
            "invoice_price": entry.invoice_price,
            "gross_basis": entry.gross_basis,
            "carry": entry.carry,
            "net_basis": entry.net_basis,
            "implied_repo": entry.implied_repo,
            "financed_balance": entry.financed_balance,
            "breakeven_futures_price": entry.breakeven_futures_price,
            "interim_coupons": [
                {"date": day.isoformat(), "amount": amount}
                for day, amount in entry.interim_coupons
            ],
        }
        for entry in ranked.basis
    ]
    result: dict[str, object] = {
        "contract": future.name,
        "first_delivery": future.first_delivery.isoformat(),
        "delivery": future.delivery.isoformat(),
        "settlement": settlement.isoformat(),
        "notional_coupon": future.notional_coupon,
        "quoted_price": arguments.price,
        "repo": arguments.repo,
        "deliverables": deliverables,
        "cheapest_to_deliver": {
            "by_implied_repo": ranked.by_implied_repo.bond.name,
            "by_net_basis": ranked.by_net_basis.bond.name,
            "by_futures_price": ranked.by_futures_price.bond.name,
            "unanimous": ranked.unanimous,
            # The price the basket justifies, against the price it is quoted at.
            # A quote below this is the delivery option being paid for; a quote
            # above it is an arbitrage or a stale price.
            "implied_futures_price": ranked.implied_futures_price,
            "quote_less_implied": arguments.price - ranked.implied_futures_price,
        },
    }
    if arguments.switch:
        levels = delivery_switch(
            bonds,
            future,
            settlement=settlement,
            repo=arguments.repo,
            yields=arguments.switch,
        )
        result["switch"] = {
            "priced_at": "a flat yield, not the quoted prices",
            "levels": [
                {
                    "yield": level.yield_level,
                    "cheapest": level.cheapest,
                    "runner_up": level.runner_up,
                    "futures_price": level.futures_price,
                    "margin": level.margin,
                }
                for level in levels
            ],
        }
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="tenor", description="Build a curve from quotes and measure things on it."
    )
    # Worth having for its own sake, and it doubles as the cheapest possible smoke
    # test of an install: it imports the package and prints something.
    root.add_argument("--version", action="version", version=f"tenor {__version__}")
    root.add_argument("--json", action="store_true", help="machine readable output")
    subcommands = root.add_subparsers(dest="command", required=True)

    def shared(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("quotes", help="a file of instrument quotes")
        sub.add_argument(
            "--reference", required=True, help="the curve's reference date, ISO"
        )
        sub.add_argument(
            "--basis",
            required=True,
            choices=[one.name for one in Basis],
            help="the day count the curve measures its own year fractions in",
        )
        sub.add_argument(
            "--interpolation",
            default=Interpolation.LOG_LINEAR_DISCOUNT.name,
            choices=[one.name for one in Interpolation],
        )

    def bond_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--maturity", required=True, help="the bond's maturity, ISO")
        sub.add_argument("--coupon", required=True, type=float, help="annual, as 0.05")
        sub.add_argument(
            "--frequency",
            default="SEMI_ANNUAL",
            choices=[one.name for one in Frequency],
        )
        sub.add_argument(
            "--bond-basis",
            default=Basis.THIRTY_360_BOND.name,
            choices=[one.name for one in Basis],
        )

    curve = subcommands.add_parser("curve", help="bootstrap and print the curve")
    shared(curve)
    curve.set_defaults(run=run_curve)

    price = subcommands.add_parser("price", help="price a bond and measure its risk")
    shared(price)
    bond_arguments(price)
    price.add_argument(
        "--quote", type=float, help="a clean price, to extract spreads against"
    )
    price.set_defaults(run=run_price)

    risk = subcommands.add_parser("risk", help="break the risk down two ways")
    shared(risk)
    bond_arguments(risk)
    risk.add_argument(
        "--buckets",
        nargs="+",
        required=True,
        help="key rate buckets, in years",
    )
    risk.set_defaults(run=run_risk)

    ahead = subcommands.add_parser(
        "horizon",
        help="carry and roll-down over a holding period",
        description=(
            "Splits a holding-period return into the part that is free and the "
            "part that is not. Carry — coupon income plus the forward price "
            "change — is identically the financing cost on an arbitrage-free "
            "curve, so the whole of the expected excess return is roll-down."
        ),
    )
    shared(ahead)
    bond_arguments(ahead)
    ahead.add_argument("--horizon", required=True, help="end of the holding period, ISO")
    ahead.set_defaults(run=run_horizon)

    floating = subcommands.add_parser(
        "floating",
        help="price a floating rate note and report both of its durations",
        description=(
            "A floater at its quoted margin is worth exactly par on any curve, and "
            "its rate duration is exactly zero. What it has instead is spread "
            "duration, over its whole life. Both are reported because reporting "
            "one as 'the duration' would report the wrong one."
        ),
    )
    shared(floating)
    floating.add_argument("--maturity", required=True, help="the note's maturity, ISO")
    floating.add_argument(
        "--quoted-margin",
        required=True,
        type=float,
        help="the spread over the index the note pays, as 0.0075 for 75bp",
    )
    floating.add_argument(
        "--issued",
        default=None,
        help="the note's original effective date, ISO. Defaults to the reference "
        "date, which makes settlement a reset date.",
    )
    floating.add_argument(
        "--frequency", default="QUARTERLY", choices=[one.name for one in Frequency]
    )
    floating.add_argument(
        "--note-basis", default=Basis.ACT_360.name, choices=[one.name for one in Basis]
    )
    floating.add_argument(
        "--margin",
        type=float,
        default=None,
        help="the discount margin to price at. Defaults to the quoted margin, "
        "which prices the note at par.",
    )
    floating.add_argument(
        "--quote",
        type=float,
        help="a clean price to solve the discount margin against instead",
    )
    floating.add_argument(
        "--fixing",
        type=float,
        default=None,
        help="the index rate set for the current period. Required in practice on "
        "any settlement date that is not a reset date, because the curve does not "
        "reach back to it.",
    )
    floating.add_argument(
        "--coupons",
        action="store_true",
        help="list every projected coupon with the rate and discount behind it",
    )
    floating.set_defaults(run=run_floating)

    linked = subcommands.add_parser(
        "linker",
        help="price an index-linked bond and split its accretion",
        description=(
            "An index-linked bond is quoted in real terms and settles in money, "
            "and the two differ by a ratio that is one at issue and drifts for "
            "the life of the bond. Everything here says which space it is in. "
            "The reference index carries the three-month lag, so part of the "
            "accretion over the coming year is already published; the report "
            "says how much."
        ),
    )
    shared(linked)
    linked.add_argument(
        "--index", required=True, help="a file of published index levels"
    )
    linked.add_argument("--maturity", required=True, help="the bond's maturity, ISO")
    linked.add_argument(
        "--coupon", required=True, type=float, help="the real annual rate, as 0.015"
    )
    linked.add_argument(
        "--base-index",
        required=True,
        type=float,
        help="the reference index at the bond's dated date, which every index "
        "ratio divides by. A property of the bond, fixed at issue.",
    )
    linked.add_argument(
        "--issued",
        default=None,
        help="the bond's dated date, ISO. Defaults to the reference date.",
    )
    linked.add_argument(
        "--frequency", default="SEMI_ANNUAL", choices=[one.name for one in Frequency]
    )
    linked.add_argument(
        "--linker-basis",
        default=Basis.ACT_ACT_ISDA.name,
        choices=[one.name for one in Basis],
    )
    real = linked.add_mutually_exclusive_group(required=True)
    real.add_argument(
        "--real-yield", type=float, help="price at this real yield, as 0.012"
    )
    real.add_argument(
        "--real-quote",
        type=float,
        help="a real clean price to solve the real yield against instead",
    )
    linked.add_argument(
        "--projection",
        required=True,
        type=float,
        help="annual inflation for every flow past the published index. Required "
        "rather than defaulted, because it is an assumption about the future "
        "price level and a default of zero would be a silent one.",
    )
    linked.add_argument(
        "--nominal-yield",
        type=float,
        default=None,
        help="a comparable nominal bond's yield, to report breakeven inflation "
        "against in both the exact and the quoted form",
    )
    linked.add_argument(
        "--no-floor",
        action="store_true",
        help="no deflation floor on the principal, as on an index-linked gilt. "
        "The default floors it at par in index terms, as on a US linker.",
    )
    linked.add_argument(
        "--flows",
        action="store_true",
        help="list every remaining flow in real and nominal terms",
    )
    linked.set_defaults(run=run_linker)

    option = subcommands.add_parser("option", help="value an embedded option")
    shared(option)
    bond_arguments(option)
    option.add_argument("--volatility", required=True, type=float)
    option.add_argument("--first-call", required=True, help="first exercise date, ISO")
    option.add_argument("--call-price", default=100.0, type=float)
    option.add_argument(
        "--put", action="store_true", help="the holder's option rather than the issuer's"
    )
    option.set_defaults(run=run_option)

    credit = subcommands.add_parser(
        "credit",
        help="strip a survival curve from credit default swap spreads",
        description=(
            "Bootstraps a piecewise-constant hazard rate from par spreads quoted "
            "in basis points, one per line as 'maturity,spread', and reports the "
            "survival probabilities it implies. The credit triangle's answer is "
            "printed beside each stripped rate, because the two differ by an "
            "amount that depends on the interest rate and the premium frequency "
            "rather than on the credit."
        ),
    )
    shared(credit)
    credit.add_argument(
        "--spreads", required=True, help="a file of maturity,spread-in-basis-points rows"
    )
    credit.add_argument(
        "--recovery", type=float, default=0.4, help="fraction of face recovered"
    )
    credit.add_argument(
        "--frequency",
        default="QUARTERLY",
        choices=[one.name for one in Frequency],
        help="premium frequency",
    )
    credit.add_argument(
        "--premium-basis",
        dest="premium_basis",
        default="ACT_360",
        choices=[one.name for one in Basis],
        help="day count the premium accrues on, which is not the curve's",
    )
    credit.add_argument(
        "--bond-maturity", dest="bond_maturity", default=None, help="also price a bond"
    )
    credit.add_argument("--bond-coupon", dest="bond_coupon", type=float, default=0.05)
    credit.set_defaults(run=run_credit)

    futures = subcommands.add_parser(
        "futures",
        help="price the basis of a deliverable bond future",
        description=(
            "Takes a basket file of 'maturity, coupon, clean price[, label]' rows "
            "and a quoted futures price, and reports every deliverable's "
            "conversion factor, gross basis, carry, net basis and implied repo "
            "rate, then the cheapest to deliver by all three of the conventional "
            "criteria. It takes no curve: the calculation is the quoted price, "
            "the factor and a money-market financing rate, and nothing else. "
            "Coupons are decimals -- 0.0475, not 4.75 -- and a value above one is "
            "refused rather than used, because a basket in per cent parses "
            "perfectly and is wrong by a factor of a hundred."
        ),
    )
    futures.add_argument("basket", help="a file of maturity,coupon,clean-price rows")
    futures.add_argument(
        "--settlement", required=True, help="when the cash bond is bought, ISO"
    )
    futures.add_argument(
        "--first-delivery",
        dest="first_delivery",
        required=True,
        help="first day of the delivery month, which is when the factor is computed",
    )
    futures.add_argument(
        "--delivery", required=True, help="the delivery date assumed, ISO"
    )
    futures.add_argument(
        "--price", required=True, type=float, help="the quoted futures price, per 100"
    )
    futures.add_argument(
        "--repo",
        required=True,
        type=float,
        help="the financing rate as a decimal, simple interest on the money basis",
    )
    futures.add_argument(
        "--notional-coupon",
        dest="notional_coupon",
        type=float,
        default=DEFAULT_NOTIONAL_COUPON,
        help="the yield the conversion factors are computed at",
    )
    futures.add_argument(
        "--digits",
        type=int,
        default=DEFAULT_DIGITS,
        help="published precision of the conversion factor",
    )
    futures.add_argument(
        "--money-basis",
        dest="money_basis",
        default=Basis.ACT_360.name,
        choices=[one.name for one in Basis],
        help="day count the financing accrues on, which is not the bond's",
    )
    futures.add_argument(
        "--bond-basis",
        dest="bond_basis",
        default=Basis.ACT_ACT_ISDA.name,
        choices=[one.name for one in Basis],
        help="day count the deliverables accrue on",
    )
    futures.add_argument(
        "--frequency",
        default=Frequency.SEMI_ANNUAL.name,
        choices=[one.name for one in Frequency],
        help="coupon frequency of the deliverables",
    )
    futures.add_argument(
        "--switch",
        nargs="+",
        type=float,
        default=None,
        metavar="YIELD",
        help=(
            "also walk these flat yields and report the cheapest bond at each, "
            "which is where the short's delivery option comes from"
        ),
    )
    futures.set_defaults(run=run_futures)

    multicurve = subcommands.add_parser(
        "multicurve",
        help="forecast on one curve and discount on another",
        description=(
            "Builds the discount curve from --quotes, then solves a projection "
            "curve out of --forecast against it, and reports the forward basis "
            "between the two. The order is the market's: discounting is settled "
            "first, from overnight-indexed swaps, and the forecast curve is built "
            "on top of it. With --maturity it also prices a par swap both ways "
            "and splits its risk between the curves, which is the number worth "
            "looking at: on a par rate the discount curve is worth a fraction of "
            "a basis point and the forecast curve is worth all of it."
        ),
    )
    shared(multicurve)
    multicurve.add_argument(
        "--forecast",
        required=True,
        help="a file of forward, swap or basis quotes on the index",
    )
    multicurve.add_argument(
        "--index-tenor",
        dest="index_tenor",
        default=Frequency.QUARTERLY.name,
        choices=[one.name for one in Frequency],
        help="reset tenor of the index being forecast",
    )
    multicurve.add_argument(
        "--index-basis",
        dest="index_basis",
        default=Basis.ACT_360.name,
        choices=[one.name for one in Basis],
        help="day count the index accrues on, which is not the fixed leg's",
    )
    multicurve.add_argument(
        "--payment-lag",
        dest="payment_lag",
        type=int,
        default=0,
        help=(
            "business days between a period's end and its payment. Zero is the "
            "money-market convention and the only value under which a leg "
            "forecast on its own discount curve telescopes"
        ),
    )
    multicurve.add_argument(
        "--flat-frequency",
        dest="flat_frequency",
        default=Frequency.SEMI_ANNUAL.name,
        choices=[one.name for one in Frequency],
        help="tenor of the flat leg of a basis quote, whose curve is solved for",
    )
    multicurve.add_argument(
        "--maturity",
        default=None,
        help="also price a par swap to this date and split its risk, ISO",
    )
    multicurve.add_argument(
        "--fixed-frequency",
        dest="fixed_frequency",
        default=Frequency.SEMI_ANNUAL.name,
        choices=[one.name for one in Frequency],
        help="payment frequency of that swap's fixed leg",
    )
    multicurve.add_argument(
        "--fixed-basis",
        dest="fixed_basis",
        default=Basis.THIRTY_360_BOND.name,
        choices=[one.name for one in Basis],
        help="day count that swap's fixed leg accrues on",
    )
    multicurve.set_defaults(run=run_multicurve)

    return root


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        payload = arguments.run(arguments)
    except (ValueError, OSError) as bad:
        print(f"{type(bad).__name__}: {bad}", file=sys.stderr)
        return 2
    try:
        print(_report(payload, arguments.json))
    except BrokenPipeError:  # pragma: no cover - needs a closed pipe to reach
        # Piping into head closes the pipe early. Reporting that as a traceback
        # makes a tool look broken for doing exactly what it was asked to.
        sys.stderr.close()
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
