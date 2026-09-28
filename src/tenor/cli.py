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
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from . import __version__
from .bond import Bond
from .bootstrap import Bootstrapped, bootstrap
from .curve import Interpolation
from .daycount import Basis
from .floating import FloatingNote
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
from .rates import Compounding
from .risk import buckets_from, instrument_risk, key_rates, level, shape_duration
from .schedule import Frequency
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


def _report(payload: dict[str, object], as_json: bool) -> str:
    if as_json:
        return json.dumps(payload, indent=2, default=str)
    lines = []
    for key, item in payload.items():
        if isinstance(item, list):
            lines.append(f"{key}:")
            for entry in item:
                if isinstance(entry, dict):
                    lines.append(
                        "  " + "  ".join(f"{k}={_show(v)}" for k, v in entry.items())
                    )
                else:  # pragma: no cover - every list here holds dicts
                    lines.append(f"  {entry}")
        else:
            lines.append(f"{key}: {_show(item)}")
    return "\n".join(lines)


def _show(item: object) -> str:
    if isinstance(item, float):
        return f"{item:.10g}"
    return str(item)


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
