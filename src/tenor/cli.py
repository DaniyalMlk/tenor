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

from .bond import Bond
from .bootstrap import Bootstrapped, bootstrap
from .curve import Interpolation
from .daycount import Basis
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


def _bond(arguments: argparse.Namespace, reference: date) -> Bond:
    return Bond(
        reference,
        date.fromisoformat(arguments.maturity),
        arguments.coupon,
        Frequency[arguments.frequency.upper()],
        _basis(arguments.bond_basis),
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
