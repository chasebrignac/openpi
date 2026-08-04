#!/usr/bin/env python3
"""Reject AWS runs that would exceed the reproduction's approved spend.

This is deliberately usable without boto3.  The durable ledger is a small JSON
file that launch wrappers update *before* they start a paid instance.  Actual
costs can later replace reservations without weakening the hard-cap check.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import json
import math
import pathlib
import sys
import tempfile
from typing import Any
import uuid

DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
DEFAULT_LEDGER = pathlib.Path(".repro/cost-ledger.json")
UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- direct-script compatibility with macOS Python 3.9.


class BudgetError(RuntimeError):
    """Raised when a proposed paid run violates a reproduction constraint."""


@dataclasses.dataclass(frozen=True)
class Projection:
    category: str
    instance_type: str
    instance_count: int
    hours: float
    hourly_usd: float
    projected_usd: float
    category_committed_usd: float
    non_compute_reserved_usd: float
    total_committed_usd: float
    remaining_after_usd: float


NON_COMMITTED_STATES = frozenset({"cancelled", "superseded"})


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BudgetError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise BudgetError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive and finite" if positive else "finite and non-negative"
        raise BudgetError(f"{label} must be {qualifier}")
    return result


def validate_ledger_entries(ledger: Any) -> list[dict[str, Any]]:
    """Validate all values that contribute to a budget decision.

    Unknown states remain committed; only the two explicit terminal bookkeeping
    states are excluded. Invalid or non-finite costs fail closed rather than
    poisoning comparisons with ``NaN``.
    """
    if not isinstance(ledger, dict) or not isinstance(ledger.get("entries"), list):
        raise BudgetError("cost ledger must be an object with an entries list")
    entries = ledger["entries"]
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise BudgetError(f"cost ledger entry {index} must be an object")
        state = entry.get("state", "reserved")
        if not isinstance(state, str) or not state:
            raise BudgetError(f"cost ledger entry {index} has an invalid state")
        category = entry.get("category")
        if not isinstance(category, str) or not category:
            raise BudgetError(f"cost ledger entry {index} has an invalid category")
        _finite_number(entry.get("usd"), f"cost ledger entry {index} usd")
    return entries


def load_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def committed_cost(ledger: dict[str, Any], *, category: str | None = None) -> float:
    if category is not None and (not isinstance(category, str) or not category):
        raise BudgetError("cost category must be a non-empty string")
    entries = validate_ledger_entries(ledger)
    try:
        total = math.fsum(
            _finite_number(entry["usd"], "ledger usd")
            for entry in entries
            if entry.get("state", "reserved") not in NON_COMMITTED_STATES
            and (category is None or entry["category"] == category)
        )
    except OverflowError as exc:
        raise BudgetError("committed cost is not finite") from exc
    if not math.isfinite(total):
        raise BudgetError("committed cost is not finite")
    return total


def project_run(
    config: dict[str, Any],
    ledger: dict[str, Any],
    *,
    category: str,
    instance_type: str,
    instance_count: int,
    hours: float,
) -> Projection:
    if not isinstance(config, dict) or not isinstance(config.get("aws"), dict):
        raise BudgetError("config must contain an aws object")
    aws = config["aws"]
    approved = aws["approved_instances"]
    if instance_type not in approved:
        raise BudgetError(f"instance type is not approved: {instance_type}")
    if category not in aws["category_caps_usd"]:
        raise BudgetError(f"unknown budget category: {category}")
    if isinstance(instance_count, bool) or not isinstance(instance_count, int) or instance_count < 1:
        raise BudgetError("instance_count must be an integer >= 1")
    hours = _finite_number(hours, "hours", positive=True)

    rate = _finite_number(approved[instance_type]["hourly_usd"], "approved hourly rate", positive=True)
    projected = rate * instance_count * hours
    if not math.isfinite(projected):
        raise BudgetError("projected cost is not finite")
    category_committed = committed_cost(ledger, category=category)
    # Storage/logging and explicit headroom are withheld from EC2 launch
    # capacity up front. This keeps delayed gp3/S3/ECR/log charges inside the
    # project cap even though AWS does not expose them synchronously.
    hard_cap = _finite_number(aws["hard_cap_usd"], "hard cap")
    non_compute_reserved = _finite_number(
        aws.get("hard_cap_non_compute_reserve_usd", 0.0),
        "hard_cap_non_compute_reserve_usd",
    )
    if non_compute_reserved > hard_cap:
        raise BudgetError("hard_cap_non_compute_reserve_usd must be within the hard cap")
    total_committed = committed_cost(ledger) + non_compute_reserved
    if not math.isfinite(total_committed):
        raise BudgetError("total committed cost is not finite")
    category_cap = _finite_number(aws["category_caps_usd"][category], f"category cap for {category}")

    if category_committed + projected > category_cap + 1e-9:
        raise BudgetError(f"category cap exceeded: {category_committed:.2f} + {projected:.2f} > {category_cap:.2f} USD")
    if total_committed + projected > hard_cap + 1e-9:
        raise BudgetError(f"hard cap exceeded: {total_committed:.2f} + {projected:.2f} > {hard_cap:.2f} USD")

    return Projection(
        category=category,
        instance_type=instance_type,
        instance_count=instance_count,
        hours=hours,
        hourly_usd=rate,
        projected_usd=projected,
        category_committed_usd=category_committed,
        non_compute_reserved_usd=non_compute_reserved,
        total_committed_usd=total_committed,
        remaining_after_usd=hard_cap - total_committed - projected,
    )


@contextlib.contextmanager
def _locked_ledger(ledger_path: pathlib.Path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reserve(ledger_path: pathlib.Path, projection: Projection, *, label: str, command: str) -> str:
    """Atomically append a local reservation only if its projection is still current."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_ledger(ledger_path):
        ledger = load_json(ledger_path, {"schema_version": 1, "entries": []})
        category_committed = committed_cost(ledger, category=projection.category)
        total_committed = committed_cost(ledger) + _finite_number(
            projection.non_compute_reserved_usd,
            "projection non-compute reserve",
        )
        if not math.isclose(category_committed, projection.category_committed_usd, rel_tol=0.0, abs_tol=1e-9) or not (
            math.isclose(total_committed, projection.total_committed_usd, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise BudgetError("cost ledger changed after projection; re-project before reserving")
        reservation_id = str(uuid.uuid4())
        ledger["entries"].append(
            {
                "id": reservation_id,
                "created_at": dt.datetime.now(UTC).isoformat(),
                "state": "reserved",
                "label": label,
                "command": command,
                "category": projection.category,
                "instance_type": projection.instance_type,
                "instance_count": projection.instance_count,
                "hours": projection.hours,
                "hourly_usd": projection.hourly_usd,
                "usd": projection.projected_usd,
            }
        )
        _atomic_write_json(ledger_path, ledger)
        return reservation_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger", type=pathlib.Path, default=DEFAULT_LEDGER)
    parser.add_argument("--category", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--label", default="unnamed-run")
    parser.add_argument("--command", default="")
    parser.add_argument("--reserve", action="store_true", help="write a reservation after the check passes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config, None)
        if config is None:
            raise BudgetError(f"missing config: {args.config}")
        ledger = load_json(args.ledger, {"schema_version": 1, "entries": []})
        projection = project_run(
            config,
            ledger,
            category=args.category,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            hours=args.hours,
        )
        result: dict[str, Any] = dataclasses.asdict(projection)
        if args.reserve:
            result["reservation_id"] = reserve(args.ledger, projection, label=args.label, command=args.command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (BudgetError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"COST GUARD REJECTED RUN: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
