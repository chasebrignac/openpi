#!/usr/bin/env python3
"""Reject AWS runs that would exceed the reproduction's approved spend.

This is deliberately usable without boto3.  The durable ledger is a small JSON
file that launch wrappers update *before* they start a paid instance.  Actual
costs can later replace reservations without weakening the hard-cap check.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import sys
import uuid
from typing import Any


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
DEFAULT_LEDGER = pathlib.Path(".repro/cost-ledger.json")


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
    total_committed_usd: float
    remaining_after_usd: float


def load_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def committed_cost(ledger: dict[str, Any], *, category: str | None = None) -> float:
    entries = ledger.get("entries", [])
    return sum(
        float(entry["usd"])
        for entry in entries
        if entry.get("state", "reserved") not in {"cancelled", "superseded"}
        and (category is None or entry.get("category") == category)
    )


def project_run(
    config: dict[str, Any],
    ledger: dict[str, Any],
    *,
    category: str,
    instance_type: str,
    instance_count: int,
    hours: float,
) -> Projection:
    aws = config["aws"]
    approved = aws["approved_instances"]
    if instance_type not in approved:
        raise BudgetError(f"instance type is not approved: {instance_type}")
    if category not in aws["category_caps_usd"]:
        raise BudgetError(f"unknown budget category: {category}")
    if instance_count < 1 or hours <= 0:
        raise BudgetError("instance_count must be >= 1 and hours must be > 0")

    rate = float(approved[instance_type]["hourly_usd"])
    projected = rate * instance_count * hours
    category_committed = committed_cost(ledger, category=category)
    total_committed = committed_cost(ledger)
    category_cap = float(aws["category_caps_usd"][category])
    hard_cap = float(aws["hard_cap_usd"])

    if category_committed + projected > category_cap + 1e-9:
        raise BudgetError(
            f"category cap exceeded: {category_committed:.2f} + {projected:.2f} > {category_cap:.2f} USD"
        )
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
        total_committed_usd=total_committed,
        remaining_after_usd=hard_cap - total_committed - projected,
    )


def reserve(ledger_path: pathlib.Path, projection: Projection, *, label: str, command: str) -> str:
    ledger = load_json(ledger_path, {"schema_version": 1, "entries": []})
    reservation_id = str(uuid.uuid4())
    ledger["entries"].append(
        {
            "id": reservation_id,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "state": "reserved",
            "label": label,
            "command": command,
            "category": projection.category,
            "instance_type": projection.instance_type,
            "instance_count": projection.instance_count,
            "hours": projection.hours,
            "hourly_usd": projection.hourly_usd,
            "usd": round(projection.projected_usd, 6),
        }
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    temporary.replace(ledger_path)
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
