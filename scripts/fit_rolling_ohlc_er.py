"""Research-only quarterly OHLC ER parameter search and checkpoint export.

This command is intentionally separate from ``run_daily.py``.  It reads a
research config, performs the walk-forward search, and writes a complete
strategy YAML that can be reviewed and committed before it is referenced as a
notification shadow config.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml

from data.store import query
from strategy.rolling_ohlc_er import resolve_current_weights


def _date(value: str) -> date:
    return date.today() if value.lower() in {"today", "latest"} else date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="research YAML")
    parser.add_argument("--template", type=Path, required=True, help="strategy YAML to update")
    parser.add_argument("--output", type=Path, required=True, help="versioned checkpoint YAML")
    parser.add_argument("--as-of", default="today", help="signal date, ISO date or today")
    parser.add_argument("--state-dir", type=Path, help="isolated research cache directory")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        research = yaml.safe_load(handle)
    with args.template.open(encoding="utf-8") as handle:
        output = yaml.safe_load(handle)

    as_of = _date(args.as_of)
    assets = list(research["asset_pool"])
    start = _date(str(research.get("start", "2013-07-01")))
    asset_data = {
        asset: query(asset, start, as_of)
        for asset in assets
    }
    state = resolve_current_weights(
        asset_data,
        assets,
        as_of,
        research["strategy_name"],
        research,
        args.state_dir,
    )
    weights = dict(zip(("close", "gap", "body", "range"), state.values, strict=True))
    output["factors"][0].setdefault("params", {})["weights"] = weights
    output["parameter_checkpoint"] = {
        "effective_date": state.effective_date.isoformat(),
        "training_start": state.training_start.isoformat(),
        "training_end": state.training_end.isoformat(),
        "history_days": int(research["history_days"]),
        "selection": "prior-quarter walk-forward search; mean of training-Sharpe Top10",
        "price_basis": "后复权 OHLC",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, allow_unicode=True, sort_keys=False)
    print(f"Wrote reviewed checkpoint: {args.output}")


if __name__ == "__main__":
    main()
