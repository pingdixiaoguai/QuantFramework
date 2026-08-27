"""Local parameter refinement around the first-pass Gold override winner."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from research.momentum_defender_gold_override import (
    build_gold_override_context,
    search_grid,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_gold_override_refinement.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260823_momentum_defender_gold_override/refinement_grid.csv"
)


def run_refinement(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    periods = {
        label: (date.fromisoformat(values[0]), date.fromisoformat(values[1]))
        for label, values in config["periods"].items()
    }
    context = build_gold_override_context(root)
    grid = search_grid(context, periods, config["grid"])
    selection = config["selection"]
    eligible = grid.loc[
        grid["gold_override_entries"].ge(
            int(selection["minimum_gold_override_entries"])
        )
        & grid["gold_override_days"].ge(
            int(selection["minimum_gold_override_days"])
        )
    ].copy()
    robust = eligible.sort_values(selection["primary_sort"], ascending=False).iloc[0]
    best_annual = eligible.sort_values(
        ["full_annualized_return_252", "full_sharpe"], ascending=False
    ).iloc[0]
    best_sharpe = eligible.sort_values(
        ["full_sharpe", "full_annualized_return_252"], ascending=False
    ).iloc[0]
    best_mdd = eligible.sort_values(
        ["full_max_drawdown", "full_sharpe"], ascending=False
    ).iloc[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values(selection["primary_sort"], ascending=False).to_csv(
        output, index=False
    )
    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(grid)),
        "eligible_candidate_count": int(len(eligible)),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "best_robust": robust.to_dict(),
        "best_annual": best_annual.to_dict(),
        "best_sharpe": best_sharpe.to_dict(),
        "best_mdd": best_mdd.to_dict(),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_refinement(args.root.resolve(), args.config, args.output)
    print(
        f"searched={summary['candidate_count']} "
        f"best_robust={summary['best_robust']['candidate_id']} "
        f"best_annual={summary['best_annual']['candidate_id']}"
    )


if __name__ == "__main__":
    main()
