"""Generate standard HTML reports for the requested 2019 combination."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from backtest.runner import run as run_backtest
from research.momentum_defender_occam import (
    _load_momentum_config,
    performance,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/w40_defender_qm_signed_exit_combination_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_w40_defender_qm_signed_exit_combination_2019"
)
LEGACY_CONFIG = Path(
    "strategy/configs/quality_momentum_top1_legacy_simple_price.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    end = date.fromisoformat(str(config["experiment"]["evidence_cutoff"]))
    daily_path = output / "factorial_daily_returns.parquet"
    daily = pd.read_parquet(daily_path)
    requested = daily["requested_all_three"].astype(float)
    current = daily["baseline"].astype(float)

    legacy_config = _load_momentum_config(root / LEGACY_CONFIG, end)
    legacy = run_backtest(legacy_config).daily_returns.reindex(requested.index)
    if legacy.isna().any():
        raise AssertionError("Original Momentum does not cover the report calendar")

    vs_current = output / "requested_combination_vs_current_v3.html"
    vs_original = output / "requested_combination_vs_original_momentum.html"
    generate_standard_report(
        requested,
        current,
        "Current Production v3",
        vs_current,
        config,
    )
    generate_standard_report(
        requested,
        legacy.astype(float),
        "Original Momentum (Simple MOM × Price ER)",
        vs_original,
        config,
    )

    reports = {}
    for path in (vs_current, vs_original):
        document = path.read_text(encoding="utf-8")
        if '<body onload="save()">' in document:
            raise AssertionError(f"dead QuantStats onload handler remains: {path}")
        if "EOY Returns" not in document or path.stat().st_size < 100_000:
            raise AssertionError(f"incomplete standard HTML report: {path}")
        reports[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    audit = {
        "status": "passed",
        "strategy": "requested_all_three",
        "start": requested.index[0].date().isoformat(),
        "end": requested.index[-1].date().isoformat(),
        "observations": int(len(requested)),
        "strategy_metrics": performance(requested),
        "current_v3_metrics": performance(current),
        "original_momentum_metrics": performance(legacy.astype(float)),
        "reports": reports,
    }
    (output / "html_report_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    result = generate(root, args.config, output)
    print(json.dumps(result["reports"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
