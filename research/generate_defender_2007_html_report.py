"""Generate the standard QuantStats report and repair only its EOY SVG."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

from research.standard_report import generate_standard_report


DEFAULT_RETURNS = Path(
    "experiments/20260826_momentum_defender_dividend_universe_2007_validation/"
    "standalone_returns.parquet"
)
DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_dividend_universe_2007_validation.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_dividend_universe_2007_validation/"
    "fixed_candidate_vs_original_defender_2007_to_2026.html"
)


def _annual_returns(returns: pd.DataFrame) -> pd.DataFrame:
    annual = returns.groupby(returns.index.year).apply(
        lambda sample: (1.0 + sample).prod() - 1.0,
        include_groups=False,
    )
    annual.index = annual.index.astype(int)
    return annual


def _corrected_eoy_svg(returns: pd.DataFrame) -> str:
    annual = _annual_returns(returns)
    years = annual.index.to_numpy(int)
    positions = np.arange(len(years), dtype=float)
    width = 0.34
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar(
        positions - width / 2,
        annual["baseline"],
        width,
        color="#F9CF70",
        label="benchmark",
    )
    axis.bar(
        positions + width / 2,
        annual["fixed_candidate"],
        width,
        color="#348DC1",
        label="Strategy",
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(
        float(annual["fixed_candidate"].mean()),
        color="red",
        linestyle="--",
        linewidth=1.2,
    )
    axis.set_title("EOY Returns  vs Benchmark", fontsize=16, fontweight="bold", pad=18)
    axis.set_xticks(positions, [str(year) for year in years], rotation=35, ha="right")
    axis.set_xlim(-0.7, len(years) - 0.3)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="both", color="#d9d9d9", linewidth=0.6, alpha=0.85)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index("<svg") :]


def _replace_eoy_svg(report: Path, replacement: str) -> None:
    document = report.read_text(encoding="utf-8")
    blocks = re.findall(r"<svg\b[^>]*>.*?</svg>", document, flags=re.DOTALL)
    targets = [block for block in blocks if "EOY Returns" in block]
    if len(targets) != 1:
        raise AssertionError(f"expected one QuantStats EOY SVG, found {len(targets)}")
    repaired = document.replace(targets[0], replacement, 1)
    if repaired == document:
        raise AssertionError("EOY SVG replacement did not modify the report")
    report.write_text(repaired, encoding="utf-8")


def generate(returns_path: Path, config_path: Path, output_path: Path) -> Path:
    returns = pd.read_parquet(returns_path)[["fixed_candidate", "baseline"]].astype(float)
    if returns.isna().any().any() or not returns.index.is_monotonic_increasing:
        raise ValueError("standalone returns must be complete and ordered")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generate_standard_report(
        returns["fixed_candidate"],
        returns["baseline"],
        "original_six_etf_defender_pool_2007_to_2026",
        output_path,
        config,
    )
    _replace_eoy_svg(output_path, _corrected_eoy_svg(returns))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", type=Path, default=DEFAULT_RETURNS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.returns, args.config, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
