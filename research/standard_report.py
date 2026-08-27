"""Project-standard QuantStats report adapter for research strategies."""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from backtest.report import generate as generate_backtest_report
from backtest.runner import BacktestResult


def _annual_returns(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    annual = frame.groupby(frame.index.year).apply(
        lambda sample: (1.0 + sample).prod() - 1.0,
        include_groups=False,
    )
    annual.index = annual.index.astype(int)
    return annual


def _corrected_eoy_svg(strategy: pd.Series, benchmark: pd.Series) -> str:
    annual = _annual_returns(strategy, benchmark)
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
        annual["benchmark"],
        width,
        color="#F9CF70",
        label="benchmark",
    )
    axis.bar(
        positions + width / 2,
        annual["strategy"],
        width,
        color="#348DC1",
        label="Strategy",
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(
        float(annual["strategy"].mean()),
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


def _repair_eoy_chart(
    output_path: Path,
    strategy: pd.Series,
    benchmark: pd.Series,
) -> None:
    document = output_path.read_text(encoding="utf-8")
    blocks = re.findall(r"<svg\b[^>]*>.*?</svg>", document, flags=re.DOTALL)
    targets = [block for block in blocks if "EOY Returns" in block]
    if len(targets) != 1:
        raise AssertionError(f"expected one QuantStats EOY SVG, found {len(targets)}")
    replacement = _corrected_eoy_svg(strategy, benchmark)
    output_path.write_text(
        document.replace(targets[0], replacement, 1), encoding="utf-8"
    )


def generate_standard_report(
    returns: pd.Series,
    benchmark: pd.Series,
    benchmark_name: str,
    output_path: Path,
    config: dict[str, object],
) -> Path:
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) != len(returns):
        raise ValueError(
            f"standard report lost observations after aligning {benchmark_name}: "
            f"{len(aligned)} != {len(returns)}"
        )
    result = BacktestResult(
        daily_returns=aligned["strategy"],
        benchmark_returns=aligned["benchmark"],
        positions=pd.DataFrame(index=aligned.index),
        train_end=date(2024, 12, 31),
        config=config,
        baseline_strategy_name=benchmark_name,
    )
    generated = generate_backtest_report(
        result,
        output_path,
        benchmark_title=benchmark_name,
    )
    _repair_eoy_chart(
        generated,
        aligned["strategy"],
        aligned["benchmark"],
    )
    return generated
