"""Research core for causal full-allocation Momentum/Defender rotation.

The accepted Defender handoff already provides exact net return segments for
continuous holding, fresh entry at the open, and fresh exit at the open.  This
module builds the same interface for the production Momentum sleeve and joins
the two with a small, explicit state machine.  It deliberately keeps signal
research separate from the production strategy path until a rule has passed
the declared robustness gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from backtest.runner import BacktestResult, run
from data.store import query


MOMENTUM_ASSETS = ("510300.SH", "159915.SZ", "513100.SH", "518880.SH")
MOMENTUM_COST_RATES = {asset: 0.0001 for asset in MOMENTUM_ASSETS}
DEFENDER_ASSETS = (
    "512890",
    "159545",
    "513530",
    "515080",
    "510880",
    "563020",
    "511260",
)
FORMAL_DEFENDER_ID = "relative_defender_rotation_anchor_100pct"
FORMAL_PRICE_ADJUSTMENT = "HFQ_FIXED_BASELINE"

HELD_RETURN = "daily_net_return_if_held"
ENTER_RETURN = "enter_open_to_close_net_return"
EXIT_RETURN = "exit_prev_close_to_open_net_return"
INTERNAL_COST = "internal_cost_rate_at_open"
ENTRY_COST = "fresh_entry_cost_rate_at_open"
EXIT_COST = "fresh_exit_cost_rate_at_open"


@dataclass(frozen=True)
class OccamParams:
    """Slow market regime plus an optional Momentum-loss control ablation."""

    lookback: int = 40
    risk_on_threshold: float = 0.025
    min_hold_days: int = 30
    emergency_daily_loss: float | None = None


@dataclass
class ResearchInputs:
    calendar: pd.DatetimeIndex
    momentum: pd.DataFrame
    defender: pd.DataFrame
    risk_close: pd.Series
    momentum_result: BacktestResult


@dataclass
class DefenderBundle:
    """The three files in the accepted Defender handoff, on one calendar."""

    daily_returns: pd.DataFrame
    indicators: pd.DataFrame
    switch_returns: pd.DataFrame
    audit: pd.DataFrame


def _load_momentum_config(path: Path, end: date) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("start", "end"):
        value = config.get(key)
        if isinstance(value, str):
            config[key] = date.today() if value.lower() == "today" else date.fromisoformat(value)
    config["end"] = end
    config["transaction_cost_rate"] = 0.0001
    return config


def _momentum_target_schedule(
    result: BacktestResult,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    positions = result.positions.copy().sort_index()
    for asset in MOMENTUM_ASSETS:
        if asset not in positions:
            positions[asset] = 0.0
    positions = positions[list(MOMENTUM_ASSETS)].fillna(0.0)
    targets = (
        positions.reindex(positions.index.union(calendar))
        .sort_index()
        .ffill()
        .reindex(calendar)
        .fillna(0.0)
    )
    totals = targets.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-12):
        bad = totals.loc[~np.isclose(totals, 1.0, atol=1e-12)]
        raise AssertionError(f"Momentum target is not fully invested: {bad.head().to_dict()}")
    positive = targets.gt(1e-14).sum(axis=1)
    if not positive.eq(1).all():
        bad = positive.loc[positive.ne(1)]
        raise AssertionError(f"Momentum Top1 target is not one-hot: {bad.head().to_dict()}")
    selected = targets.max(axis=1)
    if not np.allclose(selected, 1.0, atol=1e-12):
        raise AssertionError("Momentum Top1 selected weight is not 100%")
    return targets


def _load_prices(
    assets: tuple[str, ...],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for asset in assets:
        frame = query(asset, start, end).sort_values("date").drop_duplicates("date")
        if frame.empty:
            raise RuntimeError(f"no local prices for {asset}")
        indexed = frame.set_index("date")[["open", "close"]].astype(float)
        if indexed.isna().any().any():
            raise ValueError(f"missing stored Momentum OHLC for {asset}")
        prices[asset] = indexed
    return prices


def _weights_equal(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    assets = set(left) | set(right)
    return all(abs(left.get(asset, 0.0) - right.get(asset, 0.0)) <= 1e-12 for asset in assets)


def _execute_target(
    cash: float,
    shares: dict[str, float],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
    cost_rates: Mapping[str, float],
) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    """Sell then buy a target at the open with capital-aware one-way costs."""

    nav_open = cash + sum(shares.get(asset, 0.0) * open_prices[asset] for asset in shares)
    desired = {asset: nav_open * float(weight) for asset, weight in target.items()}
    executions: list[dict[str, float | str]] = []

    for asset in sorted(set(shares) | set(target)):
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        sell_value = max(0.0, current_value - desired.get(asset, 0.0))
        if sell_value <= 1e-14:
            continue
        rate = float(cost_rates[asset])
        shares[asset] = shares.get(asset, 0.0) - sell_value / open_prices[asset]
        cash += sell_value * (1.0 - rate)
        executions.append(
            {
                "asset": asset,
                "side": "sell",
                "notional": sell_value,
                "cost": sell_value * rate,
            }
        )

    needs: dict[str, float] = {}
    for asset, desired_value in desired.items():
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        needs[asset] = max(0.0, desired_value - current_value)
    cash_needed = sum(value * (1.0 + cost_rates[asset]) for asset, value in needs.items())
    scale = min(1.0, cash / cash_needed) if cash_needed > 0.0 else 0.0
    for asset in sorted(needs, key=lambda item: target[item], reverse=True):
        buy_value = needs[asset] * scale
        if buy_value <= 1e-14:
            continue
        rate = float(cost_rates[asset])
        shares[asset] = shares.get(asset, 0.0) + buy_value / open_prices[asset]
        cash -= buy_value * (1.0 + rate)
        executions.append(
            {
                "asset": asset,
                "side": "buy",
                "notional": buy_value,
                "cost": buy_value * rate,
            }
        )

    shares = {asset: quantity for asset, quantity in shares.items() if quantity > 1e-14}
    return cash, shares, executions


def build_momentum_interface(
    targets: pd.DataFrame,
    prices: Mapping[str, pd.DataFrame],
    cost_rates: Mapping[str, float] = MOMENTUM_COST_RATES,
) -> pd.DataFrame:
    """Return the exact held/entry/exit interface for a target schedule.

    Missing bars are treated as suspension marks only when the target is
    unchanged: the most recent close is carried for valuation and no trade is
    allowed in an asset without a current open.
    """

    calendar = pd.DatetimeIndex(targets.index)
    if calendar.duplicated().any() or not calendar.is_monotonic_increasing:
        raise ValueError("Momentum target calendar must be unique and increasing")
    assets = tuple(targets.columns)
    raw_open = {asset: prices[asset]["open"].reindex(calendar) for asset in assets}
    raw_close = {asset: prices[asset]["close"].reindex(calendar) for asset in assets}
    mark_open = {
        asset: raw_open[asset].combine_first(raw_close[asset].shift(1)).ffill()
        for asset in assets
    }
    mark_close = {asset: raw_close[asset].ffill() for asset in assets}

    cash = 1.0
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_close_nav = 1.0
    rows: list[dict[str, object]] = []

    for position, timestamp in enumerate(calendar):
        target = {
            asset: float(weight)
            for asset, weight in targets.loc[timestamp].items()
            if float(weight) > 1e-14
        }
        opens = {asset: float(mark_open[asset].loc[timestamp]) for asset in assets}
        closes = {asset: float(mark_close[asset].loc[timestamp]) for asset in assets}
        traded = {asset: pd.notna(raw_open[asset].loc[timestamp]) for asset in assets}
        quoted = {
            asset: traded[asset] and pd.notna(raw_close[asset].loc[timestamp])
            for asset in assets
        }
        required_quotes = set(target) | set(shares)
        unavailable_quotes = sorted(asset for asset in required_quotes if not quoted[asset])
        if unavailable_quotes:
            raise AssertionError(
                f"Momentum held/target assets lack OHLC on {timestamp.date()}: "
                f"{unavailable_quotes}"
            )
        shares_before = shares.copy()
        cash_before = cash
        open_nav_before = cash + sum(
            quantity * opens[asset] for asset, quantity in shares.items()
        )
        overnight = (
            open_nav_before / previous_close_nav - 1.0 if position > 0 else 0.0
        )

        target_changed = not _weights_equal(target, previous_target)
        if target_changed:
            changed_assets = {
                asset
                for asset in set(previous_target) | set(target)
                if abs(previous_target.get(asset, 0.0) - target.get(asset, 0.0)) > 1e-12
            }
            unavailable = sorted(asset for asset in changed_assets if not traded[asset])
            if unavailable:
                raise AssertionError(
                    f"cannot rebalance suspended Momentum assets on {timestamp.date()}: {unavailable}"
                )
            cash, shares, executions = _execute_target(
                cash, shares, target, opens, cost_rates
            )
            previous_target = target
        else:
            executions = []

        internal_cost = sum(float(item["cost"]) for item in executions)
        internal_cost_rate = internal_cost / open_nav_before if open_nav_before else 0.0
        post_open_nav = cash + sum(
            quantity * opens[asset] for asset, quantity in shares.items()
        )
        close_nav = cash + sum(
            quantity * closes[asset] for asset, quantity in shares.items()
        )
        intraday = close_nav / post_open_nav - 1.0
        held_net = close_nav / previous_close_nav - 1.0

        entry_cash, entry_shares, entry_executions = _execute_target(
            1.0, {}, target, opens, cost_rates
        )
        entry_post_open_nav = entry_cash + sum(
            quantity * opens[asset] for asset, quantity in entry_shares.items()
        )
        entry_close_nav = entry_cash + sum(
            quantity * closes[asset] for asset, quantity in entry_shares.items()
        )
        entry_cost = sum(float(item["cost"]) for item in entry_executions)
        entry_intraday = entry_close_nav / entry_post_open_nav - 1.0

        if position > 0:
            exit_cost = sum(
                quantity * opens[asset] * float(cost_rates[asset])
                for asset, quantity in shares_before.items()
            )
            exit_cost_rate = exit_cost / open_nav_before if open_nav_before else 0.0
            exit_net = (open_nav_before - exit_cost) / previous_close_nav - 1.0
        else:
            exit_cost_rate = np.nan
            exit_net = np.nan

        row: dict[str, object] = {
            "date": timestamp,
            "overnight_gross_return": overnight,
            "intraday_gross_return_if_held": intraday,
            INTERNAL_COST: internal_cost_rate,
            HELD_RETURN: held_net,
            "intraday_gross_return_if_entered": entry_intraday,
            ENTRY_COST: entry_cost,
            ENTER_RETURN: entry_close_nav - 1.0,
            EXIT_COST: exit_cost_rate,
            EXIT_RETURN: exit_net,
            "nav_if_held": close_nav,
        }
        for asset in assets:
            row[f"target_weight_{asset}"] = target.get(asset, 0.0)
        rows.append(row)
        previous_close_nav = close_nav

        # Guard against accidental mutation of the pre-open snapshot used by
        # the exit leg.  These variables are intentionally retained only for
        # the calculations above.
        del shares_before, cash_before

    frame = pd.DataFrame(rows).set_index("date")
    validate_sleeve_interface(frame, name="Momentum")
    return frame


def validate_sleeve_interface(frame: pd.DataFrame, name: str) -> None:
    required = {HELD_RETURN, ENTER_RETURN, EXIT_RETURN, INTERNAL_COST, ENTRY_COST, EXIT_COST}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} interface missing columns: {sorted(missing)}")
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} interface dates must be unique and increasing")
    if frame.empty:
        raise ValueError(f"{name} interface cannot be empty")
    for column in (HELD_RETURN, ENTER_RETURN, INTERNAL_COST, ENTRY_COST):
        if frame[column].isna().any():
            dates = frame.index[frame[column].isna()].tolist()[:5]
            raise ValueError(f"{name} interface has missing {column} on {dates}")
    for column in (EXIT_RETURN, EXIT_COST):
        missing_dates = frame.index[frame[column].isna()]
        allowed = len(missing_dates) == 0 or (
            len(missing_dates) == 1 and missing_dates[0] == frame.index[0]
        )
        if not allowed:
            raise ValueError(
                f"{name} interface has unexpected missing {column}: "
                f"{missing_dates.tolist()[:5]}"
            )
    numeric = frame[list(required)].to_numpy(float)
    if np.isinf(numeric).any():
        raise ValueError(f"{name} interface contains infinite values")
    for column in (INTERNAL_COST, ENTRY_COST, EXIT_COST):
        if (frame[column].dropna() < 0.0).any():
            raise ValueError(f"{name} interface has negative {column}")


def _indexed_csv(path: Path, parse_dates: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=list(parse_dates))
    if "date" not in frame:
        raise ValueError(f"{path.name} has no date column")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].head().tolist()
        raise ValueError(f"{path.name} has duplicate dates: {duplicates}")
    frame = frame.set_index("date")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} dates are not increasing")
    return frame


def _audit_row(
    check: str,
    actual: object,
    expected: object,
    tolerance: float | None,
    passed: bool,
    notes: str = "",
) -> dict[str, object]:
    return {
        "check": check,
        "actual": actual,
        "expected": expected,
        "tolerance": tolerance,
        "passed": bool(passed),
        "notes": notes,
    }


def load_defender_bundle(deliverable_dir: Path, end: date | None = None) -> DefenderBundle:
    """Load and mechanically reconcile the accepted three-file handoff.

    The checks are deliberately run on the complete export before an optional
    research cutoff is applied.  This prevents a truncated slice from hiding
    calendar or formula defects elsewhere in the delivered files.
    """

    daily = _indexed_csv(deliverable_dir / "relative_defender_rotation_daily_returns.csv")
    indicators = _indexed_csv(
        deliverable_dir / "relative_defender_rotation_daily_indicators.csv",
        ("date", "signal_observation_date", "signal_effective_next_open_date"),
    )
    switch = _indexed_csv(
        deliverable_dir / "relative_defender_rotation_switch_returns.csv",
        ("date", "signal_date", "execution_date"),
    )
    validate_sleeve_interface(switch, name="Defender")

    rows: list[dict[str, object]] = []
    same_calendar = daily.index.equals(indicators.index) and daily.index.equals(switch.index)
    rows.append(
        _audit_row(
            "three_file_calendar_identity",
            len(daily.index.intersection(indicators.index).intersection(switch.index)),
            len(daily),
            0.0,
            same_calendar,
        )
    )
    rows.append(
        _audit_row(
            "full_export_row_count",
            len(daily),
            1840,
            0.0,
            len(daily) == 1840,
        )
    )
    rows.append(
        _audit_row(
            "full_export_start",
            daily.index.min().date().isoformat(),
            "2019-01-18",
            None,
            daily.index.min() == pd.Timestamp("2019-01-18"),
        )
    )
    rows.append(
        _audit_row(
            "full_export_end",
            daily.index.max().date().isoformat(),
            "2026-08-20",
            None,
            daily.index.max() == pd.Timestamp("2026-08-20"),
        )
    )
    if not same_calendar:
        raise AssertionError("Defender handoff files do not share an identical calendar")

    def max_error(left: pd.Series, right: pd.Series) -> float:
        if not left.index.equals(right.index):
            raise AssertionError("audit series use different indexes")
        if not left.isna().equals(right.isna()):
            raise AssertionError("audit series have different missing-value patterns")
        difference = (left.astype(float) - right.astype(float)).dropna().abs()
        return float(difference.max()) if len(difference) else 0.0

    held_error = max_error(daily["daily_net_return"], switch[HELD_RETURN])
    rows.append(
        _audit_row(
            "daily_return_equals_switch_held",
            held_error,
            0.0,
            1e-12,
            held_error <= 1e-12,
        )
    )
    daily_nav_error = max_error(daily["nav"], switch["nav_if_held"])
    rows.append(
        _audit_row(
            "daily_nav_equals_switch_nav",
            daily_nav_error,
            0.0,
            1e-12,
            daily_nav_error <= 1e-12,
        )
    )
    indicator_return_error = max_error(indicators["daily_net_return"], daily["daily_net_return"])
    indicator_nav_error = max_error(indicators["nav"], daily["nav"])
    rows.append(
        _audit_row(
            "indicator_compact_fields_equal_daily",
            max(indicator_return_error, indicator_nav_error),
            0.0,
            1e-12,
            max(indicator_return_error, indicator_nav_error) <= 1e-12,
        )
    )

    held_reconstructed = (
        (1.0 + switch["overnight_gross_return"].astype(float))
        * (1.0 - switch[INTERNAL_COST].astype(float))
        * (1.0 + switch["intraday_gross_return_if_held"].astype(float))
        - 1.0
    )
    enter_reconstructed = (
        (1.0 - switch[ENTRY_COST].astype(float))
        * (1.0 + switch["intraday_gross_return_if_entered"].astype(float))
        - 1.0
    )
    exit_reconstructed = (
        (1.0 + switch["overnight_gross_return"].astype(float))
        * (1.0 - switch[EXIT_COST].astype(float))
        - 1.0
    )
    for column in (
        "overnight_gross_return",
        "intraday_gross_return_if_held",
        "intraday_gross_return_if_entered",
    ):
        if switch[column].isna().any():
            raise AssertionError(f"Defender handoff has unexpected missing {column}")
    for check, actual in (
        ("held_segment_formula", max_error(held_reconstructed, switch[HELD_RETURN])),
        ("entry_segment_formula", max_error(enter_reconstructed, switch[ENTER_RETURN])),
        ("exit_segment_formula", max_error(exit_reconstructed.dropna(), switch[EXIT_RETURN].dropna())),
    ):
        rows.append(_audit_row(check, actual, 0.0, 1e-12, actual <= 1e-12))

    nav_reconstructed = (1.0 + switch[HELD_RETURN].astype(float)).cumprod()
    nav_error = max_error(nav_reconstructed, switch["nav_if_held"])
    rows.append(
        _audit_row(
            "held_return_reconstructs_nav",
            nav_error,
            0.0,
            1e-12,
            nav_error <= 1e-12,
        )
    )

    weight_errors: list[float] = []
    for prefix, cash_column in (
        ("previous_closing_weight_", "previous_closing_cash_weight"),
        ("target_weight_", "target_cash_weight"),
        ("post_open_weight_", "post_open_cash_weight"),
        ("closing_weight_", "closing_cash_weight"),
    ):
        columns = [f"{prefix}{asset}" for asset in DEFENDER_ASSETS]
        missing = [column for column in columns + [cash_column] if column not in switch]
        if missing:
            raise ValueError(f"Defender switch interface lacks weight fields: {missing}")
        weights = switch[columns + [cash_column]].astype(float)
        if weights.isna().any().any():
            raise AssertionError(f"Defender handoff has missing weights in {prefix}")
        if (weights < -1e-12).any().any():
            raise AssertionError(f"Defender handoff has negative weights in {prefix}")
        total = weights.sum(axis=1, skipna=False)
        weight_errors.append(float((total - 1.0).abs().max()))
    weight_error = max(weight_errors)
    rows.append(
        _audit_row(
            "all_weight_sets_sum_to_one",
            weight_error,
            0.0,
            1e-12,
            weight_error <= 1e-12,
        )
    )
    target_cash_error = float(switch["target_cash_weight"].astype(float).abs().max())
    rows.append(
        _audit_row(
            "target_cash_weight_is_zero",
            target_cash_error,
            0.0,
            1e-12,
            target_cash_error <= 1e-12,
        )
    )
    expected_costs = {
        **{asset: 0.0001 for asset in DEFENDER_ASSETS if asset != "511260"},
        "511260": 0.00001,
    }
    cost_rate_error = 0.0
    for asset, expected_cost in expected_costs.items():
        column = f"transaction_cost_rate_{asset}"
        if switch[column].isna().any():
            raise AssertionError(f"Defender handoff has missing {column}")
        cost_rate_error = max(
            cost_rate_error,
            float((switch[column].astype(float) - expected_cost).abs().max()),
        )
    rows.append(
        _audit_row(
            "asset_transaction_cost_rates",
            cost_rate_error,
            0.0,
            1e-15,
            cost_rate_error <= 1e-15,
        )
    )

    strategy_ids = set(switch["formal_strategy_id"].dropna().astype(str))
    price_adjustments = set(switch["price_adjustment"].dropna().astype(str))
    rows.append(
        _audit_row(
            "formal_strategy_id",
            ",".join(sorted(strategy_ids)),
            FORMAL_DEFENDER_ID,
            None,
            strategy_ids == {FORMAL_DEFENDER_ID},
        )
    )
    rows.append(
        _audit_row(
            "price_adjustment",
            ",".join(sorted(price_adjustments)),
            FORMAL_PRICE_ADJUSTMENT,
            None,
            price_adjustments == {FORMAL_PRICE_ADJUSTMENT},
        )
    )
    execution_matches = pd.DatetimeIndex(switch["execution_date"]).equals(switch.index)
    rows.append(
        _audit_row(
            "date_equals_execution_date",
            int((pd.DatetimeIndex(switch["execution_date"]) != switch.index).sum()),
            0,
            0.0,
            execution_matches,
        )
    )

    effective = pd.DatetimeIndex(indicators["signal_effective_next_open_date"].dropna())
    expected_effective = indicators.index[1:]
    next_open_matches = effective.equals(expected_effective)
    rows.append(
        _audit_row(
            "indicator_effective_date_is_next_master_open",
            int(len(effective)),
            int(len(expected_effective)),
            0.0,
            next_open_matches,
        )
    )
    observation = pd.to_datetime(indicators["signal_observation_date"])
    observation_not_future = bool((observation <= indicators.index).all())
    effective_series = pd.to_datetime(indicators["signal_effective_next_open_date"])
    effective_after_row = bool(
        (effective_series.dropna() > indicators.index[:-1]).all()
    )
    switch_signal = pd.to_datetime(switch["signal_date"])
    switch_signal_not_future = bool(
        (switch_signal.dropna() < switch_signal.dropna().index).all()
    )
    expected_switch_signal = pd.Series(
        observation.iloc[:-1].to_numpy(),
        index=switch.index[1:],
    )
    actual_switch_signal = switch_signal.iloc[1:]
    prior_observation_matches = bool(
        actual_switch_signal.equals(expected_switch_signal)
    )
    rows.append(
        _audit_row(
            "signal_timing_is_causal",
            int(
                observation_not_future
                and effective_after_row
                and switch_signal_not_future
                and prior_observation_matches
            ),
            1,
            0.0,
            observation_not_future
            and effective_after_row
            and switch_signal_not_future
            and prior_observation_matches,
            "observation <= row < effective; switch signal equals prior row observation",
        )
    )

    october = switch.loc[pd.Timestamp("2021-10-22")]
    october_ok = (
        abs(float(october["target_weight_511260"]) - 1.0) <= 1e-12
        and all(
            abs(float(october[f"target_weight_{asset}"])) <= 1e-12
            for asset in DEFENDER_ASSETS
            if asset != "511260"
        )
        and abs(float(october["overnight_gross_return"]) - (-0.00063938)) <= 5e-9
        and abs(float(october["intraday_gross_return_if_held"]) - 0.00107800) <= 5e-9
        and abs(float(october[HELD_RETURN]) - 0.00043793) <= 5e-9
    )
    rows.append(
        _audit_row(
            "suspension_day_2021_10_22",
            float(october[HELD_RETURN]),
            0.00043793,
            5e-9,
            october_ok,
            "512890 signal anchor suspended; 511260 target held at 100%",
        )
    )

    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise AssertionError(f"Defender handoff audit failed: {failed}")

    if end is not None:
        cutoff = pd.Timestamp(end)
        daily = daily.loc[:cutoff].copy()
        indicators = indicators.loc[:cutoff].copy()
        switch = switch.loc[:cutoff].copy()
        if not (daily.index.equals(indicators.index) and daily.index.equals(switch.index)):
            raise AssertionError("Defender cutoff created inconsistent calendars")
    return DefenderBundle(daily, indicators, switch, audit)


def load_defender_interface(path: Path, end: date) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    frame = frame.loc[: pd.Timestamp(end)].copy()
    validate_sleeve_interface(frame, name="Defender")
    return frame


def build_inputs(
    root: Path,
    defender_switch_path: Path,
    end: date = date(2026, 8, 17),
) -> ResearchInputs:
    defender = load_defender_interface(defender_switch_path, end)
    return build_inputs_from_defender_interface(root, defender, end)


def build_inputs_from_defender_interface(
    root: Path,
    defender: pd.DataFrame,
    end: date,
    *,
    start: date | None = None,
) -> ResearchInputs:
    """Build aligned sleeve inputs from an in-memory Defender interface.

    Production integration uses this path so the composite strategy consumes
    the locally vendored Defender implementation directly instead of requiring
    a pre-generated CSV from another repository.  ``start`` intentionally
    slices only after interface validation, preserving the upstream strategy's
    full-history state while choosing the composite evaluation window.
    """
    defender = defender.copy().sort_index()
    validate_sleeve_interface(defender, name="Defender")
    defender = defender.loc[: pd.Timestamp(end)]
    if start is not None:
        defender = defender.loc[pd.Timestamp(start) :]
    validate_sleeve_interface(defender, name="Defender")
    calendar = pd.DatetimeIndex(defender.index)
    config = _load_momentum_config(
        root / "strategy/configs/quality_momentum_top1.yaml", end
    )
    momentum_result = run(config)
    prior_dates = momentum_result.daily_returns.index[
        momentum_result.daily_returns.index < calendar.min()
    ]
    if len(prior_dates) == 0:
        raise AssertionError("Momentum interface needs a prior trading day for warm start")
    warmup_date = prior_dates.max()
    replay_calendar = pd.DatetimeIndex([warmup_date]).append(calendar)
    targets = _momentum_target_schedule(momentum_result, replay_calendar)
    prices = _load_prices(MOMENTUM_ASSETS, date(2013, 1, 1), end)
    momentum = build_momentum_interface(targets, prices).reindex(calendar)
    if not momentum.index.equals(defender.index):
        raise AssertionError("Momentum and Defender interfaces do not share one calendar")
    risk_close = prices["510300.SH"]["close"]
    return ResearchInputs(calendar, momentum, defender, risk_close, momentum_result)


def indicator_at_effective_open(
    indicators: pd.DataFrame,
    column: str,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Map a close-known Defender signal to its declared next-open date."""

    required = {column, "signal_effective_next_open_date"}
    missing = required - set(indicators.columns)
    if missing:
        raise ValueError(f"Defender indicators missing columns: {sorted(missing)}")
    mapping = indicators[[column, "signal_effective_next_open_date"]].dropna(
        subset=["signal_effective_next_open_date"]
    )
    effective = pd.to_datetime(mapping["signal_effective_next_open_date"])
    in_scope = effective.isin(calendar)
    effective = effective.loc[in_scope]
    values = mapping.loc[in_scope, column]
    if effective.duplicated().any():
        duplicates = effective.loc[effective.duplicated()].head().tolist()
        raise ValueError(f"signal has duplicate effective open dates: {duplicates}")
    result = pd.Series(np.nan, index=calendar, dtype=object, name=column)
    result.loc[pd.DatetimeIndex(effective)] = values.to_numpy()
    return result


def volatility_cap_at_open(
    indicators: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cap_threshold: float = 1.0,
) -> pd.Series:
    """Return the frozen Defender volatility-cap alert at its eligible open."""

    cap = indicator_at_effective_open(indicators, "signal_volatility_cap", calendar)
    numeric = pd.to_numeric(cap, errors="coerce")
    return numeric.lt(cap_threshold).where(numeric.notna(), False).astype(bool).rename(
        "volatility_cap_active_at_open"
    )


def quantile_volatility_alert_at_open(
    indicators: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    quantile: float,
    min_periods: int = 20,
) -> pd.Series:
    """Rebuild cap alerts for a quantile sensitivity without duplicating halts.

    The expanding threshold is evaluated on unique anchor observations and is
    strictly lagged.  Rows added to the master calendar while the signal anchor
    is suspended therefore repeat the prior observation but do not count it a
    second time in the expanding distribution.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    required = {
        "signal_observation_date",
        "signal_effective_next_open_date",
        "signal_realized_volatility_20",
    }
    missing = required - set(indicators.columns)
    if missing:
        raise ValueError(f"Defender indicators missing columns: {sorted(missing)}")
    working = indicators.reset_index()[
        [
            "date",
            "signal_observation_date",
            "signal_effective_next_open_date",
            "signal_realized_volatility_20",
        ]
    ].copy()
    for _, group in working.groupby("signal_observation_date", dropna=False):
        if group["signal_realized_volatility_20"].nunique(dropna=False) > 1:
            raise AssertionError("repeated signal observation has inconsistent volatility")
    observations = (
        working.sort_values("date")
        .drop_duplicates("signal_observation_date", keep="last")
        .set_index("signal_observation_date")["signal_realized_volatility_20"]
        .astype(float)
    )
    threshold = observations.shift(1).expanding(min_periods=min_periods).quantile(quantile)
    alert_by_observation = observations.gt(threshold) & observations.notna() & threshold.notna()
    working["quantile_alert"] = working["signal_observation_date"].map(alert_by_observation)
    mapped = working.set_index("date")
    alert = indicator_at_effective_open(mapped, "quantile_alert", calendar)
    return alert.fillna(False).astype(bool).rename(
        f"volatility_q{quantile:.2f}_alert_at_open"
    )


def slow_regime_at_open(
    risk_close: pd.Series,
    calendar: pd.DatetimeIndex,
    lookback: int,
    threshold: float,
) -> pd.Series:
    """Compute a close-known trend gate and make it effective next open."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    close = risk_close.astype(float).sort_index()
    trailing = close / close.shift(lookback) - 1.0
    after_close = trailing.gt(threshold).where(trailing.notna())
    at_open = after_close.shift(1).reindex(calendar).ffill()
    return at_open.rename("slow_regime_at_open")


def apply_state_schedule(
    slow_at_open: pd.Series,
    emergency_at_open: pd.Series,
    calendar: pd.DatetimeIndex,
    min_hold_days: int,
    emergency_override: bool = True,
    initial_risk_on: bool = True,
) -> pd.DataFrame:
    """Apply minimum-hold hysteresis plus a risk-off emergency override."""

    if min_hold_days < 1:
        raise ValueError("min_hold_days must be positive")
    slow = slow_at_open.reindex(calendar)
    emergency = emergency_at_open.reindex(calendar).fillna(False).astype(bool)
    state = bool(initial_risk_on)
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        previous = state
        reason = "hold"
        if bool(emergency.loc[timestamp]):
            if state and (emergency_override or held_days >= min_hold_days):
                state = False
                held_days = 0
                reason = "emergency_exit"
            elif state:
                reason = "emergency_blocked_by_min_hold"
            else:
                reason = "emergency_hold"
        else:
            wanted = slow.loc[timestamp]
            if pd.notna(wanted) and bool(wanted) != state and held_days >= min_hold_days:
                state = bool(wanted)
                held_days = 0
                reason = "slow_regime_switch"
        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous,
                "state_reason": reason,
                "slow_signal_asof_previous_close": slow.loc[timestamp],
                "emergency_asof_previous_close": bool(emergency.loc[timestamp]),
                "held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def state_schedule(
    risk_close: pd.Series,
    momentum_held_returns: pd.Series,
    calendar: pd.DatetimeIndex,
    params: OccamParams,
    *,
    emergency_at_open: pd.Series | None = None,
    emergency_override: bool = True,
) -> pd.DataFrame:
    """Generate the open-effective sleeve state with no same-close execution."""

    if params.lookback < 1 or params.min_hold_days < 1:
        raise ValueError("lookback and min_hold_days must be positive")
    if params.emergency_daily_loss is not None and params.emergency_daily_loss >= 0.0:
        raise ValueError("emergency_daily_loss must be negative or None")

    slow_at_open = slow_regime_at_open(
        risk_close,
        calendar,
        params.lookback,
        params.risk_on_threshold,
    )
    if params.emergency_daily_loss is None:
        loss_emergency = pd.Series(False, index=calendar)
    else:
        loss_emergency = (
            momentum_held_returns.le(params.emergency_daily_loss)
            .shift(1)
            .reindex(calendar)
            .fillna(False)
        )
    external = (
        pd.Series(False, index=calendar)
        if emergency_at_open is None
        else emergency_at_open.reindex(calendar).fillna(False).astype(bool)
    )
    combined = loss_emergency.astype(bool) | external
    return apply_state_schedule(
        slow_at_open,
        combined,
        calendar,
        params.min_hold_days,
        emergency_override=emergency_override,
    )


def _chain(left: float, right: float) -> float:
    return (1.0 + float(left)) * (1.0 + float(right)) - 1.0


def scale_interface_costs(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Apply a proportional cost sensitivity to an existing sleeve interface.

    Gross overnight/intraday legs are held fixed.  The selected 1x result does
    not use this approximation; it consumes the exact supplied/replayed net
    segments.  This helper is only for deliberately conservative stress tests.
    """

    if multiplier < 0.0:
        raise ValueError("cost multiplier cannot be negative")
    required = {
        "overnight_gross_return",
        "intraday_gross_return_if_held",
        "intraday_gross_return_if_entered",
        INTERNAL_COST,
        ENTRY_COST,
        EXIT_COST,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cost stress interface missing columns: {sorted(missing)}")
    stressed = frame.copy()
    internal = stressed[INTERNAL_COST].astype(float) * multiplier
    entry = stressed[ENTRY_COST].astype(float) * multiplier
    exit_ = stressed[EXIT_COST].astype(float) * multiplier
    if (
        internal.dropna().ge(1.0).any()
        or entry.dropna().ge(1.0).any()
        or exit_.dropna().ge(1.0).any()
    ):
        raise ValueError("stressed cost rate must remain below 100%")
    stressed[INTERNAL_COST] = internal
    stressed[ENTRY_COST] = entry
    stressed[EXIT_COST] = exit_
    stressed[HELD_RETURN] = (
        (1.0 + stressed["overnight_gross_return"].astype(float))
        * (1.0 - internal)
        * (1.0 + stressed["intraday_gross_return_if_held"].astype(float))
        - 1.0
    )
    stressed[ENTER_RETURN] = (
        (1.0 - entry)
        * (1.0 + stressed["intraday_gross_return_if_entered"].astype(float))
        - 1.0
    )
    stressed[EXIT_RETURN] = (
        (1.0 + stressed["overnight_gross_return"].astype(float))
        * (1.0 - exit_)
        - 1.0
    )
    validate_sleeve_interface(stressed, name="cost-stressed sleeve")
    return stressed


def simulate_switch(
    momentum: pd.DataFrame,
    defender: pd.DataFrame,
    risk_on: pd.Series,
    initial_previous_state: str | None = "momentum",
) -> pd.DataFrame:
    """Compose two sleeve interfaces using exact open-switch state semantics."""

    if not momentum.index.equals(defender.index):
        raise ValueError("sleeve interfaces must have identical calendars")
    calendar = momentum.index
    state = risk_on.reindex(calendar)
    if state.isna().any():
        raise ValueError("risk state does not cover the sleeve calendar")

    rows: list[dict[str, object]] = []
    if initial_previous_state not in {None, "momentum", "defender"}:
        raise ValueError("initial_previous_state must be momentum, defender, or None")
    previous: bool | None = (
        None if initial_previous_state is None else initial_previous_state == "momentum"
    )
    nav = 1.0
    for timestamp in calendar:
        current = bool(state.loc[timestamp])
        held_leg = np.nan
        exit_leg = np.nan
        enter_leg = np.nan
        if previous is None:
            source = momentum if current else defender
            enter_leg = float(source.at[timestamp, ENTER_RETURN])
            daily_return = enter_leg
            transition = "cash_to_momentum" if current else "cash_to_defender"
            cost_rate = float(source.at[timestamp, ENTRY_COST])
        elif previous == current:
            source = momentum if current else defender
            held_leg = float(source.at[timestamp, HELD_RETURN])
            daily_return = held_leg
            transition = "momentum_hold" if current else "defender_hold"
            cost_rate = float(source.at[timestamp, INTERNAL_COST])
        elif previous and not current:
            exit_leg = float(momentum.at[timestamp, EXIT_RETURN])
            enter_leg = float(defender.at[timestamp, ENTER_RETURN])
            daily_return = _chain(exit_leg, enter_leg)
            transition = "momentum_to_defender"
            exit_cost_rate = float(momentum.at[timestamp, EXIT_COST])
            entry_cost_rate = float(defender.at[timestamp, ENTRY_COST])
            cost_rate = 1.0 - (1.0 - exit_cost_rate) * (1.0 - entry_cost_rate)
        else:
            exit_leg = float(defender.at[timestamp, EXIT_RETURN])
            enter_leg = float(momentum.at[timestamp, ENTER_RETURN])
            daily_return = _chain(exit_leg, enter_leg)
            transition = "defender_to_momentum"
            exit_cost_rate = float(defender.at[timestamp, EXIT_COST])
            entry_cost_rate = float(momentum.at[timestamp, ENTRY_COST])
            cost_rate = 1.0 - (1.0 - exit_cost_rate) * (1.0 - entry_cost_rate)
        if not np.isfinite(daily_return) or daily_return <= -1.0:
            raise ValueError(f"invalid composed return on {timestamp.date()}: {daily_return}")
        nav *= 1.0 + daily_return
        rows.append(
            {
                "date": timestamp,
                "return": daily_return,
                "nav": nav,
                "risk_on": current,
                "sleeve": "momentum" if current else "defender",
                "transition": transition,
                "sleeve_switch": previous is not None and previous != current,
                "cost_rate_at_open": cost_rate,
                "held_return_leg_used": held_leg,
                "exit_return_leg_used": exit_leg,
                "enter_return_leg_used": enter_leg,
            }
        )
        previous = current
    return pd.DataFrame(rows).set_index("date")


def performance(returns: pd.Series) -> dict[str, float | int | str]:
    values = returns.astype(float)
    if values.isna().any():
        dates = values.index[values.isna()].tolist()[:5]
        raise ValueError(f"performance input contains missing returns on {dates}")
    if len(values) < 2:
        raise ValueError("performance requires at least two returns")
    curve = (1.0 + values).cumprod()
    anchored_curve = np.concatenate(([1.0], curve.to_numpy(float)))
    drawdown = anchored_curve / np.maximum.accumulate(anchored_curve) - 1.0
    volatility = float(values.std(ddof=1))
    # The first return spans the close before the first label through the first
    # label's close.  Inclusive day count avoids dropping that first interval.
    years = ((values.index[-1] - values.index[0]).days + 1) / 365.2425
    return {
        "start": values.index[0].date().isoformat(),
        "end": values.index[-1].date().isoformat(),
        "observations": int(len(values)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "cagr_calendar": float(curve.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_return_252": float(curve.iloc[-1] ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": float(volatility * np.sqrt(252.0)),
        "sharpe": float(values.mean() / volatility * np.sqrt(252.0)) if volatility else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
