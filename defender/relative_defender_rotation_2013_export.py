"""Integration-ready CSV exports for the listing-aware 2013 research version."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .defender_opt_v2 import (
    _asof_price,
    _asset_cost_rate,
    _execute_portfolio_target,
    _indexed_market,
)
from .grid_reproduction import INITIAL_CAPITAL
from .relative_defender_rotation import (
    BASE_PRIMARY_ASSET,
    DEFENSIVE_ASSET,
    ROTATION_COST_RATES,
    rotation_params,
)
from .relative_defender_rotation_2013_report import (
    BRIDGE_SIGNAL_ASSET,
    PROMOTION_DATE,
    START_DATE,
    STRATEGY_ID,
    _clean_market,
    _trailing_return_panel,
    _union_calendar,
    hybrid_champion_schedule,
    run_backtest as run_2013_backtest,
)


DELIVERABLE_DIR = Path(__file__).parent / "deliverable"
RETURNS_FILENAME = "relative_defender_rotation_2013_daily_returns.csv"
INDICATORS_FILENAME = "relative_defender_rotation_2013_daily_indicators.csv"
SWITCH_RETURNS_FILENAME = "relative_defender_rotation_2013_switch_returns.csv"
SWITCH_HANDOFF_FILENAME = "relative_defender_rotation_2013_switch_handoff.md"

_SIGNAL_COLUMNS = {
    "range_location": "signal_range_location_40",
    "realized_volatility_20": "signal_realized_volatility_20",
    "cap_volatility_threshold": "signal_cap_volatility_threshold",
    "signal_grid_target": "signal_grid_target",
    "signal_volatility_cap": "signal_volatility_cap",
    "signal_base_target": "signal_base_target",
    "signal_base_reason": "signal_base_reason",
    "factor_return_15": "signal_factor_return_15",
    "path_efficiency_15": "signal_path_efficiency_15",
    "realized_volatility_5": "signal_realized_volatility_5",
    "low_volatility_anchor": "signal_low_volatility_anchor",
    "low_volatility_score": "signal_low_volatility_score",
    "champion_score": "signal_weighted_champion_score",
    "entry_score_threshold": "signal_entry_score_threshold",
    "exit_score_threshold": "signal_exit_score_threshold",
    "regime_return_60": "signal_regime_return_60",
    "signal_full_override_active": "signal_full_override_active",
    "signal_primary_target": "signal_primary_target_next_open",
}


def _context(
    market: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[
    object,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    params = rotation_params()
    prices = _clean_market(market, None, params)
    daily, _, metrics, selection, _ = run_2013_backtest(
        market=prices,
        start=START_DATE,
        params=params,
    )
    full_start = pd.Timestamp(prices[BRIDGE_SIGNAL_ASSET]["date"].min())
    full_calendar = _union_calendar(
        prices,
        full_start,
        pd.Timestamp(daily.index.max()),
    )
    full_schedule = hybrid_champion_schedule(prices, full_calendar)
    schedule = full_schedule.reindex(daily.index)
    return params, prices, daily, metrics, selection, schedule


def _policy_target(
    daily: pd.DataFrame,
    timestamp: pd.Timestamp,
    assets: tuple[str, ...],
) -> dict[str, float]:
    return {
        asset: float(daily.at[timestamp, f"target_{asset}"])
        for asset in assets
        if float(daily.at[timestamp, f"target_{asset}"]) > 1e-14
    }


def _daily_prices(
    indexed: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    open_prices = {
        asset: float(frame.at[timestamp, "open"])
        for asset, frame in indexed.items()
        if timestamp in frame.index
        and pd.notna(frame.at[timestamp, "open"])
        and float(frame.at[timestamp, "open"]) > 0.0
    }
    mark_open = {
        asset: (_asof_price(frame, timestamp, "close") or 0.0)
        for asset, frame in indexed.items()
    }
    mark_open.update(open_prices)
    close_prices = {
        asset: (_asof_price(frame, timestamp, "close") or 0.0)
        for asset, frame in indexed.items()
    }
    return open_prices, mark_open, close_prices


def _executable_target(
    policy_target: Mapping[str, float],
    open_prices: Mapping[str, float],
) -> dict[str, float]:
    return {
        asset: float(weight)
        for asset, weight in policy_target.items()
        if asset in open_prices and float(weight) > 1e-14
    }


def _normalized_previous_holdings(
    daily: pd.DataFrame,
    indexed: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
    assets: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    cash = float(daily.at[timestamp, "previous_closing_cash_weight"])
    shares: dict[str, float] = {}
    for asset in assets:
        weight = float(daily.at[timestamp, f"previous_closing_weight_{asset}"])
        if weight <= 1e-14:
            continue
        frame = indexed[asset]
        history = frame.loc[frame.index < timestamp, "close"].dropna()
        if history.empty:
            raise RuntimeError(
                f"missing previous close for held asset {asset} at {timestamp.date()}"
            )
        shares[asset] = weight / float(history.iloc[-1])
    return cash, shares


def build_switch_return_frame(
    market: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build exact held, fresh-entry and fresh-exit open-switch return legs."""
    params, prices, daily, metrics, selection, schedule = _context(market)
    indexed = _indexed_market(prices)
    calendar = pd.DatetimeIndex(daily.index)
    all_assets = (*params.assets, params.defensive_asset)
    first_dates = {
        asset: pd.Timestamp(frame["date"].min())
        for asset, frame in prices.items()
    }
    base_dates = set(pd.to_datetime(prices[BASE_PRIMARY_ASSET]["date"]))
    bridge_dates = set(pd.to_datetime(prices[BRIDGE_SIGNAL_ASSET]["date"]))
    rows: list[dict[str, object]] = []

    for position, timestamp_value in enumerate(calendar):
        timestamp = pd.Timestamp(timestamp_value)
        open_prices, mark_open, close_prices = _daily_prices(indexed, timestamp)
        policy_target = _policy_target(daily, timestamp, all_assets)
        executable_target = _executable_target(policy_target, open_prices)
        entry_policy_fully_executable = set(policy_target).issubset(open_prices)

        entry_cash, entry_shares, entry_executions = _execute_portfolio_target(
            INITIAL_CAPITAL,
            {},
            executable_target,
            open_prices,
            mark_open,
            ROTATION_COST_RATES,
        )
        entry_cost = sum(
            float(execution["cost"]) for execution in entry_executions
        )
        entry_turnover = sum(
            float(execution["turnover"]) for execution in entry_executions
        )
        entry_post_open_nav = entry_cash + sum(
            quantity * mark_open.get(asset, 0.0)
            for asset, quantity in entry_shares.items()
        )
        entry_close_nav = entry_cash + sum(
            quantity * close_prices.get(asset, 0.0)
            for asset, quantity in entry_shares.items()
        )
        entry_intraday_gross_return = (
            entry_close_nav / entry_post_open_nav - 1.0
            if entry_post_open_nav > 0.0
            else 0.0
        )
        entry_cost_rate = entry_cost / INITIAL_CAPITAL

        if position == 0:
            exit_turnover = np.nan
            exit_cost_rate = np.nan
            exit_return = np.nan
            exit_fully_executable: bool | float = np.nan
        else:
            exit_cash, exit_shares = _normalized_previous_holdings(
                daily,
                indexed,
                timestamp,
                all_assets,
            )
            held_assets = set(exit_shares)
            exit_fully_executable = held_assets.issubset(open_prices)
            open_nav = exit_cash + sum(
                quantity * mark_open.get(asset, 0.0)
                for asset, quantity in exit_shares.items()
            )
            exit_cash_after, exit_shares_after, exit_executions = (
                _execute_portfolio_target(
                    exit_cash,
                    exit_shares,
                    {},
                    open_prices,
                    mark_open,
                    ROTATION_COST_RATES,
                )
            )
            exit_turnover = sum(
                float(execution["turnover"])
                for execution in exit_executions
            )
            exit_cost = sum(
                float(execution["cost"]) for execution in exit_executions
            )
            exit_cost_rate = exit_cost / open_nav if open_nav > 0.0 else 0.0
            exit_nav = exit_cash_after + sum(
                quantity * mark_open.get(asset, 0.0)
                for asset, quantity in exit_shares_after.items()
            )
            exit_return = (
                exit_nav - 1.0 if bool(exit_fully_executable) else np.nan
            )

        signal_anchor = str(schedule.at[timestamp, "signal_anchor_asset"])
        anchor_dates = base_dates if signal_anchor == BASE_PRIMARY_ASSET else bridge_dates
        row: dict[str, object] = {
            "date": timestamp,
            "signal_date": daily.at[timestamp, "signal_observation_date"],
            "execution_date": timestamp,
            "has_previous_close": bool(daily.at[timestamp, "has_previous_close"]),
            "signal_anchor_asset": signal_anchor,
            "execution_signal_source_asset": daily.at[
                timestamp, "signal_source_asset"
            ],
            "signal_anchor_traded": timestamp in anchor_dates,
            "base_anchor_traded": timestamp in base_dates,
            "strategy_id": STRATEGY_ID,
            "research_status": "retrospective_history_extension_not_oos",
            "formal_status": "production_signal_frozen",
            "formal_promotion_date": PROMOTION_DATE,
            "calendar_asset": "UNION_OF_REQUIRED_ASSETS",
            "price_adjustment": "HFQ_FIXED_BASELINE",
            "selected_asset": daily.at[timestamp, "selected_asset"],
            "selection_reason": daily.at[timestamp, "selection_reason"],
            "overnight_gross_return": float(
                daily.at[timestamp, "overnight_gross_return"]
            ),
            "intraday_gross_return_if_held": float(
                daily.at[timestamp, "intraday_gross_return_if_held"]
            ),
            "daily_gross_return_if_held": float(
                daily.at[timestamp, "daily_gross_return_if_held"]
            ),
            "internal_turnover": float(
                daily.at[timestamp, "internal_turnover"]
            ),
            "internal_cost_rate_at_open": float(
                daily.at[timestamp, "internal_cost_rate_at_open"]
            ),
            "daily_net_return_if_held": float(
                daily.at[timestamp, "daily_net_return_if_held"]
            ),
            "daily_net_return_reconstructed": float(
                daily.at[timestamp, "daily_net_return_reconstructed"]
            ),
            "nav_if_held": float(daily.at[timestamp, "nav"]),
            "strategy_nav_reference": float(daily.at[timestamp, "nav"]),
            "intraday_gross_return_if_entered": entry_intraday_gross_return,
            "fresh_entry_policy_fully_executable": entry_policy_fully_executable,
            "fresh_entry_target_turnover": float(sum(executable_target.values())),
            "fresh_entry_executed_turnover": entry_turnover,
            "fresh_entry_cost_rate_at_open": entry_cost_rate,
            "enter_open_to_close_net_return": entry_close_nav - 1.0,
            "fresh_exit_fully_executable": exit_fully_executable,
            "fresh_exit_executed_turnover": exit_turnover,
            "fresh_exit_cost_rate_at_open": exit_cost_rate,
            "exit_prev_close_to_open_net_return": exit_return,
            "previous_closing_cash_weight": float(
                daily.at[timestamp, "previous_closing_cash_weight"]
            ),
            "policy_target_cash_weight": max(
                0.0, 1.0 - sum(policy_target.values())
            ),
            "target_cash_weight": max(
                0.0, 1.0 - sum(executable_target.values())
            ),
            "post_open_cash_weight": float(
                daily.at[timestamp, "post_open_cash_weight"]
            ),
            "closing_cash_weight": float(daily.at[timestamp, "cash_weight"]),
        }
        for asset in all_assets:
            code = asset.split(".", maxsplit=1)[0]
            row[f"asset_first_date_{code}"] = first_dates[asset]
            row[f"transaction_cost_rate_{code}"] = _asset_cost_rate(
                asset, ROTATION_COST_RATES
            )
            row[f"previous_closing_weight_{code}"] = float(
                daily.at[timestamp, f"previous_closing_weight_{asset}"]
            )
            row[f"policy_target_weight_{code}"] = policy_target.get(asset, 0.0)
            row[f"target_weight_{code}"] = executable_target.get(asset, 0.0)
            row[f"post_open_weight_{code}"] = float(
                daily.at[timestamp, f"post_open_weight_{asset}"]
            )
            row[f"closing_weight_{code}"] = float(
                daily.at[timestamp, f"weight_{asset}"]
            )
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("date")
    frame.index.name = "date"
    export_metrics: dict[str, object] = {
        **metrics,
        "strategy": STRATEGY_ID,
        "switch_interface_rows": int(len(frame)),
        "calendar_method": "union_of_required_asset_trading_dates",
        "price_adjustment": "HFQ_FIXED_BASELINE",
        "pre_anchor_signal_asset": BRIDGE_SIGNAL_ASSET,
        "formal_anchor_asset": BASE_PRIMARY_ASSET,
    }
    return frame, export_metrics


def build_daily_frames(
    market: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build compact returns and the full listing-aware daily indicator audit."""
    params, prices, daily, metrics, selection, schedule = _context(market)
    switch, _ = build_switch_return_frame(prices)
    index = pd.DatetimeIndex(daily.index)

    compact = pd.DataFrame({
        "daily_net_return": switch["daily_net_return_if_held"].astype(float),
        "nav": switch["nav_if_held"].astype(float),
        "cumulative_return": switch["nav_if_held"].astype(float) - 1.0,
        "drawdown": (
            switch["nav_if_held"].astype(float)
            / switch["nav_if_held"].astype(float).cummax()
            - 1.0
        ),
    }, index=index)
    compact.index.name = "date"

    indicators = compact.copy()
    indicators["strategy_id"] = STRATEGY_ID
    indicators["research_status"] = "retrospective_history_extension_not_oos"
    indicators["formal_status"] = "production_signal_frozen"
    indicators["formal_promotion_date"] = PROMOTION_DATE
    indicators["signal_anchor_asset"] = schedule["signal_anchor_asset"].astype(str)
    indicators["signal_anchor_traded"] = switch["signal_anchor_traded"].astype(bool)
    indicators["base_anchor_traded"] = switch["base_anchor_traded"].astype(bool)
    indicators["signal_observation_date"] = schedule["signal_observation_date"]
    indicators["signal_effective_next_open_date"] = pd.Series(
        index, index=index
    ).shift(-1)
    indicators["execution_signal_source_asset"] = schedule[
        "execution_signal_source_asset"
    ].astype(str)
    indicators["execution_signal_observation_date"] = schedule[
        "execution_signal_observation_date"
    ]
    indicators["selected_asset"] = selection["selected_asset"].astype(str)
    indicators["selection_reason"] = selection["selection_reason"].astype(str)
    indicators["signal_execution_reason"] = daily[
        "signal_execution_reason"
    ].astype(str)
    indicators["execution_primary_target"] = daily["primary_target"].astype(float)
    indicators["execution_defensive_target"] = daily[
        "defensive_target"
    ].astype(float)
    indicators["execution_target_cash_weight"] = switch[
        "target_cash_weight"
    ].astype(float)
    indicators["closing_cash_weight"] = daily["cash_weight"].astype(float)

    for source, target in _SIGNAL_COLUMNS.items():
        indicators[target] = schedule[source]
    indicators["execution_full_override_active"] = schedule[
        "execution_full_override_active"
    ].astype(bool)
    indicators["execution_reason"] = schedule["execution_reason"].astype(str)

    full_start = pd.Timestamp(prices[BRIDGE_SIGNAL_ASSET]["date"].min())
    full_calendar = _union_calendar(prices, full_start, pd.Timestamp(index.max()))
    reversal = _trailing_return_panel(
        prices, params.assets, full_calendar, params.reversal_lookback_days
    ).reindex(index)
    trend = _trailing_return_panel(
        prices, params.assets, full_calendar, params.trend_lookback_days
    ).reindex(index)
    regime = _trailing_return_panel(
        prices,
        (BRIDGE_SIGNAL_ASSET, BASE_PRIMARY_ASSET),
        full_calendar,
        params.regime_lookback_days,
    ).reindex(index)
    indicators["rotation_anchor_regime_return_180"] = [
        regime.at[timestamp, schedule.at[timestamp, "signal_anchor_asset"]]
        for timestamp in index
    ]

    for asset in params.assets:
        code = asset.split(".", maxsplit=1)[0]
        first_date = pd.Timestamp(prices[asset]["date"].min())
        open_dates = set(pd.to_datetime(prices[asset]["date"]))
        indicators[f"listed_{code}"] = index >= first_date
        indicators[f"traded_{code}"] = [timestamp in open_dates for timestamp in index]
        indicators[f"rotation_reversal_return_40_{code}"] = reversal[asset]
        indicators[f"rotation_trend_return_150_{code}"] = trend[asset]
        indicators[f"rotation_reversal_eligible_{code}"] = reversal[asset].notna()
        indicators[f"rotation_trend_eligible_{code}"] = trend[asset].notna()
        indicators[f"rotation_primary_fraction_{code}"] = selection[
            f"primary_fraction_{asset}"
        ].astype(float)

    all_assets = (*params.assets, params.defensive_asset)
    for asset in all_assets:
        code = asset.split(".", maxsplit=1)[0]
        indicators[f"policy_target_weight_{code}"] = switch[
            f"policy_target_weight_{code}"
        ].astype(float)
        indicators[f"target_weight_{code}"] = switch[
            f"target_weight_{code}"
        ].astype(float)
        indicators[f"closing_weight_{code}"] = switch[
            f"closing_weight_{code}"
        ].astype(float)
    indicators.index.name = "date"

    export_metrics: dict[str, object] = {
        **metrics,
        "strategy": STRATEGY_ID,
        "export_return_observations": int(len(compact)),
        "calendar_method": "union_of_required_asset_trading_dates",
        "export_return_definition": (
            "NAV-linked net return; first row is first open-to-close"
        ),
    }
    return compact, indicators, export_metrics


def _handoff_text(
    frame: pd.DataFrame,
    metrics: Mapping[str, object],
) -> str:
    held_error = float(
        (
            frame["daily_net_return_if_held"]
            - frame["daily_net_return_reconstructed"]
        ).abs().max()
    )
    nav_error = float(
        (frame["nav_if_held"] - frame["strategy_nav_reference"]).abs().max()
    )
    entry_error = float(
        (
            (1.0 - frame["fresh_entry_cost_rate_at_open"])
            * (1.0 + frame["intraday_gross_return_if_entered"])
            - 1.0
            - frame["enter_open_to_close_net_return"]
        ).abs().max()
    )
    return f"""# Defender 2013上市感知正式策略：开盘切换数据接口交接说明

## 1. 正式版本与证据边界

本交付对应 `{STRATEGY_ID}`，已于{PROMOTION_DATE}按用户明确指令晋升为
当前主策略。回测区间为 {frame.index.min().date()} 至
{frame.index.max().date()}，共 {len(frame)} 个交易日。策略已经正式晋升，
但历史扩展结果仍是回溯证据，不是独立样本外证据。

- 2013-07-01至512890首个收盘之前，使用510880和相同冻结参数产生仓位信号；
- 512890首个收盘为2019-01-18，从2019-01-21开盘起使用512890信号；
- 股票ETF只在已经上市、执行日可交易且具备相应排名历史时参加月度排名；
- 511260在2017-08-24之前未上市，防守仓位保留现金，不伪造债券收益。

## 2. 交付文件

- `{RETURNS_FILENAME}`：连续持有Defender的每日净收益、净值和回撤；
- `{INDICATORS_FILENAME}`：完整仓位信号、加权Score、轮动指标、上市状态和持仓；
- `{SWITCH_RETURNS_FILENAME}`：开盘切换所需的隔夜/日内收益、费用和权重；
- `{SWITCH_HANDOFF_FILENAME}`：本说明。

三份CSV均以 `date` 为唯一键，可按日期一对一连接。收益和费用以
`{SWITCH_RETURNS_FILENAME}` 为唯一来源；指标CSV中的收益列仅用于核对，
不要重复计入。

## 3. 指标CSV

指标观察日为 `signal_observation_date`，最早在
`signal_effective_next_open_date` 开盘执行。主要字段：

- `signal_range_location_40`：信号锚的40日收盘区间位置；
- `signal_realized_volatility_20`：20日年化Rogers–Satchell波动率；
- `signal_cap_volatility_threshold`：因果历史80%分位波动率线；
- `signal_grid_target`、`signal_volatility_cap`、`signal_base_target`：
  网格、波动限仓和两者合成的基础仓位；
- `signal_factor_return_15`、`signal_path_efficiency_15`：15日收益和路径效率；
- `signal_realized_volatility_5`、`signal_low_volatility_anchor`、
  `signal_low_volatility_score`：5日RS波动率及低波得分；
- `signal_weighted_champion_score`：三因子加权几何得分；
- `signal_entry_score_threshold`、`signal_exit_score_threshold`：迟滞进入/退出线；
- `signal_regime_return_60`、`signal_full_override_active`：60日趋势和满仓状态；
- `rotation_reversal_return_40_*`、`rotation_trend_return_150_*`：
  六只ETF逐只的40日反转与150日趋势收益；
- `listed_*`、`traded_*`、`rotation_*_eligible_*`：上市、当日交易及排名资格；
- `policy_target_weight_*`：规则希望配置的权重；
- `target_weight_*`：当日开盘实际可执行的目标；
- `closing_weight_*`：当日收盘实际权重。

加权Champion Score公式为：

```text
max(15日收益, 0)^0.25
* max(15日路径效率, 0)^0.25
* max(低波得分, 0)^0.50
```

2019-01-18之前 `signal_anchor_asset=510880.SH`；自该日收盘指标起为
`512890.SH`。停牌日信号向前沿用，`signal_anchor_traded=False`。

## 4. 开盘切换收益口径

- `overnight_gross_return`：前收盘至当日开盘，使用开盘前旧持仓；
- `intraday_gross_return_if_held`：当日开盘内部调仓后至收盘；
- `internal_cost_rate_at_open`：Defender连续持有时自身开盘调仓成本；
- `daily_net_return_if_held`：连续持有Defender的完整当日净收益；
- `intraday_gross_return_if_entered`：外部策略当日开盘新切入后的日内毛收益；
- `fresh_entry_cost_rate_at_open`：外部策略新切入Defender的建仓费用；
- `enter_open_to_close_net_return`：开盘新进入Defender至收盘的净收益；
- `fresh_exit_cost_rate_at_open`：开盘完整退出Defender的卖出费用；
- `exit_prev_close_to_open_net_return`：前收盘持有Defender、当日开盘退出的净收益。

连续持有校验式：

```text
(1 + overnight_gross_return)
* (1 - internal_cost_rate_at_open)
* (1 + intraday_gross_return_if_held) - 1
```

开盘新进入校验式：

```text
(1 - fresh_entry_cost_rate_at_open)
* (1 + intraday_gross_return_if_entered) - 1
```

若外部策略在交易日开盘切入Defender，旧策略承担隔夜段和自身退出费用，
Defender承担新进入费用与日内段；不能把两段直接相加，也不能重复扣除
Defender的内部调仓费用。

## 5. 上市、停牌与目标权重

- `policy_target_weight_*`保留完整规则意图；
- `target_weight_*`只包含当日开盘有价格、可以实际买入的资产；
- 两者差额进入 `target_cash_weight`；
- 因此511260上市前，政策防守权重不会出现在可执行目标中，而会变成现金；
- `fresh_entry_policy_fully_executable=False`表示至少一个政策目标当日不能买入；
- `fresh_exit_fully_executable=False`表示旧持仓中存在停牌资产，完整切出不可执行，
  此时 `exit_prev_close_to_open_net_return` 留空；
- 2021-10-22为512890份额拆分停牌日，仍保留为融合交易日，持有的其他ETF
  正常产生收益。

股票ETF单边费率为0.01%，511260单边费率为0.001%；价格为固定基准后复权OHLC。

## 6. 2014年解释提示

2014年只有510880可选。策略8月26日起长期保持0%股票仓位，剩余资金因511260
未上市而留现金；第四季度510880上涨约37.94%，是该年显著落后Base的主要原因。
这是510880信号桥接规则的历史表现，不应被解释成512890正式锚定信号在2014年的表现。

## 7. 机械校验

- 期末净值：{float(metrics['final_nav']):.12f}；
- 连续持有分段重构最大绝对误差：{held_error:.3e}；
- 新进入分段重构最大绝对误差：{entry_error:.3e}；
- 与策略逐日净值最大绝对误差：{nav_error:.3e}；
- 所有政策目标权重合计为1；
- 所有可执行目标权重与目标现金合计为1；
- 所有收盘实际权重与现金合计为1。
"""


def write_deliverables(
    output_dir: Path = DELIVERABLE_DIR,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    compact, indicators, metrics = build_daily_frames()
    switch, switch_metrics = build_switch_return_frame()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / RETURNS_FILENAME
    indicators_path = output_dir / INDICATORS_FILENAME
    switch_path = output_dir / SWITCH_RETURNS_FILENAME
    handoff_path = output_dir / SWITCH_HANDOFF_FILENAME
    for frame, path in (
        (compact, returns_path),
        (indicators, indicators_path),
        (switch, switch_path),
    ):
        frame.to_csv(
            path,
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
        )
    handoff_path.write_text(
        _handoff_text(switch, switch_metrics),
        encoding="utf-8",
    )
    return returns_path, indicators_path, switch_path, handoff_path, metrics


def main() -> None:
    for path in write_deliverables()[:4]:
        print(path)


if __name__ == "__main__":
    main()
