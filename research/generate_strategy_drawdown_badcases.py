"""Generate the formal strategy's distinct Top-N drawdown badcase ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from data.store import query
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.formal_strategy_holdings import (
    ALL_ASSETS,
    build_formal_target_schedule,
    format_portfolio,
    portfolio_key,
)
from research.momentum_defender_occam import HELD_RETURN
from strategy.momentum_defender_w40_qm40_threshold import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)


DEFAULT_CONTEXT = Path(
    "research/configs/strategy_drawdown_badcase_context.yaml"
)
DEFAULT_OUTPUT = Path(
    "docs/research/momentum_defender_drawdown_badcases.md"
)
MOMENTUM_REPLAY_PARITY_TOLERANCE = 2e-5
VOLUME_STRONG_RATIO = 1.50
VOLUME_STRONG_Z = 2.0


def distinct_drawdown_episodes(
    daily: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Return the deepest non-overlapping peak-to-recovery NAV episodes."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {"return", "nav"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"formal daily ledger missing columns: {sorted(missing)}")
    if not daily.index.is_monotonic_increasing or not daily.index.is_unique:
        raise ValueError("formal daily calendar must be sorted and unique")

    returns = daily["return"].astype(float)
    nav = daily["nav"].astype(float)
    reconstructed = (1.0 + returns).cumprod()
    if float((reconstructed - nav).abs().max()) > 1e-12:
        raise AssertionError("formal NAV does not reconstruct from daily returns")

    running_peak = nav.cummax().clip(lower=1.0)
    drawdown = nav / running_peak - 1.0
    underwater = drawdown.lt(-1e-12)
    groups = underwater.ne(underwater.shift(fill_value=False)).cumsum()
    calendar = pd.DatetimeIndex(daily.index)
    rows: list[dict[str, object]] = []
    for _, sample in daily.loc[underwater].groupby(groups.loc[underwater]):
        start = pd.Timestamp(sample.index[0])
        last = pd.Timestamp(sample.index[-1])
        start_position = calendar.get_loc(start)
        peak_date = (
            pd.Timestamp(calendar[start_position - 1])
            if start_position > 0
            else pd.NaT
        )
        peak_nav = (
            float(nav.at[peak_date]) if pd.notna(peak_date) else 1.0
        )
        trough = pd.Timestamp(drawdown.loc[start:last].idxmin())
        last_position = calendar.get_loc(last)
        recovered = last_position < len(calendar) - 1
        recovery = (
            pd.Timestamp(calendar[last_position + 1]) if recovered else pd.NaT
        )
        rows.append(
            {
                "decline_start": start,
                "peak_date": peak_date,
                "trough_date": trough,
                "last_underwater_date": last,
                "recovery_date": recovery,
                "open_ended": not recovered,
                "peak_nav": peak_nav,
                "trough_nav": float(nav.at[trough]),
                "max_drawdown": float(drawdown.at[trough]),
                "decline_sessions": int(
                    calendar.get_loc(trough) - start_position + 1
                ),
                "underwater_sessions": int(len(sample)),
            }
        )
    if not rows:
        empty = pd.DataFrame(
            columns=[
                "rank",
                "decline_start",
                "peak_date",
                "trough_date",
                "last_underwater_date",
                "recovery_date",
                "open_ended",
                "peak_nav",
                "trough_nav",
                "max_drawdown",
                "decline_sessions",
                "underwater_sessions",
            ]
        )
        empty.attrs["all_episode_count"] = 0
        empty.attrs["drawdown_definition"] = (
            "NAV / running peak - 1; each underwater run is one episode"
        )
        return empty
    episodes = pd.DataFrame(rows).sort_values(
        ["max_drawdown", "decline_start"], ascending=[True, True]
    )
    episodes.attrs["all_episode_count"] = int(len(episodes))
    selected = episodes.head(top_n).copy().reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected.attrs["all_episode_count"] = episodes.attrs["all_episode_count"]
    selected.attrs["drawdown_definition"] = (
        "NAV / running peak - 1; each underwater run is one episode"
    )
    return selected


def _holding_runs(
    targets: pd.DataFrame,
    daily: pd.DataFrame,
    start: pd.Timestamp,
    trough: pd.Timestamp,
    asset_names: Mapping[str, str],
) -> list[dict[str, object]]:
    sample = targets.loc[start:trough]
    keys = sample.apply(portfolio_key, axis=1)
    groups = keys.ne(keys.shift()).cumsum()
    rows = []
    for _, run in sample.groupby(groups):
        run_start = pd.Timestamp(run.index[0])
        run_end = pd.Timestamp(run.index[-1])
        interval = pd.DatetimeIndex(run.index)
        rows.append(
            {
                "start": run_start,
                "end": run_end,
                "sessions": int(len(run)),
                "portfolio": format_portfolio(
                    run.iloc[0], asset_names, ALL_ASSETS
                ),
                "candidate": str(daily.at[run_start, "candidate"]),
                "strategy_return": float(
                    (1.0 + daily.loc[interval, "return"].astype(float)).prod()
                    - 1.0
                ),
                "first_transition": str(
                    daily.at[run_start, "transition"]
                ),
            }
        )
    return rows


def _candidate_runs(
    daily: pd.DataFrame,
    start: pd.Timestamp,
    trough: pd.Timestamp,
    asset_names: Mapping[str, str],
) -> list[dict[str, object]]:
    """Collapse a mixed path into consecutive top-level candidate runs."""

    selected = daily.loc[start:trough, "candidate"].astype(str)
    groups = selected.ne(selected.shift()).cumsum()
    rows = []
    for _, run in selected.groupby(groups):
        run_start = pd.Timestamp(run.index[0])
        run_end = pd.Timestamp(run.index[-1])
        candidate = str(run.iloc[0])
        label = (
            "Defender（内部红利目标见下方实际持仓）"
            if candidate == DEFENDER_CANDIDATE
            else f"{candidate}（{asset_names.get(candidate, candidate)}）"
        )
        rows.append(
            {
                "start": run_start,
                "end": run_end,
                "candidate": candidate,
                "label": label,
                "sessions": int(len(run)),
                "strategy_return": float(
                    (
                        1.0
                        + daily.loc[run.index, "return"].astype(float)
                    ).prod()
                    - 1.0
                ),
            }
        )
    return rows


def _volume_feature_panel(asset: str, end: date) -> pd.DataFrame:
    """Build strictly lagged local-ETF volume diagnostics."""

    frame = (
        query(asset, date(2013, 1, 1), end)
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")
    )
    if frame.empty:
        return frame
    volume = frame["volume"].astype(float)
    if volume.le(0.0).any():
        raise ValueError(f"non-positive volume in {asset}")
    log_volume = np.log(volume)
    prior_log_mean = log_volume.shift(1).rolling(60, min_periods=60).mean()
    prior_log_std = log_volume.shift(1).rolling(60, min_periods=60).std(ddof=1)
    panel = pd.DataFrame(index=frame.index)
    panel["volume"] = volume
    panel["volume_ratio_to_prior20_median"] = volume / (
        volume.shift(1).rolling(20, min_periods=20).median()
    )
    panel["log_volume_z60"] = (
        log_volume - prior_log_mean
    ) / prior_log_std.replace(0.0, np.nan)
    raw = volume.to_numpy(float)
    percentile = np.full(len(volume), np.nan, dtype=float)
    for position in range(252, len(volume)):
        prior = raw[position - 252 : position]
        percentile[position] = float(np.mean(prior <= raw[position]))
    panel["volume_percentile_prior252"] = percentile
    close = frame["close"].astype(float)
    panel["asset_return_1d"] = close.pct_change()
    panel["asset_return_5d"] = close.pct_change(5)
    panel["asset_return_20d"] = close.pct_change(20)
    return panel


def _volume_classification(record: Mapping[str, object]) -> str:
    ratio = float(record["volume_ratio_to_prior20_median"])
    z_score = float(record["log_volume_z60"])
    percentile = float(record["volume_percentile_prior252"])
    recent_max = float(record["prior5_max_volume_ratio"])
    if ratio >= VOLUME_STRONG_RATIO and z_score >= VOLUME_STRONG_Z:
        return "峰值日强放量"
    if ratio >= 1.45 and z_score >= VOLUME_STRONG_Z:
        return "临界强放量"
    if percentile >= 0.90 or z_score >= 1.50 or ratio >= 1.25:
        return "高量平台 / 量能偏高"
    if recent_max >= VOLUME_STRONG_RATIO:
        return "峰前放量、峰日回落"
    return "无明显放量（例外）"


def _peak_volume_record(
    episode: pd.Series,
    targets: pd.DataFrame,
    volume_panels: Mapping[str, pd.DataFrame],
    asset_names: Mapping[str, str],
    *,
    allow_unavailable: bool = False,
) -> dict[str, object]:
    peak = pd.Timestamp(episode["peak_date"])
    weights = targets.loc[peak, list(ALL_ASSETS)].astype(float)
    active = weights.loc[weights.gt(1e-12)]
    if active.empty:
        raise AssertionError(f"peak {peak.date()} has no invested asset")
    asset = str(active.idxmax())
    weight = float(active.max())
    panel = volume_panels[asset]
    if peak not in panel.index:
        raise AssertionError(f"volume missing for {asset} on {peak.date()}")
    position = panel.index.get_loc(peak)
    prior5 = panel.iloc[max(0, position - 4) : position + 1]
    row = panel.loc[peak]
    record: dict[str, object] = {
        "asset": asset,
        "asset_name": asset_names.get(asset, asset),
        "weight": weight,
        "volume": float(row["volume"]),
        "volume_ratio_to_prior20_median": float(
            row["volume_ratio_to_prior20_median"]
        ),
        "log_volume_z60": float(row["log_volume_z60"]),
        "volume_percentile_prior252": float(
            row["volume_percentile_prior252"]
        ),
        "prior5_max_volume_ratio": float(
            prior5["volume_ratio_to_prior20_median"].max()
        ),
        "asset_return_1d": float(row["asset_return_1d"]),
        "asset_return_5d": float(row["asset_return_5d"]),
        "asset_return_20d": float(row["asset_return_20d"]),
    }
    finite = [
        record["volume_ratio_to_prior20_median"],
        record["log_volume_z60"],
        record["volume_percentile_prior252"],
        record["prior5_max_volume_ratio"],
    ]
    if not all(np.isfinite(float(value)) for value in finite):
        if not allow_unavailable:
            raise AssertionError(f"peak volume features unavailable for {asset}")
        record["classification"] = "历史不足（无法分类）"
        return record
    record["classification"] = _volume_classification(record)
    return record


def _volume_population_summary(
    all_episodes: pd.DataFrame,
    targets: pd.DataFrame,
    volume_panels: Mapping[str, pd.DataFrame],
    asset_names: Mapping[str, str],
    *,
    top_n: int,
) -> dict[str, object]:
    rows = []
    feature_available_count = 0
    for _, episode in all_episodes.iterrows():
        if pd.isna(episode["peak_date"]):
            continue
        try:
            record = _peak_volume_record(
                episode, targets, volume_panels, asset_names
            )
        except AssertionError as error:
            if "peak volume features unavailable" not in str(error):
                raise
            continue
        feature_available_count += 1
        if float(record["weight"]) < 0.80:
            continue
        rows.append(
            {
                "rank": int(episode["rank"]),
                "drawdown_depth": -float(episode["max_drawdown"]),
                **record,
            }
        )
    frame = pd.DataFrame(rows)
    top = frame.loc[frame["rank"].le(top_n)]
    other = frame.loc[frame["rank"].gt(top_n)]
    strong = frame["volume_ratio_to_prior20_median"].ge(
        VOLUME_STRONG_RATIO
    ) & frame["log_volume_z60"].ge(VOLUME_STRONG_Z)
    top_strong = strong.loc[top.index]
    other_strong = strong.loc[other.index]
    return {
        "feature_available_episode_count": feature_available_count,
        "eligible_episode_count": int(len(frame)),
        "top_count": int(len(top)),
        "other_count": int(len(other)),
        "top_median_volume_ratio": float(
            top["volume_ratio_to_prior20_median"].median()
        ),
        "other_median_volume_ratio": float(
            other["volume_ratio_to_prior20_median"].median()
        ),
        "top_median_volume_z": float(top["log_volume_z60"].median()),
        "other_median_volume_z": float(other["log_volume_z60"].median()),
        "top_strong_volume_share": float(top_strong.mean()),
        "other_strong_volume_share": float(other_strong.mean()),
        "spearman_volume_ratio_vs_depth": float(
            frame["volume_ratio_to_prior20_median"]
            .rank()
            .corr(frame["drawdown_depth"].rank())
        ),
        "spearman_volume_z_vs_depth": float(
            frame["log_volume_z60"]
            .rank()
            .corr(frame["drawdown_depth"].rank())
        ),
    }


def _worst_days(
    daily: pd.DataFrame,
    start: pd.Timestamp,
    trough: pd.Timestamp,
    *,
    count: int = 3,
) -> list[dict[str, object]]:
    rows = []
    for timestamp, value in (
        daily.loc[start:trough, "return"].astype(float).nsmallest(count).items()
    ):
        row = daily.loc[timestamp]
        rows.append(
            {
                "date": pd.Timestamp(timestamp),
                "return": float(value),
                "transition": str(row["transition"]),
                "exit_leg": row["exit_return_leg_used"],
                "enter_leg": row["enter_return_leg_used"],
            }
        )
    return rows


def _sleeve_classification(candidate: pd.Series) -> str:
    defender_days = int(candidate.eq(DEFENDER_CANDIDATE).sum())
    if defender_days == 0:
        return "纯Momentum"
    if defender_days == len(candidate):
        return "纯Defender"
    return "混合路径"


def fixed_sleeve_returns(
    formal_run,
    start: pd.Timestamp,
    trough: pd.Timestamp,
) -> tuple[float, float]:
    """Compound always-Momentum and always-Defender held-net returns.

    Both counterfactual sleeves are assumed to be held before ``start``.  This
    keeps the comparison free of an arbitrary first-day fresh-entry leg while
    preserving each sleeve's internal turnover and transaction costs.
    """

    inputs = formal_run.context.integrated.result.inputs
    momentum = inputs.momentum.loc[start:trough, HELD_RETURN].astype(float)
    defender = formal_run.context.interfaces[DEFENDER_CANDIDATE].loc[
        start:trough, HELD_RETURN
    ].astype(float)
    if momentum.empty or defender.empty or len(momentum) != len(defender):
        raise AssertionError("fixed-sleeve counterfactual interval is incomplete")
    if not (
        np.isfinite(momentum.to_numpy()).all()
        and np.isfinite(defender.to_numpy()).all()
    ):
        raise AssertionError("fixed-sleeve counterfactual contains non-finite returns")
    return (
        float((1.0 + momentum).prod() - 1.0),
        float((1.0 + defender).prod() - 1.0),
    )


def build_drawdown_evidence(
    formal_run,
    *,
    top_n: int,
    asset_names: Mapping[str, str],
) -> pd.DataFrame:
    """Attach actual holdings, switch legs and W40 state to Top-N episodes."""

    daily = formal_run.daily
    state = formal_run.state
    targets = build_formal_target_schedule(formal_run)
    all_episodes = distinct_drawdown_episodes(daily, top_n=len(daily))
    episodes = all_episodes.head(top_n).copy()
    episodes.attrs.update(all_episodes.attrs)
    cutoff = pd.Timestamp(daily.index.max()).date()
    volume_panels = {
        asset: _volume_feature_panel(asset, cutoff) for asset in ALL_ASSETS
    }
    details: dict[str, dict[str, object]] = {}
    enriched = []
    for _, episode in episodes.iterrows():
        start = pd.Timestamp(episode["decline_start"])
        trough = pd.Timestamp(episode["trough_date"])
        interval = daily.loc[start:trough]
        holdings = _holding_runs(
            targets, daily, start, trough, asset_names
        )
        candidate_runs = _candidate_runs(
            daily, start, trough, asset_names
        )
        primary_loss = min(
            holdings, key=lambda item: float(item["strategy_return"])
        )
        candidates = interval["candidate"].astype(str)
        classification = _sleeve_classification(candidates)
        defender_days = int(candidates.eq(DEFENDER_CANDIDATE).sum())
        momentum_days = int(len(candidates) - defender_days)
        formal_return = float(
            (1.0 + interval["return"].astype(float)).prod() - 1.0
        )
        pure_momentum_return, pure_defender_return = fixed_sleeve_returns(
            formal_run, start, trough
        )
        if abs(formal_return - float(episode["max_drawdown"])) > 1e-12:
            raise AssertionError("peak-to-trough return does not equal drawdown")
        if (
            classification == "纯Momentum"
            and abs(formal_return - pure_momentum_return)
            > MOMENTUM_REPLAY_PARITY_TOLERANCE
        ):
            raise AssertionError("pure Momentum episode does not match Momentum sleeve")
        if (
            classification == "纯Defender"
            and abs(formal_return - pure_defender_return) > 1e-12
        ):
            raise AssertionError("pure Defender episode does not match Defender sleeve")
        momentum_gap = formal_return - pure_momentum_return
        defender_gap = formal_return - pure_defender_return
        if abs(momentum_gap) <= MOMENTUM_REPLAY_PARITY_TOLERANCE:
            momentum_gap = 0.0
        if abs(defender_gap) <= 1e-12:
            defender_gap = 0.0
        next_changes = state.loc[state.index > trough]
        next_changes = next_changes.loc[
            next_changes["state_changed"].astype(bool)
        ]
        next_change = None
        if not next_changes.empty:
            timestamp = pd.Timestamp(next_changes.index[0])
            next_change = {
                "date": timestamp,
                "risk_on": bool(next_changes.iloc[0]["risk_on"]),
                "reason": str(next_changes.iloc[0]["state_reason"]),
                "w40_percentile": float(
                    formal_run.score_at_open.at[timestamp]
                ),
            }
        changes = state.loc[start:trough]
        changes = changes.loc[changes["state_changed"].astype(bool)]
        state_changes = [
            {
                "date": pd.Timestamp(timestamp),
                "risk_on": bool(row["risk_on"]),
                "reason": str(row["state_reason"]),
                "w40_percentile": float(
                    formal_run.score_at_open.at[timestamp]
                ),
            }
            for timestamp, row in changes.iterrows()
        ]
        case_start = start.date().isoformat()
        row = dict(episode)
        row.update(
            {
                "case_start": case_start,
                "sleeve_classification": classification,
                "momentum_days": momentum_days,
                "defender_days": defender_days,
                "formal_interval_return": formal_return,
                "pure_momentum_return": pure_momentum_return,
                "pure_defender_return": pure_defender_return,
                "formal_vs_momentum_gap": momentum_gap,
                "formal_vs_defender_gap": defender_gap,
                "primary_loss_portfolio": str(primary_loss["portfolio"]),
                "primary_loss_return": float(primary_loss["strategy_return"]),
                "start_w40_loss": float(
                    formal_run.raw_loss_at_open.at[start]
                ),
                "start_w40_percentile": float(
                    formal_run.score_at_open.at[start]
                ),
                "trough_w40_loss": float(
                    formal_run.raw_loss_at_open.at[trough]
                ),
                "trough_w40_percentile": float(
                    formal_run.score_at_open.at[trough]
                ),
                "start_risk_on": bool(state.at[start, "risk_on"]),
                "trough_risk_on": bool(state.at[trough, "risk_on"]),
            }
        )
        enriched.append(row)
        details[case_start] = {
            "holdings": holdings,
            "candidate_runs": candidate_runs,
            "worst_days": _worst_days(daily, start, trough),
            "state_changes": state_changes,
            "next_state_change": next_change,
            "peak_volume": _peak_volume_record(
                episode,
                targets,
                volume_panels,
                asset_names,
                allow_unavailable=True,
            ),
        }
    result = pd.DataFrame(enriched)
    result.attrs.update(episodes.attrs)
    result.attrs["details"] = details
    result.attrs["volume_population"] = _volume_population_summary(
        all_episodes,
        targets,
        volume_panels,
        asset_names,
        top_n=top_n,
    )
    return result


def _date(value: object) -> str:
    if pd.isna(value):
        return "开放"
    return pd.Timestamp(value).date().isoformat()


def _date_range(start: object, end: object) -> str:
    return f"{_date(start)}—{_date(end)}"


def _number_or_na(value: object, digits: int = 2, suffix: str = "") -> str:
    number = float(value)
    return f"{number:.{digits}f}{suffix}" if np.isfinite(number) else "N/A"


def _percent_or_na(value: object) -> str:
    number = float(value)
    return f"{number:+.2%}" if np.isfinite(number) else "N/A"


def _state_change_text(change: dict[str, object]) -> str:
    sleeve = "Momentum" if bool(change["risk_on"]) else "Defender"
    return (
        f"{_date(change['date'])}切至{sleeve}（`{change['reason']}`，"
        f"W40分位{float(change['w40_percentile']):.2%}）"
    )


def _render_document(
    episodes: pd.DataFrame,
    context_config: dict,
) -> str:
    contexts = context_config["cases"]
    asset_names = context_config["asset_names"]
    details = episodes.attrs["details"]
    volume_population = episodes.attrs["volume_population"]
    observed = set(episodes["case_start"].astype(str))
    configured = set(map(str, contexts))
    if configured != observed:
        raise AssertionError(
            "drawdown context coverage mismatch; "
            f"missing={sorted(observed - configured)}, "
            f"stale={sorted(configured - observed)}"
        )

    class_counts = Counter(episodes["sleeve_classification"])
    volume_class_counts = Counter(
        details[str(case_start)]["peak_volume"]["classification"]
        for case_start in episodes["case_start"]
    )
    comparison_tolerance = 1e-7
    pure_momentum_better = int(
        episodes["pure_momentum_return"].gt(
            episodes["formal_interval_return"] + comparison_tolerance
        ).sum()
    )
    formal_better_than_momentum = int(
        episodes["formal_interval_return"].gt(
            episodes["pure_momentum_return"] + comparison_tolerance
        ).sum()
    )
    momentum_equal = len(episodes) - pure_momentum_better - formal_better_than_momentum
    pure_defender_better = int(
        episodes["pure_defender_return"].gt(
            episodes["formal_interval_return"] + comparison_tolerance
        ).sum()
    )
    overlap = [
        str(contexts[key]["related_defender_badcase"])
        for key in episodes["case_start"]
        if contexts[key].get("related_defender_badcase")
    ]
    lines = [
        "# 正式整体策略 Top 10 最大回撤 Badcase 台账",
        "",
        f"- 正式策略：`{context_config['strategy_id']}`",
        f"- 证据截止：{context_config['evidence_cutoff']}",
        f"- 全部独立回撤：{int(episodes.attrs['all_episode_count'])}段；本表取最深{len(episodes)}段。",
        "- 回撤定义：正式复合NAV相对此前历史最高点的跌幅；连续水下期只记一段，嵌套低点不重复计数。",
        "- 持仓区间：从峰值后的第一个水下交易日到谷底，使用正式候选账本和Defender可执行目标还原。",
        "- 固定袖套对照：同期纯Momentum/纯Defender均假设区间开始前已经持有该袖套，复合各自"
        "每日held-net收益；包含袖套内部换仓费用，不人为加入首日新建仓腿。",
        "",
        "## 与 Defender 跑输 Momentum 台账的边界",
        "",
        "本台账记录的是**绝对资本损失**：整体策略净值从历史高点跌了多少、当时实际持有什么。",
        "现有[`momentum_defender_badcases.md`](momentum_defender_badcases.md)记录的是**反事实机会成本**：",
        "每一段Defender持仓与原Momentum同期比较，只有Momentum领先严格超过1个百分点才入账；",
        "即使正式策略本身上涨，也可能成为那本台账的badcase。两本台账可以在日期上重叠，但排名和",
        "归因口径互不替代。",
        "",
        f"Top 10中纯Momentum {class_counts.get('纯Momentum', 0)}段、混合路径"
        f" {class_counts.get('混合路径', 0)}段、纯Defender {class_counts.get('纯Defender', 0)}段。",
        f"同区间固定袖套对照中，纯Momentum优于正式路径{pure_momentum_better}段、正式路径优于"
        f"纯Momentum {formal_better_than_momentum}段、两者相同{momentum_equal}段；纯Defender优于"
        f"正式路径{pure_defender_better}段，其余为同一路径。",
        "注意：Top 10是按正式策略自身最差区间事后筛选，纯Defender在这些区间看起来更强是"
        "条件选择结果，不能外推为全历史应永久持有Defender。",
        f"峰值到谷底与旧台账事件相交的只有{len(overlap)}段（{', '.join(overlap) if overlap else '无'}）；",
        "这只是日期交集，不表示两类失败原因相同。",
        "",
        "市场背景均为事后解释，不是模型输入。无法从可靠来源确认单一催化剂的事件会降低置信度，",
        "不会用叙事替代正式收益和持仓证据。",
        "",
        "## 总览",
        "",
        "|ID|峰值日|谷底日|实际持仓类型|正式实际|同期纯Momentum|同期纯Defender|恢复日 / 水下期|主要持仓 / 原因|",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for _, row in episodes.iterrows():
        lines.append(
            f"|DD-{int(row['rank']):02d}|{_date(row['peak_date'])}|"
            f"{_date(row['trough_date'])}|{row['sleeve_classification']}|"
            f"{row['formal_interval_return']:+.2%}|"
            f"{row['pure_momentum_return']:+.2%}|"
            f"{row['pure_defender_return']:+.2%}|"
            f"{_date(row['recovery_date'])} / {int(row['underwater_sessions'])}日|"
            f"{row['primary_loss_portfolio']}（最大亏损段{row['primary_loss_return']:+.2%}）；"
            f"{contexts[str(row['case_start'])]['summary_reason']}|"
        )

    lines.extend(
        [
            "",
            "## 峰值量能：规律、例外与边界",
            "",
            "量能使用峰值日实际目标中权重最高ETF的本地成交量。`量/20日中位`以严格早于峰值日的"
            "20个交易日为基准；`Z60`用严格滞后的60日对数成交量均值和标准差；`P252`是峰值日"
            "成交量相对此前252日的历史分位。所有价格涨幅和成交量都只用于事后诊断。",
            "",
            "|ID|峰值实际持仓|峰值日 / 5日 / 20日涨幅|量/20日中位|Z60|P252|峰前5日最大量比|量能类型|",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in episodes.iterrows():
        volume = details[str(row["case_start"])]["peak_volume"]
        lines.append(
            f"|DD-{int(row['rank']):02d}|{volume['asset']}（{volume['asset_name']}）"
            f"{float(volume['weight']):.0%}|{_percent_or_na(volume['asset_return_1d'])} / "
            f"{_percent_or_na(volume['asset_return_5d'])} / "
            f"{_percent_or_na(volume['asset_return_20d'])}|"
            f"{_number_or_na(volume['volume_ratio_to_prior20_median'], suffix='×')}|"
            f"{_number_or_na(volume['log_volume_z60'])}|"
            f"{_number_or_na(float(volume['volume_percentile_prior252']) * 100.0, digits=1, suffix='%')}|"
            f"{_number_or_na(volume['prior5_max_volume_ratio'], suffix='×')}|"
            f"{volume['classification']}|"
        )
    lines.extend(
        [
            "",
            f"Top 10中明确峰值日强放量{volume_class_counts.get('峰值日强放量', 0)}段、"
            f"临界强放量{volume_class_counts.get('临界强放量', 0)}段、高量平台/量能偏高"
            f"{volume_class_counts.get('高量平台 / 量能偏高', 0)}段、峰前放量但峰值日回落"
            f"{volume_class_counts.get('峰前放量、峰日回落', 0)}段、无明显放量例外"
            f"{volume_class_counts.get('无明显放量（例外）', 0)}段。",
            "",
            f"为避免只看Top 10产生见顶错觉，又对全部{int(episodes.attrs['all_episode_count'])}段中"
            f"可计算峰值量能的{int(volume_population['feature_available_episode_count'])}段进行筛选；其中"
            f"{int(volume_population['eligible_episode_count'])}段峰值主资产权重不低于80%的事件做了"
            f"相同计算。Top 10峰值量比中位数为{float(volume_population['top_median_volume_ratio']):.2f}×、"
            f"Z中位数{float(volume_population['top_median_volume_z']):.2f}，其余事件分别为"
            f"{float(volume_population['other_median_volume_ratio']):.2f}×和"
            f"{float(volume_population['other_median_volume_z']):.2f}；强放量占比为"
            f"{float(volume_population['top_strong_volume_share']):.1%}对"
            f"{float(volume_population['other_strong_volume_share']):.1%}。",
            "",
            f"但在这{int(volume_population['eligible_episode_count'])}段中，峰值量比与后续回撤深度的"
            f"Spearman相关仅{float(volume_population['spearman_volume_ratio_vs_depth']):.3f}，"
            f"Z60与深度也只有{float(volume_population['spearman_volume_z_vs_depth']):.3f}。因此放量"
            "在最严重回撤里更常见，却不存在稳定的单调预测关系：它更像趋势加速、拥挤参与和"
            "价格发现的背景变量，不能单独当作见顶信号。",
            "",
            "两点口径限制必须保留：第一，成交量是中国场内ETF自身成交量；纳指ETF和黄金ETF的"
            "量能还受时区、申赎、额度与溢折价影响，不等于纳斯达克或全球黄金市场总成交量。"
            "第二，Top 10按正式策略事后最深回撤筛选，放量比例天然带有条件选择，不能据此修改"
            "冻结策略。",
        ]
    )

    lines.extend(
        [
            "",
            "## 横向结论",
            "",
            "- 最大风险仍是Top-1在极端反转或跳空中100%集中暴露；黄金逃生只覆盖黄金成为Top1"
            "且QM20差值达标的情形，不能处理创业板和纳指自身尾部。",
            "- 510300单锚W40对纳指和创业板独立尾部事件天然失明；黄金专用覆盖修复一部分避险趋势，"
            "但不能把观察对象与所有实际持仓完全对齐。",
            "- 基础Defender使用5日最低观察、QM40连续10日早退和30日W40保底；确认与黄金硬持有"
            "仍会带来执行延迟；"
            "收盘信号仍无法提前规避当日跳空或暴跌。",
            "- 黄金逃生与100%红利Defender都不是现金：两种风险资产交替持有时仍可能形成连续"
            "水下期。",
            "- 多个Top 10事件是复合路径：Momentum、红利Defender和黄金逃生先后受损。只给整段"
            "贴一个宏观标签会掩盖真正的候选切换与损失来源。",
            "",
            "## 逐案记录",
            "",
        ]
    )
    for _, row in episodes.iterrows():
        case_id = str(row["case_start"])
        context = contexts[case_id]
        event = details[case_id]
        recovery = _date(row["recovery_date"])
        lines.extend(
            [
                f"### DD-{int(row['rank']):02d}：{_date(row['peak_date'])}峰值，"
                f"{_date(row['trough_date'])}谷底",
                "",
                f"置信度：**{str(context['confidence']).upper()}**。",
                "",
                "|项目|结果|",
                "|---|---:|",
                f"|最大回撤|{row['max_drawdown']:.2%}|",
                f"|首个水下日|{_date(row['decline_start'])}|",
                f"|跌至谷底|{int(row['decline_sessions'])}个交易日|",
                f"|恢复前高|{recovery}（水下{int(row['underwater_sessions'])}个交易日）|",
                f"|Momentum / Defender日|{int(row['momentum_days'])} / {int(row['defender_days'])}|",
                f"|实际持仓类型|{row['sleeve_classification']}|",
                f"|正式实际收益|{row['formal_interval_return']:+.2%}|",
                f"|同期纯Momentum收益|{row['pure_momentum_return']:+.2%}|",
                f"|同期纯Defender收益|{row['pure_defender_return']:+.2%}|",
                f"|正式相对纯Momentum|{row['formal_vs_momentum_gap']:+.2%}|",
                f"|正式相对纯Defender|{row['formal_vs_defender_gap']:+.2%}|",
            ]
        )
        if row["sleeve_classification"] == "混合路径":
            lines.extend(["", "**混合路径顶层持仓变化**", ""])
            for candidate_run in event["candidate_runs"]:
                lines.append(
                    f"- {_date_range(candidate_run['start'], candidate_run['end'])}："
                    f"{candidate_run['label']}，{int(candidate_run['sessions'])}日，"
                    f"正式策略该段{float(candidate_run['strategy_return']):+.2%}。"
                )
            lines.extend(
                [
                    "",
                    "以上按正式候选连续持有段切分；每次Defender段内部的红利ETF目标变化"
                    "继续在下方逐日目标分段中展开。",
                ]
            )
        lines.extend(["", "**峰值到谷底的实际持仓**", ""])
        for holding in event["holdings"]:
            lines.append(
                f"- {_date_range(holding['start'], holding['end'])}："
                f"{holding['portfolio']}，{int(holding['sessions'])}日，"
                f"正式策略该段{float(holding['strategy_return']):+.2%}。"
            )
        lines.extend(
            [
                "",
                "切换日收益会复合旧持仓的昨收→今开退出腿与新持仓的今开→收盘进入腿，因此上述",
                "分段收益属于正式策略账本，不应误读为新持仓自身的纯价格收益。",
                "",
                "**最差交易日**",
                "",
            ]
        )
        for worst in event["worst_days"]:
            legs = ""
            if pd.notna(worst["exit_leg"]) or pd.notna(worst["enter_leg"]):
                legs = (
                    f"；退出腿{float(worst['exit_leg']):+.2%}，"
                    f"进入腿{float(worst['enter_leg']):+.2%}"
                )
            lines.append(
                f"- {_date(worst['date'])}：{float(worst['return']):+.2%}，"
                f"`{worst['transition']}`{legs}。"
            )

        volume = event["peak_volume"]
        lines.extend(
            [
                "",
                "**峰值量能诊断**",
                "",
                f"峰值日实际主持仓为{volume['asset']}（{volume['asset_name']}）"
                f"{float(volume['weight']):.0%}；当日、5日、20日涨幅分别为"
                f"{_percent_or_na(volume['asset_return_1d'])}、"
                f"{_percent_or_na(volume['asset_return_5d'])}、"
                f"{_percent_or_na(volume['asset_return_20d'])}。成交量为此前20日中位数的"
                f"{_number_or_na(volume['volume_ratio_to_prior20_median'])}倍，60日对数成交量"
                f"Z={_number_or_na(volume['log_volume_z60'])}，位于此前252日"
                f"{_number_or_na(float(volume['volume_percentile_prior252']) * 100.0, digits=1, suffix='%')}分位；峰前5日最大量比为"
                f"{_number_or_na(volume['prior5_max_volume_ratio'])}倍。分类为"
                f"**{volume['classification']}**。",
                "",
                str(context["volume_context"]),
            ]
        )

        start_sleeve = "Momentum" if bool(row["start_risk_on"]) else "Defender"
        trough_sleeve = "Momentum" if bool(row["trough_risk_on"]) else "Defender"
        lines.extend(
            [
                "",
                "**门控与机械归因**",
                "",
                f"首个水下日处于{start_sleeve}，510300 W40下跌幅度/分位为"
                f"{row['start_w40_loss']:.2%}/{row['start_w40_percentile']:.2%}；"
                f"谷底处于{trough_sleeve}，对应数值为"
                f"{row['trough_w40_loss']:.2%}/{row['trough_w40_percentile']:.2%}。"
                f"最大亏损持仓段为{row['primary_loss_portfolio']}，正式策略该段"
                f"{row['primary_loss_return']:+.2%}。",
                "",
            ]
        )
        if event["state_changes"]:
            lines.append(
                "峰值至谷底状态变化："
                + "；".join(
                    _state_change_text(change)
                    for change in event["state_changes"]
                )
                + "。"
            )
        else:
            lines.append("峰值至谷底没有发生顶层Momentum/Defender状态切换。")
        if event["next_state_change"] is not None:
            lines.append(
                "谷底后的下一次顶层状态变化为"
                + _state_change_text(event["next_state_change"])
                + "。"
            )
        lines.extend(
            [
                "",
                "**历史背景与为什么下跌**",
                "",
                str(context["market_context"]),
                "",
                str(context["why_fell"]),
                "",
                f"可复用薄弱环节：**{context['weakness']}**",
            ]
        )
        related = context.get("related_defender_badcase")
        if related:
            lines.extend(
                [
                    "",
                    f"与旧台账的关系：本段与`{related}`日期相交，但旧事件衡量Defender相对"
                    "Momentum的机会成本；本事件衡量整体净值相对前高的绝对损失。",
                ]
            )
        sources = context.get("sources", [])
        if sources:
            lines.extend(["", "参考背景："])
            for source in sources:
                lines.append(f"- [{source['title']}]({source['url']})")
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## 更新规则",
            "",
            "正式策略状态机、参数、标的池、Defender实现、费用、行情或证据截止日变化后，必须重新运行：",
            "",
            "```bash",
            "uv run python -m research.generate_strategy_drawdown_badcases",
            "uv run python -m research.generate_strategy_drawdown_badcases --check",
            "```",
            "",
            "生成器会重新识别全部独立水下期并选最深十段；上下文配置必须恰好覆盖当前Top 10。"
            "排名、起点或事件集合变化时，校验会失败，必须人工重审持仓、原因、历史背景和来源。",
            "",
            "本台账用于识别机制边界，不授权在同一历史上继续调整冻结的W40窗口、阈值或锁。",
            "",
        ]
    )
    return "\n".join(lines)


def generate(
    root: Path,
    context_path: Path,
) -> tuple[str, pd.DataFrame]:
    config = yaml.safe_load(context_path.read_text(encoding="utf-8"))
    if config["strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("drawdown context strategy ID is not formal")
    cutoff = pd.Timestamp(config["evidence_cutoff"])
    formal_run = run_formal_strategy(root, end=cutoff.date())
    actual_cutoff = formal_run.daily.index.max().date().isoformat()
    if str(config["evidence_cutoff"]) != actual_cutoff:
        raise AssertionError(
            f"context cutoff {config['evidence_cutoff']} != data cutoff {actual_cutoff}"
        )
    episodes = build_drawdown_evidence(
        formal_run,
        top_n=int(config["top_n"]),
        asset_names=config["asset_names"],
    )
    return _render_document(episodes, config), episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    context_path = args.context if args.context.is_absolute() else root / args.context
    output_path = args.output if args.output.is_absolute() else root / args.output
    document, episodes = generate(root, context_path)
    if args.check:
        if not output_path.exists():
            raise SystemExit(f"drawdown badcase document missing: {output_path}")
        if output_path.read_text(encoding="utf-8") != document:
            raise SystemExit(
                "drawdown badcase document is stale; regenerate and review context"
            )
        print(
            f"drawdown badcase document current: "
            f"{episodes.attrs['all_episode_count']} episodes, Top {len(episodes)}"
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(
        f"wrote {output_path}: {episodes.attrs['all_episode_count']} episodes, "
        f"Top {len(episodes)}"
    )


if __name__ == "__main__":
    main()
