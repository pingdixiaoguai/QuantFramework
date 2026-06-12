# -*- coding: utf-8 -*-
"""
PTrade 策略 — 质量动量 Top1（quality_momentum_top1 的实盘迁移版）
==================================================================

本文件是 QuantFramework 中 `strategy/configs/quality_momentum_top1.yaml`
策略在恒生 PTrade 平台上的等价实现。直接粘贴到 PTrade 策略编辑器即可运行
（回测 / 模拟 / 实盘）。

策略逻辑（与框架严格对齐）
--------------------------
1. 因子：质量动量 = 动量 × Kaufman 效率比率（window=20）
       momentum = close_t / close_{t-20} - 1
       ER       = |close_t - close_{t-20}| / Σ|close.diff()|（窗口内 20 个日变动绝对值之和）
       score    = momentum × ER
   ER ∈ [0,1]：奖励路径平滑的「干净趋势」，惩罚靠少数大阳线拉起的颠簸趋势。
2. 选股：Top1 —— 全仓 score 最高的单一 ETF。
3. 调仓时序：T 日收盘后用 T 及之前数据算 score，T+1 开盘成交。
   本文件用 run_daily(time='09:30') 在开盘执行，get_history(include=False)
   取到「昨日及之前」的收盘价，从而复现「信号 T、成交 T+1 开盘」的约定。
4. 最短持有期：rebalance_days=2，min_hold 模式 —— 建仓后至少持有 2 个交易日
   才允许重新评估切换，降低 whipsaw 摩擦。
5. 复权：get_history(fq='post') 后复权，对应框架 2026-05-25 起的 HFQ 口径。
   动量/ER 都是窗口内的比率与差分，后复权能正确连续化分红/拆分，与框架等价。

代码后缀转换（框架 .SH → PTrade .SS）
--------------------------------------
PTrade 用 .SS（上交所）/ .SZ（深交所）。框架里的 .SH 统一改成 .SS。

⚠️ 已知口径分歧（迁移时如实保留，部署前请确认）
------------------------------------------------
- rebalance_days：当前部署的 yaml 是 2，但 strategy_changelog.md 的 v0 基准
  记录是 5。此处默认取 2（与最新 yaml 一致）。要切回 5 改 REBALANCE_DAYS 即可。
- 交易成本：框架回测当前未扣成本（已知缺陷，§3.2 最高优先级待办）。PTrade
  实盘按券商真实费率成交；回测里用 set_commission 模拟，见 initialize()。
- 基准：框架基准是「四资产等权」，PTrade 的 set_benchmark 只能设单一标的，
  此处用沪深300指数（000300.SS）作约定基准，绩效对比口径与框架不同。
"""

import numpy as np

# ============================ 可调参数 ============================
# 资产池（框架 .SH → PTrade .SS）。顺序不影响结果。
SECURITIES = [
    "510300.SS",   # 沪深300 ETF
    "159915.SZ",   # 创业板 ETF
    "513100.SS",   # 纳斯达克 ETF
    "518880.SS",   # 黄金 ETF
]

WINDOW = 20                 # 动量 / ER 回看窗口（交易日）
REBALANCE_DAYS = 2          # 最短持有期；≥该天数后才允许重新评估切换（见上方口径分歧说明）
REBALANCE_MODE = "min_hold" # "min_hold"（满 N 日后每日评估）或 "fixed_cycle"（仅第 N、2N… 日评估）
INVEST_RATIO = 0.98         # 目标仓位占总资产比例。<1 留缓冲，避免费用/价格波动导致下单资金不足。
                            # 如券商成交可靠、想更贴近框架的 100% 满仓，可调到 0.99~1.0。
REBALANCE_TIME = "09:30"    # 开盘调仓时间。如需走集合竞价可改 '09:25' 并配合限价单。
BENCHMARK = "000300.SS"     # 约定基准（沪深300指数）；框架真实基准为四资产等权。
HISTORY_BARS = WINDOW + 20  # 每次取的历史 bar 数。因子只需 WINDOW+1，多取是安全垫：
                            # get_history(count) 按交易日历对齐，标的停牌当日会缺失（研究环境
                            # 探针实测 159915.SZ 在 25 日窗口里只返回 24 条），多取避免缓冲不足。


# ====================== 调仓时序判定（移植自 strategy/rebalance.py）======================
def _should_hold(held, holding_days, rebalance_days, mode):
    """返回 True 表示应压制今日信号、保持当前持仓（最短持有期 / 固定周期约束）。"""
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be >= 1, got %s" % rebalance_days)
    if rebalance_days <= 1:
        return False               # 每日调仓，从不压制
    if held is None:
        return False               # 空仓，允许首次建仓
    if holding_days is None:
        return True                # 刚建仓、今日 bar 尚未反映，保持以避免当日反复
    if mode == "min_hold":
        return holding_days < rebalance_days
    # fixed_cycle：仅在持仓第 N、2N、3N… 个交易日评估
    return holding_days % rebalance_days != 0


# ====================== 质量动量因子（移植自 factors/quality_momentum.py）======================
def _quality_momentum_score(closes, window):
    """用一段后复权收盘价序列计算质量动量 score 的最新值。

    closes: 按时间升序的收盘价（list / ndarray），长度需 >= window + 1。
    返回 float；数据不足或路径长度为 0（ER 无定义）时返回 None（该资产本日不参与排序）。
    """
    if closes is None or len(closes) < window + 1:
        return None
    c = np.asarray(closes, dtype=float)
    c = c[-(window + 1):]                       # 取最后 window+1 个点
    if np.any(np.isnan(c)) or c[0] == 0:
        return None
    momentum = c[-1] / c[0] - 1.0               # close_t / close_{t-window} - 1
    displacement = abs(c[-1] - c[0])            # |总位移|
    path_length = float(np.sum(np.abs(np.diff(c))))  # 窗口内每日变动绝对值之和（路径总长度）
    if path_length == 0.0:
        return None                             # 对应框架 path_length.replace(0, nan)
    er = displacement / path_length             # Kaufman 效率比率 ∈ [0,1]
    return momentum * er


# ============================ PTrade 框架函数 ============================
def initialize(context):
    """策略初始化（启动时执行一次）。"""
    set_universe(SECURITIES)
    set_benchmark(BENCHMARK)

    # —— 以下 set_commission / set_slippage 仅影响回测；实盘按券商真实费率成交 ——
    # ETF 无印花税，佣金按 ETF 类型设置（此处万分之二、单笔最低 5 元，按需调整）。
    try:
        set_commission(commission_ratio=0.0002, min_commission=5.0, type="ETF")
    except Exception as e:
        log.info("set_commission skipped: %s" % e)
    # 框架假设通过集合竞价实现 ≈0 滑点，这里设固定滑点 0；做敏感性测试时可调大。
    try:
        set_fixed_slippage(0.0)
    except Exception as e:
        log.info("set_fixed_slippage skipped: %s" % e)

    # —— 策略状态（PTrade 会序列化 g，跨交易日 / 重启持久化）——
    g.held = None        # 当前持有的标的代码；None 表示空仓
    g.held_days = 0      # 当前持仓已持有的交易日数（含建仓当日的口径见 rebalance 注释）

    # 开盘定时调仓，复现「T 收盘信号 → T+1 开盘成交」
    run_daily(context, rebalance, time=REBALANCE_TIME)
    log.info("initialized: universe=%s rd=%d mode=%s" % (SECURITIES, REBALANCE_DAYS, REBALANCE_MODE))


def handle_data(context, data):
    """PTrade 必须定义；本策略全部逻辑在 run_daily 的 rebalance 中，这里留空。"""
    pass


def rebalance(context):
    """每个交易日开盘执行一次：算因子 → Top1 → 最短持有期过滤 → 调仓。"""
    # ---- 0. 与券商真实持仓对账（处理重启 / 手工干预 / 未成交）----
    actual_held = [
        s for s in SECURITIES
        if s in context.portfolio.positions
        and getattr(context.portfolio.positions[s], "amount", 0) > 0
    ]
    if g.held is None and actual_held:
        # 重启后 g 丢失但实际有仓：采用实际持仓；持有天数未知，按「已满窗口」处理以正常评估
        g.held = actual_held[0]
        g.held_days = REBALANCE_DAYS
    elif g.held is not None and g.held not in actual_held:
        # g 认为有仓但实际没有（外部平仓 / 上一笔买单未成交）：以实际为准
        g.held = actual_held[0] if actual_held else None
        g.held_days = REBALANCE_DAYS if actual_held else 0

    # ---- 1. 计算「截至今日开盘」的持有天数 ----
    if g.held is not None:
        g.held_days += 1            # 又持有了一个交易日；建仓当日置 0，故次日为 1，与框架 holding_days 对齐
        holding_days = g.held_days
    else:
        holding_days = None

    # ---- 2. 最短持有期 / 固定周期过滤 ----
    if _should_hold(g.held, holding_days, REBALANCE_DAYS, REBALANCE_MODE):
        log.info("hold window active: held=%s holding_days=%s/%d mode=%s — 不调仓"
                 % (g.held, holding_days, REBALANCE_DAYS, REBALANCE_MODE))
        return

    # ---- 3. 计算各标的质量动量 score（仅用昨日及之前的后复权收盘价）----
    scores = {}
    for s in SECURITIES:
        try:
            df = get_history(HISTORY_BARS, "1d", "close", s, fq="post", include=False)
        except Exception as e:
            log.info("get_history failed for %s: %s" % (s, e))
            continue
        if df is None or len(df) < WINDOW + 1:
            continue
        try:
            closes = df["close"].values
        except Exception:
            closes = np.asarray(df).reshape(-1)   # 兜底：极少数版本列名不同
        val = _quality_momentum_score(closes, WINDOW)
        if val is not None:
            scores[s] = val

    if not scores:
        log.info("无可用 score（数据不足），跳过本日调仓。")
        return

    best = max(scores, key=lambda k: scores[k])
    log.info("scores=%s -> best=%s" % ({k: round(v, 4) for k, v in scores.items()}, best))

    # ---- 4. 若 Top1 变化则切换 ----
    if best != g.held:
        positions = context.portfolio.positions
        # 待换出：除 best 外所有有持仓的标的
        to_sell = [s for s in positions
                   if s != best and getattr(positions[s], "amount", 0) > 0]
        # A 股 T+1：当日买入的股票当日不可卖（enable_amount=0）。正常情况下因有最短持有期，
        # 待换出的都是过了 T+1 的旧仓；但若遇到不可卖（如重启后对账采用了当日新仓），本日不切换、
        # 下个交易日重试，避免卖单被取消却又买入新仓导致超额持仓。
        not_sellable = [s for s in to_sell
                        if getattr(positions[s], "enable_amount", 0) <= 0]
        if not_sellable:
            log.info("待换出标的今日不可卖（T+1）：%s — 本日不切换，下个交易日重试。" % not_sellable)
            return
        for s in to_sell:
            order_target(s, 0)                     # 全部卖出旧标的（卖出资金当日可用于买入）
            log.info("卖出 %s（清仓）" % s)
        # 用 order_target_value 按市值下单：回测/实盘的真实成交价与 get_history 后复权价基准不同，
        # 按市值下单由平台内部用真实价折算股数，避免复权因子错配。
        target_value = context.portfolio.total_value * INVEST_RATIO
        order_target_value(best, target_value)     # 买入新标的到目标市值
        log.info("买入 %s 到目标市值 %.2f（总资产 %.2f × %.2f）"
                 % (best, target_value, context.portfolio.total_value, INVEST_RATIO))
        g.held = best
        g.held_days = 0                            # 建仓当日置 0；次日 +1 变 1
    else:
        log.info("Top1 未变（%s），维持持仓。" % best)


def after_trading_end(context, data):
    """收盘后记录持仓快照，便于核对。"""
    held = [(s, p.amount) for s, p in context.portfolio.positions.items()
            if getattr(p, "amount", 0) > 0]
    log.info("收盘持仓: %s | g.held=%s held_days=%s | 总资产=%.2f 可用现金=%.2f"
             % (held, g.held, g.held_days, context.portfolio.total_value, context.portfolio.cash))
