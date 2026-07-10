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
   本文件用 run_daily(time='09:31') 在开盘执行，get_history(include=False)
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
- rebalance_days：本实盘当前 = 2，与框架基准 rd=5 刻意不一致（非配置漂移）。
  框架 2026-06-13 经三闸门评估已否决 rd=2、回滚 rd=5（changelog §3.7/§3.8）；
  此处 2026-06-15 经决策保留 rd=2 作外部对照样本。详见 PTRADE_MIGRATION.md §5.1。
  要与框架对齐改 REBALANCE_DAYS = 5（切换须在 §5.1 补记日期与理由）。
- 交易成本：框架回测当前未扣成本（已知缺陷，§3.2 最高优先级待办）。PTrade
  实盘按券商真实费率成交；回测里用 set_commission 模拟，见 initialize()。
- 基准：框架基准是「四资产等权」，PTrade 的 set_benchmark 只能设单一标的，
  此处用沪深300指数（000300.SS）作约定基准，绩效对比口径与框架不同。
"""

import numpy as np
import traceback

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
REBALANCE_TIME = "09:31"    # 开盘后调仓时间。探针实测 run_daily 在日线回测固定 09:31 触发；
                            # 避免用 09:30 整点（部分版本会拒绝整点开盘，导致 initialize 抛错）。
BENCHMARK = "000300.SS"     # 约定基准（沪深300指数）；框架真实基准为四资产等权。
COMMISSION_RATIO = 0.00005  # 回测佣金率，仅影响回测（实盘按券商真实费率成交）。
                            # 取万0.5 = 本账户真实 ETF 佣金。注意 rd 之间的优劣对摩擦敏感，
                            # 必须用真实费率才能做准（早先用万2 回测高估了高换手配置的拖累）。
HISTORY_BARS = WINDOW + 20  # 每次取的历史 bar 数。因子只需 WINDOW+1，多取是安全垫：
                            # get_history(count) 按交易日历对齐，标的停牌当日会缺失（研究环境
                            # 探针实测 159915.SZ 在 25 日窗口里只返回 24 条），多取避免缓冲不足。
ORDER_POLL_SECONDS = 10     # 交易模式轮询柜台持仓/未完成订单。文档提示持仓同步通常约需 6 秒。
PRICE_PROTECTION_RATIO = 0.002  # 限价相对对手一档放宽 0.2%，兼顾成交概率与滑点上限。

_ORDER_FAILURE_STATUSES = ("5", "6", "9")  # 部撤 / 已撤 / 废单


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


# ============================ 交易模式执行状态机 ============================
def _set_state_default(name, value):
    """只初始化缺失字段，避免 initialize 重启时覆盖 PTrade 已序列化的 g 状态。"""
    if not hasattr(g, name):
        setattr(g, name, value)


def _ensure_execution_state():
    _set_state_default("held", None)
    _set_state_default("held_days", 0)
    _set_state_default("pending_phase", None)       # None / selling / buying
    _set_state_default("pending_target", None)
    _set_state_default("pending_buy_required", False)
    _set_state_default("pending_sell_orders", {})   # security -> order_id
    _set_state_default("pending_buy_order", None)
    _set_state_default("pending_error", None)
    _set_state_default("pending_buy_failed", False)
    _set_state_default("pending_stale_checks", 0)
    _set_state_default("processing_pending", False)
    _set_state_default("needs_top_up", False)


def _is_trade_mode():
    try:
        checker = is_trade
    except NameError:
        return False
    return bool(checker())


def _normalize_security(security):
    if not security:
        return security
    return str(security).replace(".XSHG", ".SS").replace(".XSHE", ".SZ")


def _positive_positions(context):
    return {
        _normalize_security(s): p
        for s, p in context.portfolio.positions.items()
        if getattr(p, "amount", 0) > 0
    }


def _primary_actual_held(context):
    """返回真实主仓；多仓时优先保留 g.held，否则取市值最大者。"""
    positions = _positive_positions(context)
    if not positions:
        return None
    if g.held in positions:
        return g.held

    def _position_value(item):
        position = item[1]
        value = getattr(position, "market_value", None)
        if value is not None:
            return float(value)
        return float(getattr(position, "amount", 0)) * float(
            getattr(position, "last_sale_price", 0)
        )

    return max(positions.items(), key=_position_value)[0]


def _sync_held_from_actual(context):
    """以真实持仓校正 g.held；未知入场日一律按 0 天保守恢复。"""
    actual = _primary_actual_held(context)
    if actual != g.held:
        log.info("持仓状态校正: g.held=%s -> actual=%s（持有天数按 0 恢复）" % (g.held, actual))
        g.held = actual
        g.held_days = 0
    elif actual is None:
        g.held_days = 0
    return actual


def _order_field(order_obj, *names):
    for name in names:
        if isinstance(order_obj, dict) and name in order_obj:
            return order_obj[name]
        value = getattr(order_obj, name, None)
        if value is not None:
            return value
    return None


def _strategy_open_order_ids():
    """只返回本策略资产池的未完成订单，避免与其他策略/人工订单互相干扰。"""
    if not _is_trade_mode():
        return set()
    try:
        orders = get_open_orders() or []
    except Exception as e:
        log.info("get_open_orders failed: %s" % e)
        return None  # fail-closed：无法确认在途订单时禁止生成新单

    ids = set()
    for order_obj in orders:
        security = _normalize_security(_order_field(order_obj, "stock_code", "symbol", "sid"))
        order_id = _order_field(order_obj, "order_id", "id")
        if security in SECURITIES and order_id:
            ids.add(str(order_id))
    return ids


def _tracked_order_ids():
    ids = set(str(v) for v in g.pending_sell_orders.values() if v)
    if g.pending_buy_order:
        ids.add(str(g.pending_buy_order))
    return ids


def _quote_level_price(group, level=1):
    if not isinstance(group, dict):
        return 0.0
    row = group.get(level, group.get(str(level)))
    if not row:
        return 0.0
    try:
        return float(row[0])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _trade_limit_price(security, side):
    """从实时盘口生成显式 ETF 限价；无有效正价格时拒绝下单。"""
    try:
        snapshot = get_snapshot(security) or {}
    except Exception as e:
        log.info("get_snapshot failed for %s: %s" % (security, e))
        return None

    quote = snapshot.get(security, snapshot)
    if not isinstance(quote, dict):
        return None

    if side == "buy":
        base = _quote_level_price(quote.get("offer_grp"))
        if base <= 0:
            base = float(quote.get("last_px", 0) or 0)
        price = base * (1.0 + PRICE_PROTECTION_RATIO)
        up_limit = float(quote.get("up_px", 0) or 0)
        if up_limit > 0:
            price = min(price, up_limit)
    else:
        base = _quote_level_price(quote.get("bid_grp"))
        if base <= 0:
            base = float(quote.get("last_px", 0) or 0)
        price = base * (1.0 - PRICE_PROTECTION_RATIO)
        down_limit = float(quote.get("down_px", 0) or 0)
        if down_limit > 0:
            price = max(price, down_limit)

    # 本策略资产池全部为 ETF，委托价格精度为三位小数。
    price = round(price, 3)
    if base <= 0 or price <= 0:
        log.info("%s 无有效实时价格，取消本次%s委托。snapshot=%s" % (security, side, quote))
        return None
    return price


def _clear_pending(error=None):
    g.pending_phase = None
    g.pending_target = None
    g.pending_buy_required = False
    g.pending_sell_orders = {}
    g.pending_buy_order = None
    g.pending_error = error
    g.pending_buy_failed = False
    g.pending_stale_checks = 0


def _submit_trade_buy(context, target):
    """提交有显式限价的买单；只记录 pending，不提前修改 g.held。"""
    price = _trade_limit_price(target, "buy")
    if price is None:
        g.pending_error = "买入实时价格无效"
        return False

    target_value = float(context.portfolio.total_value) * INVEST_RATIO
    available_cash = float(context.portfolio.cash)
    positions = _positive_positions(context)
    current_position = positions.get(target)
    current_value = 0.0
    if current_position is not None:
        current_value = getattr(current_position, "market_value", None)
        if current_value is None:
            current_value = float(getattr(current_position, "amount", 0)) * float(
                getattr(current_position, "last_sale_price", 0)
            )
        current_value = float(current_value)
    required_cash = max(target_value - current_value, 0.0)
    if available_cash + 0.01 < required_cash:
        g.pending_error = "等待卖出资金到账"
        log.info("买入 %s 暂缓：目标增量 %.2f，可用资金 %.2f；等待下一次柜台同步"
                 % (target, required_cash, available_cash))
        return False

    g.pending_phase = "buying"
    g.pending_target = target
    g.pending_buy_order = None
    g.pending_buy_failed = False
    g.pending_error = None
    g.pending_stale_checks = 0
    order_id = order_target_value(target, target_value, limit_price=price)
    if not order_id:
        _clear_pending("买单创建失败")
        log.info("买入 %s 订单创建失败（limit_price=%.3f）" % (target, price))
        return False

    g.pending_buy_order = str(order_id)
    log.info("买入委托已提交 %s 到目标市值 %.2f，限价 %.3f，order_id=%s；等待真实持仓确认"
             % (target, target_value, price, order_id))
    return True


def _start_trade_switch(context, target, to_sell, buy_required=True):
    """交易模式换仓第一阶段：有旧仓时只卖，不在同一回调中买入。"""
    open_ids = _strategy_open_order_ids()
    if open_ids is None:
        log.info("无法确认未完成订单，本次不下单。")
        return False
    if open_ids:
        log.info("检测到本策略未完成订单 %s，本次不重复下单。" % sorted(open_ids))
        return False

    if not to_sell:
        return _submit_trade_buy(context, target) if buy_required else True

    g.pending_phase = "selling"
    g.pending_target = target
    g.pending_buy_required = bool(buy_required)
    g.pending_sell_orders = {}
    g.pending_buy_order = None
    g.pending_error = None
    g.pending_buy_failed = False
    g.pending_stale_checks = 0

    for security in to_sell:
        price = _trade_limit_price(security, "sell")
        if price is None:
            g.pending_error = "%s 卖出实时价格无效" % security
            continue
        order_id = order_target(security, 0, limit_price=price)
        if not order_id:
            g.pending_error = "%s 卖单创建失败" % security
            continue
        g.pending_sell_orders[security] = str(order_id)
        log.info("卖出委托已提交 %s（清仓/清理杂仓），限价 %.3f，order_id=%s；成交前不买新仓"
                 % (security, price, order_id))

    if not g.pending_sell_orders:
        error = g.pending_error or "没有卖单成功创建"
        _clear_pending(error)
        log.info("换仓已中止：%s" % error)
        return False
    return True


def _finish_confirmed_buy(context, target):
    g.held = target
    g.held_days = 0
    g.needs_top_up = bool(g.pending_buy_failed)
    error = g.pending_error
    _clear_pending(error)
    log.info("买入持仓已确认: held=%s；needs_top_up=%s" % (g.held, g.needs_top_up))


def _process_pending_switch(context):
    """由 run_interval 驱动：以真实持仓和未完成订单推进卖出→买入状态机。"""
    _ensure_execution_state()
    if not _is_trade_mode() or g.pending_phase is None or g.processing_pending:
        return

    g.processing_pending = True
    try:
        positions = _positive_positions(context)
        target = g.pending_target
        open_ids = _strategy_open_order_ids()
        if open_ids is None:
            return
        tracked_open = bool(_tracked_order_ids() & open_ids)

        if g.pending_phase == "selling":
            remaining = [s for s in positions if s != target]
            if remaining:
                if tracked_open:
                    g.pending_stale_checks = 0
                    return
                g.pending_stale_checks += 1
                if g.pending_stale_checks < 2 and not g.pending_error:
                    return  # 给已成交后的柜台持仓同步再留一个轮询周期
                error = g.pending_error or "卖单已终结但旧仓仍存在: %s" % remaining
                log.info("换仓卖出阶段未完成，停止本日链路：%s" % error)
                _clear_pending(error)
                _sync_held_from_actual(context)
                return

            # 真实旧仓已归零，才允许进入买入阶段。
            g.held = target if target in positions else None
            g.held_days = 0
            if not g.pending_buy_required and target in positions:
                _clear_pending()
                return
            _submit_trade_buy(context, target)
            return

        if g.pending_phase == "buying":
            if target in positions:
                if tracked_open and not g.pending_buy_failed:
                    # 部分成交仍在继续，先反映真实持仓但等待订单终态。
                    g.held = target
                    g.held_days = 0
                    return
                _finish_confirmed_buy(context, target)
                return

            if tracked_open:
                g.pending_stale_checks = 0
                return
            g.pending_stale_checks += 1
            if g.pending_stale_checks < 2 and not g.pending_buy_failed:
                return
            error = g.pending_error or "买单已终结但未形成持仓"
            log.info("买入阶段未完成：%s" % error)
            _clear_pending(error)
            _sync_held_from_actual(context)
    finally:
        g.processing_pending = False


def process_pending_switch(context):
    """PTrade run_interval 回调包装；异常只记录，不终止策略线程。"""
    try:
        _process_pending_switch(context)
    except Exception:
        log.info("process_pending_switch FAILED:\n" + traceback.format_exc())


def on_order_response(context, order_list):
    """记录委托终态；状态推进由轮询真实持仓完成，避免回调与柜台同步竞态。"""
    _ensure_execution_state()
    tracked = _tracked_order_ids()
    for update in order_list or []:
        order_id = _order_field(update, "order_id", "id")
        if not order_id or str(order_id) not in tracked:
            continue
        status = str(_order_field(update, "status") or "")
        error_info = _order_field(update, "error_info") or ""
        log.info("订单回报: order_id=%s status=%s error=%s" % (order_id, status, error_info))
        if status in _ORDER_FAILURE_STATUSES:
            g.pending_error = "order_id=%s status=%s %s" % (order_id, status, error_info)
            if g.pending_buy_order and str(order_id) == str(g.pending_buy_order):
                g.pending_buy_failed = True


def on_trade_response(context, trade_list):
    """成交主推仅触发一次快速核对；真正买单仍须等待真实旧仓归零。"""
    log.info("成交回报: %s" % (trade_list or []))
    process_pending_switch(context)


# ============================ PTrade 框架函数 ============================
def initialize(context):
    # 薄包装：捕获并打印初始化异常的完整 traceback（平台只报「空错误」时用于定位根因）。
    log.info("initialize: start")
    try:
        _initialize_impl(context)
    except Exception:
        log.info("initialize FAILED:\n" + traceback.format_exc())
        raise


def _initialize_impl(context):
    """策略初始化（启动时执行一次）。"""
    _ensure_execution_state()
    # 线程锁和同步计数不跨进程恢复，避免上次异常退出留下 processing_pending=True。
    g.processing_pending = False
    g.pending_stale_checks = 0
    set_universe(SECURITIES)
    set_benchmark(BENCHMARK)

    # —— 以下 set_commission / set_slippage 仅影响回测；实盘按券商真实费率成交 ——
    # ETF 无印花税，佣金率取 COMMISSION_RATIO（万0.5，本账户真实费率），单笔最低 5 元。
    try:
        set_commission(commission_ratio=COMMISSION_RATIO, min_commission=5.0, type="ETF")
    except Exception as e:
        log.info("set_commission skipped: %s" % e)
    # 框架假设通过集合竞价实现 ≈0 滑点，这里设固定滑点 0；做敏感性测试时可调大。
    try:
        set_fixed_slippage(0.0)
    except Exception as e:
        log.info("set_fixed_slippage skipped: %s" % e)
    # 成交比例：回测默认 0.25（单笔最多吃当期可成交量的 25%，超出部分直接丢弃，不挂单）。
    # 对「小账户交易大规模 ETF」过于保守，会导致大单部分成交、资金闲置、旧仓清不掉。设为 1.0
    # 允许足额成交（这些 ETF 的真实日成交量远大于本策略订单，对小账户合理）。
    # 注意：2014 年初 513100 等早期流动性极低的时段，即便 1.0 仍可能因真实成交量不足而部分成交
    # ——这是真实约束而非设置问题，必要时把回测起点设到 2015+ 规避早期低流动性失真。
    try:
        set_volume_ratio(1.0)
    except Exception as e:
        log.info("set_volume_ratio skipped: %s" % e)

    # 开盘定时调仓，复现「T 收盘信号 → T+1 开盘成交」
    run_daily(context, rebalance, time=REBALANCE_TIME)
    if _is_trade_mode():
        # 交易柜台的委托、成交、持仓更新是异步的。轮询只推进已有 pending，不生成新信号。
        run_interval(context, process_pending_switch, seconds=ORDER_POLL_SECONDS)
    log.info("initialized: universe=%s rd=%d mode=%s" % (SECURITIES, REBALANCE_DAYS, REBALANCE_MODE))


def handle_data(context, data):
    """PTrade 必须定义；本策略全部逻辑在 run_daily 的 rebalance 中，这里留空。"""
    pass


def rebalance(context):
    # 薄包装：单日异常不应中断整个回测（与框架「记录警告而非中断」一致），并打印 traceback。
    try:
        _rebalance_impl(context)
    except Exception:
        log.info("rebalance FAILED:\n" + traceback.format_exc())


def _rebalance_impl(context):
    """每个交易日开盘执行一次：算因子 → Top1 → 最短持有期过滤 → 调仓。"""
    _ensure_execution_state()
    trade_mode = _is_trade_mode()

    # ---- 0. 与券商真实持仓/在途订单对账（处理重启、手工干预、拒单、部分成交）----
    if trade_mode and g.pending_phase is not None:
        _process_pending_switch(context)
        log.info("pending switch active: phase=%s target=%s sell_orders=%s buy_order=%s — 本次不重复下单"
                 % (g.pending_phase, g.pending_target, g.pending_sell_orders, g.pending_buy_order))
        return
    if trade_mode:
        open_ids = _strategy_open_order_ids()
        if open_ids is None:
            log.info("无法确认未完成订单，本次调仓跳过。")
            return
        if open_ids:
            log.info("检测到本策略未完成订单 %s，本次调仓跳过以避免重复委托。" % sorted(open_ids))
            return
    _sync_held_from_actual(context)

    # ---- 1. 计算「截至今日开盘」的持有天数 ----
    if g.held is not None:
        g.held_days += 1            # 又持有了一个交易日；建仓当日置 0，故次日为 1，与框架 holding_days 对齐
        holding_days = g.held_days
    else:
        holding_days = None

    # 部分买入后订单已终结：优先补足目标仓位，不受最短持有期压制。
    if trade_mode and g.needs_top_up and g.held is not None:
        log.info("检测到 %s 上次仅部分成交，本日尝试补足至目标仓位。" % g.held)
        if _submit_trade_buy(context, g.held):
            g.needs_top_up = False
        return

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

    # ---- 4. 调仓：始终保证只持有 best ----
    positions = context.portfolio.positions
    switching = best != g.held

    # 待清理：除 best 外所有仍有持仓的标的（换仓的旧主仓 + 历史部分成交残留的杂仓）
    to_sell = [s for s in positions
               if s != best and getattr(positions[s], "amount", 0) > 0]

    # A 股 T+1：当日买入当日不可卖。换仓时若旧主仓今日不可卖，本日不切、下个交易日重试，
    # 避免卖不掉却又买入新仓导致超额持仓。（正常因有最短持有期不会触发。）
    if switching and g.held is not None:
        held_pos = positions.get(g.held)
        if (held_pos is not None
                and getattr(held_pos, "amount", 0) > 0
                and getattr(held_pos, "enable_amount", 0) <= 0):
            log.info("旧仓 %s 今日不可卖（T+1）— 本日不切换，下个交易日重试。" % g.held)
            return

    sellable = [s for s in to_sell if getattr(positions[s], "enable_amount", 0) > 0]

    if trade_mode:
        # 交易模式必须分阶段：卖单真实成交并同步为零仓后，轮询回调才会提交买单。
        if sellable:
            _start_trade_switch(context, best, sellable, buy_required=switching)
        elif switching:
            _submit_trade_buy(context, best)
        else:
            log.info("Top1 未变（%s），维持持仓。" % best)
        return

    # 回测模式撮合同 bar 同步完成，保留既有行为与历史对账口径。
    for s in sellable:
        order_target(s, 0)
        log.info("卖出 %s（清仓/清理杂仓）" % s)
    if switching:
        target_value = context.portfolio.total_value * INVEST_RATIO
        order_target_value(best, target_value)
        log.info("买入 %s 到目标市值 %.2f（总资产 %.2f × %.2f）"
                 % (best, target_value, context.portfolio.total_value, INVEST_RATIO))
        g.held = best
        g.held_days = 0
    else:
        log.info("Top1 未变（%s），维持持仓。" % best)


def after_trading_end(context, data):
    """收盘后记录持仓快照，便于核对。"""
    _ensure_execution_state()
    target = g.pending_target
    buy_failed = g.pending_buy_failed
    pending_error = g.pending_error
    _sync_held_from_actual(context)
    if g.pending_phase is not None:
        # 当日订单不跨日推进；保留真实持仓，次日重新按信号和可卖数量生成新订单。
        if target is not None and g.held == target and buy_failed:
            g.needs_top_up = True
        log.info("收盘清理未完成链路: phase=%s target=%s error=%s；次日从真实持仓恢复"
                 % (g.pending_phase, target, pending_error))
        _clear_pending(pending_error)
    held = [(s, p.amount) for s, p in context.portfolio.positions.items()
            if getattr(p, "amount", 0) > 0]
    log.info("收盘持仓: %s | g.held=%s held_days=%s needs_top_up=%s | 总资产=%.2f 可用现金=%.2f"
             % (held, g.held, g.held_days, g.needs_top_up,
                context.portfolio.total_value, context.portfolio.cash))
