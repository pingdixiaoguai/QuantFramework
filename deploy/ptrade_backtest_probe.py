# -*- coding: utf-8 -*-
"""
PTrade 回测环境 API 探针（交易侧）
===================================
用途：把本文件作为**策略**，在 PTrade 里跑一个**很短的回测**（如最近 5 个交易日、日线、
起始资金随意），然后把**日志输出**复制回来。用于验证研究环境无法测的交易上下文接口：

  - context.portfolio 的属性名（total_value / cash / positions）
  - positions 的结构（dict？key 格式？Position 对象有哪些属性）
  - get_history 在策略上下文里的返回结构（fq='post' / include=False）
  - order_target_value / order_target 是否可用、下单后持仓如何体现
  - set_commission(type="ETF") 是否被支持
  - run_daily(time=...) 回调签名是否为 func(context)

本探针只会下一笔**极小的测试单**（总资产的 1%），仅在回测里运行，不接实盘。
探测完成后即清仓并停止重复探测。
"""

import numpy as np

PROBE_SEC = "510300.SS"     # 探测用标的；若 .SS 取不到数，改 '510300.SH' 再试
SECOND_SEC = "159915.SZ"


def initialize(context):
    log.info("===== initialize 开始 =====")
    set_universe([PROBE_SEC, SECOND_SEC])

    # set_benchmark
    try:
        set_benchmark("000300.SS")
        log.info("[OK] set_benchmark('000300.SS')")
    except Exception as e:
        log.info("[FAIL] set_benchmark: %s: %s" % (type(e).__name__, e))

    # set_commission（ETF 类型是否支持）
    try:
        set_commission(commission_ratio=0.0002, min_commission=5.0, type="ETF")
        log.info("[OK] set_commission(type='ETF')")
    except Exception as e:
        log.info("[FAIL] set_commission(type='ETF'): %s: %s" % (type(e).__name__, e))

    # set_fixed_slippage
    try:
        set_fixed_slippage(0.0)
        log.info("[OK] set_fixed_slippage(0.0)")
    except Exception as e:
        log.info("[FAIL] set_fixed_slippage: %s: %s" % (type(e).__name__, e))

    # run_daily 回调签名探测
    try:
        run_daily(context, probe_once, time="09:35")
        log.info("[OK] run_daily(context, probe_once, time='09:35') 注册成功")
    except Exception as e:
        log.info("[FAIL] run_daily: %s: %s" % (type(e).__name__, e))

    g.done = False
    log.info("===== initialize 结束 =====")


def handle_data(context, data):
    # 兜底：万一 run_daily 没触发，这里也跑一次探测
    if not g.done:
        probe_once(context)


def probe_once(context):
    if g.done:
        return
    g.done = True
    log.info("########## 交易侧 API 探测开始 ##########")

    # --- 1. context.portfolio 属性 ---
    pf = context.portfolio
    log.info("[portfolio] dir(部分): %s" %
             [a for a in dir(pf) if not a.startswith("_")])
    for attr in ("total_value", "cash", "portfolio_value", "starting_cash", "positions_value"):
        log.info("[portfolio] %s = %s" % (attr, getattr(pf, attr, "<无此属性>")))

    # --- 2. positions 结构（下单前，应为空）---
    pos = pf.positions
    log.info("[positions] type=%s keys=%s" % (type(pos).__name__, list(pos.keys())))

    # --- 3. get_history 在策略上下文里 ---
    for sec in (PROBE_SEC, "510300.SH"):
        try:
            df = get_history(25, "1d", "close", sec, fq="post", include=False)
            cols = list(df.columns) if hasattr(df, "columns") else "N/A"
            tail = None
            try:
                tail = list(np.asarray(df["close"].values, dtype=float)[-3:])
            except Exception:
                tail = "无法取 'close' 列"
            log.info("[get_history %s] OK len=%s cols=%s tail=%s" % (sec, len(df), cols, tail))
        except Exception as e:
            log.info("[get_history %s] FAIL: %s: %s" % (sec, type(e).__name__, e))

    # --- 4. order_target_value 下一笔极小测试单（总资产 1%）---
    test_value = pf.total_value * 0.01
    try:
        oid = order_target_value(PROBE_SEC, test_value)
        log.info("[OK] order_target_value('%s', %.2f) -> 返回 %s" % (PROBE_SEC, test_value, oid))
    except Exception as e:
        log.info("[FAIL] order_target_value: %s: %s" % (type(e).__name__, e))

    # --- 5. 下单后检查 positions 与 Position 对象属性 ---
    pos2 = context.portfolio.positions
    log.info("[positions 下单后] keys=%s" % list(pos2.keys()))
    if PROBE_SEC in pos2:
        p = pos2[PROBE_SEC]
        log.info("[Position] dir(部分): %s" %
                 [a for a in dir(p) if not a.startswith("_")])
        for attr in ("amount", "enable_amount", "cost_basis", "last_sale_price",
                     "market_value", "value", "business_amount"):
            log.info("[Position.%s] = %s" % (attr, getattr(p, attr, "<无此属性>")))
    else:
        log.info("[Position] 注意：回测下单可能于下一根 bar 才成交，本 bar positions 仍为空属正常。")

    # --- 6. order_target 清仓接口 ---
    try:
        order_target(PROBE_SEC, 0)
        log.info("[OK] order_target('%s', 0) 清仓指令已下" % PROBE_SEC)
    except Exception as e:
        log.info("[FAIL] order_target: %s: %s" % (type(e).__name__, e))

    log.info("########## 交易侧 API 探测结束：请复制全部日志 ##########")


def after_trading_end(context, data):
    held = [(s, getattr(p, "amount", None)) for s, p in context.portfolio.positions.items()]
    log.info("[收盘] positions=%s total_value=%.2f cash=%.2f" %
             (held, context.portfolio.total_value, context.portfolio.cash))
