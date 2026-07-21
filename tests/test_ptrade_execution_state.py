"""PTrade 交易模式执行状态机测试。

这些测试直接加载 deploy/ptrade_quality_momentum_top1.py，并用延迟/拒单/部分成交的
柜台替身验证：卖出确认前绝不买入，订单创建不等于持仓确认，重启不覆盖持有期状态。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ptrade_recon.port import load_deploy_module


class FakeLog:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


def _position(amount, price=2.0, enable_amount=None):
    return SimpleNamespace(
        amount=amount,
        enable_amount=amount if enable_amount is None else enable_amount,
        last_sale_price=price,
        market_value=amount * price,
    )


def _context(positions=None, cash=2_000.0, total_value=100_000.0):
    portfolio = SimpleNamespace(
        positions=positions or {},
        cash=cash,
        total_value=total_value,
    )
    return SimpleNamespace(portfolio=portfolio)


@pytest.fixture
def trade_env():
    mod = load_deploy_module()
    mod.g = SimpleNamespace()
    mod.log = FakeLog()
    mod.is_trade = lambda: True
    mod.get_open_orders = lambda: []

    snapshots = {
        security: {
            "bid_grp": {1: [2.000, 100_000, 10]},
            "offer_grp": {1: [2.001, 100_000, 10]},
            "last_px": 2.000,
            "up_px": 2.200,
            "down_px": 1.800,
        }
        for security in mod.SECURITIES
    }
    mod.get_snapshot = lambda security: {security: snapshots[security]}

    sell_calls = []
    buy_calls = []
    order_seq = [0]

    def order(security, amount, limit_price=None):
        # 实盘拆单后按股数下单：正数买、负数卖。返回唯一 order_id（子单可能多笔）。
        order_seq[0] += 1
        side = "buy" if amount > 0 else "sell"
        (buy_calls if amount > 0 else sell_calls).append((security, amount, limit_price))
        return "%s-%s-%d" % (side, security, order_seq[0])

    # 目标型 API 仅回测路径使用；交易模式已全部改走 order()。保留以防加载期缺名。
    def order_target(security, amount, limit_price=None):
        return "target-%s" % security

    def order_target_value(security, value, limit_price=None):
        return "target_value-%s" % security

    mod.order = order
    mod.order_target = order_target
    mod.order_target_value = order_target_value
    mod._ensure_execution_state()
    mod.g.held = "513100.SS"
    mod.g.held_days = 2
    return mod, sell_calls, buy_calls


def test_trade_limit_price_is_positive_and_uses_etf_precision(trade_env):
    mod, _, _ = trade_env

    assert mod._trade_limit_price("513100.SS", "sell") == 1.996
    assert mod._trade_limit_price("159915.SZ", "buy") == 2.005

    mod.get_snapshot = lambda security: {security: {"last_px": 0, "bid_grp": {}, "offer_grp": {}}}
    assert mod._trade_limit_price("513100.SS", "sell") is None
    assert mod._trade_limit_price("159915.SZ", "buy") is None


def test_switch_submits_only_sell_until_old_position_is_really_gone(trade_env):
    mod, sell_calls, buy_calls = trade_env
    context = _context({"513100.SS": _position(43_800)}, cash=2_180.82, total_value=101_431.62)

    assert mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])

    assert len(sell_calls) == 1
    assert sell_calls[0][0] == "513100.SS"
    assert sell_calls[0][1] == -43_800   # 卖出全部可卖数量（负数），单笔未超上限
    assert sell_calls[0][2] > 0
    assert buy_calls == []
    assert mod.g.held == "513100.SS"
    assert mod.g.pending_phase == "selling"

    # 卖出持仓先同步为 0、但资金尚未到账时，仍不得提交买单。
    context.portfolio.positions = {}
    mod._process_pending_switch(context)
    assert buy_calls == []
    assert mod.g.pending_phase == "selling"

    # 可用资金也同步完成后，轮询才提交买单。
    context.portfolio.cash = 101_000.0
    mod._process_pending_switch(context)

    assert len(buy_calls) == 1
    assert buy_calls[0][0] == "159915.SZ"
    assert buy_calls[0][2] > 0
    assert mod.g.held is None
    assert mod.g.pending_phase == "buying"

    # 订单号仍不等于成交；真实持仓出现后才更新 g.held。
    context.portfolio.positions = {"159915.SZ": _position(23_200, price=4.2)}
    mod._process_pending_switch(context)
    assert mod.g.held == "159915.SZ"
    assert mod.g.held_days == 0
    assert mod.g.pending_phase is None


def test_rejected_sell_never_triggers_buy(trade_env):
    mod, _, buy_calls = trade_env
    context = _context({"513100.SS": _position(43_800)})
    mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])
    sell_id = mod.g.pending_sell_orders["513100.SS"][0]

    mod.on_order_response(
        context,
        [{"order_id": sell_id, "status": "9", "error_info": "委托价格必须大于0"}],
    )
    mod._process_pending_switch(context)

    assert buy_calls == []
    assert mod.g.held == "513100.SS"
    assert mod.g.pending_phase is None
    assert "status=9" in mod.g.pending_error


def test_partial_sell_waits_and_terminal_partial_does_not_buy(trade_env):
    mod, _, buy_calls = trade_env
    context = _context({"513100.SS": _position(10_000)})
    mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])
    sell_id = mod.g.pending_sell_orders["513100.SS"][0]
    mod.get_open_orders = lambda: [
        {"order_id": sell_id, "stock_code": "513100.SS", "status": "7"}
    ]

    mod._process_pending_switch(context)
    assert mod.g.pending_phase == "selling"
    assert buy_calls == []

    mod.get_open_orders = lambda: []
    mod.on_order_response(
        context,
        [{"order_id": sell_id, "status": "5", "error_info": "部分成交后撤单"}],
    )
    mod._process_pending_switch(context)
    assert mod.g.pending_phase is None
    assert mod.g.held == "513100.SS"
    assert buy_calls == []


def test_partial_buy_confirms_actual_position_and_marks_top_up(trade_env):
    mod, _, buy_calls = trade_env
    context = _context({}, cash=100_000.0)
    mod.g.held = None
    mod.g.held_days = 0

    assert mod._submit_trade_buy(context, "159915.SZ")
    buy_id = mod.g.pending_buy_orders[0]
    assert len(buy_calls) == 1

    context.portfolio.positions = {"159915.SZ": _position(5_000, price=4.2)}
    mod.on_order_response(
        context,
        [{"order_id": buy_id, "status": "5", "error_info": "部分成交后撤单"}],
    )
    mod._process_pending_switch(context)

    assert mod.g.held == "159915.SZ"
    assert mod.g.pending_phase is None
    assert mod.g.needs_top_up is True


def test_existing_open_order_blocks_duplicate_submission(trade_env):
    mod, sell_calls, buy_calls = trade_env
    context = _context({"513100.SS": _position(43_800)})
    mod.get_open_orders = lambda: [
        {"order_id": "existing", "symbol": "513100.XSHG", "status": "2"}
    ]

    assert not mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])
    assert sell_calls == []
    assert buy_calls == []


def test_open_order_query_failure_is_fail_closed(trade_env):
    mod, sell_calls, buy_calls = trade_env
    context = _context({"513100.SS": _position(43_800)})

    def fail_query():
        raise RuntimeError("柜台查询超时")

    mod.get_open_orders = fail_query
    assert not mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])
    assert sell_calls == []
    assert buy_calls == []


def test_large_buy_splits_into_capped_chunks(trade_env):
    # 1000万账户买 ~2 元 ETF：需 ~490万股 > 100万单笔上限，应拆成多笔子单。
    mod, sell_calls, buy_calls = trade_env
    context = _context({}, cash=10_000_000.0, total_value=10_000_000.0)
    mod.g.held = None
    mod.g.held_days = 0

    assert mod._submit_trade_buy(context, "159915.SZ")

    assert len(buy_calls) > 1
    assert all(sec == "159915.SZ" for sec, _, _ in buy_calls)
    assert all(0 < amt <= mod.MAX_ORDER_SHARES for _, amt, _ in buy_calls)
    assert all(amt % 100 == 0 for _, amt, _ in buy_calls)          # 每笔整百股
    assert len(mod.g.pending_buy_orders) == len(buy_calls)         # 每笔都被跟踪

    price = mod._trade_limit_price("159915.SZ", "buy")
    target_shares = int((10_000_000.0 * mod.INVEST_RATIO) / price // 100) * 100
    assert sum(amt for _, amt, _ in buy_calls) == target_shares    # 子单求和 = 目标增量


def test_large_sell_splits_into_capped_chunks(trade_env):
    # 持有 250 万股 > 100万上限，换仓卖出应拆成多笔子单（否则整单被后端撤销）。
    mod, sell_calls, buy_calls = trade_env
    context = _context(
        {"513100.SS": _position(2_500_000, price=2.0)},
        cash=0.0,
        total_value=5_000_000.0,
    )

    assert mod._start_trade_switch(context, "159915.SZ", ["513100.SS"])

    assert len(sell_calls) == 3   # 100万 + 100万 + 50万
    assert all(sec == "513100.SS" for sec, _, _ in sell_calls)
    assert all(-mod.MAX_ORDER_SHARES <= amt < 0 for _, amt, _ in sell_calls)
    assert sum(-amt for _, amt, _ in sell_calls) == 2_500_000
    assert len(mod.g.pending_sell_orders["513100.SS"]) == 3
    assert buy_calls == []         # 卖出阶段绝不同时买入


def test_restart_preserves_serialized_holding_days_and_fresh_recovery_is_conservative(trade_env):
    mod, _, _ = trade_env
    mod.g.held = "513100.SS"
    mod.g.held_days = 1
    mod._ensure_execution_state()
    assert mod.g.held_days == 1

    mod.g = SimpleNamespace()
    mod._ensure_execution_state()
    context = _context({"513100.SS": _position(43_800)})
    mod._sync_held_from_actual(context)
    assert mod.g.held == "513100.SS"
    assert mod.g.held_days == 0
