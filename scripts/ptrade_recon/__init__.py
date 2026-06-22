"""PTrade 逐日对账 harness 的复用模块。

把「框架参考产出」与「PTrade 实盘导出」整理成同一张逐日表,供
`tests/test_ptrade_reconciliation.py` 做 score / Top1 / 持仓 三维对账。

- `parse.py`:解析 PTrade 导出 CSV(交易详情 / 持仓明细),产出与框架 fixture
  对齐的逐日 schema(date 索引、资产代码用框架的 `.SH/.SZ` 后缀)。

框架参考 fixture 由 `scripts/export_framework_reference.py` 离线生成,提交在
`backtest/ptrade/reference/rd{N}/` 下。设计依据见
`docs/plans/2026-06-21_ptrade_reconciliation_harness.md` 与 `CONTRACT.md`。
"""
