---
# fixed_cycle 研究归档(原 changelog §3.5,已作废降级)

> 归档日期:2026-06-13
> 来源:本文件原为 strategy_changelog.md 的 §3.5「[候选策略] 新增 fixed_cycle 调仓模式
> 与固定周期 Top1 配置」(commit 29a3ffd,2026-06-02),现作废并降级为研究归档。
>
> 作废原因:fixed_cycle(固定周期评估)属"周期性重评 / 时间维度抗换手"研究族,该族已在
> 干净数据上反转、相位稳健性显示日历位置运气、OOS 严重崩塌,整体否决。按治理维护规则第 4 条,
> 研究诊断不进 changelog,此条当初不应作为变更条目立项,现降级为研究归档。原条目附带的无成本
> 回测表(fixed_cycle2 表面优于 min_hold2)是该族典型的"无成本表面优势",加入成本与 OOS
> 检验后不成立,不作任何变更依据。
>
> 代码/配置处置:两个 fixed_cycle YAML 已于 commit 5cb3e3b 移除;引擎 runner.py 的
> fixed_cycle/rebalance_mode 代码路径保留为引擎能力(无生产引用)。

---

## 原 §3.5 条目全文(存证,不作依据)

### 3.5 [候选策略] 新增 fixed_cycle 调仓模式与固定周期 Top1 配置

- **日期**:2026-06-02
- **变更类型**:候选策略 / 回测引擎扩展
- **变更前**:
  - 仅支持 `rebalance_mode=min_hold` 隐含语义:持有满 `rebalance_days` 后每日重新评估
  - `quality_momentum_top1` 无法区分"最短持有期"与"固定周期评估"
- **变更后**:
  - 新增 `rebalance_mode=fixed_cycle`:仅在持仓第 N、2N、3N... 个交易日评估信号
  - 新增配置:
    - `strategy/configs/quality_momentum_top1_fixed_cycle2.yaml`
    - `strategy/configs/quality_momentum_top1_fixed_cycle5.yaml`
  - `backtest/runner.py`、`run_daily.py`、`backfill_ytd.py` 共用同一调仓判定 helper
- **决策依据**:
  - 用户希望单独测试固定 2/4/6 周期评估及固定 5 日评估
  - 初步无成本回测显示 fixed_cycle2 在当前样本上优于 min_hold2,但尚未加入交易成本与治理评估
- **无成本回测对比**(2013-07-30 ~ 2026-06-01):

  | 策略口径 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 切换次数 | 平均持有期 |
  |----------|--------|----------|--------|----------|----------|------------|
  | min_hold2 | +3277.11% | +32.89% | 1.23 | -28.37% | 414 | 7.50 交易日 |
  | fixed_cycle2 | +4515.71% | +36.29% | 1.31 | -28.15% | 352 | 8.82 交易日 |
  | fixed_cycle5 | +2118.45% | +28.46% | 1.15 | -30.35% | 207 | 14.98 交易日 |

- **实施后状态**:仅新增候选策略配置,不替换当前实盘配置
- **后续监测项**:
  - 与 §3.2 交易成本模型合并评估,补充单边成本后的收益、Sharpe、最大回撤、换手率
  - 若考虑实盘切换,必须先明确 `quality_momentum_top1` 当前基准参数与 changelog 中 rd=5/rd=2 不一致的问题

---
