# W40门控 + 月度40日最弱红利ETF + 100%红利正式晋升

日期：2026-08-25  
正式策略：`momentum_defender_w40_reversal_full_equity_v1`  
晋升权限：用户明确指令  
证据性质：回溯奥卡姆候选，不是独立样本外

## 正式机制

顶层完全沿用已冻结W40门控：510300的40日对数下跌幅度使用滚动504日严格滞后分位；55%
进入Defender、40%恢复Momentum，1/1日确认，Momentum/Defender双袖套锁30/30日。

Defender改为：每月第一个联合交易日开盘，用上一收盘可得的40日对数收益，对六只已上市且
当日可交易的红利ETF排名，100%持有收益最低者。511260国债权重固定为0；旧网格、波动率
上限、满仓覆盖和趋势/反转场景分支全部关闭。无法卖出当前标的或买入新标的时保持原持仓。

## 正式检查点

区间2019-01-18至2026-08-21，共1,841个交易日。

|指标|新正式100%红利|回滚W40旧Defender|差值|
|---|---:|---:|---:|
|年化收益|50.49%|49.32%|+1.18个百分点|
|Sharpe|1.676|1.734|-0.059|
|最大回撤|-25.50%|-25.50%|基本相同|
|Defender进入|20|20|0|
|Defender交易日|870|870|0|

新版本不是Sharpe升级。用户明确接受Sharpe和普通区间回撤退化，以换取Defender机制从18个
仓位字段压缩为固定100%红利、历史年化提高和完全可解释的持仓。

逐日收益SHA-256：

```text
4d4f2db08a58aa7c6cbb459d64ae5676dd6905213f338702a362d5f4bfa2c5d6
```

正式实现与研究选中路径逐日最大误差为0。

## 证据边界

两阶段测试170个参数ID、168条唯一收益路径。全局Reality Check完整/普通区间为
`p=0.8584/0.7176`，CSCV-PBO为61.0%/72.3%；Bootstrap年化差与Sharpe差区间均跨0。
因此晋升依据是用户明确治理决定，不能表述成统计上证明未来优于回滚版本。

100%红利候选的独立Defender曲线只有16.73%年化、0.957 Sharpe、-22.99% MDD，明显弱于
旧核心的24.13%、2.216、-11.36%。组合历史提升依赖W40选出的20段持有时机，可移植性较弱。

## 两本台账

- Defender机会成本台账：20段Defender中4段跑输原Momentum严格超过1个百分点，旧版为6段；
- 整体最大回撤台账：110段独立水下期，Top 10中纯Defender 4段，新增2022年与2024年
  100%红利自身回撤，明确记录取消国债缓冲的代价。

两本台账均由生成器机械重建并通过`--check`，人工背景只解释已识别事件，不修改事件集合。

## 运维、前瞻与回滚

- 正式配置：`strategy/configs/momentum_defender_w40_full_equity.yaml`；
- 默认日跑：`run_daily_momentum_defender.py`与`scripts.run_daily_job`均指向新配置；
- 正式报告：`experiments/20260825_momentum_defender_w40_reversal_full_equity_v1_formal/`；
- 前瞻账本：`state/momentum_defender_w40_reversal_full_equity_v1_forward.jsonl`；
- 回滚配置：`strategy/configs/momentum_defender_w40_loss.yaml`。

新策略使用独立策略ID，不继承旧持仓文件。首次生产执行前必须运行`--dry-run`核对旧实盘持仓
到新100%红利目标的交易差异；不得静默复制旧策略状态或前瞻账本。
