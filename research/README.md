# Momentum × Defender C2：历史冻结检查点与当前内嵌版本

## Defender高位＋QM集成确认＋固定5日部分逃生（2026-08-27）

在正式v5最终仍为Defender时，以512890的40日区间位置≥95%为高位信号；当前Momentum Top1
与连续Defender分别计算QM38/40/42等权均值，要求Top1均值为正且相对Defender高0.0225，
随后只转15%到入场Top1、固定持有5日，并要求条件失效后才能重新武装。

2019-01-18至2026-08-26年化/Sharpe由58.43%/1.886提高到58.75%/1.894，MDD保持-25.50%；
开发、验证、近期均双胜，6/6局部参数轴通过，删任一年和十倍成本仍双胜。候选仅触发6次，
合并948条唯一路径后Reality Check `p=0.675`，Bootstrap双指标区间仍跨0，且最大正贡献集中于
2024-12。因此只能视为参数稳健但统计未证明的回溯候选。用户最终判断该优化过于偶然，决定
仅归档研究报告，不替换正式策略、不建立替代正式版本，也不再同史调参。完整
报告见`docs/research/2026-08-27_conditioned_defender_high_range_momentum_escape_2019.md`，机器
证据位于`experiments/20260827_conditioned_defender_range_escape_final_candidate/`。

## Defender高位部分切换Momentum审计（2026-08-27）

在当前正式v5上冻结W40、QM40恢复、月度Defender、Momentum Top1与黄金逃生，只在正式最终
目标仍为Defender时，按Defender 30/40/50日区间高位把10%或20%切到Momentum。两种锚、
连续记忆/Defender段内记忆/无记忆三种奥卡姆规则共108条唯一路径，没有任何路径同时提高
完整年化和Sharpe，也没有稳健候选。

图示40日10%/95%/20%连续网格把年化/Sharpe从58.43%/1.886降至48.71%/1.610；全表最少
退化的“实际Defender、40日最高点、无记忆转10%”也只有58.11%/1.881。Reality Check
`p=0.9964`，固定候选删任一年双指标通过率0%。机制被否证，不修改正式配置，也不建立影子。
完整报告见`docs/research/2026-08-27_defender_high_range_partial_momentum_escape_2019.md`，机器证据
位于`experiments/20260827_momentum_defender_defender_range_escape_2019/`。

## QM40基础Defender恢复阈值寻参（2026-08-26）

冻结正式v4所有其他层，只搜索早退条件`QM40 > θ`的绝对阈值。22个参数ID去重为17条路径；
0.005、0.0075、0.010形成正向平台，稳健中央代表0.0075为58.43%年化、1.886 Sharpe、MDD
-25.50%，相邻阈值Q25为58.19%/1.882；正式0阈值为57.79%/1.862。

但差异只来自2019与2026，Bootstrap跨0、Reality Check `p=0.8408`、walk-forward/留一年重选
双指标胜率20%/12.5%。第一次最差排名聚合因完整池指标并列产生退化，最终平台筛选属于已披露
的事后规则修正。因此研究流程不支持自动晋升；用户随后明确接受平台并将0.0075晋升为正式v5。
完整报告见`docs/research/2026-08-26_qm40_recovery_threshold_search_2019.md`。

## 当前正式v5：QM40恢复阈值0.0075（2026-08-27）

正式ID为`momentum_defender_w40_qm40_threshold_v5`。相对v4只把基础Defender早退条件从
`QM40>0`改为`QM40>0.0075`；W40、Defender QM40、5/10/30规则、Momentum和黄金覆盖不变。
2013固定正式口径为39.71%年化、1.437 Sharpe、MDD -48.63%；2019重启主口径为
58.43%/1.886/-25.50%。正式收益哈希为
`6a45479ffe5da9b081e53e36c7a0b137656ed9a61cdf3c7d8044aa278700f4d3`，v4保留直接回滚。

## W40冠军＋Defender QM40＋QM40恢复早退组合（2026-08-26）

按用户指定组合756日历史、60%/35%迟滞W40，月度QM40最低Defender，以及最低5日后510300
QM40>0连续10日早退、30日35%恢复线保底。2019重启结果由56.82%年化/1.854 Sharpe提高到
57.79%/1.862，MDD保持-25.50%；完整六只池阶段为80.15%/1.908。

八格归因显示`W40＋Defender QM40`本身为57.71%/1.875，是Sharpe最高且更简单的组合；再加
早退只提高约0.08个百分点年化，却降低0.013 Sharpe。由于零阈值下`QM40>0`与`R40>0`
完全等价，路径效率并未改变早退日期。三项组合Bootstrap跨0、Reality Check `p=0.2414`、
CSCV-PBO 93.5%，研究流程不支持自动晋升；用户随后明确要求把三项完整组合晋升生产，并接受
2013固定正式口径退化及事后组合风险。完整研究见
`docs/research/2026-08-26_w40_defender_qm_signed_exit_combination_2019.md`，晋升边界见
`docs/research/2026-08-26_w40_defender_qm_signed_exit_v4_formal_promotion.md`。

## 历史正式v4：W40冠军＋Defender QM40＋QM40恢复早退（2026-08-26）

> 已由`momentum_defender_w40_qm40_threshold_v5`取代，以下保留为回滚证据。

正式ID为`momentum_defender_w40_qm40_signed_exit_v4`。W40使用756日严格滞后历史、60%/35%
迟滞；Defender每月100%持有QM40最低的合格红利ETF；基础Defender至少5日后，510300
QM40>0连续10日可提前恢复Momentum，30日后以W40≤35%保底。黄金覆盖保持v3的
0.005/-0.020、5日硬持有和即时入口否决。

2013固定正式检查点为36.89%年化、1.358 Sharpe、MDD -48.63%，低于v3的40.06%/1.447；
2019重启主研究口径为57.79%/1.862/-25.50%，高于v3的56.82%/1.854。正式逐日收益哈希为
`11bb60cbeeb013235491312dc623a0ed1ebb904120599a72309df8633c10afd0`。本次晋升来自用户明确
治理决定，不是统计显著结论；v3配置和正式报告永久保留为直接回滚。

## Defender 30日锁与灵活退出机制（2026-08-26）

2019重启样本共有20次基础Defender进入，16段在锁内已出现恢复信号，累计154个交易日被30日
锁继续阻塞。逐段提前释放反事实显示原锁11段有利、5段不利，前两大正贡献占43%，说明锁的
历史作用并非只靠一次上涨；固定锁0–30日的年化/Sharpe也总体单调改善。但30日相对35/40日
又是明显局部峰，精确天数仍有拟合风险。

更根本的问题是当前下跌幅度把所有正40日收益压成`P40=0`，退出侧缺乏恢复强度。本轮测试了
短锁＋确认、强恢复、线性门槛、有符号10/20/40/60日收益、Top1相对Defender QM20及交集。
没有灵活候选同时提高年化和Sharpe。机制最整洁的“最低5日＋510300 R40>0连续10日早退＋
30日fallback”为57.03%年化/1.844 Sharpe，较当前56.82%/1.854只提高年化、降低Sharpe，
Reality Check `p=0.8824`且Bootstrap跨0。因此正式30日锁保持不变，该规则只建议冻结为前瞻
影子。完整报告见`docs/research/2026-08-26_defender_exit_mechanism_audit_2019.md`。

## 当前正式策略奥卡姆与参数稳健性复核：2019主样本（2026-08-26）

按最新研究要求，将主样本改为2019-01-18至2026-08-26并重新初始化全部策略状态；2013结果
只保留为尾部压力测试。正式v3复现为56.82%年化、1.854 Sharpe、MDD -25.50%。整层消融
仍全部退化；W40、黄金零退出、Momentum持有7日和Defender质量动量排序虽出现表面双指标
候选，但Reality Check分别为`p=0.9760/0.9112/0.8094/0.9288`，均不支持晋升。

2019起点也不是完整六只红利ETF样本：最后一只513630直到2024-02-06才拥有40日可用分数。
因此报告额外保存2024-02-06起的完整池压力段，并明确禁止把多个回溯小赢家叠加。最终保持v3
不变；最值得前瞻观察的是复用现有QM公式的Defender 40日质量动量最弱排序，但其历史增益
集中于少量2024/2026事件。完整结论见
`docs/research/2026-08-26_current_strategy_occam_robustness_2019.md`，机器证据位于
`experiments/20260826_current_strategy_occam_robustness_audit_2019/`。

## 2013全史奥卡姆与参数稳健性尾部压力复核（2026-08-26）

在2013-01-01固定正式区间上重新复现v3逐日收益哈希，并对机制消融、W40窗口/历史/阈值/锁、
Momentum窗口与持有期、黄金X/Y和即时入口开关做统一压力测试。没有任何更简单消融同时保留
40.06%年化和1.447 Sharpe；W40网格162个参数ID中，表面冠军仅提高0.19个百分点年化和
0.005 Sharpe，年度Reality Check `p=0.9664`且Bootstrap跨0。黄金退出线改为0的奥卡姆点
也只有0.17个百分点/0.004的点估计提升，Sharpe差区间跨0。

结论是正式v3保持不变：阈值附近较平，但W40=40日、双锁=30日和Momentum=20日均是历史
局部高点，只能称为冻结的回溯较优参数，不能称为已证明的非过拟合最优解。完整报告与机器
证据见`experiments/20260826_current_strategy_occam_robustness_audit/`。

## W40进入Defender时的即时黄金否决（2026-08-26）

研究候选保持W40基础状态正常进入Defender，但若切入当日黄金已是Top1且相对连续Defender的
QM20差值高于0.005，实际持仓立即进入黄金，不先持有Defender五日。完整年化/Sharpe由
39.10%/1.416提高到40.06%/1.447，MDD由-52.71%改善到-48.63%。

2013起历史共有7次即时入口，2026零触发、与正式v2逐日相同。Bootstrap年化差区间
`[-1.05%, +3.67%]`、Sharpe差区间`[-0.028, +0.114]`，且机制由已观察的2025正事件
提出，研究流程不支持自动晋升；用户随后基于逻辑一致性明确治理晋升为v3。完整报告见
`docs/research/2026-08-26_immediate_gold_entry_veto.md`。

## Defender 2007研发段 / 2026机械验证（2026-08-26）

补齐510880自2007-01-18上市以来行情，固定2025-12-31以前为研发段、2026为验证段。固定
六只研究池在Defender单袖套中，研发/验证年化由8.99%/8.19%提高到9.14%/17.22%；在完整
W40＋黄金逃生组合中，2019–2025研发年化由54.26%提高到54.91%，2026累计收益由36.66%
提高到41.78%，Sharpe由1.679提高到1.865。

但研发段Sharpe冠军`固定池+520990`在2026略输基线，40条路径研发/验证排名相关系数-0.05，
Reality Check `p=0.7580`、CSCV-PBO 55.1%。结论是固定池方向稳定、寻优流程不稳定；2026又
已被此前研究观察，研究流程本身不支持自动晋升；用户随后明确将固定六只池晋升为v2生产。完整报告见
`docs/research/2026-08-26_defender_2007_2026_validation.md`。

## Defender红利ETF候选池研究（2026-08-26）

以2026-08-25规模至少30亿元、近60日成交额中位数优先的红利ETF为外部筛选，只改变正式
Defender候选池。研究池删除159545与同指数重复的563020，加入515450和513630；完整年化从
54.94%提高到56.33%，Sharpe从1.799提高到1.840，MDD保持-25.50%。

46条唯一路径的Reality Check为`p=0.6050`，Bootstrap年化差95%区间跨0，CSCV-PBO为
51.9%，研究流程本身不支持自动晋升；用户随后明确治理晋升。完整报告见
`docs/research/2026-08-26_defender_dividend_etf_universe.md`，机器证据位于
`experiments/20260826_momentum_defender_dividend_universe/`。

## 历史正式v3：W40 + 固定六只红利Defender + 即时黄金入口否决（2026-08-26）

> 已由`momentum_defender_w40_qm40_signed_exit_v4`取代，以下保留为回滚证据。

用户明确将即时黄金入口否决晋升为`momentum_defender_w40_gold_qm20_escape_v3`。基础W40继续使用
55%/40%、1/1确认、30/30锁；Defender在512890、513530、515080、510880、515450和
513630中每月100%持有40日收益最低的可交易ETF。
W40切入Defender的同一开盘若黄金Top1已满足`QM20(Gold)-QM20(Defender)>0.005`，实际持仓
立即黄金；其余时候实际Defender满5日后才可破锁。黄金硬持有
5日，随后差值低于-0.020或Top1不再是黄金时返回基础Defender。

正式配置区间为2013-01-01至今，首个可执行日2013-02-04；截至2026-08-26为40.06%年化、
1.447 Sharpe、MDD -48.63%；33次黄金逃生、429个逃生日、27次
实际破30日锁，逐日收益哈希为
`2e746404983f979dd638c982e3d0e9cfdc571c038f8333f04ce8de2f9016af88`。正式配置见
`strategy/configs/momentum_defender_w40_gold_escape.yaml`，机器审计与HTML位于
`experiments/20260826_momentum_defender_w40_gold_qm20_escape_v3_formal/`。

当前无黄金覆盖的直接回滚为`momentum_defender_w40_reversal_full_equity_v2`，同样使用固定六只
红利池；2013-01-01至2026-08-26为33.79%年化、1.220 Sharpe、MDD -62.81%。

该策略是回溯选择后由用户治理晋升，并非统计显著性结论：即时否决仅7次，2026零触发，
Bootstrap年化与Sharpe差区间均跨0。完整研究与晋升边界见
`docs/research/2026-08-26_immediate_gold_entry_veto.md`和
`docs/research/2026-08-26_immediate_gold_entry_veto_formal_promotion.md`。

## W40滚动500日A/B/C分位逃生（2026-08-25）

用严格滞后滚动500日分位把固定X/Y改写为`Q_A(Top1 QM20)-Q_C(Defender QM20)`和
`Q_B(Top1 QM20)-Q_C(Defender QM20)`；A/B可按ETF不同，C在联合策略内共用。700个单资产
参数ID、1,792个联合组合最终仍只启用黄金，稳健点为A=0.70、B=0.10、C=0.60。

动态分位为54.69%年化、1.796 Sharpe，略低于固定黄金候选54.94%/1.799；表面近似点
A=0.70/B=0.30/C=0.60为54.9325%/1.7998，但邻域更弱。滚动分位因252日暖机丢失2019年
两次正向黄金逃生；全局1,526条路径Reality Check为`p=0.871/0.985`，没有降低统计过拟合
证据。因此未达到“持平或更好”，固定黄金候选保持不变。完整报告见
`docs/research/2026-08-25_w40_quantile_escape_search.md`。

## W40资产专用Top1 QM20逃生（2026-08-25）

在统一X/Y实验失败后，允许四只Momentum ETF分别配置阈值或禁用。先运行548个单资产参数ID，
再将每只ETF的禁用与4个多目标代表组合为625组。完整、普通区间和联合稳健排序均选择只启用
黄金：X=0.005、Y=-0.020；沪深300、创业板和纳指禁用。

候选完整54.94%年化、1.799 Sharpe，较正式策略提高4.45个百分点和0.123；普通区间也提高
至46.86%/1.964。黄金X/Y 9点邻域为53.43%/1.756，联合17点邻域94.1%同时改善双指标。
但合并569条路径后Reality Check为`p=0.835/0.978`，Bootstrap区间跨0，联合重选
walk-forward胜率40%。因此研究流程本身不支持自动晋升；用户随后明确将固定黄金候选晋升为
生产策略。完整报告见
`docs/research/2026-08-25_w40_asset_specific_qm20_escape_search.md`。

## W40 Defender五日后Top1 QM20逃生（2026-08-25）

固定当时正式、现在作为直接回滚的W40/full-equity策略，实际Defender持有满5日后才允许以
`QM20(当前Momentum Top1) - QM20(连续Defender净值) > X`破30日锁；入场Top1硬持有5日，
之后基础仍为Defender且差值`<Y`才返回，否则交还正常Momentum轮转。固定机制只搜索X/Y，
共137个参数ID、116条唯一路径。

没有任何候选提高完整年化或Sharpe。最少退化的稳健代表X=0.06、Y=0.02为49.97%年化、
1.650 Sharpe，均低于正式50.49%/1.676；9点邻域双目标通过率为0。全局284条路径Reality
Check为`p=1.000/1.000`，故机制被否证、不修改生产。完整报告见
`docs/research/2026-08-25_w40_top1_qm20_escape_search.md`。

## 直接回滚策略：W40 + 月度40日最弱红利ETF 100%持仓（2026-08-25）

> 已由`momentum_defender_w40_gold_qm20_escape_v1`取代，以下保留为直接回滚证据。

用户明确将`momentum_defender_w40_reversal_full_equity_v1`晋升为正式策略。顶层W40参数完全
沿用55%/40%、1/1确认和30/30锁；Defender关闭旧网格、波动率上限、满仓覆盖和国债填充，
每月100%持有40日收益最低的可交易红利ETF。

正式检查点50.49%年化、1.676 Sharpe、MDD -25.50%。本次晋升接受Sharpe相对回滚W40下降
0.059，以换取机制极简和年化提高1.18个百分点。两本台账已重建：20段Defender中4个机会
成本badcase；整体110段水下期Top 10中纯Defender 4段。v1配置保存在
`experiments/20260825_momentum_defender_w40_reversal_full_equity_v1_formal/strategy_config.yaml`，正式报告位于
`experiments/20260825_momentum_defender_w40_reversal_full_equity_v1_formal/`。

## 最新W40门控下的红利/国债奥卡姆仓位研究（2026-08-25）

固定当时正式W40的55%/40%、1/1确认和30/30锁，同时固定“每月选择40日收益最低红利ETF”，
两阶段测试170个参数ID、168条唯一收益路径。联合奥卡姆候选是不使用动态核心、Defender期间
100%持有选中红利ETF：完整50.49%年化、1.676 Sharpe；普通区间44.90%/1.872。它比正式
W40的49.32%年化更高，但Sharpe低于正式1.734，普通区间Sharpe也低于1.967。

所有1–3参数动态规则都未同时提高年化与Sharpe。全局Reality Check为`p=0.8584/0.7176`，
CSCV-PBO为61.0%/72.3%。研究流程本身不支持自动晋升；用户随后明确接受trade-off并晋升
100%红利候选。完整
报告见`docs/research/2026-08-25_w40_occam_dividend_bond_position.md`。

## Defender奥卡姆简化与门控重寻优（2026-08-25）

> 本节使用已废止的加权DRAQM门控，已由上方最新W40研究取代。

逐层消融确认，Defender核心动态仓位层不能被固定仓位或单一波动率目标替代：更激进的
40日反转100%权益版本虽达到52.14%年化，Sharpe仅1.704，低于正式1.721。可接受的简化
边界是保留冻结核心，把月度选择的场景、行情和趋势分支统一为“每月选择40日收益最低的
已上市权益ETF”。选择层政策字段由7个降为1个，Defender合计约由25个降为19个。

重新搜索顶层门控后，共同绩效领先者使用45%/20%进出线、3/1日确认和30/30日锁；完整
样本51.70%年化、1.789 Sharpe、MDD -25.50%，普通区间45.50%/2.052。但相对正式策略
前两段正事件占约90%，Reality Check为`p=0.9972/0.9746`，Bootstrap区间跨0，因此只冻结
为研究候选，不替换生产。完整报告见
`docs/research/2026-08-25_occam_defender_simplification.md`。

## 单一W40下跌幅度分位双目标寻参（2026-08-25）

固定`max(-log_return_40, 0)`和严格滞后滚动504日分位，只搜索状态阈值、1/3/5日确认及
20/25/30日锁；相同1,242个候选分别按完整样本和候选无关的普通区间选择。全量候选为
65%/40%、1/1日确认、30/30日锁，完整49.03%/1.747；剔除极端行情候选为55%/40%、
1/1日确认、30/30日锁，完整49.32%/1.734。

两个目标未在进入线收敛，96个邻域的完整年化/Sharpe Q25仅约40%/1.53；合并既有研究后
73,386个候选ID、42,816条唯一路径，Reality Check为`p=0.9966/0.9932`。简单公式没有
形成比当前加权正式策略更稳定的状态平台，研究流程不支持自动晋升；用户随后明确选择
“剔除极端行情”候选作为正式策略。完整报告见
`docs/research/2026-08-25_w40_loss_occam_dual_objective.md`。

## 历史正式策略：单一W40下跌幅度分位（2026-08-25）

> 已由`momentum_defender_w40_reversal_full_equity_v1`取代，以下保留为回滚证据。

用户明确将`momentum_defender_w40_loss_excluding_extremes_v1`晋升为正式策略。它固定四ETF
双对数质量动量、listing-aware Defender，以及510300单一40日对数下跌幅度的严格滞后
504日分位；55%进入Defender、40%恢复Momentum，1/1日确认，双袖套锁30/30日且不可
绕过。路径效率、波动率调整、地板、clip、Gold和紧急覆盖全部关闭。

正式检查点年化49.32%、Sharpe 1.734、MDD -25.50%，20次Defender进入、870个Defender
交易日；正式实现与研究选中路径逐日误差为0。正式配置见
`strategy/configs/momentum_defender_w40_loss.yaml`，HTML和机器审计位于
`experiments/20260825_momentum_defender_w40_loss_excluding_extremes_v1_formal/`。当前badcase
历史台账曾识别20段Defender中的6段跑输原Momentum超过1个百分点；当前台账已按新正式
100%红利策略重建。旧加权DRAQM仍保留为更早回滚检查点。

## 固定通用门控 + Gold相对Defender覆盖（2026-08-25）

固定通用510300门控及其30/30日锁，只在Momentum Top-1为黄金时比较黄金与whole-Defender
连续净值的上一收盘有符号RAQM。Gold豁免必须同时满足自身RAQM为正、`Gold - Defender`
达到相对入场差；离开黄金立即重置，Gold转弱可绕过覆盖锁退出。

7,140个候选ID去重为2,762条路径。全样本稳健候选为20日RAQM、入/出差0.30/-0.20、
5/1日确认、5日Gold锁、单向豁免，结果51.18%年化、1.715 Sharpe；普通区间候选为
51.07%/1.713，普通区间43.34%/1.873。探索性最好路径51.83%/1.724，但没有通过预注册的
分段与邻域稳定选择。累计全部研究75,818个候选ID、21,562条唯一路径，全局Reality Check
为`p=0.9920/0.9736`，因此不替换通用门控。完整报告见
`docs/research/2026-08-25_relative_gold_overlay_search.md`。

## Gold例外低于通用门控的归因（2026-08-25）

包含极端行情候选的510300 base为50.49%/1.711，Gold层加入后降至50.26%/1.689；Gold额外
Defender日贡献为正，但从强势Defender切回“仅仅不弱”的黄金及增加45次切换造成净损失。
不包含极端行情候选的base仅48.28%/1.690，Gold豁免小幅回补至48.45%，主要问题是重新选择
的510300基础参数。

典型事件为2022-11-11至11-28：Gold score为0，黄金涨3.79%，但Defender持港股通红利上涨
12.24%。结论是绝对Gold DRAQM不能表示Gold强于Defender；下一步若研究，应固定通用base，
只增加一个`Gold - Defender`相对强弱阈值。完整报告见
`docs/research/2026-08-25_gold_exception_underperformance.md`。

## 通用510300门控 + Gold例外（2026-08-25）

按奥卡姆原则把510300基础状态机与Gold覆盖完全分层：基础状态连续运行且不被Gold修改；仅当
Top-1为黄金时，Gold可单向豁免或双向覆盖，离开黄金立即交回基础状态。搜索72组锚政策、
8,400组Gold政策对和4,851组联合锁。

包含极端行情候选为50.26%年化、1.689 Sharpe；不包含极端行情候选全样本48.45%/1.668，
普通区间43.80%/1.924。族内Reality Check为`p=0.0294/0.0370`，合并此前研究后变为
`p=0.0836/0.0634`；两者仍未超过通用510300门控50.88%/1.721，因此不晋升。完整报告见
`docs/research/2026-08-25_universal_gate_gold_exception_search.md`。

## 通用510300门控与资产专用门控归因（2026-08-25）

归因确认黄金假设成立：通用门控处于Defender而资产专用持黄金的166天贡献+0.064 log
excess，黄金自身触发Defender的53天再贡献+0.143。资产专用落后主要因为创业板/纳指完全
取消市场门控，分别损失-0.139/-0.130。

通用门控相对领先高度集中于2022-08-05至11-28：Defender上涨24.98%，其中2022年11月
港股通红利ETF 513530上涨约23.18%；资产专用同期几乎持平。删除该阶段后资产专用以
49.49%/1.664反超通用门控48.77%/1.652。通用门控附近参数均能捕获该段，故不是精确参数
尖峰，但相对优势确有单事件运气。完整报告见
`docs/research/2026-08-25_universal_vs_asset_gate_attribution.md`。

## 资产异构score与双目标寻参（2026-08-25）

本轮允许510300与518880使用不同DRAQM score，并把Momentum/Defender锁扩为
`0/5/10/15/20/25/30`日。相同候选族分别按全样本和候选无关的普通区间独立排序；在拒绝
510300确认3日的全样本孤立峰值后，两种目标收敛到同一稳定平台：510300用40日DRAQM，
黄金用30/40日25%/75%加权，进出线分别0.30/0.20与0.20/0.05，确认1/1与5/1日，
双袖套锁25/25日。

最终全样本年化46.99%、Sharpe 1.615；普通区间41.17%、1.802。37个单步邻域中12组参数
产生完全相同路径。累计55,355个候选ID、12,619条唯一路径，全样本/普通区间Reality Check
为`p=0.1602/0.1090`。它优于纯Momentum但仍弱于通用510300门控，故只保留研究候选。
配置见`configs/momentum_defender_including_extremes_selected.yaml`与
`configs/momentum_defender_excluding_extremes_selected.yaml`，完整报告见
`docs/research/2026-08-25_dual_regime_unconstrained_score.md`。

## 共同DRAQM score与极端块修剪（2026-08-24）

确认标的是510300与518880。本轮强制两只ETF完全共用`25%×DRAQM20分位 +
75%×DRAQM40分位`，只允许资产专用阈值不同；创业板和纳指不检查。寻参前用两只ETF的
原始5日绝对对数收益把历史切成固定20日块，删除最极端10%块的选参影响，但最终回测完整
保留全部日期；另用波动调整冲击块做独立敏感性。

最终参数为510300进/出0.35/0.25（1/1日确认）、黄金0.50/0.05（3/1日确认），
Momentum/Defender锁25/23日。全样本年化47.47%、Sharpe 1.621；主普通区间40.94%/
1.785，23个单步参数邻域年化/Sharpe Q25为46.22%/1.585。三轮及此前研究合计38,631个
候选ID、5,259条唯一路径；全样本Reality Check `p=0.1400`。它优于纯Momentum但仍弱于
通用510300门控，故只保留研究候选。配置见
`configs/momentum_defender_common_score_trimmed_selected.yaml`，完整报告见
`docs/research/2026-08-24_common_score_extreme_trim.md`。

## 指定Momentum Top-1资产的下行RAQM（2026-08-24）

按用户要求测试“仅当Momentum Top-1为510300或518880时才检查自身下行RAQM；创业板和
纳指不判断”。请求中的510330按当前资产池已有的510300解释。最终双启用候选对510300使用
30/40日25%/75%分位、0.35/0.25进出线；对黄金使用20/40日25%/75%分位、0.45/0.00
进出线，黄金入场需连续5日。Momentum/Defender锁为20/23日，恢复持续监控原触发资产。

历史年化47.51%、Sharpe 1.618、MDD -25.50%，三轮25,101个候选ID去重为3,766条路径；
全局Reality Check相对Momentum `p=0.0864`。它优于纯Momentum，但弱于通用510300门控的
50.88%/1.721；黄金只有2次触发事件，参数证据不足。因此只保留为机制专用研究候选，配置见
`configs/momentum_defender_selected_asset_draqm_selected.yaml`，完整报告见
`docs/research/2026-08-24_selected_asset_downside_raqm.md`。

## 510300下行RAQM切换研究与正式晋升来源（2026-08-24）

新研究候选`momentum_defender_downside_raqm_weighted_v1`固定双对数Momentum与既有
Defender，只用510300的30/40日下行RAQM严格滞后分位（25%/75%加权）决定顶层切换；
Momentum与Defender均锁定30日，不允许紧急破锁，也不使用5日桥接或Gold覆盖。完整历史
年化50.88%、Sharpe 1.721，16组同权重参数邻域全部达到45%年化；固定留一年最低年化
45.98%，删任一事件最低46.97%，3倍费用49.75%。

三轮共72,144个候选ID、42,010条全局唯一路径，年度Reality Check `p=0.1742`；研究流程
本身不支持自动晋升。2026-08-25用户明确选择该稳健版本作为正式策略。研究配置见
`configs/momentum_defender_downside_raqm_selected.yaml`，完整报告见
`docs/research/2026-08-24_downside_raqm_momentum_defender.md`，机器证据位于
`experiments/20260824_momentum_defender_downside_raqm_final_selection/`。

## 历史正式v3（2026-08-24）

正式策略为`momentum_defender_absolute_stability_raw_gold_v3`：双对数Momentum、沪深300与
当前持仓的120日双趋势门控、20/40日非对称锁、5日负趋势加20日下行波动q95紧急退出，
以及无地板无剪裁的Raw Gold RAQM-W5（入场2.0、退出0.75、硬持有5日）。正式检查点年化
47.45%、Sharpe 2.203、MDD -16.77%，Gold入场31次、持有180日。

主HTML报告以原Momentum为base，位于
`experiments/20260824_momentum_defender_absolute_stability_raw_gold_v3_formal/formal_vs_original_momentum.html`。
旧v2配置和治理记录作为历史回滚证据保留，不覆盖其状态或前瞻账本。

## 双对数Momentum与切换层稳健寻优（2026-08-24）

Momentum现固定为20日对数收益乘对数路径Kaufman ER，因子版本为`quality_momentum 2.0.0`，
不参与切换层寻参。预注册实验先测试240组慢门，再将development段Pareto候选带入805组
联合搜索，共1,040个实际候选ID、626条唯一收益路径；validation、recent和full均未参与
选参。

开发期候选全样本年化45.93%、Sharpe 2.435、MDD -17.43%，相对双对数正式基线的
48.75%、2.054、-16.96%仅改善Sharpe，收益和MDD均恶化。候选未通过validation与recent；
全网格虽然有28个全样本三指标表面赢家，但没有任何候选同时通过两段预注册门槛。
CSCV-PBO为27.9%，年度Reality Check `p=0.7522`，walk-forward收益胜率0%，配对分块
bootstrap区间跨0。因此切换层寻优被拒绝，正式v2保留原慢门、资产专用紧急退出和30日锁。
完整证据位于`experiments/20260824_momentum_defender_log_qm_switch_robust/`。

### 宽机制、集成与全局校正

随后放开原机制，额外测试跨资产门控、非对称锁、方向敏感紧急退出和多窗口/相对Defender
投票。三轮合计3,641条全局唯一路径。没有候选经年度Reality Check证明显著优于正式基线。

按绝对跨年、删事件和费用稳定性保留的研究候选使用120日“沪深300与当前持仓趋势同时为正”
门控、Momentum/Defender 20/40日非对称锁，以及“5日趋势为负且20日下行波动超过严格滞后
q95”的紧急退出。候选年化47.32%、Sharpe 2.215、MDD -16.77%；留一年最低年化40.93%、
最低Sharpe 2.028，删前三大正事件后仍为41.81%/2.100。它是低波动研究候选，不自动替换
生产。完整结论见`docs/research/2026-08-24_log_qm_robust_strategy_selection.md`。

### Gold RAQM正则化简化

在正式C2和绝对稳定候选两条基础状态上共同重搜Gold窗口、进退场阈值、波动率地板和剪裁，
共8,256个参数ID。147个候选通过跨基础稳健门槛，其中19个完全不使用地板和剪裁。复杂度
优先规则选择`raw W5 / entry 3.0 / exit 1.0 / hard-hold 5`：正式基础年化48.33%、
Sharpe 2.041、MDD -14.27%；稳定基础45.78%、2.160、-16.77%。两边Walk-forward收益
胜率均80%，但Reality Check p值为0.887/0.838，故保留为简化研究候选，不自动替换生产。
完整结论见`docs/research/2026-08-24_gold_raqm_regularization_research.md`。

绝对稳定性基础状态的专项Raw搜索另行只比较5/10/20日窗口，共258个参数ID。5日窗口有37个
稳健合格组合，10日0个、20日4个；专项选择Raw W5 / 2.0 / 0.75，年化47.45%、Sharpe
2.203、MDD -16.77%，Gold入场31次。Reality Check `p=0.5864`，故仍为研究候选。

> 2026-08-22 更新：本页主体记录的 `frozen_v2` 保留为外部交接版历史检查点。
> 当前可运行版本是 `momentum_defender_c2_defender_main_b5e3419`，入口为
> `research.run_momentum_defender_integrated` 和根目录
> `run_daily_momentum_defender.py`；它直接调用项目内 `defender/` 的固定上游实现，
> 不依赖下文所述外部交付目录。

## Defender红利目标联合门控实验（2026-08-22）

实验入口：

```bash
uv run python -m research.run_momentum_defender_dividend_gate
```

该实验测试“510300慢门控为风险关闭，且Defender下一开盘红利ETF目标达到最低仓位”
才允许切入Defender的策略族。用户指定的40日、2.5%、红利目标80%、30日锁方案显著
弱于当前集成C2；2,450组参数搜索也没有找到Sharpe或最大回撤超过当前基线的候选，
因此不替换生产策略。完整输出位于
`experiments/20260822_momentum_defender_dividend_gate/`，最佳分段稳健候选仅作为
`configs/momentum_defender_dividend_gate_best_robust.yaml`研究检查点保存。

## Defender整体曲线直接参与质量动量（2026-08-23）

```bash
uv run python -m research.run_defender_curve_momentum
```

本实验把Defender完整连续持有净值当作一个合成资产，与四只ETF完全调用同一个
`quality_momentum(window=20)`，按上一收盘五选一、下一开盘执行。每日版相对原Momentum
只小幅提高收益和Sharpe，却将最大回撤加深到约-31%；5日持有版也未改善，明显弱于当前
C2，因此不替换生产策略。输出位于
`experiments/20260823_defender_curve_quality_momentum/`。

## C2黄金趋势覆盖（2026-08-23）

```bash
uv run python -m research.run_momentum_defender_gold_override
uv run python -m research.run_momentum_defender_gold_override_refinement
uv run python -m research.finalize_momentum_defender_gold_override
```

该实验保留C2基础状态机，仅在C2处于Defender时允许黄金以风险调整收益优势突破510300
慢门控和30日锁。首轮1,800组加局部2,058组搜索后，冻结研究候选使用5日收益/年化波动率、
Gold减Defender差值>0.60入场、≤-0.40退出、黄金最短持有7日。历史年化55.26%、
Sharpe 2.301、MDD -12.79%，相对当前C2提高年化与Sharpe、MDD持平；但3,834组候选的
后续过拟合审计给出HIGH风险：多重试验校正不显著、bootstrap区间跨0、扩展式walk-forward
胜率为0%。因此该版本拒绝生产晋升，仅保留研究检查点。配置见
`configs/momentum_defender_gold_override_best.yaml`，输出见
`experiments/20260823_momentum_defender_gold_override/`。

## C2月度跑输归因（2026-08-23）

```bash
uv run python -m research.run_c2_monthly_underperformance
```

该审计逐月比较当前C2与原四ETF Momentum，并把逐日相对log收益拆成慢门控防守、Defender
退出锁延迟、紧急cap和切换日。2019-01-18至2026-08-21的92个月中C2跑输30个月；
主要负向来源是慢门控仍要求Defender时错过Momentum上涨，而不是30日锁本身。缩短退出锁
和移除紧急cap的简单反事实均显著恶化核心指标，因此不修改生产规则。输出位于
`experiments/20260823_c2_monthly_underperformance/`。

## C2统一Momentum Top1逃生门控（2026-08-23）

```bash
uv run python -m research.run_momentum_top1_defender_escape
```

该实验在C2处于Defender时，用同一X指标、窗口和阈值比较当前Momentum Top1与Defender
整体曲线；四只ETF不设独立参数。预注册20日quality momentum方案和162组小网格全部弱于
当前C2，没有候选提高年化、Sharpe或MDD。最佳候选年化48.80%、Sharpe 1.927、MDD
-15.51%；bootstrap和walk-forward也不支持。结论为机制被否证，不替换生产策略。输出位于
`experiments/20260823_momentum_top1_defender_escape/`。

## C2资产专用Top1逃生参数（2026-08-23）

```bash
uv run python -m research.run_asset_specific_top1_escape
```

该实验允许四只ETF分别使用不同X指标、窗口、差值阈值和最低持有期。先评估864组单资产
策略，再为每只资产保留前三个高年化参数与禁用选项，组合枚举256组。最高年化组合仅启用
沪深300与黄金：历史年化53.50%，但Sharpe降至2.088、MDD恶化至-16.77%；walk-forward
胜率为0%，多重试验校正p=0.985，判定为HIGH过拟合风险，不替换生产C2。输出位于
`experiments/20260823_asset_specific_top1_escape/`。

## 固定10日风险调整收益、黄金硬持有5日（2026-08-23）

```bash
uv run python -m research.run_gold_min5_risk_adjusted_escape
```

该实验固定X为10日收益/10日年化日波动率，只允许黄金从Defender逃生；黄金前5个完整
交易日无条件持有，第6个开盘起基础C2若已恢复Momentum则切原Top1，否则按差值退出线回
Defender。仅搜索入场和退出两个阈值，共1,317组；没有候选提高年化，最佳年化51.33%、
Sharpe 2.134、MDD -12.79%，仍弱于当前C2。统计审计同样不支持，因此机制被否证、不修改
生产策略。输出位于`experiments/20260823_gold_min5_risk_adjusted_escape/`。

## 注册20日风险调整动量、黄金硬持有5日（2026-08-23）

```bash
uv run python -m research.run_gold_min5_risk_adjusted_momentum
```

Gold和Defender整体NAV均调用注册的`risk_adjusted_quality_momentum(window=20,
vol_floor_annual=0.08)`，其余黄金5日硬持有与C2优先机制不变，仅搜索入退差阈值。2,012组中
53组提高年化、3组同时微升年化和Sharpe；最高结果为年化51.78%、Sharpe 2.204、MDD
-12.79%，但只触发1次、持有14日，bootstrap与多重试验校正均不显著，判定HIGH过拟合，
不替换生产。输出位于`experiments/20260823_gold_min5_risk_adjusted_momentum/`。

## 注册5日风险调整动量、黄金硬持有5日（2026-08-23）

```bash
uv run python -m research.run_gold_min5_risk_adjusted_momentum_w5
```

固定机制同上，仅把注册`risk_adjusted_quality_momentum`窗口改为5日。2,714组阈值中最佳
为入场差2.20、退出差0.60：年化56.87%、Sharpe 2.343、MDD -12.97%，20次入场、
111个黄金日。Bootstrap、PBO、分段、参数邻域和去单事件均支持历史改善，但年度块多重试验
校正p=0.657仍不显著，因此评级为MODERATE过拟合风险，只冻结为影子候选，不替换生产。
输出位于`experiments/20260823_gold_min5_risk_adjusted_momentum_w5/`。

## 当前版本

保留的历史Momentum/Defender融合研究检查点是：

- 策略ID：`momentum_defender_c2_frozen_v2`
- 参数ID：`C2_vw10_cap0.8_qc3000.70_qcyb0.95_qndx0.95_qau0.90`
- 冻结日期：2026-08-21
- 回测截止：2026-08-17
- 参数文件：[`configs/momentum_defender_c2_frozen_v2.yaml`](configs/momentum_defender_c2_frozen_v2.yaml)
- 固定实现：[`momentum_defender_c2.py`](momentum_defender_c2.py)
- 唯一正式复现入口：[`run_momentum_defender_c2_frozen.py`](run_momentum_defender_c2_frozen.py)

“冻结”表示后续回测和前瞻观察只能从上述配置读取参数，不允许从寻参结果、全样本oracle或
消融脚本动态选择参数。其他`run_momentum_held_asset_*`文件是历史研究证据，不是当前策略入口。
原q90版本保留为`momentum_defender_c2_frozen_v1`历史检查点，不被覆盖或删除。

## 固定规则

1. Momentum袖套使用`strategy/configs/quality_momentum_top1.yaml`，在沪深300、创业板、纳指和黄金四只ETF中持有Top-1。
2. 慢门控使用沪深300ETF的40日收益。上一收盘高于2.5%时希望持有Momentum，否则希望持有Defender；信号下一开盘执行。
3. 紧急cap使用每只ETF自己的10日Rogers–Satchell波动率和严格滞后一期的全历史扩展分位数：
   - 沪深300ETF：q70
   - 创业板ETF：q95
   - 纳指ETF：q95
   - 黄金ETF：q90
4. 每个开盘只检查上一收盘Momentum实际持有ETF的cap；`cap≤0.8`时紧急切入Defender。
5. 状态锁为30个交易日。紧急Momentum→Defender可以绕过锁，Defender→Momentum不能绕过。
6. 切换日使用旧袖套退出腿与新袖套进入腿复合收益，费用沿用Momentum与Defender交接接口。

分位数历史口径固定为“截至当时全部可用历史”，不是500日滚动窗口。创业板q95也不需要沪深300q70进行二次确认。

## 一键复现

在项目根目录运行：

```bash
uv run python -m research.run_momentum_defender_c2_frozen
```

默认读取Defender交接目录：

```text
/Users/hujiaoyuan/Desktop/Quant/Defender/defender/deliverable
```

如果交接文件移动，可显式覆盖目录，但文件内容必须与版本检查点一致：

```bash
uv run python -m research.run_momentum_defender_c2_frozen \
  --defender-dir /path/to/defender/deliverable
```

输出目录为`experiments/20260821_momentum_defender_c2_frozen_v2/`，包含：

- `momentum_defender_c2_vs_original_base.html`
- `momentum_defender_c2_vs_original_momentum.html`
- `momentum_defender_c2_vs_no_cap_fusion.html`
- `daily_backtest.csv`
- `defender_periods.csv`
- `strategy_period_metrics.csv`
- `calendar_year_returns.csv`
- `checkpoint_audit.json`
- `experiment_manifest.json`
- `research_report.md`
- `strategy_config.yaml`

HTML必须由项目标准QuantStats报告生成器生成。`checkpoint_audit.json`同时锁定样本日期、
1,837个逐日收益的float64哈希、主要收益指标、报警次数、紧急入场、Defender持有日和切换次数；
任何输入或实现漂移都会使正式复现失败。

## 冻结检查点

|指标|固定值|
|---|---:|
|样本|2019-01-18至2026-08-17|
|交易日|1,837|
|年化收益|51.5746%|
|Sharpe|2.1992|
|最大回撤|-12.7917%|
|报警日|223|
|紧急入场|10|
|Defender持有日|1,230|
|袖套切换|43|

## 版本历史

|版本|创业板分位数|状态|年化收益|Sharpe|MDD|
|---|---:|---|---:|---:|---:|
|`frozen_v1`|q90|历史检查点|49.1434%|2.1919|-12.7917%|
|`frozen_v2`|q95|当前生效|51.5746%|2.1992|-12.7917%|

## 研究结论与边界

- q95在2019—2023年与q90产生完全相同路径；其优势来自已经观察到的2024—2026后段历史。它原本属于全样本oracle，本次升级为v2是明确的用户决策，不应改写成开发期已经选出q95。
- v2相对无cap的全样本年化高约2.95个百分点，但主要正向贡献仍集中于少数事件；冻结后必须从下一未观察交易日起评价，不能继续使用同一后段历史修改参数。
- 既有Reality Check、PBO、bootstrap和事件级重选均不足以证明cap可以稳定增强未来收益。因此当前定位是“冻结后前瞻验证的回撤控制候选”，不是已证实的收益增强器。
- 取消创业板cap、取消30日锁、增加沪深300q70确认以及改用500日滚动分位数，均为消融或反事实研究，不属于当前固定版本。
- `frozen_v2` 本身仍未接入`run_daily.py`；新的内嵌 main 版本通过独立入口完成实时Defender目标、确定性跨袖套状态回放、下一开盘执行与持仓持久化。两者必须保持不同策略ID，不能用新实现改写旧检查点。

## 历史W40：价格×量能×基金份额奥卡姆逃顶（2026-08-25）

```bash
uv run python -m research.run_momentum_defender_peak_escape_occam
```

本实验以当时正式`momentum_defender_w40_loss_excluding_extremes_v1`为基线，只在正式路径请求
Momentum时检查实际候选ETF：是否突破严格滞后200日高点、20日涨幅是否达到15%/20%/25%、
当日成交量是否达到此前20日中位数1.5/2.0倍，以及拆分调整后基金份额20日增长是否达到5%。
触发后下一开盘临时切Defender，最短持有5/10日。总计24个ID、14条唯一收益路径，没有资产
专用阈值、权重拟合、机器学习或逐事件例外。

预注册development/validation资格池为0，因此**没有选中候选，不建立shadow，不修改正式策略**。
仅作诊断的领先路径为价格20日涨幅≥15%且突破200日高点、量比≥1.5、Defender最短持有10日：
全样本Top20平均回撤从-10.15%改善到-7.80%，MDD从-25.50%改善到-14.61%，但development
Top20均值恶化0.11个百分点，触发资格否决。基金份额在量能未达标时没有增加任何入场，按
奥卡姆原则不进入诊断规则。完整证据位于
`experiments/20260825_momentum_defender_peak_escape_occam/`。
