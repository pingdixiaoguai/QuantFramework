# QuantFramework

可扩展的量化策略研究与执行框架。六层架构，回测与实盘共享同一套因子代码，架构层面强制隔离未来信息。

## 架构

```
数据层 → 因子层 → 标准化层 → 策略层 → 执行层 → 通知层
                                ↑
                            回测引擎（横切所有层的运行模式）
```

| 层 | 职责 | 关键文件 |
|----|------|----------|
| 数据层 | Tushare 同步 + Parquet 本地存储 | `data/sync.py`, `data/store.py` |
| 因子层 | 接收行情，输出原始因子值 | `factors/momentum.py`, `factors/volatility.py` |
| 标准化层 | 因子值映射到可比较空间 | `standardization/methods.py` |
| 策略层 | 消费因子输出，生成目标持仓权重 | `strategy/momentum_rotation.py` |
| 执行层 | 对比目标与当前持仓，生成调仓指令 | `execution/interfaces.py` |
| 通知层 | 调仓指令推送到钉钉 | `notification/dingtalk.py` |

## 快速开始

### 1. 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone <repo-url> && cd QuantFramework
uv sync
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入真实值：

```bash
cp .env.example .env
```

```dotenv
# .env
TUSHARE_TOKEN=你的tushare_token          # 必填，https://tushare.pro 注册获取
DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=xxx  # 实盘通知用
DINGTALK_SECRET=SEC...                    # 钉钉机器人加签密钥（可选）
```

### 3. 同步数据

首次运行会拉取 2016 年至今的全量历史数据，后续只增量同步：

```bash
# 按策略配置批量同步（推荐，从 asset_pool 读取标的列表）
uv run python -m data --config strategy/configs/momentum_rotation.yaml

# 或同步单个标的
uv run python -m data 510300.SH
```

数据存储在 `data/db/` 下，每个标的一个 Parquet 文件。

> **注意：** `run_daily.py` 运行时会自动同步数据并检查新鲜度，无需手动执行此步骤。首次使用时仍需手动同步以拉取全量历史。

### 4. 运行回测

```bash
# 使用默认配置回测
uv run python run_backtest.py

# 使用自定义配置
uv run python run_backtest.py --config strategy/configs/momentum_rotation.yaml

# 从实验日志复现
uv run python run_backtest.py --from-log experiments/20260413-001.yaml

# 用另一个策略当 benchmark（替代默认池子均值）
uv run python run_backtest.py --config strategy/configs/foo.yaml --baseline-config strategy/configs/bar.yaml
```

输出示例：

```
Running backtest...

Backtest complete: 2400 trading days
Train/test split at: 2023-01-15
  Train: return=85.20%  sharpe=1.20  max_dd=-18.30%
  Test : return=32.10%  sharpe=0.80  max_dd=-22.50%
  Bench: return=45.00%  sharpe=0.60  max_dd=-30.10%

Experiment log: experiments/20260413-001.yaml
HTML report: experiments/20260413-001.html
```

### 4.1 当前正式策略：W40冠军 + QM40红利Defender + QM40恢复早退 + 黄金逃生

当前正式策略为`momentum_defender_w40_qm40_threshold_v5`。Momentum使用四ETF的20日
双对数质量动量Top-1；W40门控使用510300单一40日下跌幅度的严格滞后756日分位、60%/35%
进出线和30日Momentum锁。Defender每月100%持有QM40最低的已上市红利ETF。

W40基础状态进入Defender至少5日后，若510300的QM40严格大于0.0075连续10日，下一开盘可提前恢复
Momentum；否则基础Defender满30日后按W40分位≤35%保底恢复。基础状态及计数在黄金覆盖期间
继续运行。这里的5日是基础Defender状态计数，不是当前底层红利ETF或实际Defender连续持仓。

黄金覆盖层有两个入口：W40切入Defender的同一开盘，若Momentum Top1为黄金且
`QM20(Gold)-QM20(连续Defender净值)>0.005`，实际持仓立即黄金、不先买Defender；其他时候
遵循统一5日规则。之后若基础W40仍为Defender，黄金差值低于-0.020或Top1已不再是黄金，则返回
Defender，否则继续黄金。所有信号上一收盘确定、下一开盘执行。

正式回测区间固定为2013-01-01至最新完整交易日；20日因子暖机后的首个可执行日为
2013-02-04。当前检查点截至2026-08-26：年化39.71%、Sharpe 1.437、最大回撤-48.63%；
2019-01-18重启主研究口径为58.43%/1.886/-25.50%。全历史共35次黄金逃生、438个逃生日，
其中28次实际打破未满30日的基础Defender锁，8次在W40切入
Defender的同一开盘即时否决实际Defender持仓。收益哈希为
`6a45479ffe5da9b081e53e36c7a0b137656ed9a61cdf3c7d8044aa278700f4d3`。

- [正式配置](strategy/configs/momentum_defender_w40_gold_escape.yaml)
- [2013-01-01至2026-08-26正式HTML回测](experiments/20260827_momentum_defender_w40_qm40_threshold_v5_formal/formal_backtest_2013-01-01_to_2026-08-26.html)
- [2019-01-18至2026-08-26重启HTML回测](experiments/20260827_momentum_defender_w40_qm40_threshold_v5_formal/formal_backtest_2019-01-18_to_2026-08-26.html)
- [正式晋升报告](docs/research/2026-08-27_qm40_recovery_threshold_v5_formal_promotion.md)
- [v4直接回滚配置](experiments/20260826_momentum_defender_w40_qm40_signed_exit_v4_formal/strategy_config.yaml)
- [2007研发段与2026机械验证](docs/research/2026-08-26_defender_2007_2026_validation.md)
- [完整寻参与稳健性报告](docs/research/2026-08-25_w40_asset_specific_qm20_escape_search.md)
- [2019主样本奥卡姆与参数稳健性复核](docs/research/2026-08-26_current_strategy_occam_robustness_2019.md)
- [Defender 30日锁与灵活退出机制审计](docs/research/2026-08-26_defender_exit_mechanism_audit_2019.md)
- [W40冠军＋Defender QM40＋QM40早退组合回测](docs/research/2026-08-26_w40_defender_qm_signed_exit_combination_2019.md)
- [QM40基础Defender恢复阈值寻参](docs/research/2026-08-26_qm40_recovery_threshold_search_2019.md)
- [2013全史尾部压力复核](experiments/20260826_current_strategy_occam_robustness_audit/REPORT.md)
- [整体策略Top 10最大回撤台账](docs/research/momentum_defender_drawdown_badcases.md)
- [Defender跑输Momentum台账](docs/research/momentum_defender_badcases.md)

Defender台账必须固定包含“黄金打破Defender锁胜负台账”：同时报告2013完整历史与
2019-01-18重启至最新共同截止日两种口径下的触发次数、黄金跑赢连续Defender的次数和胜率，
并拆分“已持有Defender后突破”与“Defender入场当日被黄金否决”。每次事件从黄金触发日开盘
计至最后一个完整黄金逃生日收盘，黄金与事件期间持续Defender使用相同的前序持仓、开盘切换腿
和费用；收益严格更高才算跑赢。截止日尚未结束的事件必须标为开放、计入当期统计，并在以后
重建台账时自动更新，不能把暂时胜负误写成已完成事件结论。

重新生成正式检查点与两本台账：

```bash
uv run python -m research.generate_strategy_drawdown_badcases
uv run python -m research.generate_momentum_defender_badcases
uv run python -m research.run_formal_w40_qm40_threshold
uv run python -m research.generate_strategy_drawdown_badcases --check
uv run python -m research.generate_momentum_defender_badcases --check
```

默认日跑已指向新配置；首次生产运行前先执行：

```bash
uv run python run_daily_momentum_defender.py --dry-run
```

钉钉消息同时显示黄金逃生诊断和只读顶部预警，但预警不改变正式策略。消息每天评估Momentum
Top1；即使正式实际持有Defender，也只读展示各条件是否满足、当前值、门槛和尚差多少。普通
Momentum ETF同时满足突破此前200日最高收盘、20日涨幅不低于15%、信号日成交量不低于
此前20日中位数1.5倍时触发；创业板ETF还必须满足20日基金份额严格增长。创业板份额持平、
下降或信号日份额数据不可用时均不预警。预警不进入调仓指令、状态锁、持仓文件或前瞻策略账本。

信号末尾固定追加两组表现。`同期表现`从正式目标权重最近一次实际变化的开盘日起算：显示当前
持仓、连续Momentum袖套、连续Defender袖套，以及Momentum池内所有非当前持仓ETF，并逐项
标注相对当前持仓领先或落后。`周期表现`按自然月、自然季度和自然年截至信号日复合逐日净收益，
显示当前完整策略、原非对数Momentum模型回放、当前纯Momentum和纯Defender。原非对数项按
冻结旧因子与当前统一开盘回测口径从历史重放，不读取另一个部署目录的实盘position/YTD账本，
因此不等同于main服务器钉钉中的实盘累计值。周期第一日收益沿用正式上一收盘至当日收盘的连续
策略口径；表现计算或旧基准复现失败时只显示数据不可用，不阻断正式信号。

### 4.2 当前正式策略完整定义

#### 统一数据、信号与执行时序

- $O_{i,t},H_{i,t},L_{i,t},C_{i,t}$为ETF $i$在交易日$t$的后复权开、高、低、收；
- 所有“20日”“40日”等窗口均指有效交易观测，不是自然日；
- 交易日$t$收盘后确定的因子，只能控制下一可执行交易日$t+1$开盘目标；
- 正式仓位始终为一个候选100%，正常情况下没有现金；无法完成换仓时保留原候选；
- 股票ETF单边交易成本为0.01%。跨标的换仓分别计入旧标的卖出腿和新标的买入腿。

#### 统一5日规则及作用域

策略中涉及短期主动轮转的约束统一使用5个交易日，但它不是“任何底层ETF都必然持有满5日”：

- Momentum袖套选中的实际ETF至少持有5日，之后才按新的Top-1轮转；
- 每次黄金逃生入场后至少持有5日；
- 常规Defender→黄金逃生要求实际Defender袖套已连续持有5日；W40切入Defender同一开盘的
  即时黄金否决是唯一入口例外；
- Defender内部红利ETF按月度规则选择，计数属于Defender袖套而非每只底层红利ETF。月初换成
  新红利ETF后，顶层可能在不足5日时进入黄金，因此不能表述为“所有标的至少持有5日”。

以下各层只描述信号与状态，不再重复解释同一5日规则。

#### 第一层：Momentum四ETF质量动量Top-1

Momentum候选池为`510300.SH`沪深300ETF、`159915.SZ`创业板ETF、`513100.SH`纳指ETF和
`518880.SH`黄金ETF。对每只ETF计算注册因子`quality_momentum 2.0.0`。

20日对数动量为：

$$
M_{i,t}^{(20)}=\ln\frac{C_{i,t}}{C_{i,t-20}}.
$$

20日对数路径效率为：

$$
ER_{i,t}^{(20)}=
\frac{\left|\ln(C_{i,t}/C_{i,t-20})\right|}
{\sum_{j=t-19}^{t}\left|\ln(C_{i,j}/C_{i,j-1})\right|}.
$$

最终质量动量为：

$$
QM_{i,t}^{(20)}=M_{i,t}^{(20)}ER_{i,t}^{(20)}.
$$

$M$保留正负方向，$ER\in[0,1]$惩罚来回震荡的路径。每个信号收盘横截面选择
$QM^{(20)}$最高的ETF作为原始Top-1，再应用统一5日规则得到实际Momentum目标。

#### 第二层：510300单一W40风险门控

顶层门控只观察510300，不观察当前Momentum持仓。先计算40日对数收益和只保留下跌的幅度：

$$
R_{t}^{(40)}=\ln\frac{C_{510300,t}}{C_{510300,t-40}},
\qquad
L_t^{(40)}=\max\left(-R_t^{(40)},0\right).
$$

若510300过去40日上涨或持平，则$L_t^{(40)}=0$。对当前下跌幅度计算滚动756日历史分位，
至少需要252个历史观测：

$$
P_t^{(40)}=
\begin{cases}
0, & L_t^{(40)}=0,\\
\displaystyle\frac{1}{N_t}\sum_{s\in\mathcal H_t}
\mathbf 1\!\left(L_s^{(40)}\le L_t^{(40)}\right), & L_t^{(40)}>0,
\end{cases}
$$

其中$\mathcal H_t$为最近756个已完成门控观测，$N_t\ge252$。分位越高，表示
当前40日跌幅在约两年历史中越极端。

基础W40状态机固定为：

|当前基础状态|切换条件|最短持有锁|否则|
|---|---|---:|---|
|Momentum|$P_t^{(40)}\ge60\%$|30日|保持Momentum|
|Defender|QM40连续恢复或$P_t^{(40)}\le35\%$保底|见下节|保持Defender|

35%与60%之间是迟滞区。基础状态从Momentum开始；Momentum未满30日时即使达到进入线也只
记录证据。W40基础状态在黄金逃生期间仍独立连续运行。

#### 第三层：Defender月度40日反转

Defender候选池固定为六只红利或红利低波ETF：`512890.SH`、`513530.SH`、`515080.SH`、
`510880.SH`、`515450.SH`、`513630.SH`。

|代码|中文全名|上市日期|跟踪指数|
|---|---|---|---|
|512890.SH|华泰柏瑞中证红利低波动ETF|2019-01-18|中证红利低波动指数|
|513530.SH|华泰柏瑞中证港股通高股息投资ETF（QDII）|2022-04-25|中证港股通高股息投资指数|
|515080.SH|招商中证红利ETF|2019-12-27|中证红利指数|
|510880.SH|华泰柏瑞上证红利ETF|2007-01-18|上证红利指数|
|515450.SH|南方标普中国A股大盘红利低波50ETF|2020-02-26|标普中国A股大盘红利低波50指数|
|513630.SH|摩根标普港股通低波红利ETF|2023-12-08|标普港股通低波红利指数|

这张表是正式README的固定组成部分。Defender标的池发生增删、代码/名称/上市日/跟踪指数变化
时必须同步更新；以后整理README不得删除或用仅代码列表替代。

在每月首个可执行开盘，用前一收盘已经可知的数据计算每只ETF的40日质量动量：

$$
M_{i,t}^{(40)}=\ln\frac{C_{i,t}}{C_{i,t-40}},
\qquad
ER_{i,t}^{(40)}=
\frac{|M_{i,t}^{(40)}|}
{\sum_{j=t-39}^{t}|\ln(C_{i,j}/C_{i,j-1})|},
\qquad
QM_{i,t}^{(40)}=M_{i,t}^{(40)}ER_{i,t}^{(40)}.
$$

令$\mathcal E_t$为执行日已经上市、开盘可交易且具有完整40日历史的候选集合，月度目标为：

$$
i_t^*=\arg\min_{i\in\mathcal E_t}QM_{i,t}^{(40)}.
$$

也就是选择路径效率调整后最弱的一只；持续、平滑走弱的ETF会比震荡下跌ETF得到更负的QM40。
选中后100%持有至下次月度选择。若月初没有新的合格标的，或旧持仓无法卖出，则保留原标的；
并列时按配置中的候选顺序确定。

Defender还维护一条“始终执行上述月度规则”的连续净值$N_t^D$。即使组合实际处于Momentum
或黄金逃生，这条反事实Defender净值仍继续月度轮转；黄金逃生比较的就是这条连续净值，而
不是从进入Defender当天重新起算的临时收益。

#### 第四层：QM40基础Defender恢复

对510300计算与Defender排序完全同口径的有符号$QM_{510300,t}^{(40)}$。进入基础Defender后，
最少观察5个完整交易日；若

$$
QM_{510300,t}^{(40)}>0.0075
$$

连续10个收盘，则下一开盘提前恢复Momentum。若未早退，基础Defender满30日后，只要
$P_t^{(40)}\le35\%$，下一开盘保底恢复Momentum。基础Defender计数和QM40连续计数在实际持有
黄金逃生期间仍继续运行。

阈值0.0075要求$R40>0.0075/ER40$；路径越平滑，越容易满足同一恢复阈值。该阈值来自
0.005–0.010回溯平台的中央代表，统计证据不支持自动晋升，本次变化来自用户明确决定。

#### 第五层：仅黄金可打破Defender锁

对任意正值曲线$Z_t$定义与Momentum完全同口径的20日质量动量：

$$
Q_t^{(20)}(Z)=
\ln\frac{Z_t}{Z_{t-20}}
\frac{\left|\ln(Z_t/Z_{t-20})\right|}
{\sum_{j=t-19}^{t}\left|\ln(Z_j/Z_{j-1})\right|}.
$$

黄金使用其后复权收盘曲线$Z^G=C_{518880}$；Defender使用扣除内部月度换仓费用后的连续净值
$Z^D=N^D$。两者都以信号收盘计算，并在下一开盘使用。定义相对优势：

$$
\Delta_t=Q_t^{(20)}(Z^G)-Q_t^{(20)}(Z^D).
$$

黄金逃生有两个入场路径：

1. **即时入口否决**：基础W40在本执行开盘刚切入Defender，且当前Momentum有效Top-1为
   黄金、$\Delta_t>0.005$。W40基础状态和30日锁照常进入Defender，但实际持仓直接黄金；
2. **持有后破锁**：基础W40仍为Defender、组合已经实际连续持有Defender至少5个完整交易日，
   且当前Momentum有效Top-1为黄金、$\Delta_t>0.005$。

入场后应用统一5日规则，期间不响应W40恢复、Top-1变化或退出阈值。约束完成后：

- 若基础W40已经恢复Momentum，结束逃生并交回正常Momentum Top-1；
- 若基础W40仍为Defender且当前Top-1不再是黄金，返回Defender；
- 若基础W40仍为Defender、Top-1仍是黄金且$\Delta_t<-0.020$，返回Defender；
- 其余情况继续持有黄金。

即时否决只在基础W40切入Defender的同一开盘检查一次。返回Defender后，常规黄金逃生资格
按统一5日规则重新累计。
入场和退出分别使用严格的`>`与`<`，等于阈值时不触发。

#### 最终目标优先级

每个下一开盘按以下顺序确定唯一100%目标：

1. 黄金逃生处于统一短期持有约束：持有入场时的黄金；
2. 黄金逃生在硬持有后仍满足继续条件：持有黄金；
3. 黄金逃生结束或未激活，基础W40为Momentum：持有Momentum有效Top-1；
4. 黄金逃生结束或未激活，基础W40为Defender：持有当月Defender红利ETF。

正式换仓只在目标候选变化且旧标的可卖、新标的可买时执行；否则保留实际持仓。切换日收益由
旧标的昨收至今开退出腿与新标的今开至收盘进入腿复合，不使用同收盘成交。

#### 当前正式参数总表

|模块|参数|正式值|
|---|---|---:|
|Momentum|QM窗口|20日|
|W40|下跌窗口 / 历史窗口 / 最少历史|40日 / 756日 / 252日|
|W40|进入Defender / 保底恢复Momentum|60% / 35%|
|W40|Momentum锁 / Defender最低观察 / 保底期限|30日 / 5日 / 30日|
|Defender|频率 / 排名 / 仓位|月度 / QM40最低 / 100%|
|Defender恢复|510300 QM40 / 阈值 / 连续确认|40日 / >0.0075 / 10日|
|黄金逃生|即时否决 / 入场$X$ / 退出$Y$|W40切入日 / 0.005 / -0.020|
|统一5日规则|Momentum轮转 / 黄金持有 / 常规Defender逃生资格|5日|

#### 钉钉固定袖套状态与状态原因

钉钉中的袖套标签固定为`动量`、`防守`、`黄金逃生`三种。当前正式状态机全历史只会产生
以下固定搭配；消息中的状态原因使用这里的中文，不输出内部英文代码：

|袖套状态|状态原因|
|---|---|
|动量（保持）|W40基础状态保持动量|
|防守（保持）|W40基础状态保持防守|
|动量 → 黄金逃生|黄金已满足条件，否决实际进入防守|
|防守 → 黄金逃生|黄金满足逃生条件，打破防守锁|
|黄金逃生（保持）|黄金逃生硬持有期|
|黄金逃生（保持）|黄金逃生继续持有|
|黄金逃生 → 防守|Momentum Top1不再是黄金，返回防守|
|黄金逃生 → 防守|黄金相对Defender指标跌破退出线，返回防守|
|黄金逃生 → 动量|QM40连续恢复或30日W40保底恢复，结束黄金逃生|
|动量 → 防守|W40基础状态保持防守|
|防守 → 动量|QM40连续恢复或30日W40保底恢复动量|

### 5. 正式策略运维

正式运行会自动同步11只ETF的本地行情、检查共同最新交易日、生成下一开盘目标、发送钉钉并
持久化正式持仓。腾讯云Linux服务器的09:00定时部署、失败重试和钉钉故障告警见
[`ops/tencent-cloud/README.md`](ops/tencent-cloud/README.md)。

如果只想测试完整钉钉消息而不写持仓：

```bash
uv run python run_daily_momentum_defender.py --notification-only
```

测试消息会明确标注“通知测试”；`--dry-run`则连钉钉也不会调用。正式运行使用所有标的最新
共同完整收盘日作为信号日，并按交易所日历计算下一执行开盘。

## 策略研发、回测报告与正式晋升标准

本节是仓库内策略研发和正式版本变更的操作标准；更完整的统计要求见
[`research/DEVELOPMENT_VALIDATION.md`](research/DEVELOPMENT_VALIDATION.md)。适用范围包括策略、
因子口径、状态机、覆盖层、阈值、持有期、标的池、费用、执行时序和数据口径。回测更好不等于
可以自动修改正式策略；正式晋升必须有明确用户决定。

### 1. 研发开始前先冻结问题

搜索或编写候选前，研究配置必须写明：

- 当前正式基线及其策略ID、回滚版本和证据截止日；
- 机制假设，以及它要修复的具体事件或结构性问题；
- 唯一允许变化的参数、标的或规则，其他层必须冻结；
- 完整候选空间、选择指标、费用、样本分段和晋升门槛；
- 实际尝试过的候选ID与唯一收益路径数，失败路径同样保留。

看到结果后扩大网格、移动阈值或重新定义目标属于新实验，不能并入原实验冒充预注册。根据
完整历史提出的机制，即使随后机械切出年份，也必须披露该年份是否已经被观察；已经看过的数据
不能称为独立样本外。

### 2. 数据、信号与执行口径

- 统一使用数据层输出的本地后复权OHLC；不得在一个计算中混用原始价、前复权和后复权。
- 收盘信息最早控制下一可执行开盘，必须满足`observation < execution`，禁止同收盘成交。
- 切换日收益必须复合旧标的昨收至今开退出腿与新标的今开至收盘进入腿，并按买卖两腿扣费。
- 上市前、停牌、不可买卖、现金和未执行目标必须显式处理；不得回填尚未上市ETF或用指数
  代理伪造可交易历史。
- 关闭新增机制时必须在声明容差内逐日复现基线；目标权重与现金逐日合计为1，收益必须能重构
  净值。
- ETF规模、流动性或成分筛选必须使用当时可知的截面。用当前幸存ETF回测历史时，必须披露
  幸存者偏差；`fund_share.fund_type`等可空字段不得作为唯一ETF身份依据。
- 不得为了延长研究历史而前插正式生产Parquet并改变HFQ首个复权因子；早期扩展数据使用明确的
  研究专用market override，正式收益哈希必须保持稳定。
- 完整正式回测的配置区间固定为`2013-01-01`至最新完整交易日。20日因子暖机期间不伪造收益，
  报告从首个可执行日开始；资产按真实上市日期逐只加入候选。生成正式报告和两本台账前必须先
  按正式配置同步最新行情，三者使用同一共同截止日。

### 3. 正式回测的Base与对照组

完整Momentum × Defender正式报告的**主Base固定为非对数版本的原Momentum**：

```text
strategy/configs/quality_momentum_top1_legacy_simple_price.yaml
```

该Base冻结为简单价格收益Momentum乘价格路径ER，是“Original Momentum”历史口径。不得把当前
双对数Momentum、当前正式策略自身或重新寻优后的Momentum冒充“原Momentum”。正式主报告
`formal_backtest.html`必须以这个非对数原Momentum为benchmark。

根据研究问题还应增加以下辅助对照，但不能替代主Base：

- 当前无新增机制的直接回滚版本，例如W40＋100%红利Defender；
- 当前双对数纯Momentum，用于判断风险门控的机会成本；
- 连续纯Defender净值，用于分析Defender选择层和黄金相对优势；
- 旧正式版本，用于量化本次变更的真实增量。

Defender单袖套研究可以额外使用原Defender池作为内部Base；但一旦晋升到完整正式组合，仍必须
生成相对非对数原Momentum的标准主报告。

### 4. 回测报告必须使用仓库标准流程和格式

正式HTML统一通过：

```python
from research.standard_report import generate_standard_report
```

该适配器调用仓库的`backtest.report.generate()`和QuantStats标准模板，并处理当前已知的
QuantStats无效`onload="save()"`问题。要求：

- 主文件名固定为`formal_backtest.html`，benchmark名称清楚标注为Original Momentum；
- 同时保留带日期文件名的当前固定区间报告，以及在2019-01-18重新初始化状态的历史对照报告；
  文件名必须包含完整起止日期，不能只写`latest`或覆盖后无法辨认区间；
- 保留标准QuantStats章节、指标表、图表、字体、配色和页面结构；不得用自制页面整体替换；
- 若第三方模板只有单张图错位，只能最小修复该图，不能抛弃其余标准格式；
- 可以额外生成`formal_vs_rollback_*.html`等辅助报告，但主报告不能缺失或改换Base；
- 报告日期、交易日数、策略ID、费用、收益序列与检查点必须一致；打开HTML后还要做一次视觉
  检查，确认年份、柱线位置、表格宽度和窄窗口布局正常。

每个正式实验目录至少包含：

|文件|要求|
|---|---|
|`formal_backtest.html`|正式策略相对非对数原Momentum的标准报告|
|`formal_backtest_2013-01-01_to_<截止日>.html`|当前固定区间的日期标注副本|
|`formal_backtest_2019-01-18_to_<截止日>.html`|保留原2019起点、重新初始化状态的日期标注报告|
|`formal_vs_rollback_*.html`|相对直接回滚或旧正式版本的辅助报告|
|`daily_backtest.parquet/csv`|逐日收益、状态、目标、原因和执行信息|
|`strategy_metrics.csv`|正式、Base、回滚和辅助对照的统一指标|
|`calendar_year_returns.csv`|逐年收益，首尾非完整年度需保留|
|`checkpoint_audit.json`|检查点、收益哈希和研究路径逐日一致性|
|`experiment_manifest.json`|配置、代码、报告、台账和测试源文件SHA-256|
|`strategy_config.yaml`|本次正式版本的完整冻结配置副本|
|`formal_report.md`|中文摘要、晋升来源和证据限制|

### 5. 最低研究验证要求

任何准备晋升的候选至少完成并保存：

- `development`、`validation`、`recent`和`full`固定分段的年化、波动、Sharpe和MDD；
- 参数或离散规则邻域，报告双目标改善率和是否为孤立尖峰；
- 每次状态切换/覆盖事件的起止、双方收益和log excess；
- Leave-one-event，并报告正负事件数、前两大正事件占比、删任一事件后的最低年化/Sharpe；
- 20日成对分块Bootstrap至少5,000次，固定随机种子，报告差值区间和为正概率；
- 多候选研究执行CSCV/PBO和White式Reality Check或等价多重试验校正；
- 扩展walk-forward与leave-one-year，且明确历史是否已经污染选择过程；
- 费用倍数、执行延迟、上市时间、停牌、现金、极端行情和数据修订敏感性；
- 新机制关闭时的基线逐日parity、净值重构、目标合计与正式收益SHA-256。

事件很少、Bootstrap区间跨0、Reality Check不显著或2026等验证段零触发时，必须在结论中直说；
不能因为逻辑直觉好或完整年化略高而隐藏统计不确定性。用户仍可基于治理偏好明确晋升，但晋升
报告必须记录“统计流程不支持自动晋升，正式变化来自明确用户决定”。

### 6. 正式策略更新的同步清单

正式策略一旦更新，以下内容必须在同一次工作中完成，不能只改Python或YAML：

1. **新版本身份**：创建新的正式策略ID和治理JSON，不覆盖旧版本；旧治理记录标记
   `superseded_by`，旧正式实验中的`strategy_config.yaml`作为可执行回滚证据。
2. **正式代码与配置**：回测和日跑共用同一路径；配置列全参数、标的池、费用、证据状态、
   截止日、核心指标、事件数、逐日收益哈希和前瞻账本路径。
3. **README**：同步当前策略ID、完整规则、标的池、参数表、截止日、年化、Sharpe、MDD、
   事件数、收益哈希、正式报告和晋升报告链接。Defender资料表（代码、中文全名、上市日期、
   跟踪指数）是固定内容，任何README整理或标的池更新都必须保留并同步。
4. **模块契约**：核对并更新`strategy/AGENTS.md`、`defender/AGENTS.md`；若数据或运维行为变化，
   同步更新相应模块`AGENTS.md`和`ops/tencent-cloud/README.md`。
5. **两本台账**：重建
   [`momentum_defender_badcases.md`](docs/research/momentum_defender_badcases.md)和
   [`momentum_defender_drawdown_badcases.md`](docs/research/momentum_defender_drawdown_badcases.md)。
   同步更新两个context YAML的策略ID、截止日、资产名和人工解释；当前开放Defender观察段即使
   尚未达到自动阈值，也可按治理要求单独保留，但不得混入正式badcase计数。Defender台账还必须
   自动重算2013完整历史和2019-01-18重启两种口径的黄金破锁触发数、跑赢连续Defender次数、
   胜率及“持有后突破/入场当日否决”拆分；开放事件计入并明确标记，禁止手工维护静态汇总。
6. **标准回测报告**：重新生成`formal_backtest.html`、辅助对比、逐日账本、检查点审计和来源
   manifest；报告主Base仍是非对数原Momentum。
7. **研究与晋升文档**：研究报告保留原始拒绝/不显著结论，随后发生用户治理晋升时追加治理
   更新，不得事后改写研究统计；另写正式晋升报告说明唯一变化、trade-off和回滚路径。
8. **前瞻状态**：新ID使用新的position文件与forward ledger。不得静默继承或改名旧状态；首次
   正式运行前用券商真实持仓、旧状态和新目标人工核对。
9. **生产演练**：执行`run_daily_momentum_defender.py --dry-run --skip-sync`，核对策略ID、信号日、
   执行日、当前/目标袖套、Defender月度目标、覆盖层状态、调仓文案；演练不得发通知或写持仓。
10. **测试与最终复现**：相关测试和全项目`uv run pytest -q`通过后，再运行正式生成器，使最终
    manifest记录的是最后版本源文件；最后运行两本台账`--check`。

推荐的正式收尾顺序：

```bash
# 0. 先同步正式资产池的最新完整行情
uv run python -m data --config strategy/configs/momentum_defender_w40_gold_escape.yaml

# 1. 更新context后重建两本台账
uv run python -m research.generate_momentum_defender_badcases
uv run python -m research.generate_strategy_drawdown_badcases

# 2. 更新README、AGENTS、治理、运维和晋升文档后，生成正式检查点与标准报告
uv run python -m research.run_formal_w40_qm40_threshold

# 3. 生产入口演练
uv run python run_daily_momentum_defender.py --dry-run --skip-sync

# 4. 回归测试
uv run pytest -q

# 5. 确认台账没有在最后修改后变旧；如正式manifest需纳入最终测试文件哈希，再重跑正式生成器
uv run python -m research.generate_momentum_defender_badcases --check
uv run python -m research.generate_strategy_drawdown_badcases --check
uv run python -m research.run_formal_w40_qm40_threshold
```

### 7. 晋升、拒绝与前瞻治理

- 研究脚本不得直接修改生产配置；研究通过与否先写明，正式晋升等待明确用户决定。
- 正式版本使用递增ID；旧配置、正式报告、哈希和前瞻账本永久保留，不得用新结果覆盖旧证据。
- 正式检查点截止后的第一未观察交易日起建立新前瞻账本。同一历史不得继续反复调参并称作验证。
- 被否决或不显著的实验也要保留配置、全部候选、机器审计和报告，避免未来重复探索后只看到
  “成功”路径。
- 正式更新完成的定义不是“代码能跑”，而是代码、配置、README、两本台账、标准报告、治理、
  运维、前瞻状态、日跑演练、测试和回滚证据全部一致。

## 完整范例：从零到第一次回测

以下演示从安装到拿到回测报告的完整流程。

**第一步：安装并配置**

```bash
cd QuantFramework
uv sync
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN
```

**第二步：同步 4 只 ETF 的历史数据**

```bash
uv run python -m data --config strategy/configs/momentum_rotation.yaml
```

**第三步：编写策略配置**

创建 `strategy/configs/my_strategy.yaml`：

```yaml
strategy_name: momentum_rotation
strategy_class: strategy.momentum_rotation.MomentumRotation
asset_pool:
  - 510300.SH   # 沪深300 ETF
  - 159915.SZ   # 创业板 ETF
  - 513100.SH   # 纳斯达克 ETF
  - 518880.SH   # 黄金 ETF
start: "2018-01-01"
end: "2026-04-13"
factors:
  - name: momentum
    weight: 0.7
    params: {window: 20}
  - name: volatility
    weight: 0.3
    direction_flip: true    # 低波动更好，翻转排序方向
    params: {window: 20}
train_ratio: 0.7
rebalance_rule: daily
rebalance_mode: min_hold
rebalance_days: 5
```

配置说明：

| 字段 | 含义 |
|------|------|
| `strategy_class` | 策略类的完整路径，省略则默认 MomentumRotation |
| `asset_pool` | 标的池，Tushare 代码 |
| `start` / `end` | 回测时间范围 |
| `factors` | 使用的因子列表，`weight` 为组合权重，`direction_flip` 翻转排序 |
| `train_ratio` | 训练集占比，用于过拟合检测 |
| `rebalance_mode` | 调仓时序，`min_hold` 为持有满 N 日后每日评估，`fixed_cycle` 为仅在第 N、2N、3N... 个持仓交易日评估 |
| `rebalance_days` | 调仓窗口交易日数 N |

**第四步：运行回测**

```bash
uv run python run_backtest.py --config strategy/configs/my_strategy.yaml
```

结果自动保存到 `experiments/` 目录：
- `*.yaml` — 完整实验日志（参数、指标快照，可一键复现）
- `*.html` — quantstats 可视化报告

**第五步：配置每日实盘推送**

```bash
# 编辑 .env，添加钉钉配置
# DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=xxx
# DINGTALK_SECRET=SECxxx

uv run python run_daily.py --config strategy/configs/my_strategy.yaml
```

该通用示例用于框架开发；正式策略的信号、通知和持仓必须使用
`run_daily_momentum_defender.py`，不能由自定义示例配置替代。

## 新增因子

新增一个因子只需要改动两个文件：

**1. 复制模板并实现计算逻辑**

```bash
cp factors/_template.py factors/my_factor.py
```

编辑 `factors/my_factor.py`：

```python
"""我的因子 — 简短描述。"""

import pandas as pd

METADATA = {
    "name": "my_factor",
    "author": "your_name",
    "version": "1.0.0",
    "params": {"window": 10},
    "min_history": 11,
    "direction": "higher_better",  # 或 "lower_better"
    "description": "一句话描述",
}

def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    series = df["close"].rolling(p["window"]).mean() / df["close"] - 1
    series.index = df["date"]
    return series
```

**2. 注册因子**

在 `factors/registry.yaml` 添加一行：

```yaml
factors:
  - module: factors.momentum
  - module: factors.volatility
  - module: factors.my_factor      # ← 新增
```

**3. 验证**

```bash
uv run pytest factors/tests/
```

然后在策略配置的 `factors` 列表中引用 `my_factor` 即可。

## 新增策略

继承 `BaseStrategy`，实现 `generate_weights` 方法：

```python
# strategy/equal_weight.py
from strategy.base import BaseStrategy

class EqualWeight(BaseStrategy):
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not factor_values:
            return {}
        n = len(factor_values)
        return {asset: 1.0 / n for asset in factor_values}
```

在配置中指定 `strategy_class`：

```yaml
strategy_class: strategy.equal_weight.EqualWeight
```

## 测试

```bash
uv run pytest              # 运行全部测试
uv run pytest factors/     # 只测因子层
uv run pytest -v           # 详细输出
```

## 目录结构

```
QuantFramework/
├── data/                    # 数据层
│   ├── sync.py              #   Tushare 增量同步
│   ├── store.py             #   Parquet 读写 + 查询接口
│   ├── config.py            #   Token 配置
│   └── db/                  #   Parquet 数据文件
├── factors/                 # 因子层
│   ├── _template.py         #   因子模板
│   ├── momentum.py          #   动量因子（20日收益率）
│   ├── volatility.py        #   波动率因子（20日滚动标准差）
│   ├── registry.yaml        #   因子注册表
│   └── validator.py         #   因子输出校验器
├── standardization/         # 标准化层
│   └── methods.py           #   z-score, percentile
├── strategy/                # 策略层
│   ├── base.py              #   策略基类 (BaseStrategy)
│   ├── momentum_rotation.py #   动量轮动策略
│   ├── loader.py            #   策略动态加载器
│   └── configs/             #   策略配置 YAML
├── execution/               # 执行层
│   ├── interfaces.py        #   Order + diff()
│   └── position.py          #   当前持仓读写
├── notification/            # 通知层
│   ├── interfaces.py        #   Notifier 基类
│   ├── dingtalk.py          #   钉钉适配器
│   └── formatter.py         #   调仓信号格式化
├── backtest/                # 回测引擎
│   ├── runner.py            #   时间序列遍历 + 未来信息截断
│   ├── report.py            #   quantstats HTML 报告
│   └── experiment_log.py    #   实验日志
├── experiments/             # 实验日志存储
├── state/                   # 运行状态（当前持仓）
├── run_backtest.py          # 回测入口
├── run_daily.py             # 每日实盘入口
└── docs/DESIGN.md           # 架构设计文档
```
