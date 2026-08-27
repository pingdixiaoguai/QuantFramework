# 腾讯云 09:00 早盘信号部署

适用于一台长期运行的 Linux CVM。任务在周一至周五 09:00（`Asia/Shanghai`）
启动，以最新完整交易日行情计算正式的`momentum_defender_w40_qm40_threshold_v5`，
在钉钉提示当天目标持仓、510300 W40严格滞后756日分位、60%/35%迟滞、QM40连续恢复进度、
QM40月度Defender和黄金QM20逃生状态。
代码还会查询上交所交易日历，法定休市日成功跳过且不发送信号。

整个生产链路是确定性的Python程序，不依赖大模型、Agent、Codex、ChatGPT或任何自然语言
推理服务。服务器只需本仓库锁定的Python依赖、Tushare凭据和钉钉Webhook；策略状态、条件
判断、阈值差距和消息文案均由固定公式及模板生成。模型服务不可用不会影响每日任务，因为
生产环境根本不会调用模型服务。

## 运行语义

- `quant-daily.timer` 只负责 09:00 调度，不启用错过后的延迟补跑。服务器若在开盘后
  才恢复，不应再把早盘指令当作准时信号发送。
- `scripts/run_daily_job.py` 最多运行 3 次，失败后等待 10 分钟再试。
- 三次全部失败后，只发送一次独立的钉钉 `@所有人` 故障告警，并让 systemd
  保持失败状态以便排查。
- `/usr/bin/flock` 防止同一个任务并发写行情和持仓状态。

## 1. 安全组和服务器准备

这项任务不监听网络端口。安全组只需允许你的固定公网 IP 访问 SSH 22 端口，
不需要开放 80、443 或数据库端口。建议使用 SSH 密钥登录。

2 核、1GB 内存、40GB 云硬盘足够当前11个ETF与全历史状态回放。先运行`free -h`；如果Swap为
0，可额外配置 1GB Swap，降低首次安装 Python 依赖时被系统终止的概率。

## 2. 放置代码

将仓库克隆或上传到固定目录：

```bash
sudo mkdir -p /opt/QuantFramework
sudo chown "$(id -un)":"$(id -gn)" /opt/QuantFramework
git clone <你的仓库地址> /opt/QuantFramework
cd /opt/QuantFramework
```

不要把 `.env` 提交到 Git。代码更新也不要放进每日任务自动执行；测试完成后再手动
发布新版本。

## 3. 迁移实盘状态

从当前电脑复制行情；旧策略持仓文件只作为人工核对参考，不得重命名为新策略状态：

```text
data/db/
state/momentum_defender_w40_qm40_signed_exit_v4_position.json（仅核对，不复制为新ID）
```

行情目录可以重新同步。新策略使用独立ID，首次执行前必须先运行`--dry-run`，以券商真实持仓
和旧策略状态人工核对调仓差异；不得静默继承或改名旧JSON。确认后首次正式运行会创建新文件：

```text
/opt/QuantFramework/data/db/
/opt/QuantFramework/state/momentum_defender_w40_qm40_threshold_v5_position.json
```

## 4. 安装

第一次执行会创建 `/etc/quantframework/quant.env` 并要求填写密钥：

```bash
cd /opt/QuantFramework
sudo bash ops/tencent-cloud/install.sh
sudoedit /etc/quantframework/quant.env
sudo bash ops/tencent-cloud/install.sh
```

安装器会完成以下操作：

1. 创建无登录权限的 `quant` 系统账户；
2. 安装固定版本的 `uv`；
3. 使用 `uv.lock` 创建 Python 3.12 生产环境；
4. 设置服务器时区为 `Asia/Shanghai`；
5. 安装并启用 `quant-daily.service` 与 `quant-daily.timer`。

## 5. 首次验证

选定一个可以接受钉钉测试消息的时间，手动运行一次：

```bash
sudo systemctl start quant-daily.service
sudo systemctl status quant-daily.service --no-pager
sudo journalctl -u quant-daily.service -n 200 --no-pager
```

查看下一次计划执行时间：

```bash
systemctl list-timers quant-daily.timer --no-pager
```

## 运维命令

```bash
# 暂停/恢复每日任务
sudo systemctl disable --now quant-daily.timer
sudo systemctl enable --now quant-daily.timer

# 跟踪当天日志
sudo journalctl -u quant-daily.service -f

# 查看本次启动以来的全部任务日志
sudo journalctl -u quant-daily.service -b --no-pager
```

腾讯云侧建议开启自动续费、磁盘使用率/实例不可达告警，并给系统盘配置自动快照。
其中`state/momentum_defender_w40_qm40_threshold_v5_position.json`和对应forward
ledger是最需要备份的数据。新策略ID不继承旧正式持仓文件；首次正式执行前必须人工核对新目标并完成
状态迁移。
