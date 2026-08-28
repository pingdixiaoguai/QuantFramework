# 腾讯云 09:00 早盘信号部署

适用于一台长期运行的 Linux CVM。任务在周一至周五 09:00（`Asia/Shanghai`）
启动，以最新完整交易日行情计算 `quality_momentum_top1`，在钉钉提示当天目标持仓。
代码还会查询上交所交易日历，法定休市日成功跳过且不发送信号。

## 运行语义

- `quant-daily.timer` 只负责 09:00 调度，不启用错过后的延迟补跑。服务器若在开盘后
  才恢复，不应再把早盘指令当作准时信号发送。
- `scripts/run_daily_job.py` 最多运行 3 次，失败后等待 10 分钟再试。
- 生产任务只传入 `quality_momentum_top1.yaml`，通知不加载或展示影子策略。
- 三次全部失败后，只发送一次独立的钉钉 `@所有人` 故障告警，并让 systemd
  保持失败状态以便排查。
- `/usr/bin/flock` 防止同一个任务并发写行情和持仓状态。

## 1. 安全组和服务器准备

这项任务不监听网络端口。安全组只需允许你的固定公网 IP 访问 SSH 22 端口，
不需要开放 80、443 或数据库端口。建议使用 SSH 密钥登录。

2 核、1GB 内存、40GB 云硬盘足够当前四个 ETF。先运行 `free -h`；如果 Swap 为
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

从当前电脑复制以下内容：

```text
data/db/
state/quality_momentum_top1_position.json
```

行情目录可以重新同步，但持仓 JSON 必须迁移，否则服务器会把策略视为首次建仓，
并丢失现有 YTD 实盘账本。复制完成后确认文件位于：

```text
/opt/QuantFramework/data/db/
/opt/QuantFramework/state/quality_momentum_top1_position.json
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

## 更新已有部署

服务器拉取本次改动后，重新安装服务文件并让 systemd 读取新配置：

```bash
cd /opt/QuantFramework
git pull
sudo install -m 0644 ops/tencent-cloud/quant-daily.service /etc/systemd/system/quant-daily.service
sudo systemctl daemon-reload
sudo systemctl cat quant-daily.service
```

确认输出的 `ExecStart` 中只有
`--config strategy/configs/quality_momentum_top1.yaml`，没有 `--shadow-config`。
定时器不需要重启，下一次运行即会使用新配置；如需立即验证，可手动启动一次服务，
但会发送一条真实的钉钉通知。

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
其中 `state/quality_momentum_top1_position.json` 是最需要备份的数据。
