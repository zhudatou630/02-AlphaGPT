# V3A Stage D 简单监工

当前监工只观察一次 GPU pilot，不代替本地 Agent 做决策。

## 启动 pilot

服务器需要先由用户手动开机。确认服务器可 SSH 访问后，由本地 Agent 执行：

```bash
.pi/supervisor/launch-pilot.sh
```

这个脚本只负责上传固定 bundle/input、部署远端环境并启动 screen。它不会被 timer 自动调用。

## 观察和通知

```bash
.pi/supervisor/probe.sh | jq
.pi/supervisor/tick.sh
.pi/supervisor/supervisor.py create-session
```

启用观察 timer：

```bash
.pi/supervisor/install-timer.sh
systemctl --user status alphagpt-supervisor-3227c7e8e4.timer
```

停止观察：

```bash
systemctl --user disable --now alphagpt-supervisor-3227c7e8e4.timer
```

## 本地 Agent 的手动动作

```bash
# 只同步，不验证、不晋升、不关机
.pi/supervisor/sync-pilot.sh /tmp/v3a-stage-d-pilot-sync

# 只停止远端 screen，不关机
.pi/supervisor/stop-pilot.sh
```

任务发生异常时，timer 只把事件送回 `监工｜AlphaGPT`。后续是否重启、修改代码、同步、
停止或向用户升级，由本地 Agent 根据证据判断。

## 当前冻结身份

- commit：`24de39e5d7a86c7a225cb0c2b46e68af5baea734`
- binding：`ea96bc56c6bbcba852efbc4b944991efe171c3afbe8519f07818313e16735054`
- run ID：`v3a-stage-d-pilot-transformer-s314159-078af5f6466b`
- 配置：`.pi/supervisor/runtime.json`、`.pi/supervisor/run-spec.json`

凭据只从被忽略的 `.pi/local/autodl-login.md` 读取，不写入事件或日志。