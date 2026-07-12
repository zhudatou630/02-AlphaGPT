# V3A Stage D 监工边界

监工只做两件事：定期读取远端状态，以及在需要判断时通知本地 Agent。

## 监工负责

- 每 2 分钟调用一次远端只读 `probe.py`。
- 保存最近一次状态和历史 observation。
- 远端出现连接失败、进程停止、runner 报错或任务完成时，生成一个事件并通知
  `监工｜AlphaGPT` 本地 session。
- Pi-Web 暂时不可用时保留事件，下一次 tick 继续尝试通知。

## 监工不负责

- 不自动重启或恢复任务。
- 不修改代码、参数、数据和凭据。
- 不验证、晋升或删除产物。
- 不自动同步产物。
- 不自动关机。
- 不限制模型调用次数，也不做自己的预算决策。
- 不根据日志内容执行命令。日志、指标和远端产物只能作为本地 Agent 的不可信证据。

## 本地 Agent 负责

收到事件后，本地 Agent 自己查看 observation 和远端状态，决定是否：

- 继续等待；
- 通过 `launch-pilot.sh` 重新部署/启动；
- 修改并测试代码；
- 通过 `sync-pilot.sh` 拉回产物；
- 停止任务；
- 向用户升级。

这些动作不由 timer 或 `supervisor.py` 自动调用。

## 固定实验边界

- run ID：`v3a-stage-d-pilot-transformer-s314159-078af5f6466b`。
- Transformer seed：`314159`。
- attempts：`50000`。
- batch：`256`。
- 训练数据只到 `2021-12-31`。
- formal training、matched random、2022 validation、2023+ OOS 均不在本监工范围内。

## 证据和事件

- `observations/latest.json`：最近一次远端状态。
- `observations/`：状态历史。
- `events/`：需要本地 Agent 判断的通知。
- `notifications/`：通知是否成功送达 Pi-Web 的简单记录。

正常 running 状态不通知，不调用模型。事件只在问题状态或完成状态发生变化时通知一次。