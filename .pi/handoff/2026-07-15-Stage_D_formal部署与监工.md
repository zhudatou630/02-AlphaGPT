# Stage D formal部署与监工

时间：2026-07-15

## 当前状态

- 用户已批准启动formal训练。
- `main`已用fast-forward推进到formal commit，没有产生merge commit。
- 905机已启动，formal代码与身份输入已部署到全新目录。
- 远端16项测试通过，binding独立重建逐字节一致，CUDA最小检查通过。
- 六run队列已于`2026-07-15T09:17:31Z`启动；首个Transformer seed 101正在运行。
- 启动后核验时已推进约87万attempts，近期吞吐约7,962 attempts/s，8个CPU worker齐全，GPU约80GB。
- watchman异常唤醒、timer停启、训练中正常静默检查均已验证。

## 固定身份

```text
protocol_id       02cca48d1c8536f90a23a0e0361cb45e64a30e95623afc05ef58bd57a0dca5c1
code_commit       0df9404331b43bc8775d7ff17dab1d3172ecf186
code_fingerprint  f291520bf15d4d7f875e608e381386ac8c982ecc9b6e72a0826b345dc3d4f0ed
research_spec_id  1ab171f45467c6e8273a925a771988583c3ccecd640d6e240248dca436d7db07
train_view_id     v3a-train-view-8e48ba4ce6c2af91
binding_id        f545b622f5d8887419325cc0551483b6635e35998830e4df50c7cdb0d576da12
```

## 远端

```text
机器          西北B区905机 / RTX PRO 6000 96GB
SSH           ssh autodl-905
代码          /root/02-AlphaGPT-formal-multicpu
run根目录     /root/autodl-tmp/v3a-stage-d-formal-multicpu-20260715
队列脚本      <run根目录>/ops/run_queue.sh
健康检查      <run根目录>/ops/healthcheck.sh
队列日志      <run根目录>/queue.log
状态          <run根目录>/queue_status.json
```

远端checkout为detached HEAD `0df9404`。旧Stage5仓库和旧898训练产物未修改。

## 六run顺序

```text
1. v3a-stage-d-formal-transformer-s101-02cca48d1c85
2. v3a-stage-d-formal-matched_random-s101-02cca48d1c85
3. v3a-stage-d-formal-transformer-s102-02cca48d1c85
4. v3a-stage-d-formal-matched_random-s102-02cca48d1c85
5. v3a-stage-d-formal-transformer-s103-02cca48d1c85
6. v3a-stage-d-formal-matched_random-s103-02cca48d1c85
```

任何时刻只允许一个run。每个run完成并核验后，队列才进入下一个。

## Watchman

```text
配置          .pi/watch/formal-multicpu.json
授权          .pi/watch/formal-multicpu.md
session       019f650b-2205-7c32-b497-5969e3d1a29a
provider      sub2api
model         gpt-5.6-luna（GPT-5.6 Luna）
thinking      high
timer         pi-watch-formal-multicpu.timer
检查间隔      5分钟
```

动作授权以`.pi/watch/formal-multicpu.md`为唯一任务依据：完成后完整核验、拉回小证据包并关机；满足严格条件的进程中断最多自动resume一次；身份、数据、OOM、代码或未知异常不改参数，保存现场并关机通知。

安装验证：

- 正常`PREPARED`检查退出0，pi-watch静默。
- 模拟异常退出42，证据成功保存，指定session被唤醒并回复`WATCHMAN_TEST_ACK`。
- Pi-Web API与session日志确认`sub2api / gpt-5.6-luna / high`。
- `pi-watch-formal-multicpu.timer`已启用并active，linger开启，每5分钟执行。
- 训练启动后手工触发systemd service，健康检查为`RUNNING`、退出0，没有异常指纹。

## 启停命令

启动队列：

```bash
ssh autodl-905 "cd /root/autodl-tmp/v3a-stage-d-formal-multicpu-20260715 && nohup setsid bash ops/run_queue.sh >> queue.log 2>&1 < /dev/null &"
```

只在授权条件满足时恢复：

```bash
ssh autodl-905 "cd /root/autodl-tmp/v3a-stage-d-formal-multicpu-20260715 && nohup setsid bash ops/run_queue.sh --resume-current >> queue.log 2>&1 < /dev/null &"
```

检查：

```bash
ssh autodl-905 'bash /root/autodl-tmp/v3a-stage-d-formal-multicpu-20260715/ops/healthcheck.sh'
```

不得续跑旧seed 101，不得读取2022或2023+，不得把stage5产物混入formal。