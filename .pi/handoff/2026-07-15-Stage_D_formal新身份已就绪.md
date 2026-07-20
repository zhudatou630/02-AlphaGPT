# Stage D formal新身份已就绪

时间：2026-07-15

## 状态

- 阶段5工程验收已通过。
- 新formal protocol、最终代码、ResearchSpec、train-view和binding已重建并闭合。
- 六个formal run身份已核对。
- `launch_approved=false`，`training_started=false`。
- 没有连接远端，没有创建formal run目录，没有启动CUDA训练。
- 当前分支`multi-cpu`，尚未合并`main`。

## 正式身份链

```text
protocol_id       02cca48d1c8536f90a23a0e0361cb45e64a30e95623afc05ef58bd57a0dca5c1
code_commit       0df9404331b43bc8775d7ff17dab1d3172ecf186
code_fingerprint  f291520bf15d4d7f875e608e381386ac8c982ecc9b6e72a0826b345dc3d4f0ed
research_spec_id  1ab171f45467c6e8273a925a771988583c3ccecd640d6e240248dca436d7db07
train_view_id     v3a-train-view-8e48ba4ce6c2af91
binding_id        f545b622f5d8887419325cc0551483b6635e35998830e4df50c7cdb0d576da12
launch_manifest   be28ed4b0a992599c4a3135deb30483dedbabc12ed9beeee3d48b5eec9370cce
```

旧protocol/binding以及stage5 engineering binding均已失效，不能用于formal。

## 冻结运行配置

```text
methods                     transformer / matched_random
seeds                       101 / 102 / 103
attempts                    8,000,000 per run
batch_size                  8192
CPU workers                 8
CPU/GPU overlap             true
scorer batch chunk          4096
release CUDA cache/batch    true
VM memory fractions         0.25 / 0.25 / 0.50
fast checkpoint             1800 seconds
candidate snapshot          7200 seconds
graceful stop snapshot      true
```

formal CLI从protocol读取以上配置，拒绝命令行改变worker、关闭overlap或更换run ID。

## 六个run ID

```text
v3a-stage-d-formal-transformer-s101-02cca48d1c85
v3a-stage-d-formal-transformer-s102-02cca48d1c85
v3a-stage-d-formal-transformer-s103-02cca48d1c85
v3a-stage-d-formal-matched_random-s101-02cca48d1c85
v3a-stage-d-formal-matched_random-s102-02cca48d1c85
v3a-stage-d-formal-matched_random-s103-02cca48d1c85
```

## 身份文件

```text
configs/v3a_stage_d_formal_topn.json
data/processed/v3a/stage_d/formal_topn_multicpu_train_view/
data/processed/v3a/stage_d/formal_topn_multicpu_binding.json
data/processed/v3a/stage_d/formal_topn_multicpu_launch_manifest.json
```

文件SHA：

```text
protocol             97bc89d6f4f92cd6802a8f01619e39be57a3b4881bb7f933e53fd8f42d2c59e9
binding              135a25e5226ec7d596c1d8c155970f11cc75b92423d5e5aec3a1b91fc553bc3f
train-view manifest  b839dbcdf3275fdc158ee170cc31e57ba1ddd2cec0b13f7f8f8b5443f7df923c
launch manifest      8d1112a190bd6fd7da1fb47669cfaf3a5f48aede58a072400d4dc3eaf208a692
```

## 可部署包

```text
.pi/profile/results/v3a-formal-identity-20260715/
  alpha-gpt-formal-multicpu-0df9404.bundle
  formal-multicpu-identity-inputs.tgz
  SHA256SUMS
```

git bundle完整性、输入archive目录和SHA均已验证。

## 验证

- `tests.test_v3a_stage_d`与`tests.test_v3a_stage_d_components`共16项通过。
- protocol ID自哈希一致。
- binding第二次独立构建逐字节一致。
- launch manifest自哈希一致，6个method/seed组合完整且run ID唯一。
- 当前tracked身份代码和protocol相对HEAD干净。
- `data/processed/v3a/training/formal_multicpu_runs`不存在。

## 下一停点

下一步只能在用户明确批准“启动formal训练”后执行：选择GPU机器、部署以上bundle和identity inputs、远端复算binding、创建第一个run并启动监工。

不得续跑旧seed 101；六个run全部从头开始。