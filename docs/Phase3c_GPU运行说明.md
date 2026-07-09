# Phase3c GPU 运行说明

Phase3c 是 GPU-first 训练原型：保留 Phase3b 的 token 公式、Transformer policy、REINFORCE 和 CPU validator 复核逻辑，但把训练阶段的 VM/scorer 换成 torch batch 实现。

## 当前约定

默认配置：

```text
batch_size=1024
max_len=16
train_steps=100
d_model=64
num_layers=2
num_heads=4
ff_dim=128
dtype=float32
```

4090D preset：

```text
--preset 4090d-smoke  # batch_size=1024, train_steps=50
--preset 4090d        # batch_size=2048, train_steps=100
--preset 4090d-rms-swiglu-smoke  # batch_size=4096, train_steps=2, max_len=12, RMSNorm+SwiGLU
--preset 4090d-rms-swiglu        # batch_size=4096, train_steps=1000, max_len=12, RMSNorm+SwiGLU
```

`4090d-rms-swiglu` 是正式 GPU 训练规格。由于 `batch_size=2048,max_len=16` 已经在 4090D 上 reserved 约 22.5GB，`batch_size=4096` 有 OOM 风险；实际长训前应先跑 `4090d-rms-swiglu-smoke`。

## 重要提醒

AutoDL 服务器当前如果是关机状态，实际运行前需要先在 AutoDL 控制台开机。

登录信息只保存在本地忽略文件：

```text
.pi/local/autodl-login.md
```

不要提交、复制或外传里面的口令。

## 本地验证

```bash
.venv/bin/python -m compileall scripts src tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/phase3c_train_gpu.py \
  --device cpu \
  --batch-size 8 \
  --train-steps 1 \
  --max-len 8 \
  --min-formula-len 1 \
  --top-n 2 \
  --cpu-audit-candidates 4 \
  --skip-validator \
  --run-id local_phase3c_smoke
```

## 服务器依赖

AutoDL 镜像已确认 CUDA/PyTorch 可用，但可能缺：

```text
pandas
pyarrow
pytest
```

可先安装：

```bash
/root/miniconda3/bin/pip install pandas pyarrow pytest
```

检查：

```bash
/root/miniconda3/bin/python - <<'PY'
import torch, numpy, pandas, pyarrow
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(numpy.__version__, pandas.__version__, pyarrow.__version__)
PY
```

## 同步代码和数据

建议服务器目录：

```text
/root/autodl-tmp/02-AlphaGPT
```

同步时不要上传 `.venv/`、`.pi/local/`、`.pi-subagents/`：

```bash
rsync -av \
  --exclude .venv \
  --exclude .git \
  --exclude .pi/local \
  --exclude .pi-subagents \
  -e 'ssh -p <PORT>' \
  ./ root@<HOST>:/root/autodl-tmp/02-AlphaGPT/
```

至少需要同步：

```text
data/processed/etf_panel_qfq_latest.npz
data/processed/etf_panel_raw.npz
```

## CUDA smoke

第一条先确认框架跑通：

```bash
cd /root/autodl-tmp/02-AlphaGPT
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --preset 4090d-smoke \
  --skip-validator
```

第二条确认 4090D preset：

```bash
cd /root/autodl-tmp/02-AlphaGPT
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --preset 4090d \
  --skip-validator
```

RMSNorm + SwiGLU 正式规格先跑显存 smoke：

```bash
cd /root/autodl-tmp/02-AlphaGPT
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --preset 4090d-rms-swiglu-smoke \
  --skip-validator
```

确认不 OOM 后再跑 1000 steps：

```bash
cd /root/autodl-tmp/02-AlphaGPT
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --preset 4090d-rms-swiglu \
  --checkpoint-every-steps 10 \
  --skip-validator
```

长训会周期性写出：

```text
checkpoint_latest.pt
training_log.csv
gpu_candidate_formulas.csv
```

如果训练中断，用下面的方式恢复：

```bash
cd /root/autodl-tmp/02-AlphaGPT
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --resume-from data/processed/phase3c/runs/<run_id>/checkpoint_latest.pt \
  --skip-validator
```

默认恢复到 checkpoint 里记录的总步数。若需要把同一个 run 延长到更多步，例如 1500 steps：

```bash
/root/miniconda3/bin/python scripts/phase3c_train_gpu.py \
  --device cuda \
  --resume-from data/processed/phase3c/runs/<run_id>/checkpoint_latest.pt \
  --train-steps 1500 \
  --skip-validator
```

产物目录：

```text
data/processed/phase3c/runs/<run_id>/
```

重点看：

```text
training_log.csv
run_summary.json
gpu_candidate_formulas.csv
best_formulas.csv
best_formulas.jsonl
```

`training_log.csv` 会记录：

```text
sample_seconds
vm_seconds
score_seconds
backward_seconds
step_seconds
formulas_per_second
cuda_memory_allocated_mb
cuda_memory_reserved_mb
```

第一轮服务器 smoke 建议先 `--skip-validator`，避免 CPU validator 抢时间。训练结束后的 top formulas 已经会用原 CPU scorer 复核，validator 可以后续单独跑。
