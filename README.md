# 02-AlphaGPT

English summary: **personal research prototype** that adapts AlphaGPT-style formula discovery to China A-share multi-ETF ranking. It is **not** a trading bot, broker connector, or production strategy system. Inspired by [AlphaGPT](https://github.com/imbue-bit/AlphaGPT); this repo is an independent reimplementation/research line maintained by [zhudatou630](https://github.com/zhudatou630).

---

把 [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) 的思路迁到 **A 股多只 ETF 的横截面选股/轮动研究**：用固定公式语言生成因子公式，在训练期打分，沉淀可继续研究的候选公式库。

> **这是个人研究探索原型，不是生产系统，也不自动下单。**

## 它做什么 / 不做什么

**做：**

- 把多只 ETF 日频行情整理成统一研究面板
- 用固定词表/语法表达价格类因子公式
- 用 VM 执行公式，得到每只 ETF 的分数
- 按固定 scorer 评估“训练期相对选择质量”
- 支持随机搜索与 Transformer + REINFORCE 式搜索（V3A Stage D）
- 保存 top-N 公式库和实验协议，方便复现与后续研究

**不做：**

- 不连接券商、不执行交易
- 不提供投资建议
- 不声称已找到可实盘赚钱的“圣杯公式”
- 当前主线也尚未把完整样本外验证 / 纸面交易 / 实盘当作必经交付

## 和 AlphaGPT 的关系

| | AlphaGPT | 本仓库 |
|---|---|---|
| 定位 | 原作者开源项目（Meme 币等场景） | 独立研究仓，面向 A 股多 ETF |
| 关系 | 思想与实现参考 | **不是 fork 冒充上游**；本地参考代码不随仓库发布 |
| 许可证 | Apache-2.0 | 同样采用 Apache-2.0，并在 `NOTICE` 中致谢 |

请把本项目理解为：**AlphaGPT-inspired multi-ETF research toolkit**，维护者是本仓库作者，不是 AlphaGPT 原项目维护者。

## 当前研究主线（V3A）

```text
固定 ETF 池与数据
  -> 固定公式语言与 scorer
  -> 随机 / Transformer 生成公式
  -> 保存账本与 top-N 公式库
  -> 以后再接交易思路、样本外验证与组合
```

更细的设计与状态见：

- [`docs/V3A_整体设计与当前状态.md`](docs/V3A_整体设计与当前状态.md)
- [`docs/多ETF方案路线.md`](docs/多ETF方案路线.md)
- [`docs/V3A_单公式相对选择技术规格.md`](docs/V3A_单公式相对选择技术规格.md)

## 仓库结构

```text
src/alpha_etf/          # 核心库：数据、词表/VM、评分、V3A 训练组件
scripts/                # 各阶段可运行入口（数据构建、训练、导出）
configs/                # 冻结的研究协议 / 配置
tests/                  # 单元与契约测试
docs/                   # 研究设计与阶段说明（中文）
data/                   # 部分已处理研究数据与历史 run 产物
tools/tdx_exporter/     # 通达信导出辅助工具
```

## 快速开始

### 1. 环境

建议 Python 3.11+，GPU 训练需要 CUDA 版 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 跑测试（本机可做）

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

有研究含义的完整 GPU 训练，请在 GPU 机器上按 `docs/` 与 `scripts/v3a/` 中的协议运行；本仓库默认把“正式研究训练”和“本机工程验证”分开。

### 3. 常见入口（需已准备对应数据/协议）

```bash
# V3A 数据与训练相关脚本示例
PYTHONPATH=src python scripts/v3a/build_dataset.py --help
PYTHONPATH=src python scripts/v3a/fixed_formula_sanity.py --help
PYTHONPATH=src python scripts/v3a/run_stage_d.py --help
```

具体参数、数据路径和实验身份以对应 `docs/` 与 `configs/` 为准。历史 Phase1–3c、V2 脚本仍保留，主要用于对照和审计，不是当前默认主路径。

## 状态说明（避免误读结果）

- 已完成：数据/公式 VM/scorer 链路、GPU 工程验证、Stage D pilot、部分 formal 工程验收与结果归档
- 进行中/未完成：更完整的样本外验证、交易层摩擦建模、纸面交易与实盘
- 仓库中的实验数字是 **研究过程记录**，不是策略业绩承诺

## 数据与合规提醒

- 行情数据可能来自本地通达信导出、Tushare/AkShare 等来源；请自行遵守各数据源条款
- 仓库内已提交的部分 `data/processed` 产物仅供复现研究流程，不保证最新、也不保证可直接用于交易
- 使用本项目产生的任何研究结果，风险自负

## 维护

- 主要维护者：[@zhudatou630](https://github.com/zhudatou630)
- 问题与讨论：请开 GitHub Issue
- 当前阶段以单人研究推进为主，欢迎复现反馈，但接口与协议仍可能变化

## License

Apache License 2.0. 详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
