# 02-AlphaGPT

把 [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) 的思路迁到 A 股多 ETF 研究：用固定公式语言搜索因子公式，在训练期打分，沉淀候选公式库。

这是个人研究原型，不是交易系统。

思想受 AlphaGPT 启发，代码为独立实现；上游为 Apache-2.0，本仓库同样采用 Apache-2.0，致谢见 [`NOTICE`](NOTICE)。

## 流水线

```text
ETF 数据面板
  -> 公式语言 / VM 执行
  -> 训练期 scorer 打分
  -> 随机搜索或 Transformer 搜索
  -> top-N 公式库
```

当前主线是 V3A。设计与状态见：

- [`docs/V3A_整体设计与当前状态.md`](docs/V3A_整体设计与当前状态.md)
- [`docs/多ETF方案路线.md`](docs/多ETF方案路线.md)
- [`docs/V3A_单公式相对选择技术规格.md`](docs/V3A_单公式相对选择技术规格.md)

## 结构

```text
src/alpha_etf/   核心库
scripts/         数据构建、训练、导出入口
configs/         实验配置
tests/           测试
docs/            设计文档
data/            部分研究数据与历史结果
tools/           辅助工具
```

## 使用

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m unittest discover -s tests
```

GPU 训练与完整实验流程见 `docs/` 和 `scripts/v3a/`。

## License

Apache License 2.0. 见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
