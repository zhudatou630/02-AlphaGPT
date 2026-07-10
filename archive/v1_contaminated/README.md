# V1 冻结说明

V1 指 Phase1 至 Phase3c 的旧研究链。其核心训练框架仍有复用价值，但数据与研究产物受以下问题污染：

- TDX 前复权未处理 ETF `category=11/12` 份额拆分/合并事件。
- TDX 前复权价格舍入到两位小数。
- 公式词表包含原始 `volume/amount`。
- 少量 TDX 历史 volume 与成交额、价格不一致。
- checkpoint 和公式 artifact 没有记录 dataset identity。

因此旧 reward、top formulas、validator 收益和 checkpoint 不再作为有效研究结论，也不得在 V2 中 resume。

旧文件暂不物理移动，避免破坏历史路径。`manifest.json` 固定记录 Git 提交 `2a9ac25e971494a17f6e45a553e2cb040c743b76` 中旧代码与已跟踪产物的位置、大小和 SHA-256；本地未跟踪的诊断输出不进入冻结清单。

V2 使用：

```text
data/processed/v2/dataset/
data/processed/v2/training/
price-event-v2
```
