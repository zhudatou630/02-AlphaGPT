# Stage D formal代码与worktree收口

时间：2026-07-15

## Git状态

- Formal历史commit：`0df9404331b43bc8775d7ff17dab1d3172ecf186`。
- Annotated tag：`v3a-stage-d-formal-20260715`，本地和origin均指向上述commit。
- `main`已包含post-formal结果文档、formal handoff、Stage D设计记录和精确ignore规则，并已推送origin。
- Formal代码指纹仍为`f291520bf15d4d7f875e608e381386ac8c982ecc9b6e72a0826b345dc3d4f0ed`；历史binding仍只认`0df9404`。

## Worktree职责

```text
/home/zhujunshen/Quant/02-AlphaGPT
  branch: main
  role: 后续集成、文档和新分支起点

/home/zhujunshen/Quant/02-AlphaGPT-worktrees/multi-cpu
  branch: multi-cpu
  HEAD: 0df9404
  role: 临时冻结的formal复现环境和本地工程证据载体
  lock: Frozen V3A Stage D formal 20260715; contains local identity and engineering evidence
```

`main`与`multi-cpu`的formal代码已经通过fast-forward合并；现在不应再次merge，也不应继续在`multi-cpu`开发。

## Artifact收口

以下权威archive已从`multi-cpu`复制到`main/.pi/profile/results/`，源文件保留，SHA复核通过：

```text
v3a-formal-identity-20260715/alpha-gpt-formal-multicpu-0df9404.bundle
  70050a8787bc5da5b20a730d68e58e51009a4e7c5ade394746a8cea97edee06d

v3a-formal-identity-20260715/formal-multicpu-identity-inputs.tgz
  c79f349eea19bafb8ba8a1e38bbb62a50b5fc8af2e5268b4845027afdc77c20f

v3a-stage5-20260715/v3a-stage5-evidence-v4-20260715.tgz
  ace1fed428127f5a6a8e26baa4c4af60b0fdd042d02efa90a6d52ecdb5b6a0da

v3a-formal-runs-20260715/formal-multicpu-closeout-20260715T111422Z.tar.gz
  7409c635f7661a25938da123174d7b7a8bb04cd412c1899f5984b8ee2e8703ef
```

统一校验清单：

```text
.pi/profile/results/v3a-archive-20260715/ARTIFACTS_SHA256SUMS
```

`.pi/profile/results/`、`.pi/watch/state/`和`.pi/watch/incidents/`现已精确ignore；handoff和设计文档仍由Git跟踪。密钥继续只放`.pi/local/`。

## 删除保护

当前不要删除或clean `multi-cpu`：其中仍有46MB formal train-view、binding、launch manifest和171MB本地工程证据。普通`git status`看不到被ignore的数据。

确认正式公式库已冻结、完整证据已有外部备份且不再需要该复现环境后，才能：

```bash
git worktree unlock /home/zhujunshen/Quant/02-AlphaGPT-worktrees/multi-cpu
git worktree remove /home/zhujunshen/Quant/02-AlphaGPT-worktrees/multi-cpu
```

不得使用`git clean -fdx`清理该worktree。

## 下一步边界

- 已从post-formal `main@d25911a`创建分支`stage-d-search-v2`，worktree位于`/home/zhujunshen/Quant/02-AlphaGPT-worktrees/stage-d-search-v2`。
- 新worktree只复制了本地忽略的`AGENTS.md`项目规则；没有复制formal数据、密钥或旧运行产物。
- 当前分支尚未修改训练代码、protocol或研究设计参数。
- 尚未连接905、拉回完整ledger或打开2022 validation。
- 下一步先由用户选择`docs/V3A_StageD_formal结果与下一轮优化设计.md`中的路径A或路径B。
- 后续新搜索机制的讨论和实现只在`stage-d-search-v2`进行；不要修改冻结的`multi-cpu`。