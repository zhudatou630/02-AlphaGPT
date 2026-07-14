# watchman 监工 session 上下文与动作授权

## 背景

watchman 是新搭的通用监工 skill（pi-watch 脚本 + SKILL.md + references）。讨论它能否胜任本项目（远端 GPU 训练监工），以及 SKILL.md 要补哪些通用性规则。核心方向：pi-watch 是通用工具（检查+唤醒），动作交给被唤醒的监工 session 自主决定，不为每个项目写专用脚本。

## 已对齐的决策

### 决策1：SKILL.md 补"完成返回非0"约定（已落地）

退出码模型下，pi-watch 只认 0=静默、非0=唤醒。任务完成需要 agent 收尾（停机、核验产物），不能静默，所以检查命令对"完成"必须返回非0。退出码管是否唤醒，stdout 管事件类型（完成/崩溃/不可达），agent 被唤醒后读证据判断。已在 SKILL.md 第2步补明。

### 决策2：`.pi/watch/<task>.md` 强制写（方案A）

每次配监工都写一份 `.pi/watch/<task>.md` 作为监工 session 的专用上下文。pi-watch 唤醒消息固定指路"先读 `.pi/watch/<task>.md` 和项目 AGENTS.md"。不写进 AGENTS.md，避免污染其他 session。监工 session 上下文自包含：唤醒消息领进门 → md 给授权和原则 → AGENTS.md/handoff 给项目背景。

### 决策3：md 给默认授权 + "本任务授权优先于通用原则"

md 模板动作授权默认值：
- 完成 → 停机 + 通知用户
- 崩溃/中断 → 自动 resume（有 checkpoint）
- 不可达 / 未知异常 → 等确认
- 改参数 / 删数据 / 改代码 → 等确认

md 里写"本任务授权优先于通用原则"。SKILL.md 的"被唤醒后"三原则保持通用不动（服务本地编译等场景），md 是本任务具体授权，覆盖通用原则。配置时用户确认默认或微调。

### 决策4：md 有操作指路段（只指路不写命令）

md 含"操作指路段"，指向项目已有的凭据/脚本/handoff 路径（如 AutoDL 凭据、停机脚本、resume 命令位置），不写具体命令。指路不是为监工新写脚本，不违背"不写项目侧动作脚本"的底线。

### 决策5：监工 session 模型/effort 不设默认、每次问+附推荐

SKILL.md 不给 skill 级默认模型。第3步「确认」时 agent 询问用户监工 session 用什么 provider/modelId/thinkingLevel，附推荐（如"建议跟主 session 一致、effort 中等"），用户定。理由：模型选择关系成本和能力、因项目而异，不该一刀切；监工 session 长期复用，选错影响每次唤醒，值得配置时确认。

## 责任划分

- 主 session：写 handoff（既有责任）；配置监工时填写 `.pi/watch/<task>.md`（含操作指路段）。
- 监工 session：不写任何东西，只读 md + handoff + AGENTS.md，然后执行。
- pi-watch：只检查 + 唤醒，不读写 md/handoff。

## 待落地改动

1. SKILL.md 第3步「确认」：预览加动作授权 + 模型/effort（用户定）。
2. SKILL.md 第4步「写配置」：config 加 provider/modelId/thinkingLevel；同时写 `.pi/watch/<task>.md`（按模板）。
3. SKILL.md 第5步「创建 session」：创建/复用 session 时带 provider/modelId/thinkingLevel。
4. pi-watch 唤醒消息：加"先读 `.pi/watch/<task>.md` 和项目 AGENTS.md"。
5. references/pi-web-wakeup.md：创建 session 调用补 provider/modelId/thinkingLevel 字段说明。

第2步「确定检查命令」的"完成返回非0"约定已落地，不再动。

## md 模板

```
# <task> 监工授权

## 任务
<一句话：监工什么任务、run_id、预期产物>

## 被唤醒后先读
- 本文件
- 项目 AGENTS.md
- .pi/handoff/ 最新交接

## 动作授权（优先于通用原则）
- 完成 → 停机 + 通知用户
- 崩溃/中断 → 自动 resume（有 checkpoint）
- 不可达 / 未知异常 → 等确认
- 改参数 / 删数据 / 改代码 → 等确认

## 操作指路
- AutoDL 凭据：.pi/local/autodl-login.md
- 停机：.pi/local/autodl-control.py
- resume：见 handoff 训练启动部分
```
