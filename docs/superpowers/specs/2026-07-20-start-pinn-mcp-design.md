# Start CodexPro MCP Skill 设计

日期：2026-07-20

## 目标

创建个人 Skill `start-codexpro-mcp`，用于在 macOS 上通过 Ghostty 为任意本地文件夹启动并验证 CodexPro MCP。

默认网络配置保持固定：

- 本地 MCP：`http://127.0.0.1:8787/mcp`
- 公网 MCP：`https://mcp.lihe.show/mcp`
- Cloudflare Tunnel：`chatgpt-mcp`
- Cloudflare 配置：`/Users/lihe/.cloudflared/chatgpt-mcp.yml`

同一时间只服务一个工作区。每次启动必须解析出一个明确的绝对根目录；Skill 不再硬编码 PINN 项目路径。

## 使用示例

以下请求都应触发 Skill：

- “为 `/Users/lihe/project-a` 启动 CodexPro MCP”
- “把 MCP 切换到这个文件夹：`/absolute/path`”
- “恢复 mcp.lihe.show，工作区是 `/absolute/path`”
- “检查当前 CodexPro MCP 是否在线”

根目录解析规则：

1. 优先使用用户明确给出的绝对路径。
2. 用户明确说“当前工作区”时，可以使用当前任务的工作区根目录，但必须先向用户报告解析后的绝对路径。
3. 无法确定路径时，只问一个问题要求用户提供绝对路径。
4. 不根据目录名、最近记录或当前 shell 目录猜测目标路径。

## 非目标

- 不支持多个工作区同时在线。
- 不自动终止任何占用端口的进程。
- 不自动把现有服务切换到另一个工作区。
- 不创建 LaunchAgent；该方式可能无法获得 iCloud Drive 等受保护目录的访问权限。
- 不改变 ChatGPT 连接器名称、Server URL 或认证方式。
- 不自动关闭 Ghostty 窗口。
- 不提供强制重启功能。
- 不持久化“最近使用路径”或维护命名配置档案。

## 文件结构

```text
~/.codex/skills/start-codexpro-mcp/
├── SKILL.md
├── agents/
│   └── openai.yaml
└── scripts/
    ├── start.sh
    └── verify.mjs
```

### SKILL.md

定义触发条件、路径解析和执行流程。Skill 调用 `scripts/start.sh --root <absolute-path>`，读取退出码和结构化状态，并向用户报告本地服务、隧道和文件访问结果。

### scripts/start.sh

负责确定性启动和状态检查。

启动模式必需参数：

- `--root <absolute-path>`：本次服务的工作区根目录。

可选参数：

- `--port <port>`：默认 `8787`。主要用于测试，不改变固定域名配置。
- `--status`：只检查，不启动。此模式下 `--root` 可省略；省略时报告当前工作区，提供时比较当前根目录与目标根目录。
- `--dry-run`：显示将执行的动作，不打开 Ghostty。
- `--public-url <url>`：默认 `https://mcp.lihe.show/mcp`。主要用于测试和未来扩展。

执行流程：

1. 将根目录规范化为真实绝对路径，并使用元数据检查确认目录存在；不得在 Codex 受限环境中把直接列目录失败误判为 Ghostty 无权限。
2. 验证 Ghostty、CodexPro、cloudflared、隧道配置和凭据文件存在。Ghostty 的实际目录访问权限在服务启动后由 `verify.mjs` 的 `tree` 调用确认。
3. 检查目标端口。
4. 若端口空闲，通过 Ghostty 启动 CodexPro，并传入本次 `--root`。
5. 若端口由 CodexPro 占用，读取其命令参数和 MCP `server_config`，确认实际根目录。
6. 若实际根目录与目标根目录一致，复用现有服务。
7. 若端口由其他进程或不同工作区占用，立即退出并报告 PID、命令、当前根目录和目标根目录；不得终止进程。
8. 检查固定 Cloudflare Tunnel 进程。
9. 若隧道未运行，通过第二个 Ghostty 窗口启动；若已运行则复用。
10. 使用条件轮询等待服务就绪。
11. 调用 `verify.mjs` 完成协议级验证。

### scripts/verify.mjs

接收以下参数：

- `--url <mcp-url>`
- `--expected-root <absolute-path>`：可选；提供时校验根目录，省略时只报告服务当前根目录。

通过公网 URL 执行：

1. `initialize`
2. `notifications/initialized`
3. `open_current_workspace`
4. `tree`

验证条件：

- Server 名称为 `CodexPro`。
- 提供 `--expected-root` 时，工作区根目录必须与其规范化路径一致；未提供时输出实际根目录。
- `tree` 返回成功且不包含 `EPERM`。
- `tree` 返回可解析的文本结果；空目录只返回 `.` 也视为成功。
- HTTP、协议、超时或工具错误均返回非零退出码。

## Ghostty 启动方式

macOS 使用 Launch Services 创建 Ghostty 实例：

```bash
open -na Ghostty.app --args -e /bin/zsh -lc '<command>'
```

CodexPro 和 Cloudflare Tunnel 分别运行在独立 Ghostty 窗口。命令末尾使用 `exec`，使主进程和退出状态清晰。

必须由 Ghostty 启动 CodexPro。直接从 ChatGPT/Codex 内部启动会继承受限系统策略，访问 iCloud Drive 等目录时可能出现：

```text
EPERM: operation not permitted, scandir
```

## 单工作区切换规则

`mcp.lihe.show` 固定映射到本机 `8787`，因此一次只能连接一个工作区。

当用户请求不同目录而旧服务仍在运行时：

1. 报告旧工作区与新工作区。
2. 停止操作。
3. 指示用户关闭旧 CodexPro Ghostty 窗口。
4. 用户确认旧服务已停止后，再重新调用 Skill。

Skill 不得自动发送 `q`、`kill`、`pkill` 或 `killall`。

## 状态与错误处理

脚本输出稳定状态：

- `ROOT=<resolved-path>`
- `LOCAL=started|reused|conflict|failed`
- `TUNNEL=started|reused|failed|skipped`
- `VERIFY=passed|failed|skipped`
- `PUBLIC_URL=https://mcp.lihe.show/mcp`

失败处理：

- 路径缺失或不是绝对路径：停止并报告正确用法。
- 端口冲突：报告占用进程和根目录，不做自动恢复。
- Ghostty 无目录权限：报告 `EPERM` 和相关应用权限。
- 隧道失败：保留本地 MCP，不终止 CodexPro。
- 公网验证失败：区分本地服务、DNS、Cloudflare 和 MCP 协议错误。

## 安全边界

- 不使用模糊进程终止命令。
- 不读取或打印 Cloudflare 凭据内容。
- 不在命令行或日志中输出 token。
- 不覆盖 Ghostty 或 Cloudflare 配置。
- 不修改目标工作区文件。
- 当前公网端点保持 Authentication: None；Bearer Token 加固作为独立后续变更。

## 验证计划

1. 基线场景：记录没有 Skill 时代理直接从 Codex 启动、猜测路径、重复启动或仅检查 HTTP 的行为。
2. 静态验证：运行 Skill 验证器、`bash -n scripts/start.sh` 和 Node 语法检查。
3. 路径验证：覆盖普通路径、包含空格的路径、空目录、iCloud Drive 路径、相对路径和不存在路径。
4. 幂等验证：对同一根目录执行两次，确认复用且不增加进程或窗口。
5. 冲突验证：模拟不同工作区占用目标端口，确认脚本退出且不终止进程。
6. 干运行验证：服务缺失时使用 `--dry-run`，确认 Ghostty 命令包含正确转义后的根目录。
7. 端到端验证：通过 Ghostty 冷启动后，从公网执行 `open_current_workspace` 和 `tree`。
8. 泛化验证：至少使用 PINN 项目和一个临时测试目录，确认根目录参数生效。
9. 触发验证：使用“启动 MCP”“为这个文件夹启动 CodexPro”等自然语言确认 Skill 可发现且不会猜路径。

## 后续优化

本版优先实现同域名单工作区切换。后续可独立评估：

- 增加显式、需要用户确认的停止命令。
- 增加命名配置档案。
- 为不同工作区配置独立端口和子域名。
- 为公网 MCP 增加 Bearer Token 或 Cloudflare Access。
