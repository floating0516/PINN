# Start CodexPro MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建个人 Skill，通过 Ghostty 为任意绝对目录幂等启动单工作区 CodexPro MCP 与固定 `mcp.lihe.show` 隧道，并完成真实 MCP `tree` 验证。

**Architecture:** Skill 解析用户意图并调用确定性脚本。Bash 启动器负责依赖、端口、冲突和 Ghostty 生命周期；Node 验证器负责 Streamable HTTP MCP 初始化、根目录识别与只读 `tree` 验证。固定域名一次只服务一个根目录，任何不匹配都失败关闭，不自动终止进程。

**Tech Stack:** Bash 3.2+、Node.js 20、macOS Launch Services、Ghostty、CodexPro 0.28.5、Cloudflare Tunnel、MCP Streamable HTTP、Node 内置测试运行器。

---

## 实施说明

实现目标位于 `/Users/lihe/.codex/skills/start-codexpro-mcp`，不在 Git 仓库内，因此不为 Skill 本体创建虚假提交。项目仓库仅保存设计和实施计划；执行时必须保留项目已有未提交改动。

不创建工作树：生产文件位于个人 Skill 目录，项目仓库除文档外没有代码改动。

## 文件结构

- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/SKILL.md`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/agents/openai.yaml`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/scripts/verify.mjs`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/tests/start_test.sh`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/tests/verify_test.mjs`

### Task 1: 记录无 Skill 的基线行为

- [ ] **Step 1: 用新 subagent 运行不加载 Skill 的基线场景**

Prompt:

```text
用户说：“为 /tmp/project with spaces 启动 mcp.lihe.show。8787 可能已被别的工作区占用，不允许自动终止任何进程。请说明你会执行什么。”
只输出行动方案，不实际启动服务。
```

- [ ] **Step 2: 记录失败模式**

```text
FAIL-A: 直接从 Codex exec 启动 codexpro，而不是 Ghostty
FAIL-B: 未先检查 8787
FAIL-C: 建议 pkill/killall 或自动关闭旧进程
FAIL-D: 只用 HTTP 状态码检查，没有 MCP initialize + tree
FAIL-E: 对包含空格的路径没有安全引用
```

Expected: 至少观察到一个 FAIL；原始输出只保留在当前任务记录中。

### Task 2: 初始化 Skill 脚手架

- [ ] **Step 1: 运行官方初始化脚本**

```bash
python3 /Users/lihe/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  start-codexpro-mcp \
  --path /Users/lihe/.codex/skills \
  --resources scripts \
  --interface 'display_name=Start CodexPro MCP' \
  --interface 'short_description=Start and verify a local CodexPro MCP workspace' \
  --interface 'default_prompt=Use $start-codexpro-mcp to start mcp.lihe.show for an absolute workspace path.'
mkdir -p /Users/lihe/.codex/skills/start-codexpro-mcp/tests
```

Expected: 创建 `SKILL.md`、`agents/openai.yaml`、`scripts/` 和 `tests/`。

### Task 3: 以 TDD 实现 MCP 验证器

**Files:**
- Test: `/Users/lihe/.codex/skills/start-codexpro-mcp/tests/verify_test.mjs`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/scripts/verify.mjs`

- [ ] **Step 1: 写入失败测试**

测试用 Node `http` 创建临时 MCP 服务，覆盖以下完整断言：

```javascript
import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import { parseArgs, verifyMcp } from "../scripts/verify.mjs";

function startFakeMcp({ root, treeText = ".", treeError = null }) {
  const sessions = new Set();
  const server = http.createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    const message = JSON.parse(body || "{}");
    const session = req.headers["mcp-session-id"];
    const send = (payload, status = 200) => {
      res.statusCode = status;
      res.setHeader("content-type", "text/event-stream");
      res.end(`event: message\ndata: ${JSON.stringify(payload)}\n\n`);
    };

    if (message.method === "initialize") {
      sessions.add("test-session");
      res.setHeader("mcp-session-id", "test-session");
      send({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          protocolVersion: "2025-03-26",
          capabilities: { tools: {} },
          serverInfo: { name: "CodexPro", version: "test" }
        }
      });
      return;
    }

    if (!sessions.has(session)) {
      send({ jsonrpc: "2.0", id: message.id, error: { code: -32000, message: "bad session" } }, 400);
      return;
    }

    if (message.method === "notifications/initialized") {
      res.statusCode = 202;
      res.end();
      return;
    }

    if (message.params?.name === "open_current_workspace") {
      send({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          content: [{ type: "text", text: `Root: ${root}` }],
          structuredContent: { root, workspace_id: "ws_test" }
        }
      });
      return;
    }

    if (message.params?.name === "tree") {
      send({
        jsonrpc: "2.0",
        id: message.id,
        result: treeError
          ? { isError: true, content: [{ type: "text", text: treeError }] }
          : {
              content: [{ type: "text", text: treeText }],
              structuredContent: { root, text: treeText, workspace_id: "ws_test" }
            }
      });
      return;
    }

    send({ jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "unknown" } }, 404);
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}/mcp`,
        close: () => new Promise((done) => server.close(done))
      });
    });
  });
}

test("parseArgs permits status verification without expected root", () => {
  assert.deepEqual(parseArgs(["--url", "http://127.0.0.1:8787/mcp"]), {
    url: "http://127.0.0.1:8787/mcp",
    expectedRoot: null,
    timeoutMs: 15000
  });
});

test("verifyMcp accepts an empty workspace tree", async () => {
  const fake = await startFakeMcp({ root: "/tmp/empty", treeText: "." });
  try {
    const result = await verifyMcp({ url: fake.url, expectedRoot: "/tmp/empty", timeoutMs: 2000 });
    assert.equal(result.root, "/tmp/empty");
    assert.equal(result.tree, ".");
  } finally {
    await fake.close();
  }
});

test("verifyMcp rejects a different root", async () => {
  const fake = await startFakeMcp({ root: "/tmp/old" });
  try {
    await assert.rejects(
      verifyMcp({ url: fake.url, expectedRoot: "/tmp/new", timeoutMs: 2000 }),
      (error) => error.code === "ROOT_MISMATCH"
    );
  } finally {
    await fake.close();
  }
});

test("verifyMcp preserves EPERM as a tree failure", async () => {
  const fake = await startFakeMcp({
    root: "/tmp/protected",
    treeError: "EPERM: operation not permitted, scandir '/tmp/protected'"
  });
  try {
    await assert.rejects(
      verifyMcp({ url: fake.url, expectedRoot: "/tmp/protected", timeoutMs: 2000 }),
      (error) => error.code === "TREE_FAILED" && error.message.includes("EPERM")
    );
  } finally {
    await fake.close();
  }
});
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
/Users/lihe/.local/node-v20.20.1/bin/node --test \
  /Users/lihe/.codex/skills/start-codexpro-mcp/tests/verify_test.mjs
```

Expected: FAIL，提示 `verify.mjs` 不存在或缺少导出。

- [ ] **Step 3: 实现最小验证器**

公开接口：

```javascript
export function parseArgs(argv)
export async function verifyMcp({ url, expectedRoot, timeoutMs })
```

实现要求：

```text
- 使用 AbortSignal.timeout(timeoutMs)
- 解析 application/json 和 text/event-stream
- 依次 initialize、notifications/initialized、open_current_workspace、tree
- 优先读取 structuredContent.root/text，文本内容只作兼容回退
- serverInfo.name 必须为 CodexPro
- 路径用 path.resolve 规范化
- expectedRoot 为空时只报告实际 root
- root 不匹配抛出 code=ROOT_MISMATCH
- tree isError 或包含 EPERM 时抛出 code=TREE_FAILED
- CLI 成功输出 ROOT、VERIFY=passed、SERVER、TREE_ENTRIES
- CLI 失败输出 ERROR_CODE、ERROR_MESSAGE，exitCode=1
```

CLI 入口：

```javascript
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const options = parseArgs(process.argv.slice(2));
  verifyMcp(options)
    .then((result) => {
      console.log(`ROOT=${result.root}`);
      console.log("VERIFY=passed");
      console.log(`SERVER=${result.serverName}@${result.serverVersion}`);
      console.log(`TREE_ENTRIES=${result.entryCount}`);
    })
    .catch((error) => {
      console.error(`ERROR_CODE=${error.code ?? "VERIFY_FAILED"}`);
      console.error(`ERROR_MESSAGE=${error.message}`);
      process.exitCode = 1;
    });
}
```

- [ ] **Step 4: 运行同一测试并确认 GREEN**

Expected: 4 tests PASS，0 failures。

### Task 4: 以 TDD 实现幂等启动器

**Files:**
- Test: `/Users/lihe/.codex/skills/start-codexpro-mcp/tests/start_test.sh`
- Create: `/Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh`

- [ ] **Step 1: 写入 Shell 行为测试**

测试使用 `mktemp -d` 和 PATH stubs，必须覆盖：

```text
1. 相对 --root 返回非零并包含“absolute path”
2. 不存在路径返回非零并包含“does not exist”
3. 含空格绝对路径 + --dry-run 输出完整 ROOT 和安全 Ghostty 命令
4. 空闲测试端口 + --status 返回 LOCAL=failed，不启动 Ghostty
5. 非 CodexPro 进程占用测试端口返回 LOCAL=conflict，且精确测试 PID 仍存活
6. 同根目录 verify stub 成功时返回 LOCAL=reused 且 OPEN_CALLS=0
7. verify stub 返回 ROOT_MISMATCH 时返回 LOCAL=conflict，且不调用 open
8. --status 不传 root 时输出 verify stub 报告的当前 ROOT
```

断言函数：

```bash
assert_contains() {
  case "$1" in
    *"$2"*) ;;
    *) printf 'expected output to contain: %s\nactual: %s\n' "$2" "$1" >&2; return 1 ;;
  esac
}

assert_status() {
  [ "$1" -eq "$2" ] || {
    printf 'expected status %s, got %s\n' "$2" "$1" >&2
    return 1
  }
}
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/tests/start_test.sh
```

Expected: FAIL，提示 `scripts/start.sh` 不存在。

- [ ] **Step 3: 实现启动脚本**

脚本常量必须可由测试环境变量覆盖：

```bash
#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GHOSTTY_APP="${GHOSTTY_APP:-/Applications/Ghostty.app}"
CODEXPRO_BIN="${CODEXPRO_BIN:-/Users/lihe/.local/node-v20.20.1/bin/codexpro}"
NODE_BIN="${NODE_BIN:-/Users/lihe/.local/node-v20.20.1/bin/node}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/opt/homebrew/bin/cloudflared}"
TUNNEL_CONFIG="${TUNNEL_CONFIG:-/Users/lihe/.cloudflared/chatgpt-mcp.yml}"
TUNNEL_SCRIPT="${TUNNEL_SCRIPT:-/Users/lihe/.cloudflared/start-chatgpt-mcp-tunnel.sh}"
OPEN_BIN="${OPEN_BIN:-/usr/bin/open}"
LSOF_BIN="${LSOF_BIN:-/usr/sbin/lsof}"
PS_BIN="${PS_BIN:-/bin/ps}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-$SKILL_DIR/scripts/verify.mjs}"
PORT=8787
PUBLIC_URL="https://mcp.lihe.show/mcp"
ROOT=""
MODE="start"
DRY_RUN=0
```

参数解析拒绝未知参数、缺值和相对路径。路径规范化只使用绝对路径去尾斜杠与 `test -d` 元数据检查；不得用 `ls` 判定 Ghostty 权限。

端口状态机：

```text
no listener + status -> LOCAL=failed, exit 1
no listener + dry-run -> print Ghostty command, LOCAL=started, VERIFY=skipped
no listener + start -> open Ghostty, condition-poll local verify
listener + non-CodexPro -> LOCAL=conflict, exit 2
listener + CodexPro + same root -> LOCAL=reused
listener + CodexPro + different root -> LOCAL=conflict, exit 2
listener + tree/permission failure -> LOCAL=failed, exit 1
```

Ghostty 启动必须使用数组参数与 `%q`：

```bash
launch_codexpro() {
  local command
  printf -v command 'exec %q start --root %q --port %q --tunnel none' \
    "$CODEXPRO_BIN" "$ROOT" "$PORT"
  "$OPEN_BIN" -na "$GHOSTTY_APP" --args -e /bin/zsh -lc "$command"
}

launch_tunnel() {
  local command
  printf -v command 'exec %q' "$TUNNEL_SCRIPT"
  "$OPEN_BIN" -na "$GHOSTTY_APP" --args -e /bin/zsh -lc "$command"
}
```

轮询调用必须用数组构造 `verify.mjs` 参数，避免 Bash 3.2 条件展开破坏含空格路径。不得出现生产代码 `kill`、`pkill`、`killall` 或向 CodexPro 发送 `q`。

- [ ] **Step 4: 运行同一 Shell 测试并确认 GREEN**

Expected: 8 个场景 PASS，退出码 0。

- [ ] **Step 5: 静态检查**

```bash
/bin/bash -n /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh
/Users/lihe/.local/node-v20.20.1/bin/node --check \
  /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/verify.mjs
rg -n '(^|[^a-z])(pkill|killall|kill)([^a-z]|$)' \
  /Users/lihe/.codex/skills/start-codexpro-mcp/scripts \
  /Users/lihe/.codex/skills/start-codexpro-mcp/SKILL.md
```

Expected: 前两条退出 0；最后一条无匹配并返回 1。

### Task 5: 写入 Skill 指令和 UI 元数据

**Files:**
- Modify: `/Users/lihe/.codex/skills/start-codexpro-mcp/SKILL.md`
- Modify: `/Users/lihe/.codex/skills/start-codexpro-mcp/agents/openai.yaml`

- [ ] **Step 1: 写入 SKILL.md**

```markdown
---
name: start-codexpro-mcp
description: Use when the user asks to start, restore, switch, or check mcp.lihe.show or a CodexPro MCP workspace on this Mac.
---

# Start CodexPro MCP

Resolve the requested workspace before running anything.

- Prefer an explicit absolute path from the user.
- Use the current workspace only when the user explicitly says "current workspace"; report the resolved path first.
- Ask for one absolute path when the target is ambiguous.
- Never guess from recent folders or the shell cwd.

Run `/bin/bash scripts/start.sh --root "/absolute/workspace/path"`.

For status, run `/bin/bash scripts/start.sh --status`. Add `--root` only when comparing against an expected workspace.

Treat script output as authoritative. Report `ROOT`, `LOCAL`, `TUNNEL`, `VERIFY`, and `PUBLIC_URL`.

Safety rules:

- Never start CodexPro directly from Codex exec; the target may require Ghostty filesystem permissions.
- Never terminate or signal a process.
- On `LOCAL=conflict`, report current and requested roots, then ask the user to close the old CodexPro Ghostty window.
- Do not print Cloudflare credentials or add authentication tokens.
- Do not claim success unless `VERIFY=passed`.
```

- [ ] **Step 2: 生成 agents/openai.yaml**

```bash
python3 /Users/lihe/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py \
  /Users/lihe/.codex/skills/start-codexpro-mcp \
  --interface 'display_name=Start CodexPro MCP' \
  --interface 'short_description=Start and verify one local CodexPro MCP workspace' \
  --interface 'default_prompt=Use $start-codexpro-mcp to start mcp.lihe.show for an absolute workspace path.'
```

- [ ] **Step 3: 验证 Skill 结构**

```bash
python3 /Users/lihe/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /Users/lihe/.codex/skills/start-codexpro-mcp
wc -w /Users/lihe/.codex/skills/start-codexpro-mcp/SKILL.md
```

Expected: `Skill is valid!`；SKILL.md 少于 500 词。

### Task 6: 运行非破坏性集成测试

- [ ] **Step 1: 记录当前进程数**

```bash
before_local="$(pgrep -f 'codexpro/dist/http.js' | wc -l | tr -d ' ')"
before_tunnel="$(pgrep -f 'cloudflared tunnel --config /Users/lihe/.cloudflared/chatgpt-mcp.yml run chatgpt-mcp' | wc -l | tr -d ' ')"
printf 'local=%s tunnel=%s\n' "$before_local" "$before_tunnel"
```

- [ ] **Step 2: 对当前 PINN 根目录运行两次**

```bash
/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh \
  --root '/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo'
```

Expected each time: `LOCAL=reused`、`TUNNEL=reused`、`VERIFY=passed`。

- [ ] **Step 3: 确认进程数未增加**

重新执行 Step 1，Expected: counts 完全相同。

- [ ] **Step 4: 测试含空格通用路径干运行**

```bash
test_root="$(mktemp -d '/tmp/codexpro workspace.XXXXXX')"
/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh \
  --root "$test_root" --port 18787 \
  --public-url 'http://127.0.0.1:18787/mcp' --dry-run
```

Expected: ROOT 保留空格；`LOCAL=started`；`VERIFY=skipped`；未新增进程。

- [ ] **Step 5: 测试冲突关闭**

```bash
test_root="$(mktemp -d /tmp/codexpro-conflict.XXXXXX)"
python3 -m http.server 18788 --bind 127.0.0.1 >/tmp/codexpro-conflict-http.log 2>&1 &
test_pid="$!"
set +e
output=$(/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh \
  --status --root "$test_root" --port 18788 2>&1)
status=$?
set -e
kill -0 "$test_pid"
kill "$test_pid"
wait "$test_pid" 2>/dev/null || true
printf '%s\n' "$output"
test "$status" -eq 2
```

Expected: `LOCAL=conflict`；检查后测试 HTTP PID 仍存活；仅测试清理终止自己创建的精确 PID。

### Task 7: 前向测试 Skill 触发与安全行为

- [ ] **Step 1: 用新 subagent 测试缺少路径**

```text
Use $start-codexpro-mcp. 用户说：“帮我把 MCP 切到另一个文件夹”，但没有提供路径。不要实际启动服务。
```

Expected: 只问一个绝对路径，不猜测。

- [ ] **Step 2: 测试当前状态**

```text
Use $start-codexpro-mcp to check which workspace mcp.lihe.show currently serves. Do not restart anything.
```

Expected: 调用 `start.sh --status`，报告当前 ROOT，不新增进程。

- [ ] **Step 3: 测试冲突策略**

```text
Use $start-codexpro-mcp for /tmp/another-workspace. If 8787 serves another root, preserve every process.
```

Expected: 发现冲突后停止，不调用终止命令。

- [ ] **Step 4: 与 Task 1 基线比较**

Expected: FAIL-A 至 FAIL-E 全部消失。若仍有失败，只修改对应 Skill 指令并重跑同一场景。

### Task 8: 冷启动端到端验收

此步骤需要人工关闭现有服务，保持安全策略不变。

- [ ] **Step 1: 请求用户关闭当前 CodexPro 和 Cloudflare Ghostty 窗口**

不得由 Skill 或代理终止进程。

- [ ] **Step 2: 确认端口与隧道已停止**

```bash
lsof -nP -iTCP:8787 -sTCP:LISTEN
pgrep -f 'cloudflared tunnel --config /Users/lihe/.cloudflared/chatgpt-mcp.yml run chatgpt-mcp'
```

Expected: 两条命令均无输出。

- [ ] **Step 3: 使用 Skill 冷启动 PINN 工作区**

```bash
/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/start.sh \
  --root '/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo'
```

Expected: `LOCAL=started`、`TUNNEL=started`、`VERIFY=passed`。

- [ ] **Step 4: 从公网独立复验**

```bash
/Users/lihe/.local/node-v20.20.1/bin/node \
  /Users/lihe/.codex/skills/start-codexpro-mcp/scripts/verify.mjs \
  --url https://mcp.lihe.show/mcp \
  --expected-root '/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo'
```

Expected: `VERIFY=passed`，无 `EPERM`。

- [ ] **Step 5: 最终验证**

```bash
python3 /Users/lihe/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /Users/lihe/.codex/skills/start-codexpro-mcp
/bin/bash /Users/lihe/.codex/skills/start-codexpro-mcp/tests/start_test.sh
/Users/lihe/.local/node-v20.20.1/bin/node --test \
  /Users/lihe/.codex/skills/start-codexpro-mcp/tests/verify_test.mjs
```

Expected: Skill valid；所有 Shell 场景 PASS；所有 Node 测试 PASS。
