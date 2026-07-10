<div align="center">

# 👻 Kiro Gateway

**Kiro API 代理閘道器**

🇹🇼 繁體中文 • [🇬🇧 English](docs/en/README.md) • [🇯🇵 日本語](docs/ja/README.md) • [🇰🇷 한국어](docs/ko/README.md)

由 [@oddstab](https://github.com/oddstab) 用 ❤️ 製作

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

*透過 Claude Code、Grok Build、OpenCode、OpenClaw、Claw Code、Codex app、Cursor、Cline、Roo Code、Kilo Code、Obsidian、OpenAI SDK、LangChain、Continue 等 OpenAI 或 Anthropic 相容工具，使用 Kiro 的 Claude（Opus 4.8、Sonnet 5 等）及 Grok 相容模型*

[模型](#-可用模型) • [功能](#-功能特性) • [快速開始](#-快速開始) • [設定](#%EF%B8%8F-設定)

</div>

---

## 🤖 可用模型

> ⚠️ **重要：** 模型可用性取決於您的 Kiro 方案（免費/付費）。閘道器提供您的 IDE 或 CLI 中基於訂閱可用的所有模型。

### Claude 模型

| 模型 | 說明 |
|------|------|
| 🧠 Claude Opus 4.8 | 最新、最強大。適合複雜推理、架構設計和自主代理任務 |
| 🧠 Claude Opus 4.7 | 上一代旗艦。深度分析和多步驟問題求解的優秀選擇 |
| 🧠 Claude Opus 4.6 | 全能型選手，程式開發和推理任務均表現出色 |
| 🚀 Claude Sonnet 5 | 最新均衡模型。程式開發、寫作和通用任務的絕佳選擇 |
| 🚀 Claude Sonnet 4.5 | 快速且強大。速度與品質的絕佳平衡 |
| 📦 Claude Sonnet 4 | 上一代模型。大多數場景仍然強大可靠 |
| ⚡ Claude Haiku 4.5 | 閃電般快速。適合快速回應、簡單任務和聊天 |

### Grok 相容性

閘道器支援 Grok 模型名稱作為別名，可直接替代 xAI API：

| Grok 模型 | 預設對應 |
|-----------|---------|
| `grok-4.5` | `claude-opus-4-6[1m]` |
| `grok-4` | `claude-opus-4-6[1m]` |
| `grok-4-fast` | `claude-opus-4-6[1m]` |
| `grok-3` | `claude-opus-4-6[1m]` |

在 `.env` 中設定 `GROK_TARGET_MODEL` 可自訂對應模型：

```env
# 預設為 claude-opus-4-6[1m]，可改為任何支援的模型
GROK_TARGET_MODEL="claude-opus-4-8[1m]"
```

### 開源模型

| 模型 | 說明 |
|------|------|
| 💤 GLM-5 | 開源 MoE（744B/40B 活躍）。複雜系統工程和長期自主代理任務 |
| 🐋 DeepSeek-V3.2 | 開源 MoE（685B/37B 活躍）。程式開發、推理和通用任務均衡 |
| 🧩 MiniMax M2.5 | 開源 MoE（230B/10B 活躍）。增強版，推理和任務處理能力更強 |
| 🧩 MiniMax M2.1 | 開源 MoE（230B/10B 活躍）。複雜任務、規劃和多步驟工作流程 |
| 🤖 Qwen3-Coder-Next | 開源 MoE（80B/3B 活躍）。程式開發導向，適合大型專案 |

> 💡 **智慧模型解析：** 使用任何模型名稱格式 — `claude-sonnet-4-5`、`claude-sonnet-4.5`、`grok-4`，甚至版本化名稱如 `claude-sonnet-4-5-20250929`。閘道器會自動正規化。

---

## ✨ 功能特性

| 功能 | 說明 |
|------|------|
| 🔌 **OpenAI 相容 API** | 支援任何 OpenAI 相容工具 |
| 🔌 **Anthropic 相容 API** | 原生 `/v1/messages` 端點 |
| 🔀 **多帳號支援** | 多帳號間的智慧故障轉移 |
| 🌐 **VPN/代理支援** | 受限網路的 HTTP/SOCKS5 代理 |
| 🧠 **延伸思考** | 推理功能為本專案獨有 |
| 👁️ **視覺支援** | 向模型傳送圖片 |
| 🔍 **網路搜尋** | 搜尋網路上的最新資訊 |
| 🛠️ **工具呼叫** | 支援函式呼叫 |
| 💬 **完整訊息歷史** | 傳遞完整對話上下文 |
| 📡 **串流傳輸** | 完整 SSE 串流支援 |
| 🔄 **重試邏輯** | 錯誤時自動重試（403、429、5xx） |
| 📋 **擴充模型列表** | 包含版本化模型 |
| 🔐 **智慧權杖管理** | 到期前自動刷新 |

---

## 🚀 快速開始

**選擇部署方式：**
- 🐍 **原生 Python** - 完全控制，容易除錯
- 🐳 **Docker** - 隔離環境，輕鬆部署 → [跳至 Docker](#-docker-部署)

### 前置需求

- Python 3.10+
- 以下任一：
  - 已登入帳號的 [Kiro IDE](https://kiro.dev/)，或
  - 搭配 AWS SSO (AWS IAM Identity Center, OIDC) 的 [Kiro CLI](https://kiro.dev/cli/) - 免費 Builder ID 或企業帳號

### 安裝

```bash
# 複製儲存庫（需要 Git）
git clone https://github.com/oddstab/kiro-gateway.git
cd kiro-gateway

# 或下載 ZIP：Code → Download ZIP → 解壓縮 → 開啟 kiro-gateway 資料夾

# 安裝相依套件
pip install -r requirements.txt

# 設定（參見設定章節）
cp .env.example .env
# 複製並編輯 .env，填入您的憑證

# 啟動伺服器
python main.py

# 或使用自訂埠號（如果 8000 被占用）
python main.py --port 9000
```

伺服器將在 `http://localhost:8000` 上可用

---

## ⚙️ 設定

> 💡 **進階使用者：** 尋找多帳號支援？請參閱下方的[帳號系統](#-帳號系統進階)。

### 選項 1：JSON 憑證檔案 (Kiro IDE / Enterprise)

指定憑證檔案路徑：

適用於：
- **Kiro IDE**（標準）- 個人帳號
- **Enterprise** - 搭配 SSO 的企業帳號

```env
KIRO_CREDS_FILE="~/.aws/sso/cache/kiro-auth-token.json"

# 保護您代理伺服器的密碼（自行設定任意安全字串）
# 連線到閘道器時將用作 api_key
PROXY_API_KEY="my-super-secret-password-123"
```

<details>
<summary>📄 JSON 檔案格式</summary>

```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "expiresAt": "2025-01-12T23:00:00.000Z",
  "profileArn": "arn:aws:codewhisperer:us-east-1:...",
  "region": "us-east-1",
  "clientIdHash": "abc123..."
}
```

> **注意：** 如果 `~/.aws/sso/cache/` 中有兩個 JSON 檔案（例如 `kiro-auth-token.json` 和一個雜湊命名的檔案），請在 `KIRO_CREDS_FILE` 中使用 `kiro-auth-token.json`。閘道器會自動載入另一個檔案。

</details>

### 選項 2：環境變數（.env 檔案）

在專案根目錄建立 `.env` 檔案：

```env
# 必要
REFRESH_TOKEN="your_kiro_refresh_token"

# 保護您代理伺服器的密碼（自行設定任意安全字串）
PROXY_API_KEY="my-super-secret-password-123"

# 選擇性
PROFILE_ARN="arn:aws:codewhisperer:us-east-1:..."
KIRO_REGION="us-east-1"
```

### 選項 3：AWS SSO 憑證 (kiro-cli / Enterprise)

如果您使用搭配 AWS SSO (AWS IAM Identity Center) 的 `kiro-cli` 或 Kiro IDE，閘道器會自動偵測並使用適當的認證方式。

免費 Builder ID 帳號和企業帳號皆適用。

```env
KIRO_CREDS_FILE="~/.aws/sso/cache/your-sso-cache-file.json"

# 保護您代理伺服器的密碼
PROXY_API_KEY="my-super-secret-password-123"

# 注意：AWS SSO (Builder ID 和企業帳號) 使用者不需要 PROFILE_ARN
# 閘道器無需它即可運作
```

<details>
<summary>📄 AWS SSO JSON 檔案格式</summary>

AWS SSO 憑證檔案（位於 `~/.aws/sso/cache/`）包含：

```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "expiresAt": "2025-01-12T23:00:00.000Z",
  "region": "us-east-1",
  "clientId": "...",
  "clientSecret": "..."
}
```

**注意：** AWS SSO (Builder ID 和企業帳號) 使用者不需要 `profileArn`。閘道器無需它即可運作（如有指定會被忽略）。

</details>

<details>
<summary>🔍 運作原理</summary>

閘道器根據憑證檔案自動偵測認證類型：

- **Kiro Desktop Auth**（預設）：當 `clientId` 和 `clientSecret` 不存在時使用
  - 端點：`https://prod.{region}.auth.desktop.kiro.dev/refreshToken`
  
- **AWS SSO (OIDC)**：當 `clientId` 和 `clientSecret` 存在時使用
  - 端點：`https://oidc.{region}.amazonaws.com/token`

無需額外設定 — 只要指向您的憑證檔案即可！

</details>

### 選項 4：kiro-cli SQLite 資料庫

如果您使用 `kiro-cli` 並希望直接使用其 SQLite 資料庫：

```env
KIRO_CLI_DB_FILE="~/.local/share/kiro-cli/data.sqlite3"

# 保護您代理伺服器的密碼
PROXY_API_KEY="my-super-secret-password-123"

# 注意：AWS SSO (Builder ID 和企業帳號) 使用者不需要 PROFILE_ARN
# 閘道器無需它即可運作
```

<details>
<summary>📄 資料庫位置</summary>

| CLI 工具 | 資料庫路徑 |
|----------|-----------|
| kiro-cli | `~/.local/share/kiro-cli/data.sqlite3` |
| amazon-q-developer-cli | `~/.local/share/amazon-q/data.sqlite3` |

閘道器從 `auth_kv` 資料表讀取憑證，其中存放：
- `kirocli:odic:token` 或 `codewhisperer:odic:token` — 存取權杖、刷新權杖、到期時間
- `kirocli:odic:device-registration` 或 `codewhisperer:odic:device-registration` — 客戶端 ID 和密鑰

兩種鍵格式皆支援，以相容不同版本的 kiro-cli。

</details>

### 取得憑證

**Kiro IDE 使用者：**
- 登入 Kiro IDE 並使用上方的選項 1（JSON 憑證檔案）
- 憑證檔案在登入後會自動建立

**Kiro CLI 使用者：**
- 使用 `kiro-cli login` 登入，然後使用上方的選項 3 或選項 4
- 無需手動擷取權杖！

<details>
<summary>🔧 進階：手動擷取權杖</summary>

如果您需要手動擷取 refresh token（例如用於除錯），可以攔截 Kiro IDE 流量：
- 尋找發往以下位址的請求：`prod.us-east-1.auth.desktop.kiro.dev/refreshToken`

</details>

---

## 🔀 帳號系統（進階）

帳號系統是管理多個 Kiro 帳號並自動故障轉移的方式。未來此系統將取代 `.env` 檔案用於憑證設定，但目前是選擇性的，適用於想使用多帳號的使用者。

### 為什麼需要

如果您有多個 Kiro 帳號，閘道器可在帳號暫時不可用時自動切換。

此系統也適用於單一帳號 — 只是不會切換。

### 如何啟用

在 `.env` 中加入：

```env
ACCOUNT_SYSTEM=true
```

**會發生什麼：**
- 首次啟動時，`.env` 中的憑證會自動遷移到 `credentials.json`（僅一次）
- 之後，`.env` 中的所有帳號和區域設定都會被忽略
- 帳號管理僅透過 `credentials.json` 進行

<details>
<summary>📄 設定範例</summary>

**單一帳號：**
```json
[
  {
    "type": "json",
    "path": "~/.aws/sso/cache/kiro-auth-token.json"
  }
]
```

**多個帳號：**
```json
[
  {
    "type": "json",
    "path": "~/.aws/sso/cache/kiro-auth-token.json"
  },
  {
    "type": "sqlite",
    "path": "~/.local/share/kiro-cli/data.sqlite3"
  },
  {
    "type": "refresh_token",
    "refresh_token": "eyJhbGc...",
    "profile_arn": "arn:aws:codewhisperer:us-east-1:..."
  }
]
```

**包含檔案的資料夾：**
```json
[
  {
    "type": "json",
    "path": "C:\\MyAccs\\kiro67"
  }
]
```

閘道器會掃描資料夾中的所有檔案並將其新增為個別帳號。

</details>

### 故障轉移如何運作

當某個帳號回傳錯誤（429 速率限制、402 配額超出），閘道器會自動嘗試列表中的下一個帳號。如果帳號連續多次失敗，閘道器會暫時停止使用它，並定期檢查是否已恢復。

單一帳號時故障轉移不作用 — 您會收到 Kiro API 的原始錯誤。

完整設定範例（包含每個帳號的區域設定）請參見 [`credentials.json.example`](credentials.json.example)。

---

## 🐳 Docker 部署

> **Docker 部署方式。** 偏好原生 Python？請參閱上方的[快速開始](#-快速開始)。

### 快速開始

```bash
# 1. 複製並設定
git clone https://github.com/oddstab/kiro-gateway.git
cd kiro-gateway
cp .env.example .env
# 使用您的憑證編輯 .env

# 2. 使用 docker-compose 執行
docker-compose up -d

# 3. 檢查狀態
docker-compose logs -f
curl http://localhost:8000/health
```

### Docker Run（不使用 Compose）

<details>
<summary>🔹 使用環境變數</summary>

```bash
docker run -d \
  -p 8000:8000 \
  -e PROXY_API_KEY="my-super-secret-password-123" \
  -e REFRESH_TOKEN="your_refresh_token" \
  --name kiro-gateway \
  ghcr.io/oddstab/kiro-gateway:latest
```

</details>

<details>
<summary>🔹 使用憑證檔案</summary>

**Linux/macOS:**
```bash
docker run -d \
  -p 8000:8000 \
  -v ~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro \
  -e KIRO_CREDS_FILE=/home/kiro/.aws/sso/cache/kiro-auth-token.json \
  -e PROXY_API_KEY="my-super-secret-password-123" \
  --name kiro-gateway \
  ghcr.io/oddstab/kiro-gateway:latest
```

**Windows (PowerShell):**
```powershell
docker run -d `
  -p 8000:8000 `
  -v ${HOME}/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro `
  -e KIRO_CREDS_FILE=/home/kiro/.aws/sso/cache/kiro-auth-token.json `
  -e PROXY_API_KEY="my-super-secret-password-123" `
  --name kiro-gateway `
  ghcr.io/oddstab/kiro-gateway:latest
```

</details>

<details>
<summary>🔹 使用 .env 檔案</summary>

```bash
docker run -d -p 8000:8000 --env-file .env --name kiro-gateway ghcr.io/oddstab/kiro-gateway:latest
```

</details>

### Docker Compose 設定

編輯 `docker-compose.yml` 並取消註解適合您作業系統的 volume 掛載：

```yaml
volumes:
  # Kiro IDE 憑證（選擇您的作業系統）
  - ~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro              # Linux/macOS
  # - ${USERPROFILE}/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro  # Windows
  
  # kiro-cli 資料庫（選擇您的作業系統）
  - ~/.local/share/kiro-cli:/home/kiro/.local/share/kiro-cli  # Linux/macOS
  # - ${USERPROFILE}/.local/share/kiro-cli:/home/kiro/.local/share/kiro-cli  # Windows
  
  # 除錯日誌（選擇性）
  - ./debug_logs:/app/debug_logs
```

### 管理指令

```bash
docker-compose logs -f      # 檢視日誌
docker-compose restart      # 重新啟動
docker-compose down         # 停止
docker-compose pull && docker-compose up -d  # 更新
```

<details>
<summary>🔧 從原始碼建置</summary>

```bash
docker build -t kiro-gateway .
docker run -d -p 8000:8000 --env-file .env kiro-gateway
```

</details>

---

## 🌐 VPN/代理支援

**適用於中國大陸、企業網路或與 AWS 服務連線有問題的地區使用者。**

閘道器支援透過 VPN 或代理伺服器路由所有 Kiro API 請求。如果您遇到 AWS 端點的連線問題或需要使用企業代理，這是必要的。

### 設定

在 `.env` 檔案中加入：

```env
# HTTP 代理
VPN_PROXY_URL=http://127.0.0.1:7890

# SOCKS5 代理
VPN_PROXY_URL=socks5://127.0.0.1:1080

# 含身分驗證（企業代理）
VPN_PROXY_URL=http://username:password@proxy.company.com:8080

# 無協定（預設為 http://）
VPN_PROXY_URL=192.168.1.100:8080
```

### 支援的協定

- ✅ **HTTP** — 標準代理協定
- ✅ **HTTPS** — 安全代理連線
- ✅ **SOCKS5** — 進階代理協定（VPN 軟體中常見）
- ✅ **身分驗證** — URL 中嵌入的使用者名稱/密碼

### 何時需要

| 狀況 | 解決方案 |
|------|---------|
| 與 AWS 連線逾時 | 使用 VPN/代理路由流量 |
| 企業網路限制 | 設定公司代理 |
| 區域連線問題 | 使用支援代理的 VPN 服務 |
| 隱私需求 | 透過自己的代理伺服器路由 |

### 支援代理的常見 VPN 軟體

大多數 VPN 用戶端提供本地代理伺服器：
- **Sing-box** — 支援 HTTP/SOCKS5 代理的現代 VPN 用戶端
- **Clash** — 通常在 `http://127.0.0.1:7890` 上執行
- **V2Ray** — 可設定的 SOCKS5/HTTP 代理
- **Shadowsocks** — SOCKS5 代理支援
- **企業 VPN** — 向 IT 部門詢問代理設定

不需要代理支援時，將 `VPN_PROXY_URL` 留空（預設）。

---

## 🖥️ 搭配 AI 程式開發工具使用

### Claude Code

設定環境變數將 Claude Code 指向本閘道器：

```powershell
# PowerShell profile (~\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1)
function cc {
    $env:ANTHROPIC_BASE_URL = 'http://localhost:8000'
    $env:ANTHROPIC_API_KEY = 'kiro-gateway-local'  # 對應 .env 中的 PROXY_API_KEY
    & claude --model claude-opus-4-6[1m] --effort max --dangerously-skip-permissions @args
}
```

使用：
```powershell
cc                     # 互動模式
cc "幫我修這個 bug"    # 直接下指令
cc --model claude-sonnet-5 "快速回答"  # 臨時切換模型
```

**Linux/macOS (bash/zsh):**
```bash
# ~/.bashrc 或 ~/.zshrc
cc() {
    ANTHROPIC_BASE_URL='http://localhost:8000' \
    ANTHROPIC_API_KEY='kiro-gateway-local' \
    claude --model claude-opus-4-6[1m] --effort max --dangerously-skip-permissions "$@"
}
```

### Grok Build

設定環境變數將 Grok Build 指向本閘道器（透過 Grok 別名自動對應 Claude 模型）：

```powershell
# PowerShell profile
function gg {
    $env:GROK_XAI_API_BASE_URL = "http://localhost:8000/v1"
    $env:GROK_CODE_XAI_API_KEY = "kiro-gateway-local"  # 對應 .env 中的 PROXY_API_KEY
    & "$env:USERPROFILE\.grok\bin\grok.exe" --disable-web-search @args
}
```

Grok Build 發出的 `grok-4` 等模型請求會自動對應到 `GROK_TARGET_MODEL`（預設 `claude-opus-4-6[1m]`，可在 `.env` 中修改）。

**Linux/macOS:**
```bash
gg() {
    GROK_XAI_API_BASE_URL='http://localhost:8000/v1' \
    GROK_CODE_XAI_API_KEY='kiro-gateway-local' \
    grok --disable-web-search "$@"
}
```

### 其他工具

任何支援 OpenAI 或 Anthropic API 的工具都能使用，只要將 base URL 指向 `http://localhost:8000`（Anthropic）或 `http://localhost:8000/v1`（OpenAI），API key 設為你的 `PROXY_API_KEY` 即可。

---

## 📡 API 參考

### 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 健康檢查 |
| `/health` | GET | 詳細健康檢查 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | OpenAI Chat Completions API |
| `/v1/messages` | POST | Anthropic Messages API |

---

## 💡 使用範例

### OpenAI API

<details>
<summary>🔹 簡單 cURL 請求</summary>

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-super-secret-password-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": true
  }'
```

> **注意：** 將 `my-super-secret-password-123` 替換為您在 `.env` 中設定的 `PROXY_API_KEY`。

</details>

<details>
<summary>🔹 串流請求</summary>

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-super-secret-password-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [
      {"role": "system", "content": "你是一個有幫助的助手。"},
      {"role": "user", "content": "2+2 等於多少？"}
    ],
    "stream": true
  }'
```

</details>

<details>
<summary>🛠️ 搭配工具呼叫</summary>

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-super-secret-password-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "倫敦的天氣如何？"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "取得某個地點的天氣",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "城市名稱"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

</details>

<details>
<summary>🐍 Python OpenAI SDK</summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-super-secret-password-123"  # .env 中的 PROXY_API_KEY
)

response = client.chat.completions.create(
    model="claude-sonnet-4-5",
    messages=[
        {"role": "system", "content": "你是一個有幫助的助手。"},
        {"role": "user", "content": "你好！"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

</details>

<details>
<summary>🦜 LangChain</summary>

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-super-secret-password-123",  # .env 中的 PROXY_API_KEY
    model="claude-sonnet-4-5"
)

response = llm.invoke("你好，你好嗎？")
print(response.content)
```

</details>

### Anthropic API

<details>
<summary>🔹 簡單 cURL 請求</summary>

```bash
curl http://localhost:8000/v1/messages \
  -H "x-api-key: my-super-secret-password-123" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

> **注意：** Anthropic API 使用 `x-api-key` 標頭而非 `Authorization: Bearer`。兩者皆支援。

</details>

<details>
<summary>🔹 含系統提示</summary>

```bash
curl http://localhost:8000/v1/messages \
  -H "x-api-key: my-super-secret-password-123" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "system": "你是一個有幫助的助手。",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

> **注意：** 在 Anthropic API 中，`system` 是獨立欄位，不是訊息。

</details>

<details>
<summary>📡 串流</summary>

```bash
curl http://localhost:8000/v1/messages \
  -H "x-api-key: my-super-secret-password-123" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

</details>

<details>
<summary>🐍 Python Anthropic SDK</summary>

```python
import anthropic

client = anthropic.Anthropic(
    api_key="my-super-secret-password-123",  # .env 中的 PROXY_API_KEY
    base_url="http://localhost:8000"
)

# 非串流
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好！"}]
)
print(response.content[0].text)

# 串流
with client.messages.stream(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好！"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

</details>

---

## 🔧 除錯

除錯日誌**預設為停用**。要啟用，在 `.env` 中加入：

```env
# 除錯日誌模式：
# - off：停用（預設）
# - errors：僅儲存失敗請求的日誌（4xx、5xx）- 建議用於疑難排解
# - all：儲存每個請求的日誌（每次請求時覆寫）
DEBUG_MODE=errors
```

### 除錯模式

| 模式 | 說明 | 使用場景 |
|------|------|---------|
| `off` | 停用（預設） | 正式環境 |
| `errors` | 僅儲存失敗請求的日誌（4xx、5xx） | **建議用於疑難排解** |
| `all` | 儲存每個請求的日誌 | 開發/除錯 |

### 除錯檔案

啟用後，請求會記錄到 `debug_logs/` 資料夾：

| 檔案 | 說明 |
|------|------|
| `request_body.json` | 來自用戶端的傳入請求（OpenAI 格式） |
| `kiro_request_body.json` | 送往 Kiro API 的請求 |
| `response_stream_raw.txt` | 來自 Kiro 的原始串流 |
| `response_stream_modified.txt` | 轉換後的串流（OpenAI 格式） |
| `app_logs.txt` | 請求的應用程式日誌 |
| `error_info.json` | 錯誤詳情（僅在出錯時） |

---

## 🔧 疑難排解

### 連線問題

**錯誤：「Name or service not known」或 DNS 解析失敗**

Q API 端點在您的區域可能無法公開解析。使用 VPN 或代理：

```env
VPN_PROXY_URL=http://127.0.0.1:7890
```

詳情請參閱 [VPN/代理支援](#-vpn代理支援)。

---

**錯誤：透過代理出現「503 Service Unavailable」**

Q API 端點僅存在於特定區域。嘗試不同區域：

```env
KIRO_API_REGION="eu-central-1"  # 或 us-east-1
```

常見可達區域：`us-east-1`、`eu-central-1`

---

**OIDC 正常但 Q API 失敗**

您的 SSO 區域可能與 Q API 區域不同。閘道器會從憑證中自動偵測，但您可以覆寫：

```env
KIRO_API_REGION="eu-central-1"
```

---

## 📜 授權條款

本專案採用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 授權。

這代表：
- ✅ 您可以使用、修改和散佈此軟體
- ✅ 您可以用於商業目的
- ⚠️ 散佈軟體時**必須公開原始碼**
- ⚠️ **網路使用即為散佈** — 如果您在伺服器上執行修改版本並讓他人與之互動，必須向他們提供原始碼
- ⚠️ 修改必須以相同授權條款發布

完整授權條款文字請參見 [LICENSE](LICENSE) 檔案。

### 貢獻者授權合約 (CLA)

向本專案提交貢獻即表示您同意[貢獻者授權合約 (CLA)](CLA.md) 的條款。

---

## 👥 貢獻者

<!-- ALL-CONTRIBUTORS-LIST:START -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/oddstab"><img src="https://github.com/oddstab.png?size=100" width="100px;" alt=""/><br /><sub><b>oddstab</b></sub></a><br />💻 📖 🚧</td>
    <td align="center"><a href="https://claude.ai"><img src="https://avatars.githubusercontent.com/u/76263028?s=100" width="100px;" alt=""/><br /><sub><b>Claude (Anthropic)</b></sub></a><br />💻 📖</td>
  </tr>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## ⚠️ 免責聲明

本專案與 Amazon Web Services (AWS)、Anthropic 或 Kiro IDE 無關，未經其認可或贊助。使用風險自負，請遵守底層 API 的服務條款。

---

<div align="center">

**[⬆ 回到頂部](#-kiro-gateway)**

</div>
