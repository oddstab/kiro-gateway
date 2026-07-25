# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/oddstab/kiro-gateway
# Copyright (C) 2025 oddstab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
context window 驗證腳本
========================

一鍵核對每個模型的 context window，並排對照兩個來源：

  1. AWS 源頭    — 直接打 q.{region}.amazonaws.com 的 ListAvailableModels，
                   取 tokenLimits.maxInputTokens（附完整 KiroIDE 客戶端識別
                   header，否則 AWS 回 AccessDeniedException）。
  2. 本機 gateway — 打 http://127.0.0.1:8000/v1/models，取 contextWindow。

兩欄一致，即證明 gateway 是「原封透傳 AWS 動態值」，而非寫死。
Kiro 之後出新模型或改 context 時，重跑本腳本即可核對。

用法：
    python verify_context.py            # 兩邊都打並對照
    python verify_context.py --gateway  # 只打本機 gateway（免 AWS token）
    python verify_context.py --aws      # 只打 AWS 源頭

依賴 gateway 既有模組（get_kiro_headers 會自動帶 fingerprint 等識別 header），
所以請在專案根目錄、venv 已啟用的情況下執行。
"""

import argparse
import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

# 重用 gateway 自己的邏輯：header 組裝與 auth，確保與正式請求完全一致
from kiro.utils import get_kiro_headers
from kiro.auth import KiroAuthManager
from kiro.config import (
    PROFILE_ARN,
    KIRO_CREDS_FILE,
    get_list_models_host,
    KIRO_LIST_MODELS_TARGET,
    KIRO_LIST_MODELS_ORIGIN,
)

load_dotenv()

# 重點觀察的新模型；其餘模型仍會全部列出，這些只是加上箭頭標記
HIGHLIGHT = {"claude-opus-5", "claude-sonnet-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

GATEWAY_URL = os.getenv("VERIFY_GATEWAY_URL", "http://127.0.0.1:8000/v1/models")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")


def _fmt(n):
    """把 1000000 這種數字顯示成 1,000,000；None 顯示為 -。"""
    return "-" if n is None else f"{n:,}"


async def fetch_aws():
    """直接打 AWS ListAvailableModels，回傳 {modelId: maxInputTokens}。"""
    creds = json.loads(open(KIRO_CREDS_FILE, encoding="utf-8").read())
    auth_manager = KiroAuthManager(
        refresh_token=creds.get("refreshToken") or creds.get("refresh_token"),
        profile_arn=creds.get("profileArn") or PROFILE_ARN,
        region=creds.get("region", "us-east-1"),
    )
    # 優先用憑證檔裡現成的 access token（免刷新，避開刷新網域可能不可達的環境）；
    # 沒有才回退到 get_access_token()（會視需要刷新）。
    token = creds.get("accessToken")
    if not token:
        token = await auth_manager.get_access_token()

    headers = get_kiro_headers(auth_manager, token)
    # get_kiro_headers 預設帶對話用的 target，這裡覆寫成 ListAvailableModels
    headers["x-amz-target"] = KIRO_LIST_MODELS_TARGET

    profile = auth_manager.profile_arn
    params = {"origin": KIRO_LIST_MODELS_ORIGIN}
    body = {"origin": KIRO_LIST_MODELS_ORIGIN}
    if profile:
        params["profileArn"] = profile
        body["profileArn"] = profile

    url = f"{get_list_models_host()}/"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.post(url, headers=headers, params=params,
                                 content=json.dumps(body).encode())
    if resp.status_code != 200:
        raise RuntimeError(f"AWS HTTP {resp.status_code}: {resp.text[:200]}")

    out = {}
    for m in resp.json().get("models", []):
        tl = m.get("tokenLimits") or {}
        out[m["modelId"]] = tl.get("maxInputTokens")
    return out


def fetch_gateway():
    """打本機 gateway /v1/models，回傳 {id: contextWindow}。"""
    headers = {"Authorization": f"Bearer {PROXY_API_KEY}"} if PROXY_API_KEY else {}
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(GATEWAY_URL, headers=headers)
    resp.raise_for_status()
    return {m["id"]: m.get("contextWindow") for m in resp.json().get("data", [])}


def print_table(aws, gw):
    """並排列出兩來源；both 存在時標示是否一致。"""
    ids = sorted(set(aws) | set(gw))
    width = max((len(i) for i in ids), default=10)

    header = f"{'model':<{width}}  {'AWS maxInput':>14}  {'gateway ctx':>14}  match"
    print(header)
    print("-" * len(header))

    mismatches = 0
    for mid in ids:
        a = aws.get(mid)
        g = gw.get(mid)
        # 只有兩邊都有值才比對；任一為 None（例如 grok-* 不在 AWS）標 n/a
        if a is None or g is None:
            match = "n/a"
        elif a == g:
            match = "OK"
        else:
            match = "MISMATCH"
            mismatches += 1
        arrow = " <==" if mid in HIGHLIGHT else ""
        print(f"{mid:<{width}}  {_fmt(a):>14}  {_fmt(g):>14}  {match}{arrow}")

    print()
    if mismatches:
        print(f"[!] {mismatches} 個模型 AWS 與 gateway 不一致 — gateway 可能在用 fallback 或未刷新")
    else:
        print("[OK] 所有可比對模型：gateway contextWindow == AWS maxInputTokens（動態透傳無誤）")


def main():
    parser = argparse.ArgumentParser(description="核對模型 context window（AWS 源頭 vs 本機 gateway）")
    parser.add_argument("--aws", action="store_true", help="只打 AWS 源頭")
    parser.add_argument("--gateway", action="store_true", help="只打本機 gateway")
    args = parser.parse_args()

    do_aws = args.aws or not args.gateway
    do_gw = args.gateway or not args.aws

    aws = {}
    gw = {}

    if do_gw:
        try:
            gw = fetch_gateway()
            print(f"[gateway] 取得 {len(gw)} 個模型：{GATEWAY_URL}")
        except Exception as e:
            print(f"[gateway] 失敗：{e}")

    if do_aws:
        try:
            aws = asyncio.run(fetch_aws())
            print(f"[AWS]     取得 {len(aws)} 個模型：{get_list_models_host()}")
        except Exception as e:
            print(f"[AWS]     失敗：{e}")
            print("          （token 過期請重登 Kiro；若 AccessDeniedException 代表識別 header 有誤）")

    print()
    print_table(aws, gw)


if __name__ == "__main__":
    main()
