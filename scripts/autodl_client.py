#!/usr/bin/env python3
"""AutoDL helper: login check, wallet, instance list, and 2080-oriented deploy hints.

Official open API can only create Pro SKUs (4090 / 3090 / ...). RTX 2080 lives
on the standard marketplace and must be created in the web console. Once a
2080 instance exists, this client can discover it and print SSH details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

API_HOST = "https://api.autodl.com"
WEB_HOST = "https://www.autodl.com"


class AutoDLError(RuntimeError):
    pass


def token_from_env() -> str:
    token = (os.environ.get("AUTODL_TOKEN") or "").strip()
    if not token:
        raise AutoDLError(
            "AUTODL_TOKEN is missing. Add the developer token from "
            "AutoDL 控制台 → 账号 → 设置 → 开发者 Token."
        )
    return token


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def api_request(
    method: str,
    path: str,
    token: str,
    *,
    host: str = API_HOST,
    json_body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    url = host.rstrip("/") + path
    resp = requests.request(
        method,
        url,
        headers=_headers(token),
        json=json_body,
        timeout=timeout,
    )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise AutoDLError(f"{url} returned non-JSON ({resp.status_code}): {resp.text[:300]}") from exc
    if resp.status_code >= 400:
        raise AutoDLError(f"{url} HTTP {resp.status_code}: {payload}")
    return payload


def check_login(token: str) -> dict[str, Any]:
    wallet = api_request("POST", "/api/v1/dev/wallet/balance", token)
    if wallet.get("code") not in {"Success", "success", 0, "0", None} and wallet.get("data") is None:
        raise AutoDLError(f"wallet/balance failed: {wallet}")
    return wallet


def list_pro_instances(token: str, page_size: int = 50) -> list[dict[str, Any]]:
    payload = api_request(
        "POST",
        "/api/v1/dev/instance/pro/list",
        token,
        json_body={"page_index": 1, "page_size": page_size},
    )
    data = payload.get("data") or {}
    return list(data.get("list") or [])


def pro_snapshot(token: str, instance_uuid: str) -> dict[str, Any]:
    payload = api_request(
        "GET",
        "/api/v1/dev/instance/pro/snapshot",
        token,
        json_body={"instance_uuid": instance_uuid},
    )
    return payload.get("data") or payload


def try_standard_instance_list(token: str) -> list[dict[str, Any]]:
    """Best-effort scan of undocumented web endpoints for standard (2080) boxes."""
    candidates = [
        ("POST", "/api/v1/dev/instance/list", WEB_HOST, {"page_index": 1, "page_size": 50}),
        ("POST", "/api/v1/instance", WEB_HOST, {"page_index": 1, "page_size": 50}),
        ("GET", "/api/v1/dev/instance", WEB_HOST, None),
    ]
    found: list[dict[str, Any]] = []
    for method, path, host, body in candidates:
        try:
            payload = api_request(method, path, token, host=host, json_body=body)
        except (AutoDLError, requests.RequestException) as exc:
            print(f"[autodl] skip {host}{path}: {exc}", file=sys.stderr)
            continue
        data = payload.get("data")
        rows: list[Any]
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("list") or data.get("items") or []
        else:
            rows = []
        if rows:
            print(f"[autodl] {host}{path} returned {len(rows)} row(s)")
            found.extend(row for row in rows if isinstance(row, dict))
            break
    return found


def looks_like_2080(row: dict[str, Any]) -> bool:
    blob = json.dumps(row, ensure_ascii=False).lower()
    return "2080" in blob


def yuan(millis: Any) -> str:
    try:
        return f"{int(millis) / 1000:.3f}"
    except (TypeError, ValueError):
        return str(millis)


def cmd_status(_: argparse.Namespace) -> int:
    token = token_from_env()
    wallet = check_login(token)
    data = wallet.get("data") or {}
    print("AutoDL login: OK")
    print(f"  balance_yuan:        {yuan(data.get('assets'))}")
    print(f"  voucher_yuan:        {yuan(data.get('voucher_balance'))}")
    print(f"  accumulate_yuan:     {yuan(data.get('accumulate'))}")

    pro = list_pro_instances(token)
    print(f"Pro instances: {len(pro)}")
    for row in pro:
        print(
            f"  - {row.get('uuid')}  status={row.get('status')}  "
            f"gpu={row.get('gpu_spec_uuid')}  name={row.get('name')}"
        )

    standard = try_standard_instance_list(token)
    print(f"Standard instances (best-effort): {len(standard)}")
    matches = [row for row in standard if looks_like_2080(row)]
    for row in matches or standard:
        print(f"  - {json.dumps(row, ensure_ascii=False)[:400]}")

    if not matches:
        print(
            "\nNo RTX 2080 instance is visible through the official Pro API.\n"
            "Create one in the web console: https://www.autodl.com/market/list\n"
            "Filter GPU = RTX 2080 / 2080 Super, image = PyTorch CUDA 11.8+, 1 GPU.\n"
            "After it is running, re-run: python scripts/autodl_client.py status"
        )
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    token = token_from_env()
    snap = pro_snapshot(token, args.instance_uuid)
    host = snap.get("proxy_host")
    port = snap.get("ssh_port")
    password = snap.get("root_password")
    command = snap.get("ssh_command") or (f"ssh -p {port} root@{host}" if host and port else "")
    print(json.dumps(
        {
            "ssh_command": command,
            "host": host,
            "port": port,
            "password": password,
            "jupyter": snap.get("jupyter_domain"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoDL login / instance helper")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Verify token, print wallet and instances")
    ssh = sub.add_parser("ssh", help="Print SSH details for a Pro instance UUID")
    ssh.add_argument("instance_uuid")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.cmd == "status":
            return cmd_status(args)
        if args.cmd == "ssh":
            return cmd_ssh(args)
    except AutoDLError as exc:
        print(f"[autodl] {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
