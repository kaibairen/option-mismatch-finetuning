#!/usr/bin/env python3
"""Download the small Instruct checkpoints used for the 100-sample V100 probe."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def download_via_hf(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    endpoints = []
    current = os.environ.get("HF_ENDPOINT")
    if current:
        endpoints.append(current)
    for url in ("https://huggingface.co", "https://hf-mirror.com"):
        if url not in endpoints:
            endpoints.append(url)

    last_error: Exception | None = None
    for endpoint in endpoints:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[hf] {repo_id} via {endpoint} -> {dest}", flush=True)
        try:
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest),
                    resume_download=True,
                    token=token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            print(f"[hf] {endpoint} failed: {exc}", file=sys.stderr, flush=True)
    raise RuntimeError(f"huggingface download failed for {repo_id}") from last_error


def download_via_modelscope(repo_id: str, dest: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[ms] {repo_id} -> {dest}", flush=True)
    path = snapshot_download(repo_id, local_dir=str(dest))
    return Path(path)


def download_one(repo_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "config.json"
    if marker.exists():
        print(f"[skip] already present {dest}", flush=True)
        return dest
    prefer = (os.environ.get("MODEL_SOURCE") or "auto").lower()
    errors: list[str] = []
    order = ("modelscope", "hf") if prefer in {"auto", "modelscope"} else ("hf", "modelscope")
    if prefer in {"hf", "modelscope"}:
        order = (prefer,)
    for source in order:
        try:
            if source == "modelscope":
                return download_via_modelscope(repo_id, dest)
            return download_via_hf(repo_id, dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")
            print(f"[{source}] failed: {exc}", file=sys.stderr, flush=True)
    raise RuntimeError(f"failed to download {repo_id}: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["0.5b", "1.5b"], choices=sorted(MODELS))
    parser.add_argument("--dest-root", type=Path, default=Path(os.environ.get("MODEL_ROOT", "models")))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[hf] warning: HF_TOKEN unset; public download may be rate-limited", file=sys.stderr)

    for key in args.models:
        repo_id = MODELS[key]
        dest = args.dest_root / repo_id.split("/")[-1]
        out = download_one(repo_id, dest)
        print(f"[hf] ready {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
