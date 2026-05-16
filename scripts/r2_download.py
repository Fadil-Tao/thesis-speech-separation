#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3"]
# ///
"""Download vast.ai run artifacts from R2 to local checkpoints/.

Mirrors:
    s3://$R2_BUCKET/$prefix/skim-attention-v3-reg-transfer/  → checkpoints/3speaker/skim-attention-v3-reg-transfer/
    s3://$R2_BUCKET/$prefix/skim-attention-v3/               → checkpoints/3speaker/skim-attention-v3/

Auto-loads scripts/r2.env if present.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    pat = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pat.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ.setdefault(k, v)


_load_env_file(Path(__file__).resolve().parent / "r2.env")


RUNS = {
    "reg":     ("skim-attention-v3-reg-transfer", "checkpoints/3speaker/skim-attention-v3-reg-transfer"),
    "vanilla": ("skim-attention-v3",              "checkpoints/3speaker/skim-attention-v3"),
}


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "virtual"},
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def download_run(s3, bucket: str, src_prefix: str, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    total_bytes = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(src_prefix.rstrip("/")) + 1:]
            if not rel:
                continue
            out = dest_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            size_mb = obj["Size"] / 1e6
            print(f"  ← s3://{bucket}/{key}  ({size_mb:.1f} MB)")
            s3.download_file(bucket, key, str(out))
            count += 1
            total_bytes += obj["Size"]
    print(f"  done: {count} file(s), {total_bytes/1e6:.1f} MB → {dest_dir}")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["reg", "vanilla", "both"], default="both")
    ap.add_argument("--prefix", default=os.environ.get("R2_PREFIX", "vast-checkpoint"))
    args = ap.parse_args()

    bucket = os.environ["R2_BUCKET"]
    s3 = _client()

    targets = ["reg", "vanilla"] if args.which == "both" else [args.which]
    project_root = Path(__file__).resolve().parents[1]

    total = 0
    for t in targets:
        run_name, dest_rel = RUNS[t]
        src_prefix = f"{args.prefix.rstrip('/')}/{run_name}"
        dest = project_root / dest_rel
        print(f"\n[{t}] s3://{bucket}/{src_prefix}/ → {dest}")
        total += download_run(s3, bucket, src_prefix, dest)

    if total == 0:
        sys.exit("no objects found at given prefix(es)")


if __name__ == "__main__":
    main()
