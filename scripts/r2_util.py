#!/usr/bin/env python3
"""Cloudflare R2 helper — probe, upload, download, list.

Subcommands:
    probe                                     list bucket + put/get/delete a tiny object
    upload   SRC DEST [--include PATTERN ...] upload matching files in SRC to DEST prefix
    download SRC_PREFIX DEST_DIR [--include PATTERN ...] download objects under SRC_PREFIX to DEST_DIR
    list     PREFIX                           list keys under PREFIX

Env vars required:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    R2_ENDPOINT
    R2_BUCKET
    R2_PREFIX   (probe only; defaults to "vast-checkpoint")
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def _client():
    missing = [k for k in ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing env vars: {', '.join(missing)}")
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


def cmd_probe(_args):
    bucket = os.environ["R2_BUCKET"]
    prefix = os.environ.get("R2_PREFIX", "vast-checkpoint")
    s3 = _client()

    print(f"[1/3] head_bucket({bucket})")
    s3.head_bucket(Bucket=bucket)
    print("  ok")

    key = f"{prefix}/_probe_{int(time.time())}.txt"
    body = f"r2 probe {time.strftime('%FT%TZ', time.gmtime())}".encode()
    print(f"[2/3] put_object → s3://{bucket}/{key}")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    got = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if got != body:
        sys.exit("round-trip mismatch")
    print("  put+get OK")

    print(f"[3/3] delete_object → s3://{bucket}/{key}")
    s3.delete_object(Bucket=bucket, Key=key)
    print("  ok")


def cmd_upload(args):
    bucket = os.environ["R2_BUCKET"]
    src = Path(args.src).resolve()
    if not src.is_dir():
        sys.exit(f"src not a directory: {src}")
    dest_prefix = args.dest.rstrip("/")
    patterns = args.include or ["*"]

    s3 = _client()
    uploaded = 0
    skipped = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if not any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            skipped += 1
            continue
        rel = path.relative_to(src).as_posix()
        key = f"{dest_prefix}/{rel}"
        size_mb = path.stat().st_size / 1e6
        print(f"  → s3://{bucket}/{key}  ({size_mb:.1f} MB)")
        try:
            s3.upload_file(str(path), bucket, key)
        except ClientError as e:
            sys.exit(f"upload failed for {path}: {e}")
        uploaded += 1

    print(f"uploaded {uploaded} file(s), skipped {skipped} non-matching")
    if uploaded == 0:
        sys.exit("no files matched include patterns")


def cmd_download(args):
    bucket = os.environ["R2_BUCKET"]
    src_prefix = args.src.rstrip("/") + "/"
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    patterns = args.include or ["*"]
    flat = args.flat

    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")

    downloaded = 0
    skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            base = os.path.basename(key)
            if not base:
                continue
            if not any(fnmatch.fnmatch(base, pat) for pat in patterns):
                skipped += 1
                continue
            if flat:
                local = dest / base
            else:
                rel = key[len(src_prefix):]
                local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            size_mb = obj.get("Size", 0) / 1e6
            print(f"  ← s3://{bucket}/{key}  ({size_mb:.1f} MB) → {local}")
            try:
                s3.download_file(bucket, key, str(local))
            except ClientError as e:
                sys.exit(f"download failed for {key}: {e}")
            downloaded += 1

    print(f"downloaded {downloaded} file(s), skipped {skipped} non-matching")
    if downloaded == 0:
        sys.exit(f"no files matched under s3://{bucket}/{src_prefix}")


def cmd_list(args):
    bucket = os.environ["R2_BUCKET"]
    prefix = args.prefix.rstrip("/") + "/"
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    total_mb = 0.0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            size_mb = obj.get("Size", 0) / 1e6
            total_mb += size_mb
            print(f"  {obj['Key']}  ({size_mb:.1f} MB)")
            n += 1
    print(f"total: {n} object(s), {total_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Cloudflare R2 helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="list bucket + put/get/delete a tiny object")

    up = sub.add_parser("upload", help="upload matching files to R2")
    up.add_argument("src", help="source directory")
    up.add_argument("dest", help="dest prefix (no leading slash, no s3:// prefix)")
    up.add_argument("--include", action="append", default=[], help="glob pattern on basename, repeatable")

    dn = sub.add_parser("download", help="download objects under a prefix from R2")
    dn.add_argument("src", help="source prefix in R2 (no leading slash, no s3:// prefix)")
    dn.add_argument("dest", help="local destination directory")
    dn.add_argument("--include", action="append", default=[],
                    help="glob pattern on basename, repeatable (default: *)")
    dn.add_argument("--flat", action="store_true",
                    help="flatten nested keys into dest dir (drop subdirs)")

    ls = sub.add_parser("list", help="list keys under a prefix")
    ls.add_argument("prefix", help="prefix to list (no leading slash, no s3:// prefix)")

    args = parser.parse_args()
    if args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "upload":
        cmd_upload(args)
    elif args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
