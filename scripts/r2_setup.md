# R2 setup — Cloudflare R2 backend for `vast_run.sh`

Goal: ship results from vast.ai → Cloudflare R2 bucket `speech-separation` under prefix `vast-checkpoint/<run-name>/`.

`scripts/r2.env` is gitignored — it holds API credentials. Never commit.

## 1. Create bucket (one-time)

Cloudflare dashboard → R2 → Create bucket → name `speech-separation` → location `auto`.

If bucket already exists, skip.

## 2. Mint API token (one-time)

Cloudflare dashboard → R2 → Manage R2 API Tokens → Create API Token.

| Field | Value |
|---|---|
| Token name | `vast-checkpoint-rw` |
| Permissions | Object Read & Write |
| Specify bucket | `speech-separation` |
| TTL | as long as needed |

Save the `Access Key ID` and `Secret Access Key` — they show only once.

## 3. Populate `scripts/r2.env`

```
export AWS_ACCESS_KEY_ID=<access key id>
export AWS_SECRET_ACCESS_KEY=<secret access key>
export AWS_DEFAULT_REGION=auto
export R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
export R2_BUCKET=speech-separation
export R2_PREFIX="vast-checkpoint"
```

`<account_id>` is the 32-char hex from R2 → "Use R2 with APIs" panel.

`scripts/r2.env` is already in `.gitignore`. Do not commit.

## 4. Verify locally

Local needs `boto3` + `gdown` (in `.venv` or system Python):

```
pip install boto3 gdown
bash scripts/local_r2_test.sh
```

Expects: `head_bucket`, put/get/delete probe under `$R2_PREFIX/` all succeed, three Drive file IDs reachable.

## 5. Deploy to vast.ai

`vast_run.sh` reads `scripts/r2.env` automatically. Ship it with the repo:

```
scp scripts/r2.env vast-host:/root/<repo>/scripts/r2.env
# then on vast.ai
bash scripts/vast_run.sh
```

Or pre-export the same vars in the vast.ai shell before launching the script.

## Notes

- R2 has no egress fees — pulling results back to your laptop is free.
- Upload backend = `boto3` via `scripts/r2_util.py`. No `aws-cli` install needed.
- Override defaults at run time: `R2_BUCKET=other-bucket R2_PREFIX=other/path bash scripts/vast_run.sh`.
- Rotate the token in the Cloudflare dashboard after the job ends if creds were ever exposed.
