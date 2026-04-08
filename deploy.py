"""
deploy.py — push this project to Hugging Face Spaces.

Reads HUGGINGFACE_TOKEN from .env.
Usage: python deploy.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

TOKEN     = os.environ["HUGGINGFACE_TOKEN"]
REPO_NAME = "warehouse-fulfillment-env"
FILES     = [
    "app.py", "env.py", "models.py", "rewards.py",
    "tasks.py", "baseline.py", "openenv.yaml",
    "requirements.txt", "Dockerfile", "README.md",
]

api = HfApi(token=TOKEN)
try:
    username = api.whoami()["name"]
except Exception as e:
    raise SystemExit(f"Token error: {e}\nFix: generate a token at https://huggingface.co/settings/tokens with 'write' scope.")
repo_id  = f"{username}/{REPO_NAME}"

print(f"Creating/updating Space: {repo_id}")
try:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=TOKEN,
    )
except Exception as e:
    raise SystemExit(
        f"Cannot create Space: {e}\n"
        "Fix: go to https://huggingface.co/settings/tokens\n"
        "  → New token → Role: 'write' → copy into .env as HUGGINGFACE_TOKEN"
    )

root = Path(__file__).parent
for fname in FILES:
    fpath = root / fname
    if fpath.exists():
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="space",
            token=TOKEN,
        )
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} not found, skipping")

print(f"\nDeployed → https://huggingface.co/spaces/{repo_id}")
