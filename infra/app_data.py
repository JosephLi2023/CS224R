"""Modal app: WebShop dataset acquisition into the shared Volume.

Day 2 of the plan: bake the WebShop product index into `/vol/data/webshop/`
so subsequent training runs (Day 4+) can mount the Volume and find the
index without re-downloading.

Strategy:
  1. Shallow-clone princeton-nlp/WebShop into `/vol/code/webshop` (~50 MB
     of Python source + scripts; cached so we only do it once).
  2. Parse the upstream `setup.sh` to discover the canonical gdrive file IDs
     (avoids us hard-coding IDs that could rot).
  3. Use `gdown` to pull just the JSON data files we need into
     `/vol/data/webshop/`. Skip files that already exist (idempotent).
  4. `verify_webshop_data()` reports each file's size + record count, so we
     can confirm the index landed without re-downloading.

Cost-saving knobs:
  - `small=True` (default): pulls `items_shuffle_1000.json` (~5 MB, 1000
    products) for dev iteration. Use this for first-run verification.
  - `small=False`: pulls the full `items_shuffle.json` (~1.4 GB, 1.18 M
    products) needed for the proposal §3.3 protocol runs.

This app intentionally does NOT install pyserini/Java/spacy yet — those
heavy deps are only needed when we actually instantiate `WebAgentTextEnv`
(Day 3-4 work). Keeping data acquisition cheap and fast.
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

APP_NAME = "cs224r-hgpo-data"
app = modal.App(APP_NAME)

# Lighter-weight image variant — gdown is baked into the base image
# (see infra/image.py) so we just reuse `image` directly here.
data_image = image

WEBSHOP_REPO = "https://github.com/princeton-nlp/WebShop.git"
WEBSHOP_REPO_DIR = "/vol/code/webshop"
WEBSHOP_DATA_DIR = "/vol/data/webshop"

# Files we ever want from the upstream `setup.sh`.
# `setup.sh -d small` writes items_shuffle_1000.json + items_ins_v2_1000.json +
#                            items_human_ins.json
# `setup.sh -d all`   writes items_shuffle.json      + items_ins_v2.json +
#                            items_human_ins.json (1.18 M product index).
_COMMON_FILES = {"items_human_ins.json"}
_SMALL_FILES = _COMMON_FILES | {"items_shuffle_1000.json", "items_ins_v2_1000.json"}
_FULL_FILES = _COMMON_FILES | {"items_shuffle.json", "items_ins_v2.json"}


def _parse_gdown_ids(setup_sh_text: str) -> dict[str, dict[str, str]]:
    """Extract `{filename: {"gid": ..., "split": "small"|"all"|"common"}}` from setup.sh.

    The upstream WebShop setup.sh has no `-O` flag on its gdown lines; the
    intended filename lives in the trailing `# comment`. Bash `if/elif/else`
    determines whether a line belongs to the small or all split. We track
    that branch by walking the file line-by-line.
    """
    import re

    gdown_pat = re.compile(
        r"gdown\s+(?:https?://drive\.google\.com/uc\?id=)?([\w-]{10,});?\s*#\s*(\S+)"
    )

    out: dict[str, dict[str, str]] = {}
    in_small = False
    in_all = False
    for raw_line in setup_sh_text.splitlines():
        line = raw_line.strip()
        if 'if [ "$data" == "small" ]' in line:
            in_small, in_all = True, False
            continue
        if 'elif [ "$data" == "all" ]' in line:
            in_small, in_all = False, True
            continue
        if line.startswith("else") or line == "fi":
            in_small = in_all = False
            continue

        m = gdown_pat.search(line)
        if not m:
            continue
        gid, name_hint = m.group(1), m.group(2)
        # The comment hint is the bare filename (no extension); upstream JSON
        # files always end in .json.
        fname = name_hint.split()[0]
        if not fname.endswith(".json"):
            fname = fname + ".json"
        split = "small" if in_small else ("all" if in_all else "common")
        out[fname] = {"gid": gid, "split": split}
    return out


@app.function(
    image=data_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=10 * 60,
)
def show_setup_sh() -> str:
    """Dump the cloned upstream setup.sh so we can inspect its gdown format."""
    import os

    setup_path = os.path.join(WEBSHOP_REPO_DIR, "setup.sh")
    if not os.path.isfile(setup_path):
        return f"setup.sh not found at {setup_path}; run download_webshop_data first to clone the repo."
    with open(setup_path) as f:
        return f.read()


@app.function(
    image=data_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,  # 1 hr; the full split takes ~10-20 min on a fast link.
)
def download_webshop_data(small: bool = True) -> dict:
    """Idempotent WebShop data acquisition into the shared Volume.

    Args:
        small: If True (default), download only the 1000-product dev split
               (~5 MB total). If False, download the full 1.18 M product
               index (~1.4 GB).
    Returns:
        Manifest with which files were downloaded vs already cached.
    """
    import os
    import shutil
    import subprocess

    os.makedirs(WEBSHOP_DATA_DIR, exist_ok=True)

    # ---- 1. Shallow-clone the WebShop repo (so we always have setup.sh + py code) ----
    if not os.path.isdir(os.path.join(WEBSHOP_REPO_DIR, ".git")):
        if os.path.isdir(WEBSHOP_REPO_DIR):
            shutil.rmtree(WEBSHOP_REPO_DIR)
        os.makedirs(os.path.dirname(WEBSHOP_REPO_DIR), exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", WEBSHOP_REPO, WEBSHOP_REPO_DIR]
        )
    else:
        print(f"WebShop repo already cloned at {WEBSHOP_REPO_DIR}")

    # ---- 2. Parse setup.sh for gdrive IDs ----
    setup_path = os.path.join(WEBSHOP_REPO_DIR, "setup.sh")
    if not os.path.isfile(setup_path):
        raise FileNotFoundError(
            f"setup.sh not found at {setup_path}; the upstream repo layout may have changed."
        )
    with open(setup_path) as f:
        setup_text = f.read()
    fname_to_meta = _parse_gdown_ids(setup_text)
    if not fname_to_meta:
        raise RuntimeError(
            "Failed to parse any gdown commands from setup.sh. "
            "First 500 chars of setup.sh follow:\n" + setup_text[:500]
        )
    print(f"Parsed gdrive entries from setup.sh: "
          f"{[(f, m['split']) for f, m in fname_to_meta.items()]}")

    # ---- 3. Filter to wanted files + download ----
    wanted = _SMALL_FILES if small else _FULL_FILES
    downloaded: list[str] = []
    skipped: list[str] = []
    missing_in_setup: list[str] = []

    for fname in sorted(wanted):
        dst = os.path.join(WEBSHOP_DATA_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  skip {fname} (already in volume, {os.path.getsize(dst)} bytes)")
            skipped.append(fname)
            continue
        if fname not in fname_to_meta:
            print(f"  WARNING: {fname} not present in setup.sh gdown list; skipping.")
            missing_in_setup.append(fname)
            continue
        gid = fname_to_meta[fname]["gid"]
        url = f"https://drive.google.com/uc?id={gid}"
        print(f"  downloading {fname} (gdrive id {gid}) -> {dst}")
        # Use positional URL form (matches upstream setup.sh; --id flag removed in gdown 5.x).
        subprocess.check_call(["gdown", url, "-O", dst, "--quiet"])
        downloaded.append(fname)

    # Persist Volume changes so a subsequent run sees the data.
    volume.commit()

    return {
        "data_dir": WEBSHOP_DATA_DIR,
        "small_split": small,
        "downloaded": downloaded,
        "skipped": skipped,
        "missing_in_setup": missing_in_setup,
        "all_files": sorted(os.listdir(WEBSHOP_DATA_DIR)),
        "repo_dir": WEBSHOP_REPO_DIR,
    }


@app.function(
    image=data_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=10 * 60,
)
def verify_webshop_data() -> dict:
    """Inspect /vol/data/webshop/ — report each file's size + record count.

    Use as: `modal run infra/app_data.py::verify_webshop_data`
    """
    import json
    import os

    if not os.path.isdir(WEBSHOP_DATA_DIR):
        return {"error": f"{WEBSHOP_DATA_DIR} does not exist; run download_webshop_data first."}

    out: dict = {"data_dir": WEBSHOP_DATA_DIR, "files": []}
    for fname in sorted(os.listdir(WEBSHOP_DATA_DIR)):
        path = os.path.join(WEBSHOP_DATA_DIR, fname)
        if not os.path.isfile(path):
            continue
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
        record: dict = {"name": fname, "size_mb": size_mb}
        if fname.endswith(".json"):
            try:
                with open(path) as fh:
                    data = json.load(fh)
                record["n_records"] = len(data) if hasattr(data, "__len__") else "?"
                if isinstance(data, list) and data:
                    record["sample_keys"] = (
                        sorted(data[0].keys())[:8] if isinstance(data[0], dict) else None
                    )
            except (json.JSONDecodeError, OSError) as exc:
                record["parse_error"] = repr(exc)
        out["files"].append(record)
    return out


@app.local_entrypoint()
def main(action: str = "verify", small: bool = True) -> None:
    """`modal run infra/app_data.py --action <download|verify> --small <True|False>`"""
    if action == "download":
        result = download_webshop_data.remote(small=small)
    elif action == "verify":
        result = verify_webshop_data.remote()
    elif action == "show_setup":
        result = show_setup_sh.remote()
    else:
        raise ValueError(f"Unknown action: {action!r} (expected 'download'/'verify'/'show_setup')")

    import json as _json
    print(_json.dumps(result, indent=2, default=str))
