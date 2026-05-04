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
    elif action == "show_layout":
        result = show_webshop_layout.remote()
    elif action == "show_human_ins":
        result = show_human_ins_schema.remote()
    elif action == "download_human_trajs":
        result = download_human_trajectories.remote()
    elif action == "show_human_trajs":
        result = show_human_trajs_schema.remote()
    elif action == "show_one_traj":
        result = show_one_trajectory_full.remote()
    else:
        raise ValueError(f"Unknown action: {action!r} (expected 'download'/'verify'/'show_setup'/'show_layout')")

    import json as _json
    print(_json.dumps(result, indent=2, default=str))


@app.function(image=data_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def show_webshop_layout() -> dict:
    """Inspect /vol/code/webshop layout + requirements.txt for Track B."""
    import os
    base = WEBSHOP_REPO_DIR
    out = {"base": base, "exists": os.path.isdir(base)}
    if not out["exists"]:
        return out
    out["top_level"] = sorted(os.listdir(base))
    req = os.path.join(base, "requirements.txt")
    if os.path.isfile(req):
        with open(req) as f:
            out["requirements_txt"] = f.read()
    se = os.path.join(base, "search_engine")
    if os.path.isdir(se):
        out["search_engine"] = sorted(os.listdir(se))
        for fname in ("convert_product_file_format.py", "run_indexing.sh"):
            p = os.path.join(se, fname)
            if os.path.isfile(p):
                with open(p) as f:
                    out[f"se_{fname}"] = f.read()[:2000]
    pkg = os.path.join(base, "web_agent_site")
    if os.path.isdir(pkg):
        out["web_agent_site"] = sorted(os.listdir(pkg))
        envs = os.path.join(pkg, "envs")
        if os.path.isdir(envs):
            out["web_agent_site_envs"] = sorted(os.listdir(envs))
    return out


@app.function(image=data_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def show_human_ins_schema() -> dict:
    """Inspect items_human_ins.json to confirm whether it's full trajectories
    or just instructions."""
    import json
    p = "/vol/data/webshop/items_human_ins.json"
    with open(p) as f:
        data = json.load(f)
    out: dict = {"path": p, "type": type(data).__name__}
    if isinstance(data, list):
        out["n_records"] = len(data)
        out["sample_first"] = data[0] if data else None
        if data and isinstance(data[0], dict):
            out["sample_keys"] = sorted(list(data[0].keys()))
    elif isinstance(data, dict):
        keys = list(data.keys())[:5]
        out["n_top_keys"] = len(data)
        out["first_5_keys"] = keys
        if keys:
            sv = data[keys[0]]
            out["sample_value_type"] = type(sv).__name__
            if isinstance(sv, list) and sv:
                out["sample_value_first"] = sv[0] if not isinstance(sv[0], dict) else {k: type(v).__name__ for k, v in sv[0].items()}
            elif isinstance(sv, dict):
                out["sample_value_keys"] = sorted(list(sv.keys()))
            else:
                out["sample_value_preview"] = str(sv)[:300]
    return out


@app.function(image=data_image, volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def download_human_trajectories() -> dict:
    """Download the released human-trajectory gdrive folder used by
    setup.sh::get_human_trajs(). ~50 example trajectories the WebShop
    paper authors made public for IL warm-start.

    URL is hard-coded in WebShop's setup.sh (verified).
    """
    import os
    import gdown  # type: ignore[import-not-found]
    dst = "/vol/data/webshop/human_trajs"
    os.makedirs(dst, exist_ok=True)
    folder_url = "https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
    print(">>> gdown.download_folder", folder_url)
    files = gdown.download_folder(folder_url, output=dst, quiet=True)
    volume.commit()
    listing = sorted(os.listdir(dst))
    return {"dst": dst, "n_files_returned": len(files) if files else 0, "n_files_on_disk": len(listing), "first_5": listing[:5]}


@app.function(image=data_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def show_human_trajs_schema() -> dict:
    """Inspect the downloaded human-trajectory folder so we can build the
    SFT dataset loader against the right schema."""
    import json
    import os
    base = "/vol/data/webshop/human_trajs"
    if not os.path.isdir(base):
        return {"error": f"{base} does not exist; run download_human_trajectories first"}
    files = sorted(os.listdir(base))
    sample_files = []
    for f in files[:3]:
        p = os.path.join(base, f)
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as fh:
                content = fh.read()
            head = content[:600]
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    sample_files.append({"file": f, "len": len(content), "type": "list", "n_items": len(parsed), "first_item_keys": sorted(list(parsed[0].keys())) if parsed and isinstance(parsed[0], dict) else None, "first_item_preview": str(parsed[0])[:300] if parsed else None})
                elif isinstance(parsed, dict):
                    sample_files.append({"file": f, "len": len(content), "type": "dict", "top_keys": sorted(list(parsed.keys()))[:10]})
            except Exception:
                sample_files.append({"file": f, "len": len(content), "type": "text", "head": head})
        except Exception as e:
            sample_files.append({"file": f, "error": repr(e)})
    return {"base": base, "n_files": len(files), "first_10": files[:10], "sample_files": sample_files}


@app.function(image=data_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def show_one_trajectory_full() -> dict:
    """Dump ONE complete trajectory file (all rows) so we can build the loader."""
    import json
    import os
    base = "/vol/data/webshop/human_trajs"
    files = sorted(f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f)))
    if not files:
        return {"error": "no trajectory files on disk"}
    p = os.path.join(base, files[0])
    rows = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                rows.append({"_parse_error": repr(e), "raw": line[:200]})
    row_summaries = []
    for r in rows:
        if isinstance(r, dict):
            row_summaries.append({k: (str(v)[:120] if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in r.items()})
    return {
        "file": files[0],
        "total_files": len(files),
        "first_3_filenames": files[:3],
        "n_rows_in_file": len(rows),
        "row_summaries": row_summaries,
    }
