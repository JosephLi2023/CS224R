"""Modal app: install WebShop env + build BM25 search index into the Volume.

Entrypoints:
  modal run infra/app_webshop_install.py --action pip_install
  modal run infra/app_webshop_install.py --action download_spacy
  modal run infra/app_webshop_install.py --action build_index_1k
  modal run infra/app_webshop_install.py --action reset_smoke
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import webshop_image

app = modal.App("cs224r-hgpo-webshop-install")

WEBSHOP_REPO_DIR = "/vol/code/webshop"
WEBSHOP_DATA_DIR = "/vol/data/webshop"
WEBSHOP_PYUSER = "/vol/webshop_pyuser"   # PEP-370 user-site for editable install + spacy model
WEBSHOP_INDEX_DIR = "/vol/data/webshop/indexes_1k"
WEBSHOP_RESOURCES_DIR = "/vol/data/webshop/resources_1k"


@app.function(image=webshop_image, volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def pip_install_webshop() -> dict:
    """Install the cloned WebShop repo as editable into a Volume-resident
    user site so subsequent containers can `import web_agent_site` without
    re-installing. `--no-deps` keeps WebShop's legacy pins out of the
    modern stack."""
    import os, subprocess, sys
    os.makedirs(WEBSHOP_PYUSER, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = WEBSHOP_PYUSER
    cmd = [sys.executable, "-m", "pip", "install", "--user", "--no-deps", "-e", WEBSHOP_REPO_DIR]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    volume.commit()
    return {"pyuser": WEBSHOP_PYUSER, "site": os.listdir(WEBSHOP_PYUSER)}


@app.function(image=webshop_image, volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def download_spacy_model() -> dict:
    """Download spaCy en_core_web_lg (~600 MB) into the volume-resident user site."""
    import os, subprocess, sys
    os.makedirs(WEBSHOP_PYUSER, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = WEBSHOP_PYUSER
    cmd = [sys.executable, "-m", "spacy", "download", "en_core_web_lg"]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    volume.commit()
    return {"ok": True}


@app.function(image=webshop_image, volumes={VOLUME_MOUNT: volume}, timeout=30 * 60)
def build_index_1k() -> dict:
    """Build the BM25 lucene index for the 1000-product dev split.

    Uses items_shuffle_1000.json as DEFAULT_FILE_PATH by symlinking it as
    items_shuffle.json in the data dir (WebShop's load_products() reads from
    the canonical name). Output index lands at /vol/data/webshop/indexes_1k.
    """
    import os, subprocess, sys, shutil
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = WEBSHOP_PYUSER
    env["PYTHONPATH"] = "/workspace:" + env.get("PYTHONPATH", "")

    # Symlink small split as the canonical filename WebShop expects.
    src = os.path.join(WEBSHOP_DATA_DIR, "items_shuffle_1000.json")
    repo_data = os.path.join(WEBSHOP_REPO_DIR, "data")
    os.makedirs(repo_data, exist_ok=True)
    repo_canonical = os.path.join(repo_data, "items_shuffle.json")
    repo_attrs = os.path.join(repo_data, "items_ins_v2.json")
    if os.path.lexists(repo_canonical):
        os.remove(repo_canonical)
    os.symlink(src, repo_canonical)
    if os.path.lexists(repo_attrs):
        os.remove(repo_attrs)
    os.symlink(os.path.join(WEBSHOP_DATA_DIR, "items_ins_v2_1000.json"), repo_attrs)
    repo_human = os.path.join(repo_data, "items_human_ins.json")
    if not os.path.lexists(repo_human):
        os.symlink(os.path.join(WEBSHOP_DATA_DIR, "items_human_ins.json"), repo_human)
    # WebShop's DEFAULT_FILE_PATH directly references items_shuffle_1000.json
    # under the small-data convention; mirror every JSON we have so that
    # whichever default path the upstream picks at import time, it resolves.
    for fname in os.listdir(WEBSHOP_DATA_DIR):
        if not fname.endswith(".json"):
            continue
        link = os.path.join(repo_data, fname)
        if os.path.lexists(link):
            continue
        os.symlink(os.path.join(WEBSHOP_DATA_DIR, fname), link)

    se = os.path.join(WEBSHOP_REPO_DIR, "search_engine")
    for sub in ("resources", "resources_100", "resources_1k", "resources_100k", "indexes", "indexes_100", "indexes_1k", "indexes_100k"):
        os.makedirs(os.path.join(se, sub), exist_ok=True)

    print(">>> running convert_product_file_format.py")
    proc = subprocess.run(
        [sys.executable, "convert_product_file_format.py"],
        cwd=se, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("--- STDOUT ---")
        print(proc.stdout[-4000:])
        print("--- STDERR ---")
        print(proc.stderr[-4000:])
        raise RuntimeError(f"convert_product_file_format failed: rc={proc.returncode}")
    print(proc.stdout[-1000:])

    print(">>> running pyserini.index.lucene for resources_1k")
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "resources_1k",
        "--index", "indexes_1k",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    subprocess.check_call(cmd, cwd=se, env=env)

    # Move/copy index into Volume so subsequent containers see it.
    vol_dst = WEBSHOP_INDEX_DIR
    if os.path.exists(vol_dst):
        shutil.rmtree(vol_dst)
    shutil.copytree(os.path.join(se, "indexes_1k"), vol_dst)
    vol_resources = WEBSHOP_RESOURCES_DIR
    if os.path.exists(vol_resources):
        shutil.rmtree(vol_resources)
    shutil.copytree(os.path.join(se, "resources_1k"), vol_resources)
    volume.commit()
    return {
        "index_dir": vol_dst,
        "index_files": sorted(os.listdir(vol_dst)),
    }


@app.function(image=webshop_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def reset_smoke() -> dict:
    """Smoke: instantiate WebAgentTextEnv (small split) and call reset()."""
    import os, sys
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = WEBSHOP_PYUSER
    sys.path.insert(0, WEBSHOP_REPO_DIR)
    sys.path.insert(0, "/workspace")
    # Make WebShop's user-site importable in this process.
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    user_site = f"{WEBSHOP_PYUSER}/lib/{pyver}/site-packages"
    if os.path.isdir(user_site):
        sys.path.insert(0, user_site)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # type: ignore
    e = WebAgentTextEnv(observation_mode="text", num_products=1000)
    out = e.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    obs_text = obs if isinstance(obs, str) else str(obs)[:200]
    return {
        "obs_preview": obs_text[:300],
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
    }


@app.local_entrypoint()
def main(action: str = "pip_install") -> None:
    import json as _json
    fn = {
        "pip_install": pip_install_webshop,
        "download_spacy": download_spacy_model,
        "build_index_1k": build_index_1k,
        "build_index_full": build_index_full,
        "reset_smoke": reset_smoke,
    }.get(action)
    if fn is None:
        raise ValueError(f"Unknown action: {action!r}")
    print(_json.dumps(fn.remote(), indent=2, default=str))


@app.function(image=webshop_image, volumes={VOLUME_MOUNT: volume}, timeout=120 * 60)
def build_index_full() -> dict:
    """Build the BM25 lucene index for the full 1.18M product split.

    Symlinks `items_shuffle.json` (1.18M) + `items_ins_v2.json` as canonical
    filenames in repo/data, runs convert_product_file_format.py + pyserini
    indexing for `resources/` → `indexes/`. Persists to
    /vol/data/webshop/indexes_full.
    """
    import os, subprocess, sys, shutil
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = WEBSHOP_PYUSER
    env["PYTHONPATH"] = "/workspace:" + env.get("PYTHONPATH", "")

    repo_data = os.path.join(WEBSHOP_REPO_DIR, "data")
    os.makedirs(repo_data, exist_ok=True)

    # Repoint canonical names to the full files (1.18M products + attrs).
    full_shuffle = os.path.join(WEBSHOP_DATA_DIR, "items_shuffle.json")
    full_attrs = os.path.join(WEBSHOP_DATA_DIR, "items_ins_v2.json")
    repo_canonical = os.path.join(repo_data, "items_shuffle.json")
    repo_attrs = os.path.join(repo_data, "items_ins_v2.json")
    # WebShop's DEFAULT_FILE_PATH hard-codes `items_shuffle_1000.json`, so we
    # MUST also repoint that name (and the matching attrs name) to the full
    # file when we want to index the full split.
    repo_canonical_1k = os.path.join(repo_data, "items_shuffle_1000.json")
    repo_attrs_1k = os.path.join(repo_data, "items_ins_v2_1000.json")
    for link, target in [
        (repo_canonical, full_shuffle),
        (repo_attrs, full_attrs),
        (repo_canonical_1k, full_shuffle),
        (repo_attrs_1k, full_attrs),
    ]:
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    # human ins is shared.
    repo_human = os.path.join(repo_data, "items_human_ins.json")
    if not os.path.lexists(repo_human):
        os.symlink(os.path.join(WEBSHOP_DATA_DIR, "items_human_ins.json"), repo_human)

    se = os.path.join(WEBSHOP_REPO_DIR, "search_engine")
    for sub in ("resources", "resources_100", "resources_1k", "resources_100k", "indexes", "indexes_100", "indexes_1k", "indexes_100k"):
        os.makedirs(os.path.join(se, sub), exist_ok=True)

    print(">>> running convert_product_file_format.py (FULL 1.18M)")
    proc = subprocess.run(
        [sys.executable, "convert_product_file_format.py"],
        cwd=se, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("--- STDOUT ---"); print(proc.stdout[-4000:])
        print("--- STDERR ---"); print(proc.stderr[-4000:])
        raise RuntimeError(f"convert_product_file_format failed: rc={proc.returncode}")

    print(">>> running pyserini.index.lucene for resources (FULL)")
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "resources",
        "--index", "indexes",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    proc2 = subprocess.run(cmd, cwd=se, env=env, capture_output=True, text=True)
    if proc2.returncode != 0:
        print("--- STDOUT ---"); print(proc2.stdout[-4000:])
        print("--- STDERR ---"); print(proc2.stderr[-4000:])
        raise RuntimeError(f"pyserini indexing failed: rc={proc2.returncode}")
    print(proc2.stdout[-1500:])

    vol_dst = "/vol/data/webshop/indexes_full"
    if os.path.exists(vol_dst):
        shutil.rmtree(vol_dst)
    shutil.copytree(os.path.join(se, "indexes"), vol_dst)
    vol_resources = "/vol/data/webshop/resources_full"
    if os.path.exists(vol_resources):
        shutil.rmtree(vol_resources)
    shutil.copytree(os.path.join(se, "resources"), vol_resources)
    volume.commit()
    return {"index_dir": vol_dst, "n_files": len(os.listdir(vol_dst))}
