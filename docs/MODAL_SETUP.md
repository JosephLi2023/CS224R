# Modal Setup Walkthrough — CS224R H-GRPO

End-to-end setup from zero to a successful A100 smoke test, in order. Each
step labels who acts:

- 🟢 **YOU** — must be done by you in your browser/terminal (auth, billing).
- 🟦 **AGENT** — Devmate can run for you.

Estimated wall-clock: ~30 minutes for the human-input steps (account
creation + token), ~10 minutes for first image build (cached forever after).

---

## 0. Prereqs

- Local dev environment already bootstrapped (`bash scripts/bootstrap_local.sh`).
  This installed the `modal` Python client into `.venv`.
- Working directory: `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R`
- Activate the venv before any Modal command:
  ```bash
  cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
  source .venv/bin/activate
  ```

---

## 1. 🟢 Create your Modal account

Modal is the cloud GPU provider used by the course's $500 student credit. Once
created, the credit is automatically applied (per CS224R guidelines §2.9).

1. Go to **<https://modal.com/signup>**
2. Sign up with your `@stanford.edu` email (required for the student credit).
3. Verify the email; you'll be redirected to a workspace dashboard.
4. **Note your workspace name** (top-left of the dashboard, e.g. `shoupeili`).
   Modal URLs will use it: `https://<workspace>--<app>.modal.run`.

---

## 2. 🟢 Authenticate the local CLI

```bash
source .venv/bin/activate
modal token new
```

Expected behavior:
- A browser tab opens to `https://modal.com/token`.
- Click **"Approve"** in the browser.
- Terminal prints: `✓ Token verified successfully` and writes `~/.modal.toml`.

Verify:
```bash
modal token current
# Expected: prints workspace + token id
```

Troubleshoot:
- *"command not found: modal"* → venv isn't activated. Re-run `source .venv/bin/activate`.
- *"Token rejected"* → run `modal token new` again; sometimes the initial token expires before approval.

---

## 3. 🟦 Verify the image builds (CPU smoke)

This is the cheapest possible test. Builds the image (~10 min the first
time, cached afterward) and runs a CPU function that writes a sentinel file
to the shared Volume.

```bash
modal run infra/app_train.py::hello
```

Expected output (last few lines):
```
hello from modal at 2026-05-03T...
workspace contents: ['CS224R_Project_Proposal.pdf', 'configs', 'src', ...]
✓ App completed.
```

Side effects:
- Creates Volume `cs224r-hgpo-vol` (visible in Modal dashboard → Volumes).
- Writes `/vol/hello.sentinel` inside the Volume.
- App `cs224r-hgpo-train` appears in dashboard → Apps.

Cost: ~$0 (CPU only, <1 minute).

**🚨 If the image build fails**, the most common causes are:
- `vllm==0.6.3.post1` wheel can't find a matching torch — bump pinned versions in `infra/image.py`.
- `apt_install` package missing — drop it from `_WEBSHOP_INSTALL` if not needed yet.

---

## 4. 🟦 Verify A100 + CUDA + libraries

This is the first paid call (~$0.05 for ~1 minute on A100).

```bash
modal run infra/app_train.py::env_probe
```

Expected output (the printed dict):
```python
{
  'torch': '2.4.1',
  'cuda_available': True,
  'cuda_device': 'NVIDIA A100-SXM4-80GB',
  'cuda_capability': (8, 0),
  'bf16_supported': True,
  'transformers': '4.45.2',
  'peft': '0.13.2',
  'trl': '0.11.4',
  'vllm': '0.6.3.post1',
  'accelerate': '0.34.2'
}
```

If all 10 keys are populated and `cuda_available == True`, the trainer can be
built on top of this image.

---

## 5. 🟦 (Optional) Run the toy trainer on A100

A 30-second sanity check that our existing scaffold runs end-to-end inside
Modal. Uses the toy bandit env (no LLM), so cost is negligible.

```bash
modal run infra/app_train.py::train --train-config configs/baseline_train.json
```

This proves the train function dispatches correctly. The real LLM trainer
replaces the underlying `src.trainers.train` import on Day 6 of Week 1.

---

## 6. 🟦 Manage the Volume

Volume contains all run artifacts (logs, checkpoints, judge cache).

```bash
modal volume ls cs224r-hgpo-vol /             # list root
modal volume get cs224r-hgpo-vol /hello.sentinel ./hello.txt  # download
modal volume rm cs224r-hgpo-vol -r /manifests/<run>           # delete a run
```

The Volume persists across deploys and is shared by `app_train`,
`app_judge`, and `app_eval`.

---

## 7. 🟢 (Later, when needed) Add the OpenAI Secret

Skip until you have an OpenAI API key. Method A judge currently defaults to
the vLLM Qwen backend (`configs/method_hgpo_judge.json::judge.backend = "vllm"`).

When you're ready:

```bash
modal secret create openai-key OPENAI_API_KEY=sk-...
```

Then flip the config:
```bash
sed -i '' 's/"backend": "vllm"/"backend": "openai"/' configs/method_hgpo_judge.json
```

`infra/common.py::maybe_openai_secret()` will pick it up automatically.

---

## 8. Cost dashboard

- View live spend: <https://modal.com/settings/usage>
- Soft limits per app: set in dashboard → App → Settings → Spending limits.
- Plan budget table (target ~$200 of $500 credit) is in
  `~/.llms/plans/cs224r_hgpo_execution.plan.md` under "Compute budget".

---

## Quick reference — daily commands

```bash
# Activate env
source .venv/bin/activate

# Smoke
modal run infra/app_train.py::hello

# Probe GPU
modal run infra/app_train.py::env_probe

# Run a training job
modal run infra/app_train.py::train \
  --train-config configs/method_flat_grpo.json \
  --env-config configs/env_webshop.json \
  --eval-config configs/eval.json \
  --seed 11

# Tail logs of a running app
modal app logs cs224r-hgpo-train -f

# Stop everything (cost safety net)
modal app stop cs224r-hgpo-train
modal app stop cs224r-hgpo-judge
```
