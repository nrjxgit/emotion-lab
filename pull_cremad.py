"""Sparse CREMA-D pull: 2 actors x 6 emotions subset via git-lfs (Windows-safe)."""

import os
import subprocess

ROOT = os.path.join("eval_data_cremad", "cremad_repo")
os.makedirs("eval_data_cremad", exist_ok=True)
log = []


def run(cmd, cwd="."):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=900)
    log.append(f"$ {' '.join(cmd)}\nrc={r.returncode}\n{(r.stdout + r.stderr)[-600:]}")
    return r.returncode


if not os.path.isdir(os.path.join(ROOT, ".git")):
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/CheyneyComputerScience/CREMA-D.git",
            "cremad_repo",
        ],
        cwd="eval_data_cremad",
    )
run(["git", "sparse-checkout", "set", "VideoFlash"], cwd=ROOT)
run(
    ["git", "lfs", "pull", "--include", "VideoFlash/1001_*,VideoFlash/1002_*"], cwd=ROOT
)
try:
    files = sorted(os.listdir(os.path.join(ROOT, "VideoFlash")))
    log.append(f"VideoFlash files: {len(files)}; first: {files[:6]}")
    sizes = [os.path.getsize(os.path.join(ROOT, "VideoFlash", f)) for f in files[:6]]
    log.append(f"sample sizes: {sizes} (131 bytes = unpulled LFS pointer)")
except Exception as e:
    log.append(f"list failed: {e}")
open("eval_data_cremad/pull.log", "w").write("\n".join(log))
print("\n".join(log[-4:]))
