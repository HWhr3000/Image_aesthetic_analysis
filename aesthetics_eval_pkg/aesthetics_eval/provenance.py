import hashlib, os, subprocess, datetime
from typing import List, Dict

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit_sha(repo_path: str) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", repo_path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None

def collect_weight_hashes(repo_path: str) -> List[Dict[str, str]]:
    weight_files = []
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if fn.lower().endswith((".pt",".pth",".onnx",".ckpt",".safetensors")):
                path = os.path.join(root, fn)
                try:
                    weight_files.append({"path": os.path.relpath(path, repo_path), "sha256": sha256_file(path)})
                except Exception:
                    pass
    return sorted(weight_files, key=lambda x: x["path"])

def provenance_block(repo_path: str, script_rel_path: str) -> dict:
    return {
        "toolbox_repo": os.path.abspath(repo_path),
        "toolbox_commit": git_commit_sha(repo_path),
        "toolbox_script": script_rel_path,
        "weights": collect_weight_hashes(repo_path),
        "run_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
