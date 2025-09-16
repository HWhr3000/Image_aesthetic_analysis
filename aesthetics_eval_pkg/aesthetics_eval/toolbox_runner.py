import os, subprocess, shutil, tempfile
from typing import Optional, Tuple, List

SCRIPT_CANDIDATES = [
    "QIP_machine_script.py",
    # Alternative names (if repo renames):
    "qip_machine_script.py",
    "scripts/QIP_machine_script.py",
]

def find_script(repo_path: str) -> Optional[str]:
    for cand in SCRIPT_CANDIDATES:
        p = os.path.join(repo_path, cand)
        if os.path.exists(p):
            return p
    # last resort: search
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.lower() == "qip_machine_script.py":
                return os.path.join(root, f)
    return None

def _try_cmd(cmd: List[str]) -> bool:
    try:
        print(">>> Running Toolbox:", " ".join(cmd))
        print(">>> CWD:", repo_path)
        print(">>> Expecting CSV at:", out_csv_path)
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def run_toolbox_on_dir(repo_path: str, images_dir: str, out_csv_path: str) -> Tuple[bool, str]:
    """Invoke the toolbox to process a directory, producing a single CSV."""
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    script = find_script(repo_path)
    if not script:
        raise FileNotFoundError("Could not locate QIP_machine_script.py in the toolbox repo")

    cmd = [
        "python", script,
        "--input_dir", os.path.abspath(images_dir),
        "--out_csv", os.path.abspath(out_csv_path),
    ]
    print(">>> Running Toolbox:", " ".join(cmd))      # DEBUG
    print(">>> CWD:", repo_path)                      # DEBUG

    try:
        subprocess.run(cmd, cwd=repo_path, check=True)
        return os.path.exists(out_csv_path), script
    except subprocess.CalledProcessError as e:
        print(f"Toolbox failed with code {e.returncode}")
        return False, script

def run_toolbox_on_image(repo_path: str, image_path: str, out_csv_path: str) -> Tuple[bool, str]:
    """Process a single image by copying into a temp dir and reusing directory mode."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.basename(image_path)
        dst = os.path.join(td, fn)
        shutil.copy2(image_path, dst)
        ok, script = run_toolbox_on_dir(repo_path, td, out_csv_path)
        return ok, script
