# Aesthetics Evaluation (Week‑1, No Web UI)

**Goal:** Run the **Aesthetics‑Toolbox** evaluation with **unchanged logic / learnt parameters** and emit **JSON** (and CSV) you can use downstream.  
This repo provides:
- Python package `aesthetics_eval/` with a CLI (`aesthetics-eval`)
- **Metric JSON schema** and a **thresholds.yml**
- Optional **Celery + Redis** background jobs
- **Docker** + `docker compose` files
- Basic **tests** and CI skeleton

> We **do not reimplement** any metrics. We call the original Toolbox script so its **exact code and learned parameters** are used.

---

## 1) Quickstart (WSL + local Python)

> Works on Windows **Alienware RTX 4090** with **WSL Ubuntu**.

### 1.1 Install prerequisites
```bash
# Inside WSL
sudo apt-get update
sudo apt-get install -y python3.10-venv git redis-server

# NVIDIA GPU (optional for CNN features speed)
# Inside WSL, verify:
nvidia-smi || echo "GPU not visible in WSL yet (okay for CPU)"
```

### 1.2 Clone & set up this repo
```bash
cd ~
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# To use PyTorch (optional, for any CNN-based metrics acceleration), install per PyTorch docs for your CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Get this package's deps
pip install -r requirements.txt
pip install -e .
```

### 1.3 Clone the **Aesthetics-Toolbox**
```bash
# Clone next to this repo (or anywhere). We will point the CLI to it.
git clone https://github.com/RBartho/Aesthetics-Toolbox.git ~/Aesthetics-Toolbox
```

### 1.4 Evaluate your images (JSON output)
```bash
# Example: evaluate a folder of JPG/PNG images
aesthetics-eval evaluate   --images-dir /path/to/images   --toolbox-repo ~/Aesthetics-Toolbox   --out-dir /path to out   --thresholds path to thresholds.yml   --emit-jsonl
# JSONL at out/results.jsonl ; CSV at out/results.csv ; run metadata in out/run_meta.json
```

### 1.5 Using Celery + Redis (background jobs)
```bash
# Start redis (WSL service or docker)
sudo service redis-server start

# In one terminal
celery -A aesthetics_eval.tasks worker --loglevel=INFO

# In another terminal (submit a job)
python -c "from aesthetics_eval.tasks import enqueue_eval; print(enqueue_eval('/path/to/images','~/Aesthetics-Toolbox','thresholds.yml','./out'))"

# The worker writes outputs under ./out and returns the path
```

---

## 2) Docker (optional)

> If you want full reproducibility, build with Docker. GPU can be enabled via `--gpus all` where supported.

```bash
# Build image
docker build -f docker/Dockerfile -t aesthetics-eval:0.1 .

# Run CLI (mount images + toolbox + out)
docker run --rm -it   -v /path/to/images:/data/images   -v ~/Aesthetics-Toolbox:/opt/Aesthetics-Toolbox   -v $(pwd)/out:/workspace/out   aesthetics-eval:0.1   aesthetics-eval evaluate --images-dir /data/images --toolbox-repo /opt/Aesthetics-Toolbox --out-dir /workspace/out --emit-jsonl
```

**Compose with Redis + Celery worker**
```bash
docker compose -f docker-compose.yml up --build
# Submit jobs by exec'ing into the worker container or by importing tasks from your host
```

---

## 3) How we guarantee the original **learnt parameters** are used

1. We **call the Toolbox script directly**, not a reimplementation.  
2. At runtime we record:
   - The Toolbox **Git commit SHA** (`git rev-parse HEAD`)
   - SHA‑256 of any **weight files** found under the Toolbox repo (`*.pt`, `*.pth`, `*.onnx`)
3. These hashes are written in `out/run_meta.json` and into each JSON record’s `provenance` block.
4. To prove alignment, evaluate a known image and check that metrics match your **Appendix table** within tiny tolerance.

See: `tools/verify_toolbox_alignment.py`

---

## 4) Outputs

- `out/results.csv` and `out/results.jsonl` — one row/object per image:
  - QIPs exactly from the Toolbox
  - Compliance flags vs `thresholds.yml`
  - Provenance: Toolbox commit + weight hashes
- `out/run_meta.json` — end‑to‑end reproducibility info
- `schema.json` — JSON contract

**Thresholds (from your dissertation)**: LAB (L=88–92, a=−1–3, b=8–12), RMS 15–30 (target ≈22), EdgeDensity>500, EOE>4, ColorEntropy 4–6, Fourier slope −2..−3, Symmetry<5%, Balance<20%, (SSIM≥0.90, LPIPS≤0.20 for later pairwise comparison).

---

## 5) Developer Notes

- Config‑as‑code: thresholds in `thresholds.yml`
- Deterministic: sets seeds and, if PyTorch is present, toggles deterministic modes
- Version pinning: see `requirements.txt`
- Logging: to stdout/stderr with timestamps

---

## 6) Run a small sample set

```bash
aesthetics-eval evaluate --images-dir /path/to/sample --toolbox-repo ~/Aesthetics-Toolbox --out-dir ./out --emit-jsonl
```

Now open `out/results.csv` and confirm ranges; you can also do:
```bash
python tools/verify_toolbox_alignment.py --results ./out/results.csv
```

---

## 7) Tests & CI

```bash
pytest -q
```

---

## 8) Why this conforms to your dissertation & Toolbox

- We retain the Toolbox metrics and parameters verbatim (script call).  
- Thresholds mirror your documented targets and validation (LAB, RMS, Edge/EOE, ColorEntropy, Fourier slope, Symmetry/Balance; SSIM/LPIPS for later).

