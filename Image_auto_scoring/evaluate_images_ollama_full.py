# -------------------------------------------------------------
# Full AI-based image evaluation with Ollama (GPU-friendly)
# Pre-resizes a PREVIEW (<=1000x1000) BEFORE sending to the model
# while passing ORIGINAL width/height in the prompt for dims_ok.
# -------------------------------------------------------------

import os, re, json, base64, yaml, time
from io import BytesIO
from typing import Dict, Any, List
import pandas as pd
from PIL import Image
from tqdm import tqdm
import concurrent.futures as futures
import ollama

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ---------- thresholds & file utils ----------

def load_thresholds(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def get_image_paths(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

def sanitize_json_block(s: str) -> str:
    s = s.strip().replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m: s = m.group(0)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s

def coerce_flags(d: Dict[str, Any]) -> Dict[str, bool]:
    keys = [
        "lab_ok","rms_ok","edge_ok","eoe_ok","color_entropy_ok",
        "fourier_ok","symmetry_ok","balance_ok","dims_ok","ratio_ok",
        "ivory_ok","warmth_ok","frame_ok"
    ]
    out = {}
    for k in keys:
        v = d.get(k, False)
        if isinstance(v, str):
            v = v.strip().lower()
            v = True if v in {"true","1","yes"} else False
        out[k] = bool(v)
    return out

# ---------- GPU friendly opts for a 4090 ----------

def model_options_for_4090() -> Dict[str, Any]:
    """
    Push more work to the GPU and stop pegging the CPU.
    - num_gpu: ask Ollama to offload (nearly) all layers to the GPU.
    - num_thread: keep small so CPU isn't the bottleneck.
    - num_batch / num_ctx: big enough to engage the GPU on a 4090.
    """
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,

        # GPU offload / workload shape
        "num_gpu": 99,        # try to offload (nearly) all layers to the GPU
        "num_ctx": 4096,      # larger context drives more GPU work (watch VRAM)
        "num_batch": 128,     # increase batching to feed the GPU

        # Keep CPU from dominating
        "num_thread": 2,      # lower threads -> less CPU thrash
    }
# ---------- NEW: preview generator (<=1000×1000) ----------

def make_preview_b64(image_path: str,
                     target_side: int = 1000,
                     exact_square: bool = False) -> (str, int, int):
    """
    Returns (base64_preview_png, original_width, original_height).
    - If image is already small, it is passed through.
    - If exact_square=False (default), scales so max(H,W) <= target_side (keeps aspect).
    - If exact_square=True, pads to a square canvas after scaling (centers content).
      (We use aspect-preserving scale by default to avoid altering the scene.)
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        ow, oh = im.size

        # Downscale if needed (keep aspect)
        scale = min(1.0, float(target_side) / max(ow, oh))
        nw, nh = int(ow * scale), int(oh * scale)
        if scale < 1.0:
            im = im.resize((nw, nh), Image.LANCZOS)
        else:
            nw, nh = ow, oh

        if exact_square:
            # Pad to 1000x1000 using edge pixels to avoid biasing the background color
            canvas = Image.new("RGB", (target_side, target_side))
            # simple letterbox centering
            x0 = (target_side - nw) // 2
            y0 = (target_side - nh) // 2
            canvas.paste(im, (x0, y0))
            im = canvas

        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, ow, oh

# ---------- prompt ----------

def build_prompt(thresholds: Dict[str, Any],
                 measured_w: int,
                 measured_h: int,
                 dims_target=(1000, 1000),
                 obj_cov_target=0.75,
                 ratio_band=(0.6, 0.9)) -> str:
    # Floward: dims target 1000×1000, obj coverage ~75%, ratio 0.6–0.9. :contentReference[oaicite:2]{index=2}
    T = thresholds
    return f"""
You are a strict product-photo QA inspector.

IMPORTANT: The attached image may be a SMALL PREVIEW (resized so the longer side ≤ {dims_target[0]} px)
to reduce GPU memory. Use the ORIGINAL dimensions provided here to judge 'dims_ok' precisely:
OriginalWidth={measured_w}, OriginalHeight={measured_h}. The production target is exactly {dims_target[0]}x{dims_target[1]}.

Output ONLY JSON with these booleans:
{{
  "lab_ok": bool,
  "rms_ok": bool,
  "edge_ok": bool,
  "eoe_ok": bool,
  "color_entropy_ok": bool,
  "fourier_ok": bool,
  "symmetry_ok": bool,
  "balance_ok": bool,
  "dims_ok": bool,
  "ratio_ok": bool,
  "ivory_ok": bool,
  "warmth_ok": bool,
  "frame_ok": bool
}}

Thresholds from thresholds.yml:
- LAB L: {T["Lab"]["L_min"]}–{T["Lab"]["L_max"]}
- LAB a: {T["Lab"]["a_min"]}–{T["Lab"]["a_max"]}
- LAB b: {T["Lab"]["b_min"]}–{T["Lab"]["b_max"]}
- RMS contrast: {T["RMS"]["min"]}–{T["RMS"]["max"]}
- Edge density (min): {T["EdgeDensity"]["threshold"]}
- EOE (min): {T["EOE"]["threshold"]}
- Color entropy: {T["ColorEntropy"]["min"]}–{T["ColorEntropy"]["max"]}
- Fourier slope: {T["FourierSlope"]["min"]} to {T["FourierSlope"]["max"]}
- Symmetry (max): {T["Symmetry"]["max"]}
- Balance (max): {T["Balance"]["max"]}

Layout & framing guidelines (Floward):
- Target size exactly {dims_target[0]}x{dims_target[1]} px.
- Object coverage target ≈ {int(obj_cov_target*100)}%.
- Podium/object ratio visually in [{ratio_band[0]}, {ratio_band[1]}].
- Background clean ivory/off-white, slight warmth acceptable.
- Frame_ok: subject fully contained; right angles aligned.

Be strict and deterministic. Return ONLY the JSON.
"""

# ---------- Ollama call (now uses preview) ----------

def call_ollama(image_path: str,
                model_name: str,
                thresholds: Dict[str, Any],
                keep_alive="45m") -> Dict[str, bool]:

    # generate preview (<=1000 on longest side), but keep original dims for dims_ok
    try:
        b64_preview, ow, oh = make_preview_b64(image_path, target_side=1000, exact_square=False)
    except Exception as e:
        print(f"[WARN] preview failed for {os.path.basename(image_path)}: {e}")
        # fallback: no preview, still try to send original (may be large)
        with Image.open(image_path) as im:
            ow, oh = im.size
        with open(image_path, "rb") as f:
            b64_preview = base64.b64encode(f.read()).decode("utf-8")

    prompt = build_prompt(thresholds, measured_w=ow, measured_h=oh)
    opts = model_options_for_4090()

    last_err = None
    for attempt in range(4):
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt, "images": [b64_preview]}],
                options=opts,
                keep_alive=keep_alive
            )
            content = resp.get("message", {}).get("content", "")
            parsed = json.loads(sanitize_json_block(content))
            return coerce_flags(parsed)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    print(f"[WARN] Ollama failed for {os.path.basename(image_path)}: {last_err}")
    return coerce_flags({})

# ---------- driver ----------

def evaluate_folder(model_name: str,
                    input_folder: str,
                    output_folder: str,
                    thresholds_path: str,
                    max_workers: int = 1) -> str:

    os.makedirs(output_folder, exist_ok=True)
    thresholds = load_thresholds(thresholds_path)

    images = get_image_paths(input_folder)
    if not images:
        raise SystemExit(f"No images found in: {input_folder}")

    print(f"\nModel: {model_name}")
    print(f"Images: {len(images)}  |  Output: {output_folder}")
    print("Using 1000px PREVIEW per image to reduce VRAM & latency (dims judged from original size).\n")

    rows = []
    if max_workers <= 1:
        for p in tqdm(images, desc="Evaluating"):
            flags = call_ollama(p, model_name, thresholds)
            flags["image_name"] = os.path.basename(p)
            rows.append(flags)
    else:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(call_ollama, p, model_name, thresholds): p for p in images}
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Evaluating"):
                p = futs[fut]
                try:
                    flags = fut.result()
                except Exception as e:
                    print(f"[WARN] {os.path.basename(p)}: {e}")
                    flags = coerce_flags({})
                flags["image_name"] = os.path.basename(p)
                rows.append(flags)

    cols = [
        "image_name","lab_ok","rms_ok","edge_ok","eoe_ok","color_entropy_ok",
        "fourier_ok","symmetry_ok","balance_ok","dims_ok","ratio_ok",
        "ivory_ok","warmth_ok","frame_ok"
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = False
    df = df[cols]

    safe = model_name.replace(":", "_").replace("/", "_")
    out_xlsx = os.path.join(output_folder, f"{safe}_evaluation.xlsx")
    df.to_excel(out_xlsx, index=False)
    return out_xlsx

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Full AI evaluation (all flags) via Ollama with 1000px preview")
    ap.add_argument("--model", required=True, help="qwen2.5vl:7b | llama3.2-vision:latest | gemma3:27b")
    ap.add_argument("--input_folder", required=True, help="Folder with images")
    ap.add_argument("--output_folder", required=True, help="Where to write the Excel")
    ap.add_argument("--thresholds", required=True, help="Path to thresholds.yml")
    ap.add_argument("--max-workers", type=int, default=1, help="Use >1 if your Ollama server supports concurrency")
    args = ap.parse_args()
    print(args.input_folder)
    print(args.output_folder)
    
    out_file = evaluate_folder(
        model_name=args.model,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        thresholds_path=args.thresholds,
        max_workers=args.max_workers
    )
    print(f"\nWROTE: {out_file}")

if __name__ == "__main__":
    main()
