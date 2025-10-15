#!/usr/bin/env python3
"""
Deterministic product-photo QA that outputs **numeric metrics** per image
and (optionally) combines them with lightweight LLM judgements.

Why this approach?
- Your earlier pipeline asked a vision LLM to *measure* LAB, contrast, etc.
  Those are classic image-processing metrics that are cheaper and far more
  reliable to compute directly. We do that here.
- LLMs are useful for subjective checks (e.g., "is that a podium?", "is there
  stray text?") or as a tie-breaker. This script supports optional LLM assists
  via Ollama, but you’ll still get full numeric metrics even without it.

Key outputs (all numeric so you can compare to ground truth):
- background_lab_L, a, b (estimated from border/background model)
- background_ivory_deltaE (distance to ivory target in Lab)
- warmth_lab_b ("warmth" ~ Lab b mean)
- contrast_rms (RMS of luminance)
- sharpness_laplacian (variance of Laplacian)
- edges_density (edge px / total px)
- eoe (edge-over-entropy heuristic)
- color_entropy_bits (Shannon entropy of hue histogram)
- fourier_slope (1/f slope fitted on radial average power spectrum)
- symmetry_vertical (0..1, lower = more symmetric)
- balance_offset (px distance of saliency center from image center; normalized)
- obj_coverage (foreground mask area / total area)
- obj_bbox_touch (True if touches frame)
- podium_height_ratio (0..1, heuristic; NaN if not found)
- text_confidence (requires pytesseract; else NaN)

It writes a single Excel with one row per image.

Optional: If you enable --ollama-models, we’ll prompt selected local LLMs for
soft judgements (background_ivory_yes, warmth_level, podium_present, text_ok,
frame_ok, ratio_ok, etc.) using a small preview. We will aggregate their
scores/booleans and include averages in the Excel. (Safe even if models are
unreliable—the deterministic columns remain your source of truth.)

Dependencies: numpy, pillow, opencv-python, scikit-image, pandas, scipy (fft),
optional: pytesseract, ollama

Example:
python evaluate_images_metrics.py \
  --input_folder ./images \
  --output ./out/metrics.xlsx \
  --ivory_lab 90 0 20 \
  --ollama-models qwen2.5vl:7b llama3.2-vision:latest

"""
import os, sys, math, json, time, argparse, base64
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Optional deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from skimage import color, filters, morphology, measure, exposure
except Exception:
    color = filters = morphology = measure = exposure = None

try:
    from scipy import fftpack
except Exception:
    fftpack = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import ollama  # optional, only if --ollama-models used
except Exception:
    ollama = None

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ----------------------------- utils -----------------------------

def list_images(folder: str) -> List[str]:
    return [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED_EXTS)
    ]


def pil_to_np_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def to_lab(rgb: np.ndarray) -> np.ndarray:
    if color is None:
        raise RuntimeError("scikit-image is required for Lab conversion (pip install scikit-image)")
    return color.rgb2lab(rgb / 255.0)


def deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))

# -------------------- background / foreground --------------------

def estimate_background_lab(rgb: np.ndarray, border_px: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate background color using border pixels; return (lab_bg_mean, lab_bg_std).
    Assumes studio-like shots where borders are background.
    """
    h, w, _ = rgb.shape
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True

    lab = to_lab(rgb)
    bg_vals = lab[border_mask]
    mean = np.nanmean(bg_vals, axis=0)
    std = np.nanstd(bg_vals, axis=0)
    return mean, std


def foreground_mask(rgb: np.ndarray, lab_bg: np.ndarray, deltaE_thresh: float = 8.0) -> np.ndarray:
    """Create a rough foreground mask by thresholding Lab distance to background.
    deltaE_thresh: 6–12 works well for clean studio backgrounds.
    """
    lab = to_lab(rgb)
    d = deltaE76(lab, lab_bg[None, None, :])
    mask = d > deltaE_thresh
    if morphology is not None:
        mask = morphology.remove_small_holes(mask, area_threshold=500)
        mask = morphology.remove_small_objects(mask, min_size=800)
        mask = morphology.binary_opening(mask, morphology.disk(3))
        mask = morphology.binary_closing(mask, morphology.disk(5))
    return mask

# -------------------------- metrics ------------------------------

def luminance_r(rgb: np.ndarray) -> np.ndarray:
    # ITU-R BT.709 luma approximation
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def rms_contrast(rgb: np.ndarray) -> float:
    Y = luminance_r(rgb)
    return float(np.sqrt(np.mean((Y - Y.mean()) ** 2)))


def laplacian_sharpness(rgb: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for sharpness (pip install opencv-python)")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F)
    return float(lap.var())


def edge_density(rgb: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for edge density")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    return float(edges.sum() / 255.0) / (rgb.shape[0] * rgb.shape[1])


def hue_entropy_bits(rgb: np.ndarray, bins: int = 36) -> float:
    if color is None:
        raise RuntimeError("scikit-image is required for color entropy")
    hsv = color.rgb2hsv(rgb / 255.0)
    hist, _ = np.histogram(hsv[..., 0], bins=bins, range=(0, 1), density=True)
    hist = hist + 1e-12
    p = hist / hist.sum()
    ent = -np.sum(p * np.log2(p))
    return float(ent)


def eoe_metric(rgb: np.ndarray) -> float:
    """Edge-over-entropy heuristic: more structure with less color entropy => higher EOE.
    Scale doesn’t matter; compare relatively across set.
    """
    ed = edge_density(rgb)
    ce = hue_entropy_bits(rgb)
    return float(ed / (ce + 1e-6))


def fourier_slope(rgb: np.ndarray, sample: int = 512) -> float:
    if fftpack is None:
        raise RuntimeError("scipy is required for fourier slope (pip install scipy)")
    # center-crop/resize to sample x sample for speed
    h, w, _ = rgb.shape
    s = min(sample, h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    patch = rgb[y0:y0+s, x0:x0+s]
    gray = luminance_r(patch)
    f = fftpack.fftshift(fftpack.fft2(gray))
    psd2D = np.abs(f) ** 2

    # radial average
    y, x = np.indices(psd2D.shape)
    r = np.sqrt((x - s/2) ** 2 + (y - s/2) ** 2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / (nr + 1e-9)

    # fit slope on log-log excluding DC and extremes
    eps = 1e-9
    freqs = np.arange(1, len(radial) // 2)
    ylog = np.log(radial[1:len(freqs)+1] + eps)
    xlog = np.log(freqs + eps)
    A = np.vstack([xlog, np.ones_like(xlog)]).T
    slope, _ = np.linalg.lstsq(A, ylog, rcond=None)[0]
    return float(slope)


def symmetry_vertical(rgb: np.ndarray) -> float:
    """Mean absolute difference between image and its horizontal flip, normalized to 0..1 by 255.
    Lower is "more symmetric".
    """
    flipped = np.flip(rgb, axis=1)
    mad = np.mean(np.abs(rgb.astype(np.int16) - flipped.astype(np.int16)))
    return float(mad / 255.0)


def balance_offset(rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Distance between saliency centroid and image center, normalized by diagonal.
    Use foreground mask if available for saliency; else edge map.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for balance metric")
    h, w, _ = rgb.shape
    if mask is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        sal = edges.astype(np.float32)
    else:
        sal = mask.astype(np.float32)

    y, x = np.indices(sal.shape)
    m = sal.sum() + 1e-9
    cx = float((x * sal).sum() / m)
    cy = float((y * sal).sum() / m)
    dx = cx - w / 2.0
    dy = cy - h / 2.0
    d = math.hypot(dx, dy)
    diag = math.hypot(w, h)
    return float(d / diag)


def podium_height_ratio(rgb: np.ndarray) -> float:
    """Heuristic podium detection near bottom: detect strong horizontal line and block.
    Returns height of bottom structure / image height, else NaN.
    """
    if cv2 is None:
        return float("nan")
    h, w, _ = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Focus on bottom 35%
    y0 = int(h * 0.65)
    roi = edges[y0:]
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=100, minLineLength=int(w*0.3), maxLineGap=10)
    base_y = None
    if lines is not None:
        # pick the longest near-horizontal line
        best_len = 0
        for l in lines[:, 0, :]:
            x1,y1,x2,y2 = l
            if abs(y2 - y1) <= 3:  # near horizontal
                length = abs(x2 - x1)
                if length > best_len:
                    best_len = length
                    base_y = min(y1, y2)
    if base_y is None:
        return float("nan")
    # Estimate podium top by searching upward until edges diminish
    podium_mask = roi.copy()
    colsum = podium_mask.sum(axis=1) / 255.0
    # from base_y go upward until edge density drops below 20% of its max in ROI
    idx = base_y
    thresh = 0.2 * np.max(colsum)
    while idx > 0 and colsum[idx] > thresh:
        idx -= 1
    podium_height = (roi.shape[0] - idx)
    return float(podium_height / h)


def text_confidence_rgb(rgb: np.ndarray) -> float:
    if pytesseract is None:
        return float("nan")
    img = np_rgb_to_pil(rgb)
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if c not in ("-1", "-1.0")]
        if not confs:
            return 0.0
        return float(np.mean(confs)) / 100.0
    except Exception:
        return float("nan")

# ---------------------- LLM (optional) ---------------------------
LLM_PROMPT = """
You are a product-photo QA reviewer. Given a small preview (may be resized),
make strict but *numerical* estimates for these fields. Return **ONLY JSON**:
{
  "background_ivory_yes": bool,
  "warmth_level": 0..100,
  "podium_present": bool,
  "text_ok": bool,
  "frame_ok": bool,
  "ratio_ok": bool
}
Definitions:
- background_ivory_yes: background is clean ivory/off-white with slight warmth
- warmth_level: subjective warmth impression (0=cool, 100=very warm)
- podium_present: a base/platform under the object is visible and aligned
- text_ok: if any text exists, it is clear, non-clipped, non-mirrored
- frame_ok: subject contained, not cut off, aligned to frame
- ratio_ok: object-to-frame coverage roughly 60–90%
Be terse and deterministic.
"""


def make_preview_base64(image_path: str, target_side: int = 1000) -> str:
    im = Image.open(image_path).convert("RGB")
    ow, oh = im.size
    scale = min(1.0, target_side / max(ow, oh))
    nw, nh = int(ow * scale), int(oh * scale)
    if scale < 1.0:
        im = im.resize((nw, nh), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def llm_judge(image_path: str, models: List[str], keep_alive: str = "30m") -> Dict[str, Any]:
    out_rows = []
    if ollama is None:
        return {"models": [], "avg": {}, "raw": []}
    b64 = make_preview_base64(image_path)
    for m in models:
        try:
            resp = ollama.chat(
                model=m,
                messages=[{"role": "user", "content": LLM_PROMPT, "images": [b64]}],
                options={"temperature": 0.0},
                keep_alive=keep_alive,
            )
            content = resp.get("message", {}).get("content", "{}")
            # sanitize minimal
            s = content.strip().replace("```json", "").replace("```", "").strip()
            s = s[s.find("{") : s.rfind("}") + 1]
            data = json.loads(s)
            out_rows.append({"model": m, **data})
        except Exception as e:
            out_rows.append({"model": m, "error": str(e)})
    # aggregate simple averages
    keys_bool = ["background_ivory_yes", "podium_present", "text_ok", "frame_ok", "ratio_ok"]
    keys_num = ["warmth_level"]
    agg = {}
    if out_rows:
        for k in keys_bool:
            vals = [1.0 if r.get(k) is True else 0.0 for r in out_rows if k in r]
            agg[k + "_avg"] = float(np.mean(vals)) if vals else float("nan")
        for k in keys_num:
            vals = [float(r.get(k)) for r in out_rows if isinstance(r.get(k), (int, float))]
            agg[k + "_avg"] = float(np.mean(vals)) if vals else float("nan")
    return {"models": [r.get("model") for r in out_rows], "avg": agg, "raw": out_rows}

# ----------------------- main pipeline ---------------------------

def analyze_image(image_path: str, ivory_lab: Tuple[float,float,float], deltaE_bg_thresh: float = 8.0) -> Dict[str, Any]:
    im = Image.open(image_path).convert("RGB")
    rgb = pil_to_np_rgb(im)
    h, w, _ = rgb.shape

    # background model
    bg_mean, bg_std = estimate_background_lab(rgb)

    # foreground
    fg = foreground_mask(rgb, bg_mean, deltaE_thresh=deltaE_bg_thresh)
    coverage = float(fg.mean())

    # bbox & frame touch
    obj_touch = False
    if fg.any():
        ys, xs = np.where(fg)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        if y0 <= 2 or x0 <= 2 or (h - 1 - y1) <= 2 or (w - 1 - x1) <= 2:
            obj_touch = True
    else:
        y0 = x0 = 0; y1 = h-1; x1 = w-1

    # metrics
    lab_bg = bg_mean
    ivory = np.array(ivory_lab, dtype=np.float32)
    background_ivory_deltaE = float(np.linalg.norm(lab_bg - ivory))

    metrics = {
        "image_name": os.path.basename(image_path),
        "width": w,
        "height": h,
        "background_lab_L": float(lab_bg[0]),
        "background_lab_a": float(lab_bg[1]),
        "background_lab_b": float(lab_bg[2]),
        "background_lab_std_L": float(bg_std[0]),
        "background_lab_std_a": float(bg_std[1]),
        "background_lab_std_b": float(bg_std[2]),
        "background_ivory_deltaE": background_ivory_deltaE,
        "warmth_lab_b": float(lab_bg[2]),
        "contrast_rms": rms_contrast(rgb),
        "sharpness_laplacian": laplacian_sharpness(rgb) if cv2 is not None else float("nan"),
        "edges_density": edge_density(rgb) if cv2 is not None else float("nan"),
        "color_entropy_bits": hue_entropy_bits(rgb) if color is not None else float("nan"),
        "eoe": eoe_metric(rgb) if (cv2 is not None and color is not None) else float("nan"),
        "fourier_slope": fourier_slope(rgb) if fftpack is not None else float("nan"),
        "symmetry_vertical": symmetry_vertical(rgb),
        "balance_offset": balance_offset(rgb, mask=fg) if cv2 is not None else float("nan"),
        "obj_coverage": coverage,
        "obj_bbox_touch": bool(obj_touch),
        "podium_height_ratio": podium_height_ratio(rgb),
        "text_confidence": text_confidence_rgb(rgb),
        "obj_bbox_x0": int(x0),
        "obj_bbox_y0": int(y0),
        "obj_bbox_x1": int(x1),
        "obj_bbox_y1": int(y1),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Deterministic product-photo QA metrics + optional LLM assists")
    ap.add_argument("--input_folder", required=True)
    ap.add_argument("--output", required=True, help="Path to output Excel (e.g., ./out/metrics.xlsx)")
    ap.add_argument("--ivory_lab", nargs=3, type=float, default=[90.0, 0.0, 20.0],
                    help="Target ivory Lab (L a b), defaults to 90,0,20")
    ap.add_argument("--deltaE_bg_thresh", type=float, default=8.0,
                    help="DeltaE threshold separating foreground from background (default 8.0)")
    ap.add_argument("--ollama-models", nargs="*", default=[],
                    help="Optional list of ollama vision models for soft judgements")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    images = list_images(args.input_folder)
    if not images:
        print(f"No images in {args.input_folder}")
        sys.exit(2)

    rows = []
    llm_rows = []
    for p in images:
        try:
            m = analyze_image(p, ivory_lab=tuple(args.ivory_lab), deltaE_bg_thresh=args.deltaE_bg_thresh)
            if args.ollama-models:
                llm = llm_judge(p, args.ollama-models)
                # flatten some averages
                for k, v in llm.get("avg", {}).items():
                    m[f"llm_{k}"] = v
                m["llm_models"] = ",".join(llm.get("models", []))
                # also store raw per-model JSON for auditing in a separate sheet
                for r in llm.get("raw", []):
                    rr = {"image_name": os.path.basename(p), **r}
                    llm_rows.append(rr)
            rows.append(m)
        except Exception as e:
            rows.append({"image_name": os.path.basename(p), "error": str(e)})

    df = pd.DataFrame(rows)
    # Useful derived booleans (do not hide the numeric ground truth!)
    df["background_is_ivory"] = df["background_ivory_deltaE"] < 6.0
    df["coverage_ok_60_90"] = (df["obj_coverage"] >= 0.60) & (df["obj_coverage"] <= 0.90)
    df["frame_ok_inferred"] = (~df["obj_bbox_touch"]) & (df["balance_offset"] < 0.20)

    # Save Excel
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as xl:
        df.to_excel(xl, index=False, sheet_name="metrics")
        if llm_rows:
            pd.DataFrame(llm_rows).to_excel(xl, index=False, sheet_name="llm_raw")

    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
