import os
import cv2
import numpy as np
import yaml
from skimage import color

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_config(yaml_file: str) -> dict:
    try:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {yaml_file}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file: {e}")
        raise


# Load configuration
CONFIG = load_config("aesthetics_eval_pkg/thresholds.yml")

# ---- Ivory LAB thresholds ----
IVORY_L = (CONFIG["lab"]["L_min"], CONFIG["lab"]["L_max"])
IVORY_A = (CONFIG["lab"]["a_min"], CONFIG["lab"]["a_max"])
IVORY_B = (CONFIG["lab"]["b_min"], CONFIG["lab"]["b_max"])

dims_cfg = CONFIG.get("dims_target", {})
TGT_W = int(dims_cfg.get("width", 1000))
TGT_H = int(dims_cfg.get("height", 1000))
COV_TGT = float(CONFIG.get("coverage_target", 0.75))
ratio_cfg = CONFIG.get("ratio_band", {})
RATIO_MIN = float(ratio_cfg.get("min", 0.6))
RATIO_MAX = float(ratio_cfg.get("max", 0.9))


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    return color.rgb2lab(img_rgb.astype(np.float32) / 255.0)


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3] / 255.0
        bg = np.ones_like(img[:, :, :3]) * 255
        img = (img[:, :, :3] * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def ivory_background_mask(img_rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(img_rgb)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    m = (
        (L >= IVORY_L[0] - 6) & (L <= IVORY_L[1] + 6)
        & (A >= IVORY_A[0] - 6) & (A <= IVORY_A[1] + 6)
        & (B >= IVORY_B[0] - 8) & (B <= IVORY_B[1] + 8)
    ).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def largest_component(mask: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, None
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    comp = (labels == idx).astype(np.uint8)
    return comp, stats[idx]


def compute_object_mask(img_rgb: np.ndarray):
    bg = ivory_background_mask(img_rgb)
    obj = (1 - bg).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    comp, bbox = largest_component(obj)
    if comp is None:
        return np.zeros(obj.shape, np.uint8), None
    return comp, bbox


def podium_width_ratio(img_rgb: np.ndarray, obj_mask: np.ndarray):
    H, W, _ = img_rgb.shape
    bg = ivory_background_mask(img_rgb)
    cand = bg.copy()
    cand[: int(H * 0.65), :] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 25), np.uint8))
    comp, bbox = largest_component(cand)
    if comp is None:
        return None
    px, py, pw, ph, _ = bbox
    _, obox = largest_component(obj_mask)
    if obox is None:
        return None
    ox, oy, ow, oh, _ = obox
    band_top = oy + int(oh * 0.8)
    band = obj_mask[band_top : oy + oh, :]
    if band.size == 0:
        return None
    cols = np.where(band.sum(axis=0) > 0)[0]
    if cols.size == 0:
        return None
    obj_base_w = cols[-1] - cols[0] + 1
    return float(obj_base_w) / float(pw) if pw > 0 else None


def background_stats_without_object(img_rgb: np.ndarray, obj_mask: np.ndarray):
    inv = (1 - obj_mask).astype(np.uint8)
    lab = rgb_to_lab(img_rgb)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    Lbg = L[inv == 1]
    if Lbg.size == 0:
        return None
    meanL, stdL = np.mean(Lbg), np.std(Lbg)
    shadow = (inv == 1) & (L < (meanL - 2 * stdL))
    bg_mask = inv.copy()
    bg_mask[shadow] = 0
    if bg_mask.sum() == 0:
        bg_mask = inv
    return {
        "mean_L_bg": float(np.mean(L[bg_mask == 1])),
        "mean_a_bg": float(np.mean(A[bg_mask == 1])),
        "mean_b_bg": float(np.mean(B[bg_mask == 1])),
    }


def right_angle_frame_check(img_rgb: np.ndarray, tol_deg: float = 3.0) -> bool:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=gray.shape[1] // 4, maxLineGap=20
    )
    if lines is None:
        return False
    import math

    ok, total = 0, 0
    for ln in lines[:50]:
        x1, y1, x2, y2 = ln[0]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        for target in (0.0, 90.0, -90.0):
            if abs(((ang - target + 180) % 360) - 180) <= tol_deg:
                ok += 1
                break
        total += 1
    return (ok / max(total, 1)) >= 0.7


def evaluate_image(path: str, image_size_target=None, obj_cov_target=None, ratio_band=None) -> dict:
    img = load_rgb(path)
    H, W, _ = img.shape
    tgt_w = image_size_target[0] if image_size_target else TGT_W
    tgt_h = image_size_target[1] if image_size_target else TGT_H
    cov_tgt = obj_cov_target if obj_cov_target is not None else COV_TGT
    rb = ratio_band if ratio_band is not None else (RATIO_MIN, RATIO_MAX)
    dims_ok = (H == tgt_h and W == tgt_w)

    obj_mask, _ = compute_object_mask(img)
    obj_cov = float(obj_mask.sum()) / float(H * W)

    bg_stats = background_stats_without_object(img, obj_mask) or {
        "mean_L_bg": np.nan,
        "mean_a_bg": np.nan,
        "mean_b_bg": np.nan,
    }
    iv_ok = (
        (not np.isnan(bg_stats["mean_L_bg"]) and IVORY_L[0] <= bg_stats["mean_L_bg"] <= IVORY_L[1])
        and (not np.isnan(bg_stats["mean_a_bg"]) and IVORY_A[0] <= bg_stats["mean_a_bg"] <= IVORY_A[1])
        and (not np.isnan(bg_stats["mean_b_bg"]) and IVORY_B[0] <= bg_stats["mean_b_bg"] <= IVORY_B[1])
    )

    ratio = podium_width_ratio(img, obj_mask)
    ratio_ok = (ratio is not None and rb[0] <= ratio <= rb[1])
    frame_ok = right_angle_frame_check(img)

    # Build actionable prompt from Floward-specific checks only
    recs = []
    if not dims_ok:
        recs.append(f"Resize to {tgt_w}x{tgt_h}.")
    if not (cov_tgt * 0.9 <= obj_cov <= cov_tgt * 1.1):
        recs.append(
            f"Adjust object coverage to ~{int(cov_tgt * 100)}% (current {int(obj_cov * 100)}%)."
        )
    if not ratio_ok:
        recs.append("Fix podium alignment (object base ~75% of podium).")
    if not iv_ok:
        recs.append(
            f"Normalize background to ivory (L {IVORY_L[0]}..{IVORY_L[1]}, a {IVORY_A[0]}..{IVORY_A[1]}, b {IVORY_B[0]}..{IVORY_B[1]})."
        )
    if not frame_ok:
        recs.append("Align frame/right angles.")

    return {
        "img_file": os.path.basename(path),
        "img_H": H,
        "img_W": W,
        "dims_ok": bool(dims_ok),
        "object_coverage": obj_cov,
        "podium_object_ratio": (np.nan if ratio is None else float(ratio)),
        "ratio_ok": bool(ratio_ok),
        "bg_mean_L": float(bg_stats["mean_L_bg"]),
        "bg_mean_a": float(bg_stats["mean_a_bg"]),
        "bg_mean_b": float(bg_stats["mean_b_bg"]),
        "ivory_ok": bool(iv_ok),
        "frame_ok": bool(frame_ok),
        "floward_prompt": (" ".join(recs) if recs else "OK"),
        "error": "",
    }

