import os, argparse, cv2, numpy as np, pandas as pd
from skimage import color
from tqdm import tqdm

# ---- Ivory LAB thresholds ----
IVORY_L = (88, 92)
IVORY_A = (-1, 3)
IVORY_B = (8, 12)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def rgb_to_lab(img_rgb):
    return color.rgb2lab(img_rgb.astype(np.float32) / 255.0)

def load_rgb(path):
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

def ivory_background_mask(img_rgb):
    lab = rgb_to_lab(img_rgb)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    m = (
        (L >= IVORY_L[0]-6) & (L <= IVORY_L[1]+6) &
        (A >= IVORY_A[0]-6) & (A <= IVORY_A[1]+6) &
        (B >= IVORY_B[0]-8) & (B <= IVORY_B[1]+8)
    ).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    return m

def largest_component(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, None
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    comp = (labels == idx).astype(np.uint8)
    return comp, stats[idx]  # (x,y,w,h,area)

def compute_object_mask(img_rgb):
    bg = ivory_background_mask(img_rgb)
    obj = (1 - bg).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    comp, bbox = largest_component(obj)
    if comp is None:
        return np.zeros(obj.shape, np.uint8), None
    return comp, bbox

def podium_width_ratio(img_rgb, obj_mask):
    H, W, _ = img_rgb.shape
    bg = ivory_background_mask(img_rgb)
    cand = bg.copy()
    cand[:int(H*0.65), :] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3,25), np.uint8))
    comp, bbox = largest_component(cand)
    if comp is None:
        return None
    px, py, pw, ph, _ = bbox
    _, obox = largest_component(obj_mask)
    if obox is None:
        return None
    ox, oy, ow, oh, _ = obox
    band_top = oy + int(oh*0.8)
    band = obj_mask[band_top:oy+oh, :]
    if band.size == 0:
        return None
    cols = np.where(band.sum(axis=0) > 0)[0]
    if cols.size == 0:
        return None
    obj_base_w = cols[-1] - cols[0] + 1
    return float(obj_base_w) / float(pw) if pw > 0 else None

def background_stats_without_object(img_rgb, obj_mask):
    inv = (1 - obj_mask).astype(np.uint8)
    lab = rgb_to_lab(img_rgb)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    Lbg = L[inv==1]
    if Lbg.size == 0:
        return None
    meanL, stdL = np.mean(Lbg), np.std(Lbg)
    shadow = (inv==1) & (L < (meanL - 2*stdL))
    bg_mask = inv.copy(); bg_mask[shadow] = 0
    if bg_mask.sum() == 0:
        bg_mask = inv
    return {
        'mean_L_bg': np.mean(L[bg_mask==1]),
        'mean_a_bg': np.mean(A[bg_mask==1]),
        'mean_b_bg': np.mean(B[bg_mask==1])
    }

def right_angle_frame_check(img_rgb, tol_deg=3.0):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=gray.shape[1]//4, maxLineGap=20)
    if lines is None:
        return False
    import math
    ok, total = 0, 0
    for ln in lines[:50]:
        x1,y1,x2,y2 = ln[0]
        ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
        ang = min(ang, 180-ang)
        total += 1
        if abs(ang-0)<=tol_deg or abs(ang-90)<=tol_deg:
            ok += 1
    return (total > 0 and ok/total >= 0.7)

def evaluate_image(path, image_size_target=(1000,1000), obj_cov_target=0.75, ratio_band=(0.6,0.9)):
    img = load_rgb(path)
    H, W, _ = img.shape
    dims_ok = (H==image_size_target[1] and W==image_size_target[0])

    obj_mask, _ = compute_object_mask(img)
    obj_cov = float(obj_mask.sum()) / float(H*W)

    bg_stats = background_stats_without_object(img, obj_mask) or {'mean_L_bg': np.nan,'mean_a_bg': np.nan,'mean_b_bg': np.nan}
    iv_ok = (IVORY_L[0] <= bg_stats['mean_L_bg'] <= IVORY_L[1] and
             IVORY_A[0] <= bg_stats['mean_a_bg'] <= IVORY_A[1] and
             IVORY_B[0] <= bg_stats['mean_b_bg'] <= IVORY_B[1])
    warmth_ok = (bg_stats['mean_b_bg'] >= 8.0) if not np.isnan(bg_stats['mean_b_bg']) else False

    ratio = podium_width_ratio(img, obj_mask)
    ratio_ok = (ratio is not None and ratio_band[0] <= ratio <= ratio_band[1])
    frame_ok = right_angle_frame_check(img)

    recs = []
    if not dims_ok: recs.append("Resize to 1000x1000.")
    if not (obj_cov_target*0.9 <= obj_cov <= obj_cov_target*1.1):
        recs.append(f"Adjust object coverage to ~{int(obj_cov_target*100)}% (current {int(obj_cov*100)}%).")
    if not ratio_ok: recs.append("Fix podium alignment (object base ~75% of podium).")
    if not iv_ok: recs.append("Normalize background to ivory (L≈90, a≈1, b≈10).")
    if not warmth_ok: recs.append("Increase warmth (raise b channel).")
    if not frame_ok: recs.append("Correct perspective to right angles.")

    return {
        'img_file': os.path.basename(path),
        'img_H': H, 'img_W': W, 'dims_ok': dims_ok,
        'object_coverage': obj_cov,
        'podium_object_ratio': (np.nan if ratio is None else ratio),
        'ratio_ok': ratio_ok,
        'bg_mean_L': bg_stats['mean_L_bg'],
        'bg_mean_a': bg_stats['mean_a_bg'],
        'bg_mean_b': bg_stats['mean_b_bg'],
        'ivory_ok': iv_ok,
        'warmth_ok': warmth_ok,
        'frame_ok': frame_ok,
        'floward_prompt': " ".join(recs) if recs else "OK",
        'error': ""
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--toolbox_csv", required=True, help="results.csv from QIP script")
    ap.add_argument("--out_csv", required=True, help="merged CSV with Floward checks")
    ap.add_argument("--excel", default=None, help="optional path to save Excel workbook")
    args = ap.parse_args()

    base = pd.read_csv(args.toolbox_csv, on_bad_lines='skip')

    rows = []
    for r, _, files in os.walk(args.images_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                try:
                    rows.append(evaluate_image(os.path.join(r,f)))
                except Exception as e:
                    rows.append({'img_file': f, 'error': str(e), 'floward_prompt': f"ERROR: {e}"})

    flow = pd.DataFrame(rows)
    merged = base.merge(flow, how="left", on="img_file")
    merged.to_csv(args.out_csv, index=False)
    if args.excel:
        with pd.ExcelWriter(args.excel) as xw:
            base.to_excel(xw, sheet_name="Toolbox", index=False)
            flow.to_excel(xw, sheet_name="FlowardChecks", index=False)
            merged.to_excel(xw, sheet_name="Merged", index=False)

if __name__ == "__main__":
    main()
