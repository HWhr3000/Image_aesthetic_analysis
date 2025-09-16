from __future__ import annotations
import os, pandas as pd, numpy as np, yaml
from typing import Tuple, List, Dict

def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns: return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def load_thresholds(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def map_and_flag(toolbox_csv_path: str, thresholds: dict) -> Tuple[pd.DataFrame, List[dict]]:
    df = pd.read_csv(toolbox_csv_path)

    width_c = _col(df, "Image width (pixels)","Image width","Image size (pixels)","width")
    height_c = _col(df, "Image height (pixels)","Image height","height")
    aspect_c = _col(df, "Aspect ratio","aspect_ratio")
    img_c = _col(df, "img_file","image","file","filename","image_file")

    rms_c = _col(df, "RMS contrast","rms_contrast")
    light_entropy_c = _col(df, "Lightness entropy","lightness_entropy")
    complexity_c = _col(df, "Complexity","complexity","HOG Complexity")
    edge_den_c = _col(df, "Edge density","edge_density")
    meanL_c = _col(df, "mean L channel","mean_L")
    meana_c = _col(df, "mean a channel","mean_a")
    meanb_c = _col(df, "mean b channel (Lab)","mean b channel","mean_b")
    color_entropy_c = _col(df, "Color entropy","color_entropy")
    mirror_sym_c = _col(df, "Mirror symmetry","mirror_symmetry","Symmetry")
    balance_c = _col(df, "Balance","balance")
    fourier_slope_c = _col(df, "Fourier slope","fourier_slope")
    eoe1_c = _col(df, "1st-order EOE","EOE 1st order","eoe_1st")
    eoe2_c = _col(df, "2nd-order EOE","EOE 2nd order","eoe_2nd")

    def v(col, i):
        if col is None: return np.nan
        return df[col].iloc[i]

    rows = []
    out_rows = []
    for i in range(len(df)):
        width = int(v(width_c, i)) if width_c is not None and not pd.isna(v(width_c, i)) else None
        height = int(v(height_c, i)) if height_c is not None and not pd.isna(v(height_c, i)) else None
        aspect_ratio = float(v(aspect_c, i)) if aspect_c is not None and not pd.isna(v(aspect_c, i)) else (float(width)/float(height) if width and height else np.nan)
        image_file = str(v(img_c, i)) if img_c is not None and not pd.isna(v(img_c, i)) else f"img_{i}"

        metrics = {
            "rms_contrast": float(v(rms_c, i)) if rms_c is not None and not pd.isna(v(rms_c, i)) else np.nan,
            "lightness_entropy": float(v(light_entropy_c, i)) if light_entropy_c is not None and not pd.isna(v(light_entropy_c, i)) else np.nan,
            "complexity": float(v(complexity_c, i)) if complexity_c is not None and not pd.isna(v(complexity_c, i)) else np.nan,
            "edge_density": float(v(edge_den_c, i)) if edge_den_c is not None and not pd.isna(v(edge_den_c, i)) else np.nan,
            "mean_L": float(v(meanL_c, i)) if meanL_c is not None and not pd.isna(v(meanL_c, i)) else np.nan,
            "mean_a": float(v(meana_c, i)) if meana_c is not None and not pd.isna(v(meana_c, i)) else np.nan,
            "mean_b": float(v(meanb_c, i)) if meanb_c is not None and not pd.isna(v(meanb_c, i)) else np.nan,
            "color_entropy": float(v(color_entropy_c, i)) if color_entropy_c is not None and not pd.isna(v(color_entropy_c, i)) else np.nan,
            "mirror_symmetry": float(v(mirror_sym_c, i)) if mirror_sym_c is not None and not pd.isna(v(mirror_sym_c, i)) else np.nan,
            "balance": float(v(balance_c, i)) if balance_c is not None and not pd.isna(v(balance_c, i)) else np.nan,
            "fourier_slope": float(v(fourier_slope_c, i)) if fourier_slope_c is not None and not pd.isna(v(fourier_slope_c, i)) else np.nan,
            "eoe_1st": float(v(eoe1_c, i)) if eoe1_c is not None and not pd.isna(v(eoe1_c, i)) else np.nan,
            "eoe_2nd": float(v(eoe2_c, i)) if eoe2_c is not None and not pd.isna(v(eoe2_c, i)) else np.nan,
            "ssim": None,
            "lpips": None
        }

        t = thresholds
        lab_ok = all([
            not np.isnan(metrics["mean_L"]) and t["lab"]["L_min"] <= metrics["mean_L"] <= t["lab"]["L_max"],
            not np.isnan(metrics["mean_a"]) and t["lab"]["a_min"] <= metrics["mean_a"] <= t["lab"]["a_max"],
            not np.isnan(metrics["mean_b"]) and t["lab"]["b_min"] <= metrics["mean_b"] <= t["lab"]["b_max"],
        ])
        rms_ok = (not np.isnan(metrics["rms_contrast"]) and t["rms_contrast"]["min"] <= metrics["rms_contrast"] <= t["rms_contrast"]["max"])
        edge_ok = (not np.isnan(metrics["edge_density"]) and metrics["edge_density"] >= t["sharpness"]["edge_density_min"])
        eoe_ok = (not np.isnan(metrics["eoe_1st"]) and metrics["eoe_1st"] >= t["sharpness"]["eoe_min"])
        color_entropy_ok = (not np.isnan(metrics["color_entropy"]) and t["color_entropy"]["min"] <= metrics["color_entropy"] <= t["color_entropy"]["max"])
        fourier_ok = (not np.isnan(metrics["fourier_slope"]) and t["fourier_slope"]["min"] <= metrics["fourier_slope"] <= t["fourier_slope"]["max"])
        # Symmetry/balance may be reported as percents or 0..1 floats; accept either form
        sym = metrics["mirror_symmetry"]
        bal = metrics["balance"]
        symmetry_ok = (not np.isnan(sym) and (sym <= t["symmetry_max"]*100.0 or sym <= t["symmetry_max"]))
        balance_ok = (not np.isnan(bal) and (bal <= t["balance_max"]*100.0 or bal <= t["balance_max"]))

        flags = [lab_ok, rms_ok, edge_ok, eoe_ok, color_entropy_ok, fourier_ok, symmetry_ok, balance_ok]
        compliance_pct = float(np.mean(flags))*100.0

        out_rows.append({
            "image_file": image_file,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            **metrics,
            "lab_ok": lab_ok,
            "rms_ok": rms_ok,
            "edge_ok": edge_ok,
            "eoe_ok": eoe_ok,
            "color_entropy_ok": color_entropy_ok,
            "fourier_ok": fourier_ok,
            "symmetry_ok": symmetry_ok,
            "balance_ok": balance_ok,
            "compliance_pct": compliance_pct
        })

    out_df = pd.DataFrame(out_rows, columns=[
        "image_file","width","height","aspect_ratio",
        "rms_contrast","lightness_entropy","complexity","edge_density",
        "mean_L","mean_a","mean_b","color_entropy","mirror_symmetry",
        "balance","fourier_slope","eoe_1st","eoe_2nd","ssim","lpips",
        "lab_ok","rms_ok","edge_ok","eoe_ok","color_entropy_ok","fourier_ok","symmetry_ok","balance_ok","compliance_pct"
    ])
    # JSON rows
    json_rows : List[dict] = []
    for r in out_df.itertuples(index=False):
        json_rows.append({
            "image_file": r.image_file,
            "width": int(r.width) if r.width is not None and not np.isnan(r.width) else None,
            "height": int(r.height) if r.height is not None and not np.isnan(r.height) else None,
            "aspect_ratio": float(r.aspect_ratio) if r.aspect_ratio is not None and not np.isnan(r.aspect_ratio) else None,
            "metrics": {
                "rms_contrast": _nan_to_none(r.rms_contrast),
                "lightness_entropy": _nan_to_none(r.lightness_entropy),
                "complexity": _nan_to_none(r.complexity),
                "edge_density": _nan_to_none(r.edge_density),
                "mean_L": _nan_to_none(r.mean_L),
                "mean_a": _nan_to_none(r.mean_a),
                "mean_b": _nan_to_none(r.mean_b),
                "color_entropy": _nan_to_none(r.color_entropy),
                "mirror_symmetry": _nan_to_none(r.mirror_symmetry),
                "balance": _nan_to_none(r.balance),
                "fourier_slope": _nan_to_none(r.fourier_slope),
                "eoe_1st": _nan_to_none(r.eoe_1st),
                "eoe_2nd": _nan_to_none(r.eoe_2nd),
                "ssim": _nan_to_none(r.ssim),
                "lpips": _nan_to_none(r.lpips),
            },
            "compliance": {
                "lab_ok": bool(r.lab_ok),
                "rms_ok": bool(r.rms_ok),
                "edge_ok": bool(r.edge_ok),
                "eoe_ok": bool(r.eoe_ok),
                "color_entropy_ok": bool(r.color_entropy_ok),
                "fourier_ok": bool(r.fourier_ok),
                "symmetry_ok": bool(r.symmetry_ok),
                "balance_ok": bool(r.balance_ok),
                "compliance_pct": float(r.compliance_pct)
            }
        })
    return out_df, json_rows

def _nan_to_none(x):
    import math
    try:
        if x is None: return None
        if isinstance(x, float) and math.isnan(x): return None
        return x
    except Exception:
        return None
