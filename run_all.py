# run_all.py  (robust merge version)
# - Keeps original pipeline
# - Backs up raw results
# - Guarantees 'img_file' column exists after postprocess
# - Merges Floward checks safely

import os, sys, json, argparse, time, hashlib, shutil, re
import pandas as pd
from pathlib import Path

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_img_file_column(df: pd.DataFrame, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Ensure df has a column named 'img_file' with just the basename (e.g., 123.jpg).
    Tries several strategies and uses fallback_df if available.
    """
    # Already present?
    if 'img_file' in df.columns:
        df['img_file'] = df['img_file'].astype(str).map(lambda s: os.path.basename(s).strip())
        return df

    # Try common alternatives
    candidates = []
    for col in df.columns:
        cl = col.lower()
        if cl in {'image', 'img', 'filename', 'file', 'file_name', 'name'}:
            candidates.append(col)
    if candidates:
        c = candidates[0]
        df['img_file'] = df[c].astype(str).map(lambda s: os.path.basename(s).strip())
        return df

    # Try regex search for a column that *looks* like filenames with extensions
    for col in df.columns:
        series = df[col].astype(str)
        if series.str.contains(r'\.(jpg|jpeg|png|webp)$', case=False, na=False).any():
            df['img_file'] = series.map(lambda s: os.path.basename(s).strip())
            return df

    # Use fallback df if given
    if fallback_df is not None:
        if 'img_file' in fallback_df.columns:
            df['img_file'] = fallback_df['img_file'].astype(str).map(lambda s: os.path.basename(s).strip())
            return df
        # same heuristics on fallback
        for col in fallback_df.columns:
            series = fallback_df[col].astype(str)
            if series.str.contains(r'\.(jpg|jpeg|png|webp)$', case=False, na=False).any():
                df['img_file'] = series.map(lambda s: os.path.basename(s).strip())
                return df

    raise RuntimeError("Unable to determine 'img_file' column after postprocess.")

def main():
    ap = argparse.ArgumentParser(description="End-to-end Aesthetics + Floward pipeline (pure Python)")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--toolbox-repo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresholds", required=True, help="thresholds.yml (with L_min/L_max etc.)")
    ap.add_argument("--floward-excel", default=None, help="optional path to save Excel workbook")
    ap.add_argument("--enhanced-root", default=None, help="root containing enhanced model subfolders")
    args = ap.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    toolbox_repo = os.path.abspath(args.toolbox_repo)
    out_dir = os.path.abspath(args.out_dir)
    thresholds_path = os.path.abspath(args.thresholds)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) Run original toolbox script to produce out/results.csv ----------
    out_csv = os.path.join(out_dir, "results.csv")
    raw_csv = os.path.join(out_dir, "results_raw.csv")  # backup before postprocess

    sys.path.insert(0, os.path.abspath(os.path.join(Path(__file__).parent, "aesthetics_eval_pkg")))
    try:
        from aesthetics_eval.toolbox_runner import run_toolbox_on_dir, find_script
    except Exception as e:
        print("ERROR: cannot import aesthetics_eval.toolbox_runner. Did you `pip install -e aesthetics_eval_pkg`?", file=sys.stderr)
        raise

    ok, script = run_toolbox_on_dir(toolbox_repo, images_dir, out_csv)
    if not ok or not os.path.exists(out_csv):
        sys.exit("Aesthetics-Toolbox script did not produce CSV. Check paths/flags.")

    # Keep a raw backup of the toolbox output
    shutil.copy2(out_csv, raw_csv)

    # ---------- 2) Postprocess with thresholds ----------
    try:
        from aesthetics_eval.postprocess import load_thresholds, map_and_flag
    except Exception:
        print("[WARN] Could not import aesthetics_eval.postprocess; continuing without postprocess.")
        df = pd.read_csv(out_csv, on_bad_lines='skip')
        rows = []
    else:
        thresholds = load_thresholds(thresholds_path)
        df, rows = map_and_flag(out_csv, thresholds)

    # Guarantee 'img_file' in the postprocessed df
    try:
        raw_df = pd.read_csv(raw_csv, on_bad_lines='skip')
    except Exception:
        raw_df = None

    try:
        df = normalize_img_file_column(df, fallback_df=raw_df)
    except Exception as e:
        print(f"[ERROR] normalize_img_file_column failed: {e}", file=sys.stderr)
        print("As a fallback, using raw CSV as the authoritative dataframe.", file=sys.stderr)
        if raw_df is None:
            raise
        df = normalize_img_file_column(raw_df, fallback_df=None)

    # Save the postprocessed (or normalized) results back to out_csv
    df.to_csv(out_csv, index=False)

    # Write JSONL + run_meta (if we have postprocess rows)
    try:
        from aesthetics_eval.provenance import provenance_block
    except Exception:
        def provenance_block(repo_path, script_rel):  # fallback
            return {"repo": repo_path, "script": script_rel}

    prov = provenance_block(toolbox_repo, os.path.relpath(script, toolbox_repo))
    out_jsonl = os.path.join(out_dir, "results.jsonl")
    try:
        with open(out_jsonl, "w") as f:
            for r in rows:
                r["provenance"] = prov
                f.write(json.dumps(r) + "\n")
    except Exception:
        # If postprocess/rows not available, still write minimal provenance
        with open(out_jsonl, "w") as f:
            pass

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "toolbox_provenance": prov,
        "thresholds_path": thresholds_path,
        "thresholds_sha256": sha256_file(thresholds_path) if os.path.exists(thresholds_path) else None,
        "toolbox_csv": out_csv,
        "toolbox_csv_sha256": sha256_file(out_csv),
        "toolbox_csv_raw": raw_csv,
        "toolbox_csv_raw_sha256": sha256_file(raw_csv),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------- 3) Floward checks + merge ----------
    # Import the same evaluator used before (your version with full heuristics)
    from aesthetics_eval.floward_eval import evaluate_image

    rows_flow = []
    for r, _, files in os.walk(images_dir):
        for f in files:
            fl = f.lower()
            if fl.endswith((".jpg", ".jpeg", ".png", ".webp")):
                p = os.path.join(r, f)
                try:
                    rec = evaluate_image(p)
                    rows_flow.append(rec)
                except Exception as e:
                    rows_flow.append({'img_file': f, 'floward_prompt': f"ERROR: {e}", 'error': str(e)})

    flow = pd.DataFrame(rows_flow)

    # Normalize flow['img_file'] to basenames (just in case)
    if 'img_file' in flow.columns:
        flow['img_file'] = flow['img_file'].astype(str).map(lambda s: os.path.basename(s).strip())

    # Ensure left df has img_file
    if 'img_file' not in df.columns:
        raise SystemExit("[FATAL] Postprocess results missing 'img_file' after normalization. Cannot merge.")

    merged = df.merge(flow, how="left", on="img_file")
    merged_csv = os.path.join(out_dir, "results_floward.csv")
    merged.to_csv(merged_csv, index=False)

    # Optional Excel with three sheets
    if args.floward_excel:
        with pd.ExcelWriter(args.floward_excel) as xw:
            df.to_excel(xw, sheet_name="Toolbox", index=False)
            flow.to_excel(xw, sheet_name="FlowardChecks", index=False)
            merged.to_excel(xw, sheet_name="Merged", index=False)

    print(f"\nWROTE: {out_csv}\nWROTE: {out_jsonl}\nWROTE: {merged_csv}\nWROTE: {os.path.join(out_dir,'run_meta.json')}")

    # ---------- 4) (Optional) Enhanced similarity + selection ----------
    if args.enhanced_root:
        enhanced_root = os.path.abspath(args.enhanced_root)

        # Similarity (SSIM/LPIPS)
        from aesthetics_eval.similarity_eval import main as sim_main
        sim_out = os.path.join(out_dir, "similarity_scores.csv")
        sys.argv = ["similarity_eval.py", "--original_dir", images_dir, "--enhanced_root", enhanced_root, "--out_csv", sim_out]
        sim_main()

        # Re-evaluate per-model with Floward checks
        reval_rows = []
        for m in os.listdir(enhanced_root):
            mdir = os.path.join(enhanced_root, m)
            if not os.path.isdir(mdir): 
                continue
            for r, _, files in os.walk(mdir):
                for f in files:
                    if f.lower().endswith((".jpg",".jpeg",".png",".webp")):
                        p = os.path.join(r,f)
                        try:
                            rec = evaluate_image(p)
                            rec['model'] = m
                            # Ensure per-model re-eval also has img_file normalized
                            rec['img_file'] = os.path.basename(rec['img_file']).strip()
                            reval_rows.append(rec)
                        except Exception as e:
                            reval_rows.append({'img_file': f, 'model': m, 'floward_prompt': f"ERROR: {e}", 'error': str(e)})
        reval_df = pd.DataFrame(reval_rows)
        reval_csv = os.path.join(out_dir, "results_floward_reval.csv")
        reval_df.to_csv(reval_csv, index=False)

        # Model selection
        from aesthetics_eval.select_model import main as sel_main
        sel_out = os.path.join(out_dir, "model_selection.csv")
        sys.argv = ["select_model.py", "--reval_csv", reval_csv, "--sim_csv", sim_out, "--out_csv", sel_out]
        sel_main()
        print(f"WROTE: {sim_out}\nWROTE: {reval_csv}\nWROTE: {sel_out}")

if __name__ == "__main__":
    main()
