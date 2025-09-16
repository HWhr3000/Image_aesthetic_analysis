#!/usr/bin/env python
"""
Verification helper:
- Reads results.csv produced by the toolbox through this wrapper.
- Optionally compares a given image row to expected values (e.g., from your Appendix table).
"""
import argparse, pandas as pd, math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to out/results.csv")
    ap.add_argument("--image", help="Specific image filename to check")
    ap.add_argument("--expect-L", type=float)
    ap.add_argument("--expect-RMS", type=float)
    ap.add_argument("--expect-EdgeDensity", type=float)
    ap.add_argument("--tol", type=float, default=1e-3, help="Tolerance for comparisons")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    if args.image:
        row = df[df["image_file"].astype(str).str.endswith(args.image)].head(1)
        if row.empty:
            print(f"Image {args.image} not found in results.")
            return 1
    else:
        row = df.head(1)

    r = row.iloc[0]
    print("Row preview:", r.to_dict())

    # Optional comparisons
    ok = True
    if args.expect_L is not None:
        if not (abs(float(r["mean_L"]) - args.expect_L) <= args.tol):
            ok = False
            print(f"L mismatch: got {r['mean_L']} vs expect {args.expect_L}")
    if args.expect_RMS is not None:
        if not (abs(float(r["rms_contrast"]) - args.expect_RMS) <= args.tol):
            ok = False
            print(f"RMS mismatch: got {r['rms_contrast']} vs expect {args.expect_RMS}")
    if args.expect_EdgeDensity is not None:
        if not (abs(float(r["edge_density"]) - args.expect_EdgeDensity) <= args.tol):
            ok = False
            print(f"EdgeDensity mismatch: got {r['edge_density']} vs expect {args.expect_EdgeDensity}")

    if ok:
        print("Verification OK (within tolerance).")
        return 0
    else:
        print("Verification FAILED.")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
