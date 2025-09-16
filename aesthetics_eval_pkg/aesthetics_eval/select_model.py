import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reval_csv", required=True, help="Floward-merged CSV after re-evaluation of enhanced images")
    ap.add_argument("--sim_csv", required=True, help="similarity_scores.csv")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ssim_w", type=float, default=0.4)
    ap.add_argument("--lpips_w", type=float, default=0.3)
    ap.add_argument("--mos_w", type=float, default=0.3)
    args = ap.parse_args()

    reval = pd.read_csv(args.reval_csv)
    sim   = pd.read_csv(args.sim_csv)

    if 'compliance_score' not in reval.columns:
        flags = []
        for c in ['ivory_ok','warmth_ok','frame_ok','ratio_ok']:
            if c in reval.columns: flags.append(reval[c].astype(float))
        reval['compliance_score'] = pd.concat(flags, axis=1).mean(axis=1) if flags else np.nan

    sim['ssim_norm'] = sim['SSIM'].clip(0,1)
    sim['lpips_norm'] = (1.0 - sim['LPIPS']).clip(0,1)
    if 'MOS' not in sim.columns: sim['MOS'] = np.nan
    sim['mos_norm'] = (sim['MOS']/5.0).clip(0,1)

    sw, lw, mw = args.ssim_w, args.lpips_w, args.mos_w
    totw = sw + lw + mw
    sim['similarity_score'] = (sw*sim['ssim_norm'] + lw*sim['lpips_norm'] + mw*sim['mos_norm']) / totw

    if 'model' not in reval.columns:
        raise SystemExit("reval_csv must contain 'model' column per enhanced image row")

    merged = sim.merge(reval[['img_file','model','compliance_score']], on=['img_file','model'], how='left')
    merged['total_score'] = 0.5*merged['similarity_score'].fillna(0) + 0.5*merged['compliance_score'].fillna(0)
    merged['rank_per_image'] = merged.groupby('img_file')['total_score'].rank(ascending=False, method='min')

    merged.to_csv(args.out_csv, index=False)
    overall = merged.groupby('model')['total_score'].mean().reset_index().sort_values('total_score', ascending=False)
    print("\nOverall model ranking (mean total_score):")
    print(overall.to_string(index=False))

if __name__ == "__main__":
    main()
