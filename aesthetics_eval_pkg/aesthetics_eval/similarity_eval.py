import os, argparse, glob, numpy as np, pandas as pd, torch, lpips, cv2
from skimage.metrics import structural_similarity as ssim

def load_rgb(path):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None: 
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", required=True)
    ap.add_argument("--enhanced_root", required=True, help="root containing model subfolders (e.g., sdxl/, qwen/, kandinsky/)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    models = [d for d in os.listdir(args.enhanced_root) if os.path.isdir(os.path.join(args.enhanced_root, d))]
    loss_fn = lpips.LPIPS(net='vgg').eval().cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg').eval()

    rows = []
    for o_path in glob.glob(os.path.join(args.original_dir, "*")):
        base = os.path.basename(o_path)
        if not base.lower().endswith((".jpg",".jpeg",".png")): 
            continue
        try:
            orig = load_rgb(o_path)
            for m in models:
                cand = os.path.join(args.enhanced_root, m, base)
                if not os.path.exists(cand):
                    continue
                enh = load_rgb(cand)
                H = min(orig.shape[0], enh.shape[0]); W = min(orig.shape[1], enh.shape[1])
                o = cv2.resize(orig, (W,H), interpolation=cv2.INTER_AREA)
                e = cv2.resize(enh, (W,H), interpolation=cv2.INTER_AREA)
                s = ssim(o, e, channel_axis=2, data_range=255)
                to_t = lambda x: torch.tensor(x).permute(2,0,1)[None].float()/127.5-1.0
                t_o, t_e = to_t(o), to_t(e)
                if torch.cuda.is_available():
                    t_o, t_e = t_o.cuda(), t_e.cuda()
                l = float(loss_fn(t_o, t_e).detach().cpu().numpy().squeeze())
                rows.append({'img_file': base, 'model': m, 'SSIM': s, 'LPIPS': l})
        except Exception:
            rows.append({'img_file': base, 'model': 'ERROR', 'SSIM': np.nan, 'LPIPS': np.nan})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
