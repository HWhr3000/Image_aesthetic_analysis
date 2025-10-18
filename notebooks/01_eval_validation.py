"""
Fallback script version of the validation notebook.
Run with:  python notebooks/01_eval_validation.py
"""
import os
import pandas as pd
import yaml

# Configure paths (edit if needed)
RESULTS_CSV = 'out/results.csv'             # written by run_all.py (postprocessed)
MANUAL_XLSX = 'Image_evaluation_manual.xlsx'
THRESHOLDS_YML = 'aesthetics_eval_pkg/thresholds.yml'
OUT_XLSX = 'out/eval_validation_merged.xlsx'

if not os.path.exists(RESULTS_CSV):
    raise SystemExit(f'Not found: {RESULTS_CSV}. Please run run_all.py first.')

metrics = pd.read_csv(RESULTS_CSV)
manual = pd.read_excel(MANUAL_XLSX)
with open(THRESHOLDS_YML, 'r') as f:
    T = yaml.safe_load(f)

def base(s):
    try:
        return os.path.basename(str(s)).strip()
    except Exception:
        return str(s)

# Normalize filename in metrics
if 'img_file' in metrics.columns:
    metrics['image_name'] = metrics['img_file'].map(base)
elif 'image_file' in metrics.columns:
    metrics['image_name'] = metrics['image_file'].map(base)
else:
    cands = [c for c in metrics.columns if metrics[c].astype(str).str.contains(r'\.(jpg|jpeg|png|webp)$', case=False, na=False).any()]
    if not cands:
        raise SystemExit('Could not find filename column in results.csv')
    metrics['image_name'] = metrics[cands[0]].map(base)

# Normalize filename in manual
manual_cols = [c for c in manual.columns if 'image' in c.lower()]
if not manual_cols:
    raise SystemExit('Manual sheet must contain an image filename column')
manual = manual.rename(columns={manual_cols[0]: 'image_name'})
manual['image_name'] = manual['image_name'].map(base)

df = metrics.merge(manual, on='image_name', how='inner')
print('Merged rows:', len(df))

Lmin, Lmax = T['lab']['L_min'], T['lab']['L_max']
bmin, bmax = T['lab']['b_min'], T['lab']['b_max']

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

cL    = pick(df, 'mean_L', 'background_lab_L')
cb    = pick(df, 'mean_b', 'background_lab_b', 'warmth_lab_b')
crms  = pick(df, 'rms_contrast')
cedge = pick(df, 'edge_density')
cent  = pick(df, 'color_entropy', 'color_entropy_bits')
cfs   = pick(df, 'fourier_slope')
csym  = pick(df, 'mirror_symmetry', 'symmetry_vertical')
cbal  = pick(df, 'balance')

if cL:   df['bright_ok'] = df[cL].between(Lmin, Lmax)
if cb:   df['warmth_ok'] = df[cb].between(bmin, bmax)
if crms: df['contrast_ok'] = df[crms].between(T['rms_contrast']['min'], T['rms_contrast']['max'])
if cedge and 'sharpness' in T: df['edge_ok'] = df[cedge] >= T['sharpness']['edge_density_min']
if cent: df['color_entropy_ok'] = df[cent].between(T['color_entropy']['min'], T['color_entropy']['max'])
if cfs:  df['fourier_ok'] = df[cfs].between(T['fourier_slope']['min'], T['fourier_slope']['max'])
if csym: df['symmetry_ok'] = (df[csym] <= T['symmetry_max']*100.0) | (df[csym] <= T['symmetry_max'])
if cbal: df['balance_ok']  = (df[cbal] <= T['balance_max']*100.0)  | (df[cbal] <= T['balance_max'])

# TODO: rename manual columns here to standardized names if necessary, e.g.:
# df = df.rename(columns={
#   'Manual Brightness OK': 'manual_bright_ok',
#   'Manual Warmth OK':     'manual_warmth_ok',
#   'Manual Contrast OK':   'manual_contrast_ok',
#   'Manual Edge OK':       'manual_edge_ok',
# })

def report(true, pred):
    t = pd.crosstab(true, pred, rownames=['manual'], colnames=['pred'], dropna=False)
    acc = (true == pred).mean()
    return t, acc

pairs = [
  ('manual_bright_ok','bright_ok'),
  ('manual_warmth_ok','warmth_ok'),
  ('manual_contrast_ok','contrast_ok'),
  ('manual_edge_ok','edge_ok'),
]
for mt, mp in pairs:
    if mt in df.columns and mp in df.columns:
        t, acc = report(df[mt].astype(bool), df[mp].astype(bool))
        print(f"\n== {mt} vs {mp} ==")
        print(t)
        print(f'acc: {acc:.3f}')

os.makedirs('out', exist_ok=True)
df.to_excel(OUT_XLSX, index=False)
print('WROTE:', OUT_XLSX)

