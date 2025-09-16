# Import the required Libraries
import numpy as np
from PIL import Image
from skimage import color
import os, argparse, csv
import pandas as pd
from tqdm import tqdm

# custom imports (ORIGINAL TOOLBOX LOGIC)
from AT import (
    balance_qips, CNN_qips, color_and_simple_qips,
    edge_entropy_qips, fourier_qips,
    fractal_dimension_qips, PHOG_qips
)

# -------------------- CLI arguments --------------------
parser = argparse.ArgumentParser(description="Aesthetics-Toolbox QIP Machine (patched, multi-image, robust CSV)")
parser.add_argument("--input_dir", type=str, required=True, help="Folder with images")
parser.add_argument("--out_csv", type=str, default="./out/results.csv", help="Path to results CSV")
args = parser.parse_args()

results_path, csv_name = os.path.split(args.out_csv)
if results_path == "":
    results_path = "./"
os.makedirs(results_path, exist_ok=True)
out_file = os.path.join(results_path, csv_name)

# -------------------- Which QIPs to compute (ORIGINAL SET) --------------------
check_dict = {
    'Image size (pixels)': True,
    'Aspect ratio': True,
    'RMS contrast': True,
    'Luminance entropy': True,
    'Complexity': True,
    'Edge density': True,
    'Color entropy': True,
    'means RGB': True,
    'means Lab': True,
    'means HSV': True,
    'std RGB': True,
    'std Lab': True,
    'std HSV': True,
    'Mirror symmetry': True,
    'DCM': True,
    'Balance': True,
    'left-right': True,
    'up-down': True,
    'left-right & up-down': True,
    'Slope Redies': True,
    'Slope Spehar': True,
    'Slope Mather': True,
    'Sigma': True,
    '2-dimensional': True,
    '3-dimensional': True,
    'PHOG-based': True,
    'CNN-based': True,
    'Anisotropy': True,
    'Homogeneity': True,
    '1st-order': True,
    '2nd-order': True,
    'Sparseness': True,
    'Variability': True,
}

dict_of_multi_measures = {
    'means RGB': ['mean R channel', 'mean G channel', 'mean B channel (RGB)'],
    'means Lab': ['mean L channel', 'mean a channel', 'mean b channel (Lab)'],
    'means HSV': ['mean H channel', 'mean S channel', 'mean V channel'],
    'std RGB': ['std R channel', 'std G channel', 'std B channel'],
    'std Lab': ['std L channel', 'std a channel', 'std b channel (Lab)'],
    'std HSV': ['std H channel', 'std S channel', 'std V channel'],
    'DCM': ['DCM distance', 'DCM x position', 'DCM y position'],
}

dict_full_names_QIPs = {
    'left-right': 'CNN symmetry left-right',
    'up-down': 'CNN symmetry up-down',
    'left-right & up-down': 'CNN symmetry left-right & up-down',
    '2-dimensional': '2D Fractal dimension',
    '3-dimensional': '3D Fractal dimension',
    'Sigma': 'Fourier sigma',
    'PHOG-based': 'Self-similarity (PHOG)',
    'CNN-based': 'Self-similarity (CNN)',
    '1st-order': '1st-order EOE',
    '2nd-order': '2nd-order EOE',
}

def custom_round(num):
    try:
        if num < 1:
            scientific_notation = "{:e}".format(num)
            e_val = int(scientific_notation.split("e")[-1])
            return np.round(num, 3 + abs(e_val))
        else:
            return np.round(num, 3)
    except Exception:
        return num

def build_header():
    cols = ['img_file']
    for key in check_dict:
        if check_dict[key]:
            if key in dict_of_multi_measures:
                cols.extend(dict_of_multi_measures[key])
            else:
                cols.append(dict_full_names_QIPs.get(key, key))
    cols.append('error')
    return cols

header = build_header()

# create CSV if needed
if not os.path.exists(out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        csv.writer(f).writerow(header)

# list images & resume
def list_images(root_dir):
    out = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            out.append(os.path.join(r, fn))
    return out

all_paths = list_images(args.input_dir)

already = set()
try:
    df_prev = pd.read_csv(out_file, on_bad_lines='skip')
    if 'img_file' in df_prev.columns:
        already = set(df_prev['img_file'].astype(str).tolist())
except Exception:
    pass

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png')  # keep stable; add .webp once tested

# learned parameters (unchanged)
[kernel, bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)

with open(out_file, 'a', newline='') as f:
    w = csv.writer(f)
    for path in tqdm(all_paths, total=len(all_paths)):
        base = os.path.basename(path).replace(',', '_')
        if not base.lower().endswith(SUPPORTED_EXTS):
            continue
        if base in already:
            continue

        error_flag = ""
        row = [base]

        try:
            # robust load (alpha PNG -> white background)
            img_in = Image.open(path)
            if img_in.mode in ('RGBA', 'LA') or (img_in.mode == 'P' and 'transparency' in img_in.info):
                bg = Image.new("RGBA", img_in.size, (255, 255, 255, 255))
                img_in = Image.alpha_composite(bg, img_in.convert("RGBA")).convert("RGB")
            else:
                img_in = img_in.convert("RGB")

            img_rgb = np.asarray(img_in)
            img_lab = color.rgb2lab(img_rgb)
            img_hsv = color.rgb2hsv(img_rgb)
            img_gray = np.asarray(img_in.convert("L"))

            first_ord = sec_ord = edge_d = None
            sym_lr = sym_ud = sym_lrud = None
            sigma = slope = None
            self_sim = complexity = anisotropy = None

            for key in check_dict:
                if not check_dict[key]:
                    continue
                try:
                    if key == 'Image size (pixels)':
                        row.append(custom_round(color_and_simple_qips.image_size(img_rgb)))

                    elif key == 'Aspect ratio':
                        row.append(custom_round(color_and_simple_qips.aspect_ratio(img_rgb)))

                    elif key == 'RMS contrast':
                        row.append(custom_round(color_and_simple_qips.std_channels(img_lab)[0]))

                    elif key == 'Luminance entropy':
                        row.append(custom_round(color_and_simple_qips.shannonentropy_channels(img_lab[:, :, 0])))

                    elif key == 'Complexity':
                        if self_sim is None:
                            self_sim, complexity, anisotropy = PHOG_qips.PHOGfromImage(
                                img_rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=[1,1,1]
                            )
                        row.append(custom_round(complexity))

                    elif key == 'Edge density':
                        if first_ord is None:
                            first_ord, sec_ord, edge_d = edge_entropy_qips.do_first_and_second_order_entropy_and_edge_density(img_gray)
                        row.append(custom_round(edge_d))

                    elif key == 'Color entropy':
                        row.append(custom_round(color_and_simple_qips.shannonentropy_channels(img_hsv[:, :, 0])))

                    elif key == 'means RGB':
                        r = color_and_simple_qips.mean_channels(img_rgb)
                        row.extend([custom_round(v) for v in r])

                    elif key == 'means Lab':
                        r = color_and_simple_qips.mean_channels(img_lab)
                        row.extend([custom_round(v) for v in r])

                    elif key == 'means HSV':
                        circ_mean, _ = color_and_simple_qips.circ_stats(img_hsv)
                        r = color_and_simple_qips.mean_channels(img_hsv)
                        row.extend([custom_round(circ_mean), custom_round(r[1]), custom_round(r[2])])

                    elif key == 'std RGB':
                        r = color_and_simple_qips.std_channels(img_rgb)
                        row.extend([custom_round(v) for v in r])

                    elif key == 'std Lab':
                        r = color_and_simple_qips.std_channels(img_lab)
                        row.extend([custom_round(v) for v in r])

                    elif key == 'std HSV':
                        _, circ_std = color_and_simple_qips.circ_stats(img_hsv)
                        r = color_and_simple_qips.std_channels(img_hsv)
                        row.extend([custom_round(circ_std), custom_round(r[1]), custom_round(r[2])])

                    elif key in ['1st-order', '2nd-order']:
                        if first_ord is None:
                            first_ord, sec_ord, edge_d = edge_entropy_qips.do_first_and_second_order_entropy_and_edge_density(img_gray)
                        row.append(custom_round(first_ord if key == '1st-order' else sec_ord))

                    elif key == 'Mirror symmetry':
                        row.append(custom_round(balance_qips.Mirror_symmetry(img_gray)))

                    elif key == 'DCM':
                        d = balance_qips.DCM(img_gray)
                        row.extend([custom_round(v) for v in d])

                    elif key == 'Balance':
                        row.append(custom_round(balance_qips.Balance(img_gray)))

                    elif key in ['left-right', 'up-down', 'left-right & up-down']:
                        if sym_lr is None:
                            sym_lr, sym_ud, sym_lrud = CNN_qips.CNN_symmetry(img_rgb, kernel, bias)
                        row.append(custom_round({'left-right':sym_lr,'up-down':sym_ud,'left-right & up-down':sym_lrud}[key]))

                    elif key == 'Sparseness':
                        resp = CNN_qips.conv2d(img_rgb, kernel, bias)
                        _, mp = CNN_qips.max_pooling(resp, patches=22)
                        row.append(custom_round(CNN_qips.CNN_Variance(mp, kind='sparseness')))

                    elif key == 'Variability':
                        resp = CNN_qips.conv2d(img_rgb, kernel, bias)
                        _, mp = CNN_qips.max_pooling(resp, patches=12)
                        row.append(custom_round(CNN_qips.CNN_Variance(mp, kind='variability')))

                    elif key in ['Sigma', 'Slope Redies']:
                        if sigma is None:
                            sigma, slope = fourier_qips.fourier_redies(img_gray, bin_size=2, cycles_min=10, cycles_max=256)
                        row.append(custom_round(sigma if key == 'Sigma' else slope))

                    elif key == 'Slope Spehar':
                        row.append(custom_round(fourier_qips.fourier_slope_branka_Spehar_Isherwood(img_gray)))

                    elif key == 'Slope Mather':
                        row.append(custom_round(fourier_qips.fourier_slope_mather(img_rgb)))

                    elif key == '2-dimensional':
                        row.append(custom_round(fractal_dimension_qips.fractal_dimension_2d(img_gray)))

                    elif key == '3-dimensional':
                        row.append(custom_round(fractal_dimension_qips.fractal_dimension_3d(img_gray)))

                    elif key in ['PHOG-based', 'Anisotropy']:  # Complexity handled earlier
                        if self_sim is None:
                            self_sim, complexity, anisotropy = PHOG_qips.PHOGfromImage(
                                img_rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=[1,1,1]
                            )
                        row.append(custom_round(self_sim if key == 'PHOG-based' else anisotropy))

                    elif key == 'Homogeneity':
                        row.append(custom_round(balance_qips.Homogeneity(img_gray)))

                except Exception as sub_e:
                    if key in dict_of_multi_measures:
                        row.extend(['ERROR'] * len(dict_of_multi_measures[key]))
                    else:
                        row.append('ERROR')
                    error_flag = "ERROR"

        except Exception as e:
            # global failure for this image: fill placeholders for ALL enabled fields
            row = [base_safe]
            for kk in check_dict:
                if check_dict[kk]:
                    if kk in dict_of_multi_measures:
                        row.extend(['ERROR'] * len(dict_of_multi_measures[kk]))
                    else:
                        row.append('ERROR')
            error_flag = "ERROR"

        row.append(error_flag)
        w.writerow(row)
