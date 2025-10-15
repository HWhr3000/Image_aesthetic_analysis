**Aesthetic Toolbox and Image Evaluation**

## 1. Overview

This repository implements a feature extraction and evaluation pipeline for image quality and aesthetics, extended with Floward-specific photography guidelines and optional enhancement model selection. The system is designed to run as simple Python scripts (no Docker/Kubernetes) and produces results in CSV, JSONL, and Excel formats.

## 2. High-Level Workflow

1. **Aesthetics-Toolbox:** Computes quantitative image properties (QIPs) and outputs `results.csv`.
2. **Postprocessing:** Maps QIPs to thresholds defined in `thresholds.yml`, creating flag values and normalized values.
3. **Floward Evaluation:** Applies domain-specific rules for background, object coverage, podium ratio, etc., merging these checks with the QIPs.
4. **Similarity Metrics (optional):** Compares original and enhanced images using SSIM and LPIPS.
5. **Model Selection (optional):** Combines compliance and similarity scores to rank models.
6. **Run all script:** All steps are orchestrated by run_all.py.
7. **run_all.py**: The main entry point for the evaluation process.
8. **aesthetic_toolbox/**: A directory containing utility functions and classes for calculating aesthetic metrics.
9. **floward_eval.py**: A script for evaluating images using the aesthetic toolbox.

## 3. Repository Layout
Image_aesthetic_eval/
 ├─ run_all.py                     # Orchestrator (single entrypoint)
 ├─ Aesthetics-Toolbox/            # This is the download from https://github.com/RBartho/Aesthetics-Toolbox
 │  └─ QIP_machine_script.py       # Toolbox script for QIPs
 ├─ aesthetics_eval_pkg/           # custom package that has the evaluation package and the threshold evaluated 
 │  ├─ aesthetics_eval/
 │  │  ├─ postprocess.py           # Threshold mapping + JSONL export
 │  │  └─ floward_eval.py          # Floward-specific checks
 │  │  └─ thresholds.yml           # Metric thresholds
 ├─ Image_auto_scoring/            # contains python script to run different ollama hosted llm models 
 │  │  ├─ evaluate_images_ollama_full.py          # Script that runs with different parameters for running different llm model and saving output in out folder 
 └─ out/                           # Outputs (CSV, JSONL, Excel, etc.)


## 4. Commands

### 4.1 Full Run
  python run_all.py \
    --images-dir "/path/to/images" \
    --toolbox-repo "/path/to/Aesthetics-Toolbox" \
    --out-dir "/path/to/out" \
    --thresholds "/path/to/thresholds.yml" \
    --floward-excel "/path/to/out/evaluation.xlsx"

#### Produces:
##### results.csv (QIPs)
##### results_raw.csv + results.jsonl (threshold flags)
##### results_floward.csv (merged checks)
##### evaluation.xlsx (Excel workbook with three sheets: Toolbox, FlowardChecks, Merged)

### 4.2 Toolbox Only

python Aesthetics-Toolbox/QIP_machine_script.py \
  --input_dir "/path/to/images" \
  --out_csv "/path/to/out/results.csv"

### 4.3 Similarity Evaluation
python similarity_eval.py \
  --orig-dir "/path/to/originals" \
  --enhanced-dirs "/path/sdxl" "/path/qwen" "/path/kandinsky" \
  --out "/path/to/out/similarity_scores.csv"

### 4.4 Model Selection
python select_model.py \
  --floward-merged "/path/to/out/results_floward.csv" \
  --similarity "/path/to/out/similarity_scores.csv" \
  --out "/path/to/out/model_selection.csv"

## 5. Data Flow & Outputs
### 5.1 Toolbox (QIPs)
- Computed metrics include:
-   Color/Intensity: Mean & std (RGB/Lab/HSV), luminance entropy, color entropy
-   Contrast: RMS contrast
-   Edges: Edge density, 1st/2nd order EOE
-   Fourier: Slopes (Redies/Spehar/Mather), sigma
-   Fractal Dimensions: 2D, 3D
-   Symmetry & Balance: CNN symmetry, mirror symmetry, DCM, balance
-   Self-Similarity: PHOG, CNN-based
-   Texture: Anisotropy, homogeneity, sparseness, variability
-   Output: results.csv with one row per image + error column if QIPs failed.

### 5.2 Postprocess (Threshold Mapping)
- Uses thresholds.yml for pass/fail mapping.

- Example YAML schema:
-   RMS:
-     range: [20, 25]
-   EdgeDensity:
-     min: 500
-   Lab:
-     L: [88, 92]
-     a: [-1, 3]
-     b: [8, 12]

Outputs:
-   results_raw.csv (flags + normalized values)
-   results.jsonl (line-delimited JSON)
-   run_meta.json (toolbox commit, weights hash)

### 5.3 Floward Evaluation
- Checks applied per image:
-   Background Ivory: LAB mean in ivory band, shadows excluded
-   Object Coverage: ~75% ± 10%
-   Podium/Object Ratio: ~75% of podium width
-   Warmth: background b* ≥ target (e.g., ≥10)
-   Frame: right-angle alignment (Hough lines)
-   Dimensions: exactly 1000×1000
-   Each image gets a Floward prompt (actionable fixes).

Outputs: results_floward.csv + evaluation.xlsx.

### 5.4 Similarity Evaluation
- For each original vs enhanced pair:
-   SSIM: (≥0.90 target)
-   LPIPS: (≤0.20 target, pretrained VGG)
-   Output: similarity_scores.csv.

### 5.5 Model Selection
-   Compliance score: (threshold pass/fail, averaged)
-   Similarity score: (weighted SSIM/LPIPS/MOS if available)
-   Combined ranking: (50/50)
-   Output: model_selection.csv.

## 6. Debugging Guide
-   ParserError (Expected N fields): old results.csv mixed headers → delete/restart (or rely on header guard).
-   KeyError img_file: toolbox CSV empty/malformed → inspect first 2 lines.
-   Error rows in CSV: corrupted/unsupported images → now logged in error column, downstream steps still run.
-   Alpha-channel PNGs: composited over white before LAB conversion.

## 7. Dissertation Notes
- Include:
-   Diagram of pipeline (stages 1–5)
-   QIP definitions table (from toolbox headers)
-   Threshold table (copied from thresholds.yml)
-   Provenance (run_meta.json, commit hash)
-   Ablation/failure modes (podium detector failing, shadow removal effect)

## 8. Future Extensions
-   Classifier for lifestyle/model images (to relax strict checks)
-   Policy per image type (close-up vs cover vs normal) in floward_eval.py
-   GPU acceleration for LPIPS
-   Auto-report generator (model_selection.pdf)
-   Model Run Steps
-   Activating Environment
-     deactivate 2>/dev/null || true
-     python3 -m venv .venv
-     source .venv/bin/activate
-     which python # should end in .../aesthetics_eval_pkg/.venv/bin/python
-     pip install --upgrade pip
-     pip install -r requirement.txt
    # --source .imenv/bin/activate
-   Install in editable mode
-     pip install -e .
-     which aesthetics-eval
-     aesthetics-eval --help
-   Run the script
-     python run_all.py \
-       --images-dir "/path/to/images" \
-       --toolbox-repo "/path/to/Aesthetics-Toolbox" \
-       --out-dir "/path/to/out" \
-       --thresholds "/path/to/thresholds.yml" \
-       --floward-excel "/path/to/out/evaluation.xlsx"
-   Optional: --enhanced-root "/path/to/Enhanced" to compute SSIM/LPIPS + model selection.

  My Env run:
    python run_all.py \
      --images-dir "/mnt/c/Users/rahar/Documents/Sample_image" \
      --toolbox-repo "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/Aesthetics-Toolbox" \
      --out-dir "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out" \
      --thresholds "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/aesthetics_eval_pkg/thresholds.yml" \
      --floward-excel "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out/evaluation.xlsx"

##Model run steps

1. Activating Environment
   deactivate 2>/dev/null || true
   python3 -m venv .venv
   source .venv/bin/activate
   which python   # should end in .../aesthetics\_eval\_pkg/.venv/bin/python
   pip install --upgrade pip
   pip install -r requirement.txt

   --source .imenv/bin/activate

2. Install in editable mode
   pip install -e .
   which aesthetics-eval
   aesthetics-eval --help

3. Run the script
   python run\_all.py   
     --images-dir "{image path}"   
     --toolbox-repo "{Repo path}"   
     --out-dir "{Repo Path}/out"   
     --thresholds "{Repo Path}/aesthetics\_eval\_pkg/thresholds.yml"   
     --floward-excel "{"Repo Path"}/out/evaluation.xlsx"

   Optional: --enhanced-root "/mnt/c/Users/rahar/Documents/Enhanced" to compute SSIM/LPIPS + model selection

   --My Env run
   python run\_all.py   
   --images-dir "/mnt/c/Users/rahar/Documents/Sample\_image"   
   --toolbox-repo "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image\_aesthetic\_eval/Aesthetics-Toolbox"   
   --out-dir "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image\_aesthetic\_eval/out"   
   --thresholds "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image\_aesthetic\_eval/aesthetics\_eval\_pkg/thresholds.yml"   
   --floward-excel "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image\_aesthetic\_eval/out/evaluation.xlsx"

   --My Env run
python run_all.py    --images-dir "/mnt/c/Users/rahar/Documents/Sample_image"  --toolbox-repo "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/Aesthetics-Toolbox"   --out-dir "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out"   --thresholds "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/aesthetics_eval_pkg/thresholds.yml"  --floward-excel "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out/evaluation.xlsx"

--demo
python run_all.py    --images-dir "/mnt/c/Users/rahar/Documents/Image_demo"  --toolbox-repo "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/Aesthetics-Toolbox"   --out-dir "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out"   --thresholds "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/aesthetics_eval_pkg/thresholds.yml"  --floward-excel "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out/evaluation.xlsx"


python evaluate_images_ollama_full.py --model qwen2.5vl:7b --input_folder "C:\Users\rahar\Documents\Sample_image" --output_folder "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\out" --thresholds ./thresholds.yml

cd "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring" 
python evaluate_images_ollama_full.py --model qwen2.5vl-7b-gpu:latest --input_folder "C:\Users\rahar\Documents\Sample_image" --output_folder "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\out" --thresholds "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\thresholds.yml"

python evaluate_images_ollama_full.py --model llama3.2-vision:latest --input_folder "C:\Users\rahar\Documents\Sample_image" --output_folder "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\out" --thresholds "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\thresholds.yml"
 

python evaluate_images_ollama_full.py --model childof7sins/llava-llama3-f16:latest  --input_folder "C:\Users\rahar\Documents\Sample_image" --output_folder "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\out" --thresholds "C:\Users\rahar\OneDrive - Heriot-Watt University\F21M\Code\Final Code\Image_aesthetic_eval\Image_auto_scoring\thresholds.yml"

 --output_folder, --thresholds




## Detailed Code Structure

### `run_all.py`

This script is the main entry point for the evaluation process. It takes the following command-line arguments:

* `input_image`: The path to the input image.
* `output_dir`: The directory where the output files will be stored.

The script calls `aesthetic_toolbox/evaluate.py` to perform the evaluation.

### `aesthetic_toolbox/evaluate.py`

This module contains the logic for evaluating images. It uses utility functions from `aesthetic_toolbox/utils.py` to calculate aesthetic metrics.

### `aesthetic_toolbox/utils.py`

This module provides utility functions for calculating aesthetic metrics. It contains the following functions:

* `calculate_color_metrics()`: Calculates color-related metrics.
* `calculate_texture_metrics()`: Calculates texture-related metrics.
* `calculate_spatial_metrics()`: Calculates spatial-related metrics.

### `floward_eval.py`

This script is used to evaluate images using the aesthetic toolbox.

## Detailed Outline of Parameter Calculation

The following parameters are calculated:

* `colorfulness`: A measure of the colorfulness of an image.
* `contrast`: A measure of the contrast of an image.
* `sharpness`: A measure of the sharpness of an image.
* `entropy`: A measure of the entropy of an image.

### Color Metrics

The color metrics are calculated using the following functions:

* `calculate_color_metrics()`: Calculates the color metrics using the `aesthetic_toolbox/utils.py` module.

### Texture Metrics

The texture metrics are calculated using the following functions:

* `calculate_texture_metrics()`: Calculates the texture metrics using the `aesthetic_toolbox/utils.py` module.

### Spatial Metrics

The spatial metrics are calculated using the following functions:

* `calculate_spatial_metrics()`: Calculates the spatial metrics using the `aesthetic_toolbox/utils.py` module.
