1\) Introduction

You have a lightweight, script-first pipeline (no Docker, no web server) that:

1\.	runs the Aesthetics-Toolbox feature extractor on a folder of images → results.csv

2\.	maps those raw features to target ranges (thresholds) → flags \& JSONL/CSV

3\.	applies Floward-specific checks (background, object size %, podium ratio, frame alignment, etc.) → merged evaluation (CSV/Excel)

4\.	optionally computes similarity scores (SSIM/LPIPS) between original and enhanced images → similarity\_scores.csv

5\.	aggregates everything to rank models → model\_selection.csv / model\_selection.pdf (if you generate a report)

Everything is orchestrated by one command via run\_all.py.



2\) Repo layout (what to open when)

Image\_aesthetic\_eval/

├─ run\_all.py                     # Orchestrator (single entrypoint you run)

├─ Aesthetics-Toolbox/

│  └─ QIP\_machine\_script.py       # Original toolbox script (patched for CLI + robust CSV)

├─ aesthetics\_eval\_pkg/

│  ├─ aesthetics\_eval/

│  │  ├─ postprocess.py           # Maps raw QIPs to thresholds; emits JSONL/CSV

│  │  └─ thresholds.yml           # Your target ranges (LAB, RMS, EdgeDensity, etc.)

│  └─ ... (helper modules)

├─ floward\_eval.py                # Floward rules (background ivory, object coverage, podium ratio, frame)

├─ similarity\_eval.py             # SSIM/LPIPS between original and enhanced images

├─ select\_model.py                # Combines compliance + similarity to rank models

└─ out/                           # All outputs land here (CSV/JSONL/Excel)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

3\) One-line run (what you actually execute)

python run\_all.py \\

&nbsp; --images-dir "/path/to/originals" \\

&nbsp; --toolbox-repo "/path/to/Aesthetics-Toolbox" \\

&nbsp; --out-dir "/path/to/out" \\

&nbsp; --thresholds "/path/to/aesthetics\_eval\_pkg/aesthetics\_eval/thresholds.yml" \\

&nbsp; --floward-excel "/path/to/out/evaluation.xlsx"

This will:

•	call the toolbox to compute QIPs → out/results.csv

•	validate the CSV (consistent columns)

•	map \& flag by thresholds → out/results\_raw.csv, out/results.jsonl

•	run Floward checks and merge → out/results\_floward.csv and Excel with sheets:

o	Toolbox (raw QIPs)

o	FlowardChecks (your domain checks)

o	Merged (combined)

If you also pass enhanced image folders, run\_all.py can call similarity\_eval.py and select\_model.py afterward (optional; configurable).

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

4\) Data flow \& file formats

4.1 Aesthetics-Toolbox → results.csv

•	Source: Aesthetics-Toolbox/QIP\_machine\_script.py

•	Inputs: --input\_dir (folder of images), --out\_csv (target path)

•	What it computes (QIPs):

o	Color/Intensity: mean \& std of RGB/Lab/HSV, luminance entropy, color entropy

o	Contrast: RMS (std of L\*)

o	Edges \& Entropy: Edge density, 1st/2nd order EOE

o	Fourier: slope variants (Redies/Spehar/Mather), sigma

o	Fractal dimensions: 2D, 3D

o	Symmetry/Balance: CNN symmetry (LR, UD, LR\&UD), balance, mirror symmetry, DCM + coordinates

o	Self-similarity: PHOG-based, CNN-based

o	Texture-ish: anisotropy, homogeneity, sparseness, variability

•	Outputs:

o	results.csv (one row per image; fixed column header + an error column)

o	It appends only when the header matches exactly; otherwise it rotates the old file and starts fresh (your patch).

Why this matters for reproducibility

We rotate on header mismatch so you never get mixed-schema CSVs that crash pandas later. We also write using csv.writer to avoid commas in values breaking columns.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

4.2 Threshold mapping → results\_raw.csv, results.jsonl

•	Source: aesthetics\_eval\_pkg/aesthetics\_eval/postprocess.py

•	Input: results.csv

•	Config: thresholds.yml

•	What it does:

o	reads raw QIPs

o	maps metrics to Pass/Fail/WithinRange using your thresholds.yml

o	emits:

	results\_raw.csv (pass/fail flags + normalized values if defined)

	results.jsonl (stream-friendly line-delimited JSON)

	run\_meta.json (provenance: commit, weight hashes if available)

About thresholds.yml

•	It’s the single source of truth for target windows (e.g., L: \[88, 92], EdgeDensity: ">= 500", etc.).

•	We use closed intervals for clarity and fewer off-by-one surprises:

o	was: L\_min, L\_max

o	now: L: \[88, 92]

•	Changing the YAML does not change the underlying QIP computation; it only changes pass/fail mapping.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

4.3 Floward rules → results\_floward.csv (+ Excel)

•	Source: floward\_eval.py

•	Inputs: original images + toolbox CSV

•	What it checks (heuristics aligned to your brief):

o	Background ivory without the object:

	create a rough object mask = “not-ivory” pixels

	compute mean LAB of the background only (shadows discounted via L\* sigma rule)

	pass if background LAB is within the ivory band (tolerance configurable)

o	Object coverage: object mask / image area ≈ 75% ± 10% by default

o	Podium ratio: object base width vs podium width ≈ ~75% (bottom-of-image analysis with ivory-band morphology)

o	Image size: exactly 1000×1000

o	Warmth: background Lab b\* ≥ target (e.g., ≥10)

o	Frame alignment: edges near borders mostly horizontal/vertical (Hough lines)

o	Emits a human-readable prompt listing what to fix for each image (“Resize to 1000×1000”, “Increase warmth”, …)

•	Outputs:

o	results\_floward.csv (merged with toolbox by img\_file)

o	Excel workbook (optional) with Toolbox / FlowardChecks / Merged sheets

Image format handling

•	Reads JPG/JPEG/PNG/WEBP/JFIF; converts GRAY→RGB; RGBA→RGB with white background.

•	This is important because your earlier runs had PNG/WEBP/JFIF alongside JPGs.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

4.4 Similarity \& Model selection (optional)

•	similarity\_eval.py: for each (original, enhanced) pair

o	SSIM: target ≥ 0.90

o	LPIPS: target ≤ 0.20 (using a pretrained VGG backbone)

o	writes similarity\_scores.csv

•	select\_model.py:

o	builds a compliance score from re-evaluated QIPs (1.0 if within range; 0 otherwise; averaged)

o	builds a similarity score from SSIM/LPIPS(/MOS if provided)

o	combines them (e.g., 50/50) to rank models per image and overall

o	writes model\_selection.csv (and optionally a PDF report)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

5\) Configuration knobs you’ll actually touch

•	thresholds.yml — metric bands:

o	LAB (background) windows

o	RMS target (~22)

o	EdgeDensity (≥ 500)

o	EOE (> 4)

o	ColorEntropy (4–6)

o	Fourier slope (−2 to −3)

o	Symmetry (< 5%), Balance (< 20%)

o	SSIM (≥ 0.90), LPIPS (≤ 0.20)

•	floward\_eval.py — domain tolerance \& geometry:

o	object coverage target (default 0.75 ± 10%)

o	“ivory” tolerance margins around the LAB window

o	podium band depth (bottom % of image to analyze)

o	Hough line tolerance for “right-angle” framing

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

6\) Reproducibility \& provenance

•	Fresh runs: run\_all.py deletes previous results.\* by default to avoid mixed headers.

•	Header guard in QIP\_machine\_script.py: rotates old CSV if header differs.

•	run\_meta.json: includes commit hash/path to record which toolbox commit and weights were used.

•	Version pinning: your requirements\*.txt lock major libs (numpy, pandas, scikit-image, torch, statsmodels, etc.) compatible with the toolbox.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

7\) Common errors you saw (and what fixes them)

•	ParserError: Expected N fields…

Cause: old results.csv appended with a different header/schema.

Fix: the fresh-run cleanup + header guard (now in place). If it reappears, inspect the failing line by printing column counts for header vs that row.

•	KeyError: 'img\_file' on merge

Cause: toolbox CSV didn’t write header/rows, or a different schema.

Fix: same as above; verify the first two lines of results.csv.

•	“error occurred. QIPs not calculated!” lines

Cause: corrupted/incompatible images (e.g., some WEBP variants), or temporary I/O errors.

Fix: we now keep an error column per row; these images still appear in CSV with the error message so downstream steps don’t crash.

•	PNG with alpha channel looked “wrong”

Fix: we composite RGBA over white in floward\_eval.py before LAB computations.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

8\) How to justify “we used the learned parameters” (dissertation)

•	The QIP computations are exactly the ones from Aesthetics-Toolbox (original code under AT/… imported by QIP\_machine\_script.py).

•	You are not retraining anything; you use:

o	the published convolution kernels (e.g., AT/bvlc\_alexnet\_conv1.npy) and the same hand-crafted metrics as the paper/toolbox

o	identical hyper-parameters for PHOG, Fourier slopes, EOE, etc.

•	Provenance: include run\_meta.json + the toolbox commit hash in your appendix. If needed, hash your AT/ files and store those hashes in the meta.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

9\) Extending to your full Floward spec

Already covered in floward\_eval.py (heuristics you can refine):

•	Background ivory with shadow exclusion

•	Object coverage (~75%)

•	Podium/object width ratio (~75%)

•	Warmth via b\* channel

•	1000×1000 strict size

•	Frame alignment via Hough

•	“Lifestyle images” or “models” → add a class detector if you need to exclude certain types from strict checks.

•	Close-ups: skip podium ratio/object-to-image checks; keep brightness/contrast/sharpness/edges — there’s a stub in floward\_eval.py to add per-type policy if you pass a --image-type-map.

Re-evaluation + model selection:

•	After enhancement (e.g., Qwen/Kandinsky/SDXL), run the same pipeline on enhanced images; write per-pair SSIM/LPIPS; compute compliance deltas; rank models with select\_model.py.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

10\) Minimal run recipes

Toolbox only (sanity):

python Aesthetics-Toolbox/QIP\_machine\_script.py \\

&nbsp; --input\_dir "/path/to/images" \\

&nbsp; --out\_csv "/path/to/out/results.csv"

Map + Floward (no similarity yet):

python run\_all.py \\

&nbsp; --images-dir "/path/to/images" \\

&nbsp; --toolbox-repo "/path/to/Aesthetics-Toolbox" \\

&nbsp; --out-dir "/path/to/out" \\

&nbsp; --thresholds "/path/to/thresholds.yml" \\

&nbsp; --floward-excel "/path/to/out/evaluation.xlsx"

Similarity for enhanced images (optional example):

python similarity\_eval.py \\

&nbsp; --orig-dir "/path/to/originals" \\

&nbsp; --enhanced-dirs "/path/sdxl" "/path/qwen" "/path/kandinsky" \\

&nbsp; --out "/path/to/out/similarity\_scores.csv"

Model selection (optional):

python select\_model.py \\

&nbsp; --floward-merged "/path/to/out/results\_floward.csv" \\

&nbsp; --similarity "/path/to/out/similarity\_scores.csv" \\

&nbsp; --out "/path/to/out/model\_selection.csv"

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

11\) Troubleshooting checklist

•	results.csv exists but postprocess fails → open line 1 \& the failing line; column counts must match.

•	Running QIP script alone doesn’t write → check that --input\_dir actually contains readable images; try a single .jpg.

•	SSIM/LPIPS compute too slowly → batch size/CPU/GPU toggles in similarity\_eval.py (LPIPS benefits from GPU).

•	Different machines → re-install with your pinned requirements\*.txt; verify numpy, opencv-python(-headless), statsmodels, torch/torchvision versions.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

12\) What to put in your dissertation

•	Architecture diagram (the 5 numbered stages above).

•	A table listing all QIPs and short definitions (you can lift names from the CSV header).

•	A table of thresholds (copied from thresholds.yml).

•	Provenance (run\_meta.json, commit hash).

•	Ablations for Floward heuristics (e.g., show effect of shadow removal on background LAB).

•	Failure modes: list examples where the podium detector fails (e.g., non-ivory podium) and how you handle it (error column + prompt).







\##Model run steps

1. Activating Environment
   deactivate 2>/dev/null || true
   python3 -m venv .venv
   source .venv/bin/activate
   which python   # should end in .../aesthetics\_eval\_pkg/.venv/bin/python
   pip install --upgrade pip
   pip install -r requirements.txt

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

