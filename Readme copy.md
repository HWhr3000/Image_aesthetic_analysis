##Model run steps
1) Activating Environment
	deactivate 2>/dev/null || true
	python3 -m venv .venv
	source .venv/bin/activate
	which python   # should end in .../aesthetics_eval_pkg/.venv/bin/python
	pip install --upgrade pip
	pip install -r requirements.txt
    
	--source .imenv/bin/activate

2) Install in editable mode
	pip install -e .
	which aesthetics-eval
	aesthetics-eval --help

3. Run the script
python run_all.py \
  --images-dir "{image path}" \
  --toolbox-repo "{Repo path}" \
  --out-dir "{Repo Path}/out" \
  --thresholds "{Repo Path}/aesthetics_eval_pkg/thresholds.yml" \
  --floward-excel "{"Repo Path"}/out/evaluation.xlsx"

Optional: --enhanced-root "/mnt/c/Users/rahar/Documents/Enhanced" to compute SSIM/LPIPS + model selection

--My Env run
python run_all.py \
  --images-dir "/mnt/c/Users/rahar/Documents/Sample_image" \
  --toolbox-repo "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/Aesthetics-Toolbox" \
  --out-dir "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out" \
  --thresholds "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/aesthetics_eval_pkg/thresholds.yml" \
  --floward-excel "/mnt/c/Users/rahar/OneDrive - Heriot-Watt University/F21M/Code/Final Code/Image_aesthetic_eval/out/evaluation.xlsx"