from __future__ import annotations
import os, sys, json, tempfile, argparse, time
from pathlib import Path
from typing import Optional
import pandas as pd
from jsonschema import validate

from .postprocess import load_thresholds, map_and_flag
from .toolbox_runner import run_toolbox_on_dir, run_toolbox_on_image, find_script
from .provenance import provenance_block
from .version import __version__

def _write_jsonl(rows, out_path, schema_path: str | None = None):
    if schema_path:
        import json as _json
        schema = _json.load(open(schema_path, "r"))
    with open(out_path, "w") as f:
        for r in rows:
            if schema_path:
                validate(instance=r, schema=schema)
            f.write(json.dumps(r) + "\n")

def _save_meta(out_dir: str, provenance: dict, thresholds_path: str, csv_path: str):
    import hashlib, yaml
    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "toolbox_provenance": provenance,
        "thresholds_path": os.path.abspath(thresholds_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "toolbox_csv": os.path.abspath(csv_path),
        "toolbox_csv_sha256": sha256_file(csv_path),
        "package_version": __version__,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

def evaluate(images_dir: Optional[str], image: Optional[str], toolbox_repo: str, out_dir: str, thresholds_path: str, emit_jsonl: bool, stdout_json: bool):
    os.makedirs(out_dir, exist_ok=True)
    # 1) Create csv via toolbox
    out_csv = os.path.join(out_dir, "results.csv")
    if images_dir:
        ok, script = run_toolbox_on_dir(toolbox_repo, images_dir, out_csv)
    elif image:
        ok, script = run_toolbox_on_image(toolbox_repo, image, out_csv)
    else:
        print("Either --images-dir or --image must be provided", file=sys.stderr)
        sys.exit(2)

    if not ok or not os.path.exists(out_csv):
        print("Aesthetics-Toolbox script did not produce CSV. Check paths/flags.", file=sys.stderr)
        sys.exit(3)

    # 2) Postprocess & thresholds
    thresholds = load_thresholds(thresholds_path)
    df, rows = map_and_flag(out_csv, thresholds)

    # 3) Attach provenance to JSON rows
    prov = provenance_block(toolbox_repo, os.path.relpath(script, toolbox_repo))
    for r in rows:
        r["provenance"] = prov

    # 4) Save outputs
    df.to_csv(out_csv, index=False)
    schema_path = os.path.join(Path(__file__).resolve().parent.parent, "schema.json")
    out_jsonl = os.path.join(out_dir, "results.jsonl")
    _write_jsonl(rows, out_jsonl, schema_path=schema_path)

    _save_meta(out_dir, prov, thresholds_path, out_csv)

    if stdout_json:
        for r in rows:
            print(json.dumps(r))

    if emit_jsonl:
        print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_csv}")
    print(f"Provenance: {os.path.join(out_dir, 'run_meta.json')}")

def main():
    parser = argparse.ArgumentParser(description="Aesthetics-Toolbox wrapper (no web). Emit JSON + CSV with thresholds & provenance.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="Run evaluation for an image or directory")
    g = p_eval.add_mutually_exclusive_group(required=True)
    g.add_argument("--images-dir", type=str, help="Directory with images")
    g.add_argument("--image", type=str, help="Single image path")
    p_eval.add_argument("--toolbox-repo", type=str, required=True, help="Path to cloned Aesthetics-Toolbox repo")
    p_eval.add_argument("--out-dir", type=str, default="./out", help="Where to write results")
    p_eval.add_argument("--thresholds", type=str, default=str(Path(__file__).resolve().parent.parent / "thresholds.yml"))
    p_eval.add_argument("--emit-jsonl", action="store_true", help="Print path to results.jsonl at the end")
    p_eval.add_argument("--stdout-json", action="store_true", help="Also print each JSON record to stdout")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        evaluate(args.images_dir, args.image, args.toolbox_repo, args.out_dir, args.thresholds, args.emit_jsonl, args.stdout_json)

if __name__ == "__main__":
    main()
