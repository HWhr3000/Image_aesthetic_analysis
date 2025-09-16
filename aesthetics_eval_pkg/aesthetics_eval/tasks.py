from __future__ import annotations
import os
from celery import Celery
from .config import REDIS_URL
from .cli import evaluate as eval_cli

app = Celery("aesthetics_eval", broker=REDIS_URL, backend=REDIS_URL)

@app.task(name="aesthetics_eval.evaluate")
def evaluate_task(images_dir: str | None, image: str | None, toolbox_repo: str, out_dir: str, thresholds_path: str = "thresholds.yml"):
    eval_cli(images_dir, image, toolbox_repo, out_dir, thresholds_path, emit_jsonl=True, stdout_json=False)
    return {"out_dir": os.path.abspath(out_dir)}

def enqueue_eval(images_or_dir: str, toolbox_repo: str, thresholds_path: str, out_dir: str):
    if os.path.isdir(images_or_dir):
        return evaluate_task.delay(images_dir=images_or_dir, image=None, toolbox_repo=toolbox_repo, out_dir=out_dir, thresholds_path=thresholds_path).id
    else:
        return evaluate_task.delay(images_dir=None, image=images_or_dir, toolbox_repo=toolbox_repo, out_dir=out_dir, thresholds_path=thresholds_path).id
