import csv
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.run_dir / "tb")
        self.jsonl = open(self.run_dir / "metrics.jsonl", "a")
        eval_path = self.run_dir / "eval.csv"
        new_eval = not eval_path.exists()
        self.eval_f = open(eval_path, "a", newline="")
        self.eval_w = csv.writer(self.eval_f)
        if new_eval:
            self.eval_w.writerow(["step", "mean", "std", "min", "max"])
            self.eval_f.flush()

    def log_scalar(self, key, val, step):
        self.tb.add_scalar(key, float(val), step)

    def log_dict(self, d, step):
        for k, v in d.items():
            self.log_scalar(k, v, step)
        rec = {"step": int(step), **{k: float(v) for k, v in d.items()}}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()

    def log_eval(self, step, returns):
        m = float(np.mean(returns))
        s = float(np.std(returns))
        mn = float(np.min(returns))
        mx = float(np.max(returns))
        self.tb.add_scalar("eval/return_mean", m, step)
        self.tb.add_scalar("eval/return_std", s, step)
        self.eval_w.writerow([int(step), m, s, mn, mx])
        self.eval_f.flush()

    def close(self):
        self.tb.close()
        self.jsonl.close()
        self.eval_f.close()
