import csv
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.run_dir / "tb")
        eval_path = self.run_dir / "eval.csv"
        new_eval = not eval_path.exists()
        self.eval_f = open(eval_path, "a", newline="")
        self.eval_w = csv.writer(self.eval_f)
        if new_eval:
            self.eval_w.writerow(["step", "mean", "std", "min", "max"])
            self.eval_f.flush()

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
        self.eval_f.close()
