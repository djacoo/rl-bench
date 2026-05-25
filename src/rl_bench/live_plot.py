import time

import matplotlib.pyplot as plt


class LivePlot:
    """Interactive matplotlib window: train episode return + eval mean over env steps."""

    def __init__(self, title="training", min_redraw_interval=0.5):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title(title)
        self.ax.set_xlabel("env steps")
        self.ax.set_ylabel("return")
        self.ax.grid(True, alpha=0.3)
        (self.line_train,) = self.ax.plot(
            [], [], color="tab:blue", alpha=0.5, label="train ep return"
        )
        (self.line_eval,) = self.ax.plot(
            [], [], color="tab:red", marker="o", label="eval mean"
        )
        self.ax.legend(loc="lower right")
        self.train_x, self.train_y = [], []
        self.eval_x, self.eval_y = [], []
        self._last_draw = 0.0
        self._min_dt = float(min_redraw_interval)
        plt.show(block=False)

    def add_train(self, step, ret):
        self.train_x.append(int(step))
        self.train_y.append(float(ret))
        self._maybe_redraw()

    def add_eval(self, step, mean):
        self.eval_x.append(int(step))
        self.eval_y.append(float(mean))
        self._maybe_redraw(force=True)

    def _maybe_redraw(self, force=False):
        now = time.time()
        if not force and now - self._last_draw < self._min_dt:
            return
        self._last_draw = now
        self.line_train.set_data(self.train_x, self.train_y)
        self.line_eval.set_data(self.eval_x, self.eval_y)
        self.ax.relim()
        self.ax.autoscale_view()
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self):
        plt.ioff()
        try:
            plt.close(self.fig)
        except Exception:
            pass
