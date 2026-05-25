import matplotlib

matplotlib.use("Agg")  # headless backend for CI

from rl_bench.live_plot import LivePlot


def test_live_plot_accepts_updates_and_closes():
    lp = LivePlot(title="test", min_redraw_interval=0.0)
    for t in range(10):
        lp.add_train(t * 100, float(t))
    lp.add_eval(1000, 5.0)
    assert len(lp.train_x) == 10
    assert len(lp.eval_x) == 1
    lp.close()
