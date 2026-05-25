import csv
import json

from rl_bench.logger import Logger


def test_logger_writes_jsonl_and_tb(tmp_path):
    log = Logger(tmp_path)
    log.log_dict({"loss/critic": 1.5, "loss/actor": -0.2}, step=10)
    log.close()
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["step"] == 10
    assert rec["loss/critic"] == 1.5
    assert (tmp_path / "tb").exists()


def test_logger_eval_csv(tmp_path):
    log = Logger(tmp_path)
    log.log_eval(step=100, returns=[1.0, 2.0, 3.0])
    log.close()
    with open(tmp_path / "eval.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == 100
    assert abs(float(rows[0]["mean"]) - 2.0) < 1e-6
    assert int(float(rows[0]["min"])) == 1
    assert int(float(rows[0]["max"])) == 3
