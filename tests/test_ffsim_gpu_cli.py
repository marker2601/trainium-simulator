"""`python -m ffsim gpu <sub> ...`: the route-2 group of ffsim/cli.py delegates to ffsim.gpu.fleet /
calibrate / make_train_gpu / launch.sh with the arguments passed through, and imports them lazily so
the base CLI never needs torch (numpy + stdlib only here; torch is not installed on the dev box)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ffsim import cli
from ffsim.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "ffsim" / "gpu" / "configs" / "gate-k60.json"
K60_TRAIN = REPO / "recipes" / "K60" / "train.py"


def test_group_is_registered_with_every_delegate():
    ap = cli.build_parser()
    names = {a.dest: a for a in ap._subparsers._group_actions[0]._get_subactions()}  # type: ignore[union-attr]
    assert "gpu" in names
    assert set(cli.GPU_SUBS) == {"plan", "schedule", "run", "report", "verdict", "from-search", "prices",
                                 "calibrate", "make-train", "launch"}
    assert set(cli.GPU_FLEET_SUBS) < set(cli.GPU_SUBS)


def test_gpu_argv_accepts_singular_config_and_drops_separator():
    assert cli.gpu_argv(["--config", "x.json", "--json", "o"]) == ["--configs", "x.json", "--json", "o"]
    assert cli.gpu_argv(["--config=x.json"]) == ["--configs=x.json"]
    assert cli.gpu_argv(["--", "--config", "x"]) == ["--configs", "x"]
    assert cli.gpu_argv(["--configs", "x", "--"]) == ["--configs", "x", "--"]     # only a LEADING -- is dropped
    assert cli.gpu_argv([]) == []


def test_base_cli_import_is_lazy_no_gpu_modules_no_torch():
    code = ("import sys, ffsim.cli; print(sorted(m for m in sys.modules if m.startswith('ffsim.gpu') or m == 'torch'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_plan_passes_through_config_and_launches_nothing(capsys, tmp_path):
    out_json = tmp_path / "plan.json"
    rc = cli_main(["gpu", "plan", "--config", str(GATE), "--quote-prices", "--seconds-per-step", "4.0",
                   "--json", str(out_json)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "39 runs from" in text and "16 pairs" in text and "Nothing above was launched" in text
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(plan["runs"]) == 39 and plan["cost"]["steps_total"] == 90_336
    assert "torch" not in sys.modules


def test_schedule_prints_the_k60_walk(capsys):
    assert cli_main(["gpu", "schedule"]) == 0
    text = capsys.readouterr().out
    assert "stop (pinned)" in text and "2361" in text and "phase 2 (k=2" in text


def test_sub_help_goes_to_the_delegate(capsys):
    with pytest.raises(SystemExit) as e:
        cli_main(["gpu", "plan", "--help"])
    assert e.value.code == 0
    assert "usage: ffsim.gpu.fleet plan" in capsys.readouterr().out


def test_unknown_sub_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli_main(["gpu", "bogus"])
    assert e.value.code == 2


def test_calibrate_freeze_round_trip(tmp_path, capsys):
    frozen = tmp_path / "rule.json"
    rc = cli_main(["gpu", "calibrate", "--configs", str(GATE), "--freeze", str(frozen)])
    assert rc in (0, 2)                                # 2 = a frozen row could not be resolved; still written
    rule = json.loads(frozen.read_text(encoding="utf-8"))
    assert rule["digest"] and rule["n_decisive"] >= 1
    assert "frozen rule" in capsys.readouterr().out


def test_verdict_without_results_is_pending(tmp_path, capsys):
    results = tmp_path / "empty.jsonl"
    results.write_text("", encoding="utf-8")
    rc = cli_main(["gpu", "verdict", "--results", str(results), "--configs", str(GATE)])
    assert rc == 2                                     # pending: nothing scored yet
    assert "pending" in capsys.readouterr().out.lower()


@pytest.mark.skipif(not K60_TRAIN.exists(), reason="K60 submission train.py not checked out")
def test_make_train_builds_the_cuda_fork(tmp_path, capsys):
    out = tmp_path / "train_gpu.py"
    rc = cli_main(["gpu", "make-train", "--source", str(K60_TRAIN), "--out", str(out), "--quiet"])
    assert rc == 0 and out.exists() and out.stat().st_size > 100_000
    compile(out.read_text(encoding="utf-8"), str(out), "exec")     # syntax only; torch is not installed here


def test_from_search_writes_a_candidate_config(tmp_path, capsys):
    out = tmp_path / "cands.json"
    rc = cli_main(["gpu", "from-search", "--top-k", "2", "--out", str(out)])
    assert rc == 0 and out.exists()
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["runs"] and all("env" in r and "label" in r and "arm" in r for r in cfg["runs"])
    assert cli_main(["gpu", "plan", "--config", str(out), "--quote-prices"]) == 0   # fleet accepts what it wrote
    assert "Nothing above was launched" in capsys.readouterr().out


def test_launch_delegates_to_bash_without_touching_aws(monkeypatch, tmp_path, capsys):
    """`gpu launch <args>` execs `bash ffsim/gpu/launch.sh <args>`; here bash is a stub on PATH that
    only records its argv, so no aws call happens. The real `plan` is exercised by test_ffsim_gpu_launch."""
    log = tmp_path / "argv.txt"
    if os.name == "nt":
        stub = tmp_path / "bash.bat"
        stub.write_text("@echo off\r\necho %* > \"" + str(log) + "\"\r\nexit /b 7\r\n", encoding="utf-8")
    else:
        stub = tmp_path / "bash"
        stub.write_text("#!/bin/sh\necho \"$@\" > \"" + str(log) + "\"\nexit 7\n", encoding="utf-8")
        stub.chmod(0o755)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: str(stub) if name == "bash" else None)
    rc = cli_main(["gpu", "launch", "plan", "--region", "us-west-2"])
    assert rc == 7
    argv = log.read_text(encoding="utf-8")
    assert "launch.sh" in argv and "plan --region us-west-2" in argv
    assert "$ " in capsys.readouterr().out
