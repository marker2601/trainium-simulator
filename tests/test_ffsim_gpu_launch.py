"""Offline tests for ffsim/gpu/launch.sh and bootstrap_gpu.sh.

``bash -n`` on both scripts, then every subcommand that touches AWS is driven against a fake ``aws`` CLI
(a bash shim on PATH that logs each call). The shim answers with CRLF where aws.exe on Windows does, and
its ``run-instances`` handler is native Python, so the ``--user-data file://...`` argument is opened with
exactly the rules the real (native) aws CLI applies under Git Bash (cygpath -m, not a bare POSIX path).

The invariants under test: ``plan`` never issues a call that spends or destroys; ``up`` and ``down`` refuse
without ``--yes``; ``up`` refuses when an instance tagged ffsim-gpu is already up or the quota is short;
``down`` is idempotent; the rendered user-data carries the --max-hours / --auto-down knobs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / "ffsim" / "gpu"
LAUNCH = GPU / "launch.sh"
BOOTSTRAP = GPU / "bootstrap_gpu.sh"
WINDOWS = os.name == "nt"

SPENDING_CALLS = ("run-instances", "terminate-instances", "create-security-group",
                  "authorize-security-group-ingress", "delete-security-group", "request-service-quota-increase")

FAKE_AWS_SH = r"""#!/usr/bin/env bash
# stand-in for the aws CLI: logs every call, answers what launch.sh asks. CRLF where aws.exe emits it.
printf '%s\n' "$*" >> "$FAKE_AWS_LOG"
case " $* " in
  *" sts get-caller-identity "*) echo "123456789012";;
  *" get-service-quota "*" L-DB2E81BA "*) printf '64.0\r\n';;
  *" get-service-quota "*" L-3819A6DF "*) printf '8.0\r\n';;
  *" ssm get-parameters-by-path "*)
     printf '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.6-ubuntu-22.04/latest/ami-id\tami-0old26\r\n'
     printf '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id\tami-0fake27\r\n';;
  *" ec2 describe-images "*) printf '40\r\n';;
  *" ec2 describe-instances "*"--instance-ids "*) printf '203.0.113.5\r\n';;   # no leading space: it is shared with the previous segment
  *" ec2 describe-instances "*"tag:Name"*)
     [ "${FAKE_RUNNING:-0}" = 1 ] && printf 'i-0running\trunning\t203.0.113.9\tg6e.xlarge\t2026-09-29T00:00:00Z\r\n'; exit 0;;
  *" ec2 describe-instances "*) exit 0;;
  *" ec2 describe-spot-price-history "*)
     printf 'g6e.xlarge\t1.861000\r\ng6e.xlarge\t1.702400\r\ng6e.2xlarge\t1.423800\r\ng6e.12xlarge\t6.098300\r\n'
     printf 'g6e.48xlarge\t9.613000\r\ng5.xlarge\t0.486600\r\ng5.2xlarge\t0.515400\r\ng5.12xlarge\t3.621700\r\ng5.48xlarge\t5.541600\r\n';;
  *" pricing get-products "*)
     case " $* " in
       *"Value=g6e.xlarge "*) p=1.8610000000;; *"Value=g6e.2xlarge "*) p=2.2420800000;; *"Value=g6e.12xlarge "*) p=10.4926400000;;
       *"Value=g6e.48xlarge "*) p=30.1311800000;; *"Value=g5.xlarge "*) p=1.0060000000;; *"Value=g5.2xlarge "*) p=1.2120000000;;
       *"Value=g5.12xlarge "*) p=5.6720000000;; *) p=16.2880000000;;
     esac
     printf '{"terms":{"OnDemand":{"X":{"priceDimensions":{"Y":{"unit":"Hrs","pricePerUnit":{"USD":"%s"}}}}}}}\r\n' "$p";;
  *" ec2 describe-vpcs "*) printf 'vpc-0fake\r\n';;
  *" ec2 describe-security-groups "*) printf 'None\r\n';;
  *" ec2 create-security-group "*) printf 'sg-0fake\r\n';;
  *" ec2 authorize-security-group-ingress "*) exit 0;;
  *" ec2 run-instances "*) exec "$FAKE_AWS_PYTHON" "$FAKE_AWS_PY" "$@";;   # native python: opens file:// like aws.exe
  *" ec2 wait "*) exit 0;;
  *" ec2 terminate-instances "*) printf 'i-0fakegpu\tshutting-down\r\n';;
  *" ec2 delete-security-group "*) exit 0;;
  *" ec2 describe-volumes "*) exit 0;;
  *) echo "fake aws: unexpected call: $*" >&2; exit 99;;
esac
"""

FAKE_AWS_PY = r"""
import sys
args = sys.argv[1:]
def val(flag):
    return args[args.index(flag) + 1]
ud = val("--user-data")
assert ud.startswith("file://"), ud
path = ud[len("file://"):]
text = open(path, encoding="utf-8").read()          # a bare /tmp/... path raises on Windows, exactly like aws.exe
assert text.startswith("#!/bin/bash\n"), text[:40]
assert "FFSIM_MAX_HOURS=" in text and "ffsim-maxhours.timer" in text
assert text.startswith("#!/bin/bash\nFFSIM_MAX_HOURS="), "the FFSIM_* header follows the single leading shebang"
assert "\n#!/bin/bash\n# ffsim/gpu/bootstrap_gpu.sh" not in text, "bootstrap's own shebang must be dropped"
assert val("--security-group-ids") == "sg-0fake"
assert "--key-name" in args
assert val("--instance-initiated-shutdown-behavior") == "terminate"
assert "--block-device-mappings" in args and '"VolumeType":"gp3"' in val("--block-device-mappings")
assert "Value=ffsim-gpu" in val("--tag-specifications")
open(path + ".seen", "w").write("ok")
print("i-0fakegpu")
"""


def _bash() -> str | None:
    """Git Bash (MSYS) on Windows, /bin/bash elsewhere; never the WSL launcher in System32."""
    if not WINDOWS:
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        for parent in Path(git).resolve().parents:
            cand = parent / "usr" / "bin" / "bash.exe"
            if cand.exists():
                return str(cand)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


def _posix(p: Path) -> str:
    s = p.as_posix()
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/{m.group(1).lower()}/{m.group(2)}" if m else s


@pytest.fixture
def harness(tmp_path: Path):
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "aws").write_text(FAKE_AWS_SH, encoding="utf-8", newline="\n")
    (bin_dir / "aws").chmod(0o755)
    (tmp_path / "fake_aws.py").write_text(FAKE_AWS_PY, encoding="utf-8")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "state").mkdir()
    log = tmp_path / "aws.log"
    env = dict(os.environ)
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
        "FAKE_AWS_LOG": _posix(log),
        "FAKE_AWS_PY": str(tmp_path / "fake_aws.py"),
        "FAKE_AWS_PYTHON": sys.executable,
        "TMPDIR": _posix(tmp_path / "tmp"),
        "FFSIM_GPU_STATE_DIR": _posix(tmp_path / "state"),
        "FFSIM_GPU_MYIP": "198.51.100.7",
        "FFSIM_GPU_CONFIRM_SLEEP": "0",
    })
    env.pop("FAKE_RUNNING", None)

    def run(*args: str, **extra_env: str) -> subprocess.CompletedProcess:
        e = dict(env, **extra_env)
        return subprocess.run([bash, LAUNCH.as_posix(), *args], cwd=str(REPO), env=e,
                              capture_output=True, text=True, timeout=300)

    def calls() -> str:
        return log.read_text(encoding="utf-8") if log.exists() else ""

    return run, calls, tmp_path


def test_bash_syntax():
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    for script in (LAUNCH, BOOTSTRAP):
        proc = subprocess.run([bash, "-n", script.as_posix()], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"


def test_usage_and_help(harness):
    run, calls, _ = harness
    proc = run()
    assert proc.returncode == 2
    assert "plan" in proc.stdout and "down" in proc.stdout
    proc = run("--help")
    assert proc.returncode == 0 and "up" in proc.stdout
    proc = run("bogus")
    assert proc.returncode == 2 and "unknown subcommand" in proc.stderr
    assert calls() == ""


def test_plan_is_read_only(harness):
    run, calls, tmp = harness
    proc = run("plan", "--gpu-hours", "20", "--key", "mykey", "--region", "us-west-2")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "L-DB2E81BA = 64.0" in out and "L-3819A6DF = 8.0" in out
    assert "ami-0fake27" in out and "ami-0old26" not in out.split("== AMI")[1].split("\n")[0]   # newest PyTorch wins
    assert "root snapshot 40 GiB" in out and "64 GiB gp3" in out                                  # CR stripped
    assert re.search(r"g6e\.xlarge\s+1\s+4\s+L40S/48G\s+1\.8610\s+1\.7024\s+20\.0\s+37\.22\s+34\.05\s+ok", out)
    assert re.search(r"g6e\.12xlarge\s+4\s+48\s+L40S/48G\s+10\.4926\s+6\.0983\s+5\.0\s+52\.46\s+30\.49\s+ok", out)
    assert re.search(r"g6e\.48xlarge .* NO \(192 > 64\.0-0\)", out)
    assert "tcp/22 from 198.51.100.7/32" in out
    assert "ec2 run-instances --image-id ami-0fake27 --instance-type g6e.xlarge" in out
    assert "--key-name mykey" in out and "--instance-initiated-shutdown-behavior terminate" in out
    assert "Nothing was launched" in out
    log = calls()
    assert log and all(c not in log for c in SPENDING_CALLS), log
    assert "ec2 run-instances" not in log
    uds = list((tmp / "tmp").glob("ffsim-gpu-userdata-*.sh"))
    assert len(uds) == 1
    text = uds[0].read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\nFFSIM_MAX_HOURS=12\nFFSIM_AUTO_DOWN=0\nFFSIM_GRACE_MIN=30\n")
    assert "\n#!/bin/bash\n# ffsim/gpu/bootstrap_gpu.sh" not in text          # bootstrap's own shebang dropped
    assert "# ffsim/gpu/bootstrap_gpu.sh" in text and "ffsim-autodown.timer" in text
    assert "file://" + ("C:/" if WINDOWS else "/") in out or "file:///" in out


def test_plan_spot_and_type_flags(harness):
    run, calls, _ = harness
    proc = run("plan", "--type", "g6e.12xlarge", "--spot", "--max-hours", "3", "--auto-down", "--grace-min", "10", "--gpu-hours", "8")
    assert proc.returncode == 0, proc.stderr
    assert "--instance-type g6e.12xlarge" in proc.stdout
    assert "MarketType=spot" in proc.stdout
    assert "max-hours 3, auto-down 1, grace 10 min" in proc.stdout
    assert all(c not in calls() for c in SPENDING_CALLS)


def test_up_refuses_without_yes_or_key(harness):
    run, calls, _ = harness
    proc = run("up", "--key", "k")
    assert proc.returncode == 3 and "REFUSED" in proc.stderr and "--yes" in proc.stderr
    proc = run("up", "--yes")
    assert proc.returncode == 3 and "--key" in proc.stderr
    assert "run-instances" not in calls()


def test_up_refuses_when_already_running(harness):
    run, calls, _ = harness
    proc = run("up", "--yes", "--key", "k", FAKE_RUNNING="1")
    assert proc.returncode == 3, proc.stderr
    assert "already up" in proc.stderr and "i-0running" in proc.stderr
    assert "run-instances" not in calls()


def test_up_refuses_over_quota(harness):
    run, calls, _ = harness
    proc = run("up", "--yes", "--key", "k", "--type", "g6e.48xlarge")
    assert proc.returncode == 3, proc.stderr
    assert "quota" in proc.stderr and "192" in proc.stderr
    assert "run-instances" not in calls()


def test_up_then_down_roundtrip(harness):
    run, calls, tmp = harness
    pem = tmp / "k.pem"
    pem.write_text("fake")
    proc = run("up", "--yes", "--key", "k", "--ssh-key", _posix(pem), "--max-hours", "3", "--auto-down", "--grace-min", "15")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ESTIMATED COST" in proc.stdout and "3 h x 1.8610" in proc.stdout
    assert "i-0fakegpu running at 203.0.113.5" in proc.stdout
    log = calls()
    assert "ec2 create-security-group" in log
    assert "authorize-security-group-ingress --group-id sg-0fake --protocol tcp --port 22 --cidr 198.51.100.7/32" in log
    assert "ec2 run-instances --image-id ami-0fake27 --instance-type g6e.xlarge" in log
    assert "ec2 wait instance-running --instance-ids i-0fakegpu" in log
    uds = list((tmp / "tmp").glob("ffsim-gpu-userdata-*.sh"))
    assert len(uds) == 1 and (tmp / "tmp" / (uds[0].name + ".seen")).exists()   # the native handler opened file://
    text = uds[0].read_text(encoding="utf-8")
    assert "FFSIM_MAX_HOURS=3\nFFSIM_AUTO_DOWN=1\nFFSIM_GRACE_MIN=15\n" in text
    state = tmp / "state" / "us-west-2.env"
    assert state.exists()
    st = state.read_text(encoding="utf-8")
    assert "INSTANCE_ID=i-0fakegpu\n" in st and "IP=203.0.113.5\n" in st and "SG_ID=sg-0fake\n" in st
    assert "MAX_HOURS=3\n" in st

    # a second up must refuse on the state file alone (the fake tag lookup is empty)
    proc = run("up", "--yes", "--key", "k")
    assert proc.returncode == 3 and "exists" in proc.stderr

    n_before = calls().count("terminate-instances")
    proc = run("down")
    assert proc.returncode == 3 and "REFUSED" in proc.stderr
    assert calls().count("terminate-instances") == n_before and state.exists()

    proc = run("down", "--yes")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ec2 terminate-instances --instance-ids i-0fakegpu" in calls()
    assert "ec2 wait instance-terminated --instance-ids i-0fakegpu" in calls()
    assert "ec2 delete-security-group --group-id sg-0fake" in calls()
    assert not state.exists()

    proc = run("down", "--yes")                       # idempotent: nothing left, exit 0
    assert proc.returncode == 0 and "nothing to terminate" in proc.stdout
    assert calls().count("terminate-instances") == 1


def test_down_finds_instance_by_tag_without_state(harness):
    run, calls, tmp = harness
    proc = run("down", "--yes", FAKE_RUNNING="1")
    assert proc.returncode == 0, proc.stderr
    assert "ec2 terminate-instances --instance-ids i-0running" in calls()


def test_status_and_ssh_commands_need_a_box(harness):
    run, calls, _ = harness
    proc = run("status")
    assert proc.returncode == 0 and "none" in proc.stdout
    for cmd in ("push", "fetch", "ssh"):
        proc = run(cmd)
        assert proc.returncode == 2 and "no running ffsim-gpu instance" in proc.stderr, cmd
    proc = run("run")
    assert proc.returncode == 2 and "--config" in proc.stderr
    assert all(c not in calls() for c in SPENDING_CALLS)


def test_only_aws_rw_spends():
    text = LAUNCH.read_text(encoding="utf-8")
    assert 'aws_rw "${RUN_ARGS[@]}"' in text
    assert "aws_rw ec2 terminate-instances" in text
    for bad in ("aws_ro ec2 run-instances", "aws_ro ec2 terminate-instances", "aws ec2 run-instances", "aws ec2 terminate-instances"):
        assert bad not in text, bad
    body = text.split("aws_rw() {", 1)[1]
    assert '[ "$YES" = 1 ] || { echo "REFUSED' in body.split("}", 1)[0] + "}"
    assert "MSYS_NO_PATHCONV=1" in text and "cygpath -m" in text
    assert "ConnectTimeout" in text and re.search(r"^\s*timeout\s", text, re.M) is None


def test_bootstrap_guard_contents():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n")
    for needle in (': "${FFSIM_MAX_HOURS:=12}"', ': "${FFSIM_AUTO_DOWN:=0}"', ': "${FFSIM_GRACE_MIN:=30}"',
                   "ffsim-maxhours.timer", "ffsim-autodown.timer", "OnBootSec=${FFSIM_MAX_HOURS}h",
                   "OnUnitActiveSec=1min", "/sbin/poweroff", "fleet.done", "BOOTSTRAP_DONE", "BOOTSTRAP_WARN",
                   "NEURON_COMPETITION_R1_CACHE_DIR=", "python_path", "unattended-upgrades",
                   "systemctl enable --now ffsim-maxhours.timer ffsim-autodown.timer"):
        assert needle in text, needle
    assert "terminate-instances" not in text          # poweroff + shutdown-behavior=terminate: no IAM needed
    assert "pyarrow tiktoken requests" in text


def test_run_resume_filter_logic(tmp_path: Path):
    """The python that `run` executes on the box drops labels that already have a scored results row."""
    import json
    text = LAUNCH.read_text(encoding="utf-8")
    snippet = text.split("<<'PY'", 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]     # the heredoc body
    script = tmp_path / "resume.py"
    script.write_text(snippet, encoding="utf-8")
    cfg = {"_comment": "x", "defaults": {"steps": 300}, "pairs": ["a_vs_b"],
           "runs": [{"label": "base_s73", "arm": "base"}, {"label": "k60_s73", "arm": "k60"}, {"label": "k60_s58", "arm": "k60"}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    rows = [{"label": "base_s73", "val_bpb": 0.9591}, {"label": "k60_s58", "status": "exit 1"}, "not json",
            {"label": "other_set", "val_bpb": 1.0}]
    (tmp_path / "res.jsonl").write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "filtered.json"
    proc = subprocess.run([sys.executable, str(script), str(tmp_path / "cfg.json"), str(tmp_path / "res.jsonl"), str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "1 of 3 labels already scored" in proc.stdout and "base_s73" in proc.stdout
    filtered = json.loads(out.read_text(encoding="utf-8"))
    assert [r["label"] for r in filtered["runs"]] == ["k60_s73", "k60_s58"]      # a failed row is re-run
    assert filtered["pairs"] == ["a_vs_b"] and filtered["defaults"] == {"steps": 300}
    # everything scored -> exit 1 (the remote wrapper then starts nothing)
    (tmp_path / "res.jsonl").write_text("\n".join(json.dumps({"label": l, "val_bpb": 0.96}) for l in ("base_s73", "k60_s73", "k60_s58")),
                                        encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(tmp_path / "cfg.json"), str(tmp_path / "res.jsonl"), str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 1 and "nothing to run" in proc.stdout


def test_rendered_userdata_parses(harness):
    run, _, tmp = harness
    proc = run("plan", "--max-hours", "7")
    assert proc.returncode == 0, proc.stderr
    ud = next((tmp / "tmp").glob("ffsim-gpu-userdata-*.sh"))
    bash = _bash()
    proc = subprocess.run([bash, "-n", _posix(ud)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    text = ud.read_text(encoding="utf-8")
    assert "FFSIM_MAX_HOURS=7\n" in text
    # the header wins over the standalone defaults: `: "${VAR:=default}"` keeps a preset value
    assert text.index("FFSIM_MAX_HOURS=7\n") < text.index(': "${FFSIM_MAX_HOURS:=12}"')
