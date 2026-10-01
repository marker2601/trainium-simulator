"""Offline tests for ffsim/cloud: ec2_launch.sh driven against a fake aws CLI, plus the Docker build context.

The fake ``aws`` is a bash shim on PATH. Its ``run-instances`` handler is *native* Python, so the
``--user-data file://...`` argument is opened with exactly the rules the real (native) aws CLI applies.
Under Git Bash on Windows that is where ``file:///tmp/...`` used to fail after the tarball had already
been uploaded (review finding F9): MSYS rewrites bare POSIX paths for native exes but not URL-shaped ones.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLOUD = REPO / "ffsim" / "cloud"
SCRIPT = CLOUD / "ec2_launch.sh"
DOCKERFILE = CLOUD / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
WINDOWS = os.name == "nt"

FAKE_AWS_SH = r"""#!/usr/bin/env bash
# stand-in for the aws CLI: logs every call, answers the few commands ec2_launch.sh issues
printf '%s\n' "$*" >> "$FAKE_AWS_LOG"
case " $* " in
  *" sts get-caller-identity "*) echo "123456789012 arn:aws:iam::123456789012:user/pytest AIDAPYTEST";;
  *" describe-instance-type-offerings "*) echo "t3.large";;
  *" iam get-instance-profile "*) echo "arn:aws:iam::123456789012:instance-profile/pytest";;
  *" s3 ls "*) exit 0;;
  *" s3 cp "*) exit 0;;
  *" s3 presign "*) echo "https://example.invalid/ffsim/repo.tgz?X-Amz-Signature=pytest";;
  *" run-instances "*) exec "$FAKE_AWS_PYTHON" "$FAKE_AWS_PY" "$@";;   # native python: opens file:// like aws.exe
  *) echo "fake aws: unexpected call: $*" >&2; exit 99;;
esac
"""

FAKE_AWS_PY = r"""import os, sys
args = sys.argv[1:]
ud = args[args.index("--user-data") + 1]
if not ud.startswith("file://"):
    sys.exit("fake aws: --user-data is not a file:// URL: %r" % (ud,))
path = ud[len("file://"):]                       # what awscli/paramfile.py opens, verbatim
try:
    with open(os.path.expandvars(os.path.expanduser(path)), encoding="utf-8") as fh:
        fh.read()
    with open(os.path.expandvars(os.path.expanduser(path)), "rb") as fh:
        raw = fh.read()                          # byte-exact copy for the test to inspect
except OSError as exc:
    sys.exit("fake aws: cannot open user-data %r: %s" % (ud, exc))
with open(os.environ["FAKE_AWS_USERDATA_COPY"], "wb") as fh:
    fh.write(raw)
print("i-0pytest0000")
"""

READ_ONLY_CALLS = (" sts get-caller-identity ", " ec2 describe-instance-type-offerings ",
                   " iam get-instance-profile ", " s3 ls ")


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
    """The path as Git Bash / MSYS tar want it (/c/...); unchanged on POSIX."""
    s = p.resolve().as_posix()
    if WINDOWS and re.match(r"^[A-Za-z]:/", s):
        return "/" + s[0].lower() + s[2:]
    return s


def _launch(tmp_path: Path, *args: str):
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "aws").write_text(FAKE_AWS_SH, newline="\n")
    (shim / "aws").chmod(0o755)
    fake_py = tmp_path / "fake_aws.py"
    fake_py.write_text(FAKE_AWS_PY, newline="\n")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    paths = {"log": tmp_path / "aws-calls.log", "userdata_copy": tmp_path / "userdata-as-read.sh", "tmpdir": tmpdir}
    env = dict(os.environ)
    # The bash executable's own folder (Git's usr/bin on Windows) holds date, sed and the other tools the scripts
    # call; PowerShell sessions often lack it on PATH.
    env["PATH"] = os.pathsep.join([str(shim), str(Path(bash).parent), env.get("PATH", "")])
    env["TMPDIR"] = _posix(tmpdir)
    env["FAKE_AWS_LOG"] = paths["log"].as_posix()
    env["FAKE_AWS_PY"] = fake_py.as_posix()
    env["FAKE_AWS_USERDATA_COPY"] = paths["userdata_copy"].as_posix()
    env["FAKE_AWS_PYTHON"] = Path(sys.executable).as_posix()
    cmd = [bash, SCRIPT.as_posix(), "--profile", "pytest", "--region", "us-east-1", "--bucket", "pytest-bucket",
           "--stamp", "pytest", *args]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    return proc, paths


def test_cloud_scripts_parse():
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    for name in ("ec2_launch.sh", "run_batch.sh"):
        proc = subprocess.run([bash, "-n", (CLOUD / name).as_posix()], capture_output=True, text=True)
        assert proc.returncode == 0, (name, proc.stderr)


def test_dry_run_only_reads(tmp_path):
    proc, p = _launch(tmp_path, "--instance-profile", "ffsim-writer")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dry run: nothing launched" in proc.stdout
    calls = p["log"].read_text().splitlines()
    assert calls
    for call in calls:
        assert any(k in " " + call + " " for k in READ_ONLY_CALLS), call
    assert not list(p["tmpdir"].iterdir()), "dry run wrote temp files"
    assert not p["userdata_copy"].exists()
    # the printed plan shows the exact user-data argument run-instances will get
    assert "--user-data file://" in proc.stdout


def test_yes_hands_run_instances_a_user_data_file_the_native_cli_can_open(tmp_path):
    proc, p = _launch(tmp_path, "--yes", "--spot", "--instance-profile", "ffsim-writer", "--n-sims", "50", "--top", "5")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "launched i-0pytest0000" in proc.stdout
    calls = p["log"].read_text().splitlines()
    writes = [c for c in calls if any(k in " " + c + " " for k in (" s3 cp ", " s3 presign ", " run-instances "))]
    kinds = ["s3 cp" if " s3 cp " in w else "s3 presign" if " s3 presign " in w else "run-instances" for w in writes]
    assert kinds == ["s3 cp", "s3 presign", "run-instances"], writes   # upload, presign, then launch
    run = writes[-1]
    assert "MarketType=spot" in run and "Name=ffsim-writer" in run and "Value=ffsim-pytest" in run
    ud = re.search(r"--user-data (\S+)", run).group(1)
    if WINDOWS:
        # F9: a native aws.exe cannot open file:///tmp/... or file:///c/...; it needs file://C:/...
        assert re.match(r"^file://[A-Za-z]:/", ud), ud
    else:
        assert ud.startswith("file:///"), ud
    # the native handler actually opened and read it; the instance is Linux, so no CR may reach the shebang
    raw = p["userdata_copy"].read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.startswith("#!/bin/bash\n")
    assert "https://example.invalid/ffsim/repo.tgz?X-Amz-Signature=pytest" in text
    assert 'bash ffsim/cloud/run_batch.sh "ffsim/examples/space-k59-local.json" "local" "50" "5" "2"' in text
    assert "s3://pytest-bucket/ffsim/pytest/results" in text
    assert text.rstrip().endswith("shutdown -h now")
    tarball = p["tmpdir"] / "ffsim-repo-pytest.tgz"
    assert tarball.exists()
    with tarfile.open(tarball) as tf:
        names = tf.getnames()
    assert "ffsim/cloud/run_batch.sh" in names
    assert any(n.startswith("research/sim-data") for n in names)


def test_dockerignore_sends_only_what_the_dockerfile_copies():
    lines = [ln.strip() for ln in DOCKERIGNORE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines[0] == "*", lines
    copied = re.findall(r"^COPY\s+(\S+)\s", DOCKERFILE.read_text(), re.M)
    assert copied, "no COPY lines found in the Dockerfile"
    for src in copied:
        assert "!" + src in lines, (src, lines)
