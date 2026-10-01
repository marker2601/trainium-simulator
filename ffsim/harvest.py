"""ffsim.harvest: pull run directories off chips C/D over AWS SSM (read-only on the chip).

Reproduces the manual pull documented in research/sim-data/chip{C,D}/HARVEST_NOTES.md:

  1. inventory  - one light SSM invocation: run names, finished markers (t_end), file counts.
  2. build      - one invocation: filter every run dir into /tmp/simharvest/chip<X>/<run>/ on the chip
                  (train.log minus the CONTRACT.md noise lines, eval*.log reduced to val_bpb/eval/tokens/bytes/error
                  lines, small files copied verbatim, steps.log only when <= 20 KB), plus _meta/ (queue runner.log,
                  base.env, driver_* files, code dir sha256s, manifests), tar+gzip it, print size + sha256 and the
                  first base64 chunk.
  3. chunks     - the instance role cannot write S3 (PutObject denied, tested 29 Sep 2026), and SSM stdout is capped
                  at 24,000 chars, so the tarball is fetched as base64 slices of <= 21,000 chars, one invocation
                  each, strictly sequentially (parallel SSM calls serialise badly through the credential process).
  4. verify     - reassemble, check sha256, extract into research/sim-data/chip<X>/, write INDEX.json.

Pure stdlib (subprocess -> aws CLI). Nothing on the chip is written outside /root/ff-claude/cmds (the dispatcher's
script drop) and /tmp/simharvest. No systemd, no queue, no kills.

CLI:  python -m ffsim harvest --chip C [--all] [--dest DIR] [--wait S] [--chunk N] [--inventory-only]
      python -m ffsim.harvest --chip D
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Nothing account-specific is baked in; the chips are named through the environment:
#   FFSIM_AWS_PROFILE (else AWS_PROFILE, else "default"); FFSIM_AWS_ACCOUNT (optional: when set, the profile
#   must resolve to this account); FFSIM_CHIP_<X>_REGION and FFSIM_CHIP_<X>_INSTANCE for X in C, D.
PROFILE = os.environ.get("FFSIM_AWS_PROFILE") or os.environ.get("AWS_PROFILE") or "default"
ACCOUNT = os.environ.get("FFSIM_AWS_ACCOUNT", "")
CHIPS: Dict[str, Dict[str, str]] = {
    chip: {"region": os.environ.get(f"FFSIM_CHIP_{chip}_REGION", ""),
           "instance": os.environ.get(f"FFSIM_CHIP_{chip}_INSTANCE", "")}
    for chip in ("C", "D")
}
RUNS_DIR = "/root/ff-claude/runs"
QUEUE_DIR = "/root/ff-claude/queue"
WORK_DIR = "/tmp/simharvest"
CMDS_DIR = "/root/ff-claude/cmds"
SSM_STDOUT_CAP = 24000
DEFAULT_CHUNK = 21000
STEPS_LOG_MAX = 20480

# CONTRACT.md noise lines to drop from train.log (everything else is kept, so every 'step ', FF_*, phase/stage,
# val_bpb/charged/median/saved/level/exit line survives).
NOISE_RE = r"OperatorEntry|registered at|dispatch key|new kernel|previous kernel|operator:|W[0-9]{4}.*torch/distributed/run\.py|ShardIndexInjection"
# Opt-in (--aggressive) extra train.log noise, NOT in CONTRACT.md: NCCL/OFI plugin warnings, torch autocast /
# barrier() warnings and their continuation line. On chip D these were ~74% of the surviving non-step lines.
EXTRA_NOISE_RE = r"CCOM WARN|WARNING:torch_neuronx\.python_ops\.dtype_autocast|UserWarning: barrier\(\)|^  return func\(\*args, \*\*kwargs\)$"
# eval.log / eval20.log / eval_ema.log: keep only these (case-insensitive).
EVAL_KEEP_RE = r"val_bpb|eval|tokens|bytes|error"
SMALL_FILES = ("overrides", "code.sha256", "full_run.status", "timing.txt", "exit_code", "eval_exit_code", "eval_ema_exit_code")
EVAL_LOGS = ("eval.log", "eval20.log", "eval_ema.log")
QUEUE_FILES = ("runner.log", "base.env", "queue_runner.sh", "watchdog.sh")

_B64_LINE = re.compile(r"^[A-Za-z0-9+/=]+$")


class HarvestError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_dest(chip: str) -> Path:
    return _repo_root() / "research" / "sim-data" / f"chip{chip}"


# --------------------------------------------------------------------------------------------------------------- SSM


def _aws() -> str:
    exe = shutil.which("aws")
    if not exe:
        raise HarvestError("aws CLI not found on PATH")
    return exe


def _run_aws(args: Sequence[str], timeout: float = 120.0) -> str:
    """Run one aws CLI call and return stdout (CRLF stripped). Raises HarvestError on non-zero exit."""
    cmd = [_aws(), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise HarvestError(f"aws {' '.join(args[:3])} failed rc={p.returncode}: {p.stderr.strip()[:500]}")
    return p.stdout.replace("\r", "")


_identity_ok = False


def check_identity(profile: str = PROFILE, retries: int = 3) -> str:
    """Verify the profile resolves to the expected account (done once per process)."""
    global _identity_ok
    if _identity_ok:
        return ACCOUNT
    last = ""
    for _ in range(retries):
        try:
            acct = _run_aws(["sts", "get-caller-identity", "--profile", profile, "--query", "Account", "--output", "text"]).strip()
        except HarvestError as e:  # credential process hiccup: retry
            last = str(e)
            time.sleep(2)
            continue
        if not ACCOUNT or acct == ACCOUNT:
            _identity_ok = True
            return acct
        raise HarvestError(f"ABORT identity={acct!r} (expected {ACCOUNT})")
    raise HarvestError(f"ABORT could not resolve identity: {last}")


def _ssm_parameters(script_text: str) -> dict:
    """Same wrapping as the campaign's SSM dispatcher: gzip+base64 the script, drop it under /root/ff-claude/cmds, run it."""
    b64 = base64.b64encode(gzip.compress(script_text.encode("utf-8"))).decode("ascii")
    cmds = [
        f"mkdir -p {CMDS_DIR}",
        f"f=$(mktemp {CMDS_DIR}/cmd.XXXXXX)",
        "echo '" + b64 + "' | base64 -d | gunzip > \"$f\"",
        "bash -c \"set -o pipefail; bash $f 2>&1 | tr -cd '\\11\\12\\15\\40-\\176'\"",
    ]
    return {"commands": cmds, "executionTimeout": ["1800"]}


def dispatch(script_text: str, region: str, instance: str, wait: float = 150.0, profile: str = PROFILE,
             poll: float = 3.0, first_poll: float = 4.0, raise_on_fail: bool = True) -> str:
    """Run a bash script on the instance via AWS-RunShellScript and return its stdout (<= 24,000 chars).

    Polls Status every `poll` seconds (politely) until Success/Failed/Cancelled/TimedOut or `wait` seconds elapse.
    Status and stdout are fetched in the same get-command-invocation call to keep the local aws-call count low
    (each aws call costs seconds through the credential process).
    """
    check_identity(profile)
    params = _ssm_parameters(script_text)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(params, fh)
        cid = _run_aws([
            "ssm", "send-command", "--instance-ids", instance, "--document-name", "AWS-RunShellScript",
            "--comment", "claude-sim-harvest", "--parameters", f"file://{tmp}", "--region", region,
            "--profile", profile, "--query", "Command.CommandId", "--output", "text",
        ]).strip()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if not cid:
        raise HarvestError("ABORT no command id")
    deadline = time.monotonic() + wait
    status, out, err = "Pending", "", ""
    time.sleep(first_poll)
    while True:
        try:
            raw = _run_aws([
                "ssm", "get-command-invocation", "--command-id", cid, "--instance-id", instance, "--region", region,
                "--profile", profile, "--output", "json",
            ])
            inv = json.loads(raw)
            status = inv.get("Status", "Pending")
            out = inv.get("StandardOutputContent", "") or ""
            err = inv.get("StandardErrorContent", "") or ""
        except HarvestError:
            status = "Pending"  # invocation not registered yet
        except json.JSONDecodeError:
            status = "Pending"
        if status in ("Success", "Failed", "Cancelled", "TimedOut"):
            break
        if time.monotonic() > deadline:
            break
        time.sleep(poll)
    out = out.replace("\r", "")
    if status != "Success" and raise_on_fail:
        raise HarvestError(f"SSM command {cid} status={status}; stderr={err.strip()[:400]!r}; stdout_tail={out[-300:]!r}")
    return out


# -------------------------------------------------------------------------------------------------- on-chip scripts


def inventory_script() -> str:
    return f"""set -u
R={RUNS_DIR}
echo "host=${{FF_HOST_LABEL:-<HOST>}} date=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "--- runs ---"
for d in "$R"/*/; do n=$(basename "$d"); t=0; [ -f "$d/t_end" ] && t=1; l=0; [ -f "$d/train.log" ] && l=$(stat -c %s "$d/train.log"); e=0; [ -f "$d/eval.log" ] && e=1; echo "RUN $n t_end=$t train_bytes=$l eval=$e"; done
echo "--- drivers ---"; ls "$R"/driver_* 2>/dev/null
echo "--- xz ---"; which xz || echo none
"""


def build_script(chip: str, skip: Sequence[str] = (), chunk: int = DEFAULT_CHUNK, use_xz: bool = False,
                 aggressive: bool = False) -> str:
    """The filter+tar script. `skip` = run names already harvested locally (finished), left out of the tarball."""
    skip_lines = "\n".join(skip)
    comp = "xz -9 -T2" if use_xz else "gzip -6"
    ext = "txz" if use_xz else "tgz"
    noise = NOISE_RE + ("|" + EXTRA_NOISE_RE if aggressive else "")
    return f"""set -u
R={RUNS_DIR}
Q={QUEUE_DIR}
W={WORK_DIR}
C=chip{chip}
rm -rf "$W"; mkdir -p "$W/$C/_meta"
cat > "$W/skip.txt" <<'SKIP'
{skip_lines}
SKIP
NOISE='{noise}'
EVALKEEP='{EVAL_KEEP_RE}'
for d in "$R"/*/; do
  n=$(basename "$d")
  grep -q -x -F -- "$n" "$W/skip.txt" && continue
  o="$W/$C/$n"; mkdir -p "$o"
  [ -f "$d/train.log" ] && grep -a -v -E "$NOISE" "$d/train.log" > "$o/train.log"
  for e in {' '.join(EVAL_LOGS)}; do [ -f "$d/$e" ] && grep -a -i -E "$EVALKEEP" "$d/$e" > "$o/$e"; done
  for f in {' '.join(SMALL_FILES)}; do [ -f "$d/$f" ] && cp -p "$d/$f" "$o/"; done
  if [ -f "$d/steps.log" ] && [ "$(stat -c %s "$d/steps.log")" -le {STEPS_LOG_MAX} ]; then cp -p "$d/steps.log" "$o/"; fi
  for f in "$d"/t_*; do [ -f "$f" ] && cp -p "$f" "$o/"; done
done
M="$W/$C/_meta"
for f in {' '.join(QUEUE_FILES)}; do [ -f "$Q/$f" ] && cp -p "$Q/$f" "$M/"; done
for f in "$R"/driver_*.log "$R"/driver_*.sh; do [ -f "$f" ] && cp -p "$f" "$M/"; done
{{ for c in /root/ff-claude/code*/; do f="$c/train_ff.py"; if [ -f "$f" ]; then echo "$(sha256sum "$f" | cut -c1-64)	$(stat -c %s "$f")	$(stat -c %y "$f" | cut -c1-19)	$c"; else echo "-	-	-	$c (no train_ff.py)"; fi; done; }} > "$M/code_dirs.txt"
{{ for q in done pending parked held failed running; do echo "## $q"; ls -la --time-style=+%FT%TZ "$Q/$q" 2>/dev/null | awk 'NR>1 && $1 !~ /^d/ {{print $5 "\\t" $6 "\\t" $7}}'; done; }} > "$M/queue_listing.txt"
{{ echo "host=${{FF_HOST_LABEL:-<HOST>}}"; echo "chip={chip}"; echo "harvested_utc=$(date -u +%FT%TZ)"; echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"; echo "## neuron packages (dpkg)"; dpkg -l 2>/dev/null | awk '/aws-neuron/ {{print $2 "\\t" $3}}'; }} > "$M/chip_info.txt"
{{ for d in "$R"/*/; do n=$(basename "$d"); for f in "$d"/*; do [ -f "$f" ] && echo "$n/$(basename "$f")	$(stat -c %s "$f")	$(stat -c %y "$f" | cut -c1-19)"; done; done; for f in "$R"/driver_*; do [ -f "$f" ] && echo "_meta/$(basename "$f")	$(stat -c %s "$f")	$(stat -c %y "$f" | cut -c1-19)"; done; }} > "$M/on_chip_manifest.tsv"
for d in "$R"/*/; do basename "$d"; done > "$M/run_names.txt"
cd "$W" && tar -cf - "$C" | {comp} > corpus.{ext}
echo "TGZ_NAME corpus.{ext}"
echo "TGZ_BYTES $(stat -c %s corpus.{ext})"
echo "TGZ_SHA256 $(sha256sum corpus.{ext} | cut -c1-64)"
echo "B64_LEN $(base64 -w0 corpus.{ext} | wc -c)"
echo "FILTERED_BYTES $(du -sb "$C" | cut -f1) FILES $(find "$C" -type f | wc -l)"
echo "CHUNK_BEGIN 1 {chunk}"
base64 -w0 corpus.{ext} | cut -c1-{chunk}
echo
echo "CHUNK_END"
"""


def chunk_script(start: int, end: int, name: str = "corpus.tgz") -> str:
    """1-based inclusive character range of `base64 -w0 <tarball>` (what `cut -c` expects)."""
    return f"""cd {WORK_DIR} || exit 9
echo "CHUNK_BEGIN {start} {end}"
base64 -w0 {name} | cut -c{start}-{end}
echo
echo "CHUNK_END"
"""


# -------------------------------------------------------------------------------------------------- output parsing


def parse_chunk(stdout: str) -> str:
    m = re.search(r"CHUNK_BEGIN[^\n]*\n(.*?)\nCHUNK_END", stdout, re.S)
    if not m:
        raise HarvestError("chunk markers not found in SSM stdout: " + stdout[-200:].strip())
    body = "".join(m.group(1).split())
    if body and not _B64_LINE.match(body):
        raise HarvestError("chunk contains non-base64 characters")
    return body


def parse_kv(stdout: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)} (\S+)", stdout, re.M)
    return m.group(1) if m else None


def parse_inventory(stdout: str) -> Dict[str, dict]:
    runs: Dict[str, dict] = {}
    for m in re.finditer(r"^RUN (\S+) t_end=(\d) train_bytes=(\d+) eval=(\d)", stdout, re.M):
        runs[m.group(1)] = {"finished": m.group(2) == "1", "train_bytes": int(m.group(3)), "has_eval": m.group(4) == "1"}
    return runs


# ------------------------------------------------------------------------------------------------------ local side


def local_finished_runs(dest_dir: Path) -> List[str]:
    """Runs already harvested locally AND finished on the chip when harvested (t_end present)."""
    if not dest_dir.is_dir():
        return []
    return sorted(p.name for p in dest_dir.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "t_end").is_file())


def _safe_extract(tar_bytes: bytes, dest_dir: Path, top: str) -> List[str]:
    """Extract members `top/<rel>` into dest_dir/<rel>; refuse anything escaping dest_dir. Returns rel paths."""
    written: List[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if len(parts) < 2 or parts[0] != top or any(p in ("..", "") for p in parts):
                raise HarvestError(f"unexpected tar member {m.name!r}")
            rel = Path(*parts[1:])
            target = (dest_dir / rel).resolve()
            if root != target and root not in target.parents:
                raise HarvestError(f"tar member escapes dest: {m.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
            written.append(rel.as_posix())
    return written


def _count_lines(path: Path, pattern: str, flags: int = 0) -> int:
    rx = re.compile(pattern, flags)
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if rx.search(line):
                    n += 1
    except OSError:
        return 0
    return n


def write_index(chip: str, dest_dir: Path, on_chip_runs: Optional[Sequence[str]] = None, extra: Optional[dict] = None) -> dict:
    """INDEX.json: every run dir under dest_dir with the files present, their sizes and cheap sanity counts."""
    runs: Dict[str, dict] = {}
    for p in sorted(dest_dir.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        files = {f.name: f.stat().st_size for f in sorted(p.iterdir()) if f.is_file()}
        entry = {
            "files": files,
            "bytes": sum(files.values()),
            "finished": "t_end" in files,
            "has_train_log": "train.log" in files,
            "has_eval_log": "eval.log" in files,
            "has_eval20_log": "eval20.log" in files,
            "step_lines": _count_lines(p / "train.log", r"^step \d+") if "train.log" in files else 0,
            "eval_val_bpb_lines": _count_lines(p / "eval.log", r"val_bpb") if "eval.log" in files else 0,
        }
        runs[p.name] = entry
    meta_dir = dest_dir / "_meta"
    meta = {f.name: f.stat().st_size for f in sorted(meta_dir.iterdir()) if f.is_file()} if meta_dir.is_dir() else {}
    missing = sorted(set(on_chip_runs or []) - set(runs))
    index = {
        "chip": chip,
        "instance": CHIPS.get(chip, {}).get("instance"),
        "region": CHIPS.get(chip, {}).get("region"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_count": len(runs),
        "on_chip_run_count": len(on_chip_runs) if on_chip_runs is not None else None,
        "missing_runs": missing,
        "file_count": sum(len(r["files"]) for r in runs.values()) + len(meta),
        "bytes": sum(r["bytes"] for r in runs.values()) + sum(meta.values()),
        "runs": runs,
        "_meta": meta,
    }
    if extra:
        index["harvest"] = extra
    with open(dest_dir / "INDEX.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return index


# ------------------------------------------------------------------------------------------------------------ pull


def pull_chip(chip: str, region: Optional[str] = None, instance: Optional[str] = None, dest_dir: Optional[os.PathLike] = None,
              only_new: bool = True, wait: float = 150.0, chunk: int = DEFAULT_CHUNK, inventory_only: bool = False,
              retries: int = 2, log: Callable[[str], None] = print, keep_tarball: Optional[os.PathLike] = None) -> dict:
    """Harvest chip `chip` into dest_dir (default research/sim-data/chip<X>/). Returns a report dict.

    only_new=True skips run dirs that already exist locally with a t_end stamp (finished when last pulled); runs
    still in progress at the last pull are re-pulled. _meta/ is always refreshed.
    """
    chip = chip.upper()
    if chip not in CHIPS and not (region and instance):
        raise HarvestError(f"unknown chip {chip!r}; pass region and instance")
    region = region or CHIPS[chip]["region"]
    instance = instance or CHIPS[chip]["instance"]
    if not (region and instance):
        raise HarvestError(f"chip {chip}: set FFSIM_CHIP_{chip}_REGION and FFSIM_CHIP_{chip}_INSTANCE "
                           f"(or pass region and instance)")
    dest = Path(dest_dir) if dest_dir else default_dest(chip)
    if chunk > SSM_STDOUT_CAP - 500:
        raise HarvestError(f"chunk {chunk} too close to the {SSM_STDOUT_CAP}-char SSM stdout cap")
    t0 = time.monotonic()
    report: dict = {"chip": chip, "region": region, "instance": instance, "dest": str(dest), "transport": "ssm-base64-chunks",
                    "invocations": 0, "failed_chunks": 0, "problems": []}

    log(f"[harvest {chip}] inventory ...")
    inv_out = dispatch(inventory_script(), region, instance, wait=min(wait, 120), poll=3.0)
    report["invocations"] += 1
    on_chip = parse_inventory(inv_out)
    report["on_chip_runs"] = sorted(on_chip)
    report["on_chip_finished"] = sorted(n for n, v in on_chip.items() if v["finished"])
    report["on_chip_train_bytes"] = sum(v["train_bytes"] for v in on_chip.values())
    use_xz = "xz" in inv_out.split("--- xz ---", 1)[-1] and "none" not in inv_out.split("--- xz ---", 1)[-1]
    log(f"[harvest {chip}] on chip: {len(on_chip)} runs, {len(report['on_chip_finished'])} finished, "
        f"{report['on_chip_train_bytes']/1e6:.1f} MB train.log; xz={'yes' if use_xz else 'no'}")
    if inventory_only:
        report["seconds"] = round(time.monotonic() - t0, 1)
        return report

    skip = [n for n in local_finished_runs(dest) if n in on_chip] if only_new else []
    todo = sorted(set(on_chip) - set(skip))
    report["skipped_local_finished"] = skip
    report["pulled_runs"] = todo
    log(f"[harvest {chip}] pulling {len(todo)} runs (skipping {len(skip)} already harvested and finished) ...")

    log(f"[harvest {chip}] build filtered corpus on chip ...")
    build_out = dispatch(build_script(chip, skip=skip, chunk=chunk, use_xz=use_xz), region, instance, wait=wait, poll=3.0)
    report["invocations"] += 1
    name = parse_kv(build_out, "TGZ_NAME") or "corpus.tgz"
    tgz_bytes = int(parse_kv(build_out, "TGZ_BYTES") or 0)
    tgz_sha = parse_kv(build_out, "TGZ_SHA256") or ""
    b64_len = int(parse_kv(build_out, "B64_LEN") or 0)
    if not (tgz_bytes and tgz_sha and b64_len):
        raise HarvestError("build step did not report TGZ_BYTES/TGZ_SHA256/B64_LEN: " + build_out[-400:])
    report.update({"tarball": name, "tarball_bytes": tgz_bytes, "tarball_sha256": tgz_sha, "b64_len": b64_len,
                   "filtered_bytes_on_chip": int(parse_kv(build_out, "FILTERED_BYTES") or 0)})
    n_chunks = (b64_len + chunk - 1) // chunk
    report["chunks"] = n_chunks
    log(f"[harvest {chip}] tarball {tgz_bytes} bytes sha256 {tgz_sha[:12]}..., base64 {b64_len} chars -> {n_chunks} chunks")

    pieces: List[str] = [parse_chunk(build_out)]
    if len(pieces[0]) != min(chunk, b64_len):
        raise HarvestError(f"chunk 1 length {len(pieces[0])} != {min(chunk, b64_len)}")
    for i in range(2, n_chunks + 1):
        start = (i - 1) * chunk + 1
        end = min(i * chunk, b64_len)
        want = end - start + 1
        body = ""
        for attempt in range(retries + 1):
            try:
                out = dispatch(chunk_script(start, end, name), region, instance, wait=min(wait, 120), poll=3.0)
                report["invocations"] += 1
                body = parse_chunk(out)
                if len(body) == want:
                    break
                raise HarvestError(f"chunk {i} length {len(body)} != {want}")
            except HarvestError as e:
                report["failed_chunks"] += 1
                report["problems"].append(f"chunk {i} attempt {attempt + 1}: {e}")
                body = ""
                if attempt < retries:
                    time.sleep(3)
        if len(body) != want:
            raise HarvestError(f"chunk {i} could not be fetched after {retries + 1} attempts")
        pieces.append(body)
        log(f"[harvest {chip}] chunk {i}/{n_chunks} ok ({time.monotonic() - t0:.0f}s)")

    b64 = "".join(pieces)
    blob = base64.b64decode(b64, validate=True)
    sha = hashlib.sha256(blob).hexdigest()
    if len(blob) != tgz_bytes or sha != tgz_sha:
        raise HarvestError(f"tarball mismatch: got {len(blob)} bytes sha {sha}, chip said {tgz_bytes} {tgz_sha}")
    if keep_tarball:
        Path(keep_tarball).parent.mkdir(parents=True, exist_ok=True)
        Path(keep_tarball).write_bytes(blob)
    written = _safe_extract(blob, dest, top=f"chip{chip}")
    report["files_written"] = len(written)
    report["seconds"] = round(time.monotonic() - t0, 1)
    idx = write_index(chip, dest, on_chip_runs=sorted(on_chip), extra={
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "transport": report["transport"],
        "tarball_bytes": tgz_bytes, "tarball_sha256": tgz_sha, "chunks": n_chunks, "chunk_chars": chunk,
        "invocations": report["invocations"], "failed_chunks": report["failed_chunks"],
        "pulled_runs": todo, "skipped_local_finished": skip, "on_chip_runs": sorted(on_chip),
    })
    report["local_runs"] = idx["run_count"]
    report["missing_runs"] = idx["missing_runs"]
    report["local_train_logs"] = sum(1 for r in idx["runs"].values() if r["has_train_log"])
    log(f"[harvest {chip}] done: {idx['run_count']} runs local ({len(on_chip)} on chip), {report['files_written']} files written, "
        f"{report['invocations']} SSM invocations, {report['failed_chunks']} failed chunks, {report['seconds']}s")
    return report


# ------------------------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffsim harvest", description="Pull run dirs off a Trainium chip over SSM (read-only on chip).")
    p.add_argument("--chip", required=True, choices=sorted(CHIPS), help="C or D")
    p.add_argument("--dest", default=None, help="destination dir (default research/sim-data/chip<X>/)")
    p.add_argument("--all", action="store_true", help="re-pull every run (default: only runs not yet harvested or unfinished)")
    p.add_argument("--wait", type=float, default=150.0, help="seconds to wait for the build invocation")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="base64 chars per SSM invocation (<= 21000 recommended)")
    p.add_argument("--inventory-only", action="store_true", help="only list runs on the chip")
    p.add_argument("--keep-tarball", default=None, help="also save the verified tarball here")
    p.add_argument("--region", default=None)
    p.add_argument("--instance", default=None)
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "harvest":
        argv = argv[1:]
    a = build_parser().parse_args(argv)
    try:
        rep = pull_chip(a.chip, region=a.region, instance=a.instance, dest_dir=a.dest, only_new=not a.all, wait=a.wait,
                        chunk=a.chunk, inventory_only=a.inventory_only, keep_tarball=a.keep_tarball,
                        log=lambda s: print(s, file=sys.stderr, flush=True))
    except HarvestError as e:
        print(f"harvest failed: {e}", file=sys.stderr)
        return 1
    if a.json or a.inventory_only:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        print(f"chip {rep['chip']}: {rep.get('local_runs')} runs local, {len(rep['on_chip_runs'])} on chip, "
              f"{rep.get('files_written')} files written, missing={rep.get('missing_runs')}, problems={rep['problems']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
