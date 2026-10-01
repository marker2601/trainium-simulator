"""ffsim.contrib: people share the runs they already have, and the simulator refits on them.

python -m ffsim contrib validate | ingest | refit | stats | schema | issue-url  (``--help`` on each)

A CONTRIBUTED RECORD is one training run described by the fields of ``SCHEMA`` (also ``contrib/schema.json``):
hardware, time budget, the ``FF_*`` recipe (optionally on top of a published base recipe), seed, steps, the local
rehearsal val_bpb and/or the official score, and a consent flag. The pipeline:

1. ``validate``  schema (a small built-in JSON Schema interpreter, so no dependency), secret / personal-data scan
   (strings that look like keys, tokens, account or instance ids, ARNs, e-mail or IP addresses are REJECTED), sane
   ranges and cross-field checks, per-knob plausible ranges (``KNOB_BOUNDS``; non-finite numbers anywhere in the
   recipe are rejected), a finite feature vector and a finite prediction, duplicate detection by content hash, and
   an outlier guard against the current model (|z| > 4 is FLAGGED for the reviewer, never rejected: a surprising run
   is exactly what the model needs).
2. ``ingest``    converts the record to the ``RunRecord`` format the fitter reads and appends it, with the original
   submission under a ``contrib`` key that ``RunRecord.from_dict`` ignores, to
   ``research/sim-data/contrib/runs-contrib.jsonl``.
3. ``refit``     fits the three route-1 models on the built-in data plus every contribution, exports them as JSON
   (``ffsim/model/params.json``, see ``ffsim.params``; never a pickle) and writes ``contrib/stats.json``: counts,
   contributors, hardware breakdown, the held-out error on contributed runs before (built-in model only) and
   after (leave-one-out over the contributions, when there are at least ``LOO_MIN`` usable ones), and the GUARD:
   the built-in in-sample and pair-validation error with and without the contributions, and how far the published
   base recipes' predictions move. ``refit --guard`` refuses to write anything when either error gets worse by more
   than ``GUARD_MAX_WORSE`` or a base prediction moves by more than ``GUARD_MAX_BASE_SHIFT``.

Every reader of the contributed file goes through ``load_contrib``, which skips unreadable lines, drops duplicate
content hashes and drops any record whose numbers or feature vector are not finite or not plausible (``record_problem``),
so one bad merged line can never take down the app, the CLI or a workflow. ``contrib_fit_records`` then caps the
records one contributor (or the anonymous bucket) adds to a fit at ``MAX_FIT_PER_CONTRIBUTOR``.

How a contribution enters the quality model
-------------------------------------------
* ``chip`` is ``contrib-<hardware>-<base_recipe>``: every hardware type and base recipe gets one fixed effect
  (prior sd 0.003 bpb, like chip D), so a systematic difference between a contributor's setup and the team's chips
  is absorbed there, not in the knob effects. Keying it by base recipe too keeps K82s4's known model bias (its
  newest levers are outside the fitted data: the app anchors it by about -0.004 bpb) from leaking into the
  hardware effect that K60- or custom-based runs on the same hardware share. Until a (hardware, base) effect has
  data, predictions for an anchored base (``BASE_ANCHORS``) add the same anchor shift the app uses. The step-time model never sees contributions (it needs per-phase step times; ``step_seconds`` is
  kept with the submission for a future step-time route).
* ``knobs`` are the base recipe's knobs with the contributor's ``recipe`` on top (``base_recipe`` = ``custom``:
  the recipe as given, missing knobs read as train.py's code defaults), and ``knobs_complete`` is set.
* ``code_version`` is one of the known lineages (``M1`` .. ``M15``) or ``custom``; default the base recipe's
  lineage (K59 -> M12, K60 / K82s4 -> M14), else ``custom`` (its own era). A custom-base record that claims a team
  lineage is flagged for the reviewer.
* ``bpb_2m`` is ``chip_val_bpb`` when it was measured on the first 2,097,152 public tokens (``chip_eval_tokens``,
  the default). An official-only record is imputed as ``official - offset`` (tagged ``bpb_imputed_from_official``);
  a local score on another eval length with no official score is stored but not fitted.
* Runs under ``MIN_FIT_STEPS`` steps are short screens: stored, never fitted (a 120-step screen predicts early loss
  only).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import shutil
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import RunRecord

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
SIM_DATA = ROOT / "research" / "sim-data"
DEFAULT_RUNS = SIM_DATA / "runs.jsonl"
DEFAULT_PAIRS = SIM_DATA / "validation-pairs.json"
DEFAULT_UPLOADS = SIM_DATA / "official-uploads.csv"
DEFAULT_CONTRIB = SIM_DATA / "contrib" / "runs-contrib.jsonl"
DEFAULT_PARAMS = PKG / "model" / "params.json"
DEFAULT_STATS = ROOT / "contrib" / "stats.json"
SPACE_CONTRIB = ROOT / "space" / "research" / "sim-data" / "contrib" / "runs-contrib.jsonl"
EXAMPLES = PKG / "examples"

GITHUB_REPO = "marker2601/trainium-simulator"
ISSUE_TEMPLATE = "run-submission.yml"
ISSUE_FIELD = "record"            # the id of the JSON textarea in .github/ISSUE_TEMPLATE/run-submission.yml
ISSUE_URL_MAX = 7000
SCHEMA_VERSION = "1"
HARDWARE = ("trn2.3xlarge", "trn2.48xlarge", "trn1.2xlarge", "trn1.32xlarge", "gpu-proxy", "other")
BASE_RECIPES = ("K82s4", "K60", "K59", "custom")
EVAL_TOKENS_2M = 2097152
MIN_FIT_STEPS = 500
LOO_MIN = 5
OUTLIER_Z = 4.0
MAX_RECORD_CHARS = 20000
MAX_ISSUE_CHARS = 65536
MAX_JSON_DEPTH = 10
MAX_FIT_PER_CONTRIBUTOR = 25      # records one handle (or the anonymous bucket) may add to one fit
GUARD_MAX_WORSE = 0.0001          # refit --guard: max worsening of the built-in in-sample / pair-validation MAE (bpb)
GUARD_MAX_BASE_SHIFT = 0.001      # refit --guard: max move of the published base recipes' predictions on chip C (bpb)
KNOWN_LINEAGES = tuple(f"M{i}" for i in range(1, 16)) + ("custom",)
APPROVAL_LABEL = "contrib-approved"
# Bases whose newest levers are outside the fitted data: (measured rehearsal bpb_2m, steps, seed). The app anchors the
# same base to the same rehearsal (space/app.py build_bases).
BASE_ANCHORS: Dict[str, Tuple[float, float, int]] = {"K82s4": (0.954472, 2388.0, 73)}
CONTRIB_SOURCE = "contrib"
README_START = "<!-- contrib-stats:start -->"
README_END = "<!-- contrib-stats:end -->"
CONSENT_TEXT = ("I agree that this record becomes public in the trainium-simulator repository (data under CC-BY-4.0, "
                "code under MIT), that it may be used to fit and evaluate the simulator, and that it contains no "
                "secrets, account ids or personal data.")

__all__ = ["SCHEMA", "SCHEMA_VERSION", "HARDWARE", "BASE_RECIPES", "KNOB_BOUNDS", "ValidationResult", "Context",
           "schema_errors", "scan_secrets", "scan_text", "normalize_record", "content_hash", "to_run_record",
           "record_problem", "validate_record", "load_contrib", "contrib_fit_records", "load_context", "ingest_records",
           "compute_stats", "model_guard", "refit", "extract_issue_record", "issue_free_text", "issue_body",
           "build_issue_url", "build_plain_issue_url", "format_comment", "update_readme", "main"]

# --------------------------------------------------------------------------------------------- the schema
_HANDLE = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"
SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"https://github.com/{GITHUB_REPO}/blob/main/contrib/schema.json",
    "title": "ffsim contributed run record",
    "description": ("One training run shared with the FrontierForge Trainium simulator. Everything in it becomes "
                    "public (data CC-BY-4.0). No account ids, instance ids, keys, tokens, e-mail or IP addresses: "
                    "records containing anything that looks like one are rejected. Generated from "
                    "ffsim/contrib.py (python -m ffsim contrib schema)."),
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "hardware", "time_budget_s", "recipe", "steps", "consent"],
    "anyOf": [{"required": ["chip_val_bpb"]}, {"required": ["official_val_bpb"]}],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION, "description": "always \"1\""},
        "hardware": {"enum": list(HARDWARE),
                     "description": "instance type the run trained on; gpu-proxy = ffsim/gpu route-2 replay"},
        "hardware_other": {"type": "string", "maxLength": 60, "pattern": r"^[A-Za-z0-9 ._/+()-]*$",
                           "description": "free text when hardware is \"other\" (e.g. \"trn2u.48xlarge\")"},
        "time_budget_s": {"type": "number", "minimum": 60, "maximum": 7200,
                          "description": "training time budget in seconds (the challenge: 1800)"},
        "base_recipe": {"enum": list(BASE_RECIPES), "default": "custom",
                        "description": ("published recipe the run started from; recipe then lists only the knobs "
                                        "you changed. custom = recipe is the full FF_* environment")},
        "code_version": {"enum": list(KNOWN_LINEAGES),
                         "description": ("optional code lineage: one of the simulator's known lineages (M1-M15) or "
                                         "custom; default from base_recipe")},
        "recipe": {"type": "object", "maxProperties": 400,
                   "propertyNames": {"pattern": r"^FF_[A-Z0-9_]{1,60}$"},
                   "additionalProperties": {"anyOf": [{"type": "string", "maxLength": 200}, {"type": "number"},
                                                      {"type": "boolean"}]},
                   "description": "FF_* knob -> value, the names train.py and the simulator read"},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "training seed (FF_SEED)"},
        "chip_val_bpb": {"type": "number", "minimum": 0.5, "maximum": 2.5,
                         "description": "local rehearsal val_bpb of the trained weights"},
        "chip_eval_tokens": {"type": "integer", "minimum": 65536, "maximum": 1000000000, "default": EVAL_TOKENS_2M,
                             "description": "tokens the local val_bpb was measured on (default: the first 2,097,152)"},
        "official_val_bpb": {"type": "number", "minimum": 0.5, "maximum": 2.5,
                             "description": "official leaderboard val_bpb, if the run was scored"},
        "steps": {"type": "integer", "minimum": 1, "maximum": 200000, "description": "optimizer steps completed"},
        "step_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 60,
                         "description": "optional mean seconds per optimizer step"},
        "date": {"type": "string", "pattern": r"^20[2-9][0-9]-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$",
                 "description": "optional run date YYYY-MM-DD"},
        "framework_notes": {"type": "string", "maxLength": 500,
                            "description": "optional: SDK / runtime versions, anything unusual about the run"},
        "contributor": {"type": "string", "pattern": _HANDLE,
                        "description": ("optional: your GitHub login, to credit you (through an issue it must be "
                                        "the account that opens it)")},
        "consent": {"const": True, "description": CONSENT_TEXT},
    },
}


def schema_json() -> str:
    return json.dumps(SCHEMA, indent=2, ensure_ascii=False) + "\n"


_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def _key_label(k: Any, i: int) -> str:
    """A field name fit to repeat in a public comment: the name itself when it is a plain identifier that does not
    look like a secret, else its position ('field no. 3'). Untrusted keys are never echoed verbatim."""
    s = str(k)
    if _SAFE_KEY.match(s) and not any(pat.search(s) for name, pat in SECRET_PATTERNS
                                      if not name.startswith("a long opaque")) and len(s) < 40:
        return s
    return f"field no. {i + 1}"


def schema_errors(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Errors of ``value`` against the subset of JSON Schema that SCHEMA uses (type, enum, const, numeric bounds,
    maxLength, pattern, required, properties, additionalProperties, propertyNames, maxProperties, anyOf)."""
    errs: List[str] = []
    if "const" in schema and not (value == schema["const"] and type(value) is type(schema["const"])):
        return [f"{path} must be {json.dumps(schema['const'])}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {', '.join(map(str, schema['enum']))}"]
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_JSON_TYPES[t](value) for t in types):
            return [f"{path} must be of type {' or '.join(types)}"]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path} must be a finite number"]
    if _JSON_TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path} = {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path} = {value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{path} = {value} must be above {schema['exclusiveMinimum']}")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path} is longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path} has characters or a shape that is not allowed")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k} is required")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errs.append(f"{path} has more than {schema['maxProperties']} entries")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        names = schema.get("propertyNames")
        for i, (k, v) in enumerate(value.items()):
            label = _key_label(k, i)
            if names is not None:
                errs += schema_errors(k, names, f"{path}: key {label}")
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: {label} is not an allowed field")
            elif isinstance(extra, dict):
                errs += schema_errors(v, extra, f"{path}.{label}")
    if "anyOf" in schema:
        subs = [schema_errors(value, s, path) for s in schema["anyOf"]]
        if all(subs):
            if all(set(s.keys()) == {"required"} for s in schema["anyOf"]):
                need = " or ".join(r for s in schema["anyOf"] for r in s["required"])
                errs.append(f"{path} needs at least one of {need}")
            else:
                errs.append(f"{path}: " + "; ".join(subs[0]))
    return errs


# --------------------------------------------------------------------------------------------- secret scan
SECRET_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple((n, re.compile(p)) for n, p in (
    ("an AWS access key id", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA)[A-Z0-9]{16}\b"),
    ("an AWS ARN", r"(?i)\barn:aws[a-z-]*:"),
    ("an EC2 instance or resource id", r"\b(?:i|vol|sg|subnet|vpc|ami|eni)-[0-9a-f]{8,17}\b"),
    ("an account id (12 digits)", r"(?<![\d.])\d{12}(?![\d.])"),
    ("an account id (12 digits)", r"(?<![\d.-])(?<!\d )\d{4}[- ]\d{4}[- ]\d{4}(?![\d])(?![- ]\d)"),
    ("an e-mail address", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("a GitHub token", r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("a Hugging Face token", r"\bhf_[A-Za-z0-9]{20,}"),
    ("an API key", r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),
    ("a Slack token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("a JSON web token", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("a private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("a credential assignment", r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*\S+"),
    ("a URL with credentials", r"://[^/\s:@]+:[^/\s@]+@"),
    ("a long opaque string (key-like)", r"[A-Za-z0-9+/=_-]{40,}"),
))
IP_NAME = "an IP address"
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# A standalone dotted quad: not inside a longer dotted run (torch-neuronx 2.1.2.2.3.0) and not glued to a word.
_IP = re.compile(rf"(?<![\w.=])({_OCTET})\.{_OCTET}\.{_OCTET}\.{_OCTET}(?![\w])(?!\.\d)")
_PRIVATE_IP = re.compile(r"^(?:10|127)\.|^192\.168\.|^172\.(?:1[6-9]|2\d|3[01])\.|^169\.254\.")
# The word right before a version number: a package / tool name (anything with - or _ in it, or a known one).
_VERSION_WORD = re.compile(r"(?i)^(?:[a-z0-9]+[-_][\w.+-]*|sdk|version|ver|v|release|runtime|driver|compiler|neuron\w*|"
                           r"torch\w*|pytorch|jax\w*|python|cuda|cudnn|nccl|efa|libfabric|transformers|numpy|xla|"
                           r"aws\w*|lib\w*|cc|tools?|dlami|ami|image|kernel|build|pip|conda)$")
_HOST_CONTEXT = re.compile(r"(?i)(?:@|://|\b(?:ip|ips|ipv4|host|hostname|addr|address|server|node|endpoint|inet|"
                           r"peer|master|worker)\b)[\s:=]*$")


def _looks_like_ip(s: str) -> bool:
    """A dotted quad that is plausibly a host address, not a 4-part version string. Private ranges and quads after
    'host' / 'ip' / '@' / '://' always count; otherwise a quad right after a package or version word
    ('neuronx-cc 2.15.128.0', 'Neuron SDK 2.20.0.0') does not."""
    for m in _IP.finditer(s):
        quad = m.group(0)
        before = s[max(0, m.start() - 40):m.start()]
        if _PRIVATE_IP.match(quad) or _HOST_CONTEXT.search(before):
            return True
        prev = re.search(r"([A-Za-z0-9][\w.+-]*)[\s:=(]*$", before)
        if prev and _VERSION_WORD.match(prev.group(1).rstrip(".")):
            continue
        return True
    return False


def scan_text(s: str) -> Optional[str]:
    """The kind of secret / personal data ``s`` looks like it contains, or None."""
    for name, pat in SECRET_PATTERNS:
        if name.startswith("a long opaque") and _KNOB_NAME.match(s):
            continue
        if pat.search(s):
            return name
    if _looks_like_ip(s):
        return IP_NAME
    return None


def _strings(value: Any, path: str = "$") -> Iterable[Tuple[str, str]]:
    """(path, text) for every key and string / integer value. Paths name keys only when ``_key_label`` says the key
    is safe to repeat, so an error message never carries a secret that was used as a key."""
    if isinstance(value, dict):
        for i, (k, v) in enumerate(value.items()):
            label = _key_label(k, i)
            yield f"{path}: key {label}", str(k)
            yield from _strings(v, f"{path}.{label}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(value, str):
        yield path, value
    elif isinstance(value, int) and not isinstance(value, bool):
        yield path, str(value)


_KNOB_NAME = re.compile(r"^FF_[A-Z0-9_]+$")


def scan_secrets(value: Any) -> List[str]:
    """One error per field holding something that looks like a secret or personal data (the value is not echoed)."""
    out: List[str] = []
    for path, s in _strings(value):
        kind = scan_text(s)
        if kind:
            out.append(f"{path} looks like it contains {kind}: remove it (records become public)")
    return out


# --------------------------------------------------------------------------------------------- normalise / hash
def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else format(v, ".10g")
    return str(v).strip()


def _canon_value(v: Any) -> str:
    s = str(v).strip()
    try:
        f = float(s)
        if math.isfinite(f):
            return format(f, ".10g")
    except (ValueError, OverflowError):
        pass
    return s.replace(" ", "").lower()


def _finite_int(v: Any) -> Optional[int]:
    """int(float(v)) when that is a finite whole number, else None (never raises)."""
    try:
        f = float(str(v).strip())
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(f) or not f.is_integer():
        return None
    return int(f)


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """A schema-valid record with defaults filled, strings stripped, integral floats made ints and the recipe's
    values as the strings train.py reads. Field order follows SCHEMA."""
    out: Dict[str, Any] = {}
    for k in SCHEMA["properties"]:
        if k not in raw or raw[k] is None:
            continue
        v = raw[k]
        if k == "recipe":
            v = {str(kk).strip(): _fmt_value(vv) for kk, vv in sorted(v.items())}
        elif isinstance(v, str):
            v = v.strip()
        elif isinstance(v, float) and SCHEMA["properties"][k].get("type") == "integer":
            v = int(v)
        out[k] = v
    out.setdefault("base_recipe", "custom")
    if "chip_val_bpb" in out:
        out.setdefault("chip_eval_tokens", EVAL_TOKENS_2M)
    if "seed" not in out and "FF_SEED" in out.get("recipe", {}):
        seed = _finite_int(out["recipe"]["FF_SEED"])
        if seed is not None and 0 <= seed <= 2147483647:
            out["seed"] = seed
    return out


HASH_FIELDS = ("hardware", "hardware_other", "time_budget_s", "base_recipe", "code_version", "seed", "chip_val_bpb",
               "chip_eval_tokens", "official_val_bpb", "steps")


def content_hash(rec: Dict[str, Any]) -> str:
    """sha256 over the measurement (who submitted it, notes, dates and consent do not count): the same run sent
    twice, with knob values spelled differently (0.30 vs 0.3), numbers typed differently (1800 vs 1800.0) or a
    different handle, is the same hash."""
    core: Dict[str, Any] = {}
    for k in HASH_FIELDS:
        v = rec.get(k)
        if v is None:
            continue
        if k in ("chip_val_bpb", "official_val_bpb"):
            core[k] = format(float(v), ".7f")
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            core[k] = _canon_value(v)
        else:
            core[k] = v
    core["recipe"] = {k: _canon_value(v) for k, v in sorted((rec.get("recipe") or {}).items()) if k != "FF_SEED"}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- conversion
_BASE_CACHE: Dict[str, Tuple[str, Dict[str, str]]] = {}


def base_recipe(name: str) -> Tuple[str, Dict[str, str]]:
    """(code_version, knobs) of a published base recipe (ffsim/examples/recipe-<name>.json)."""
    if name not in _BASE_CACHE:
        with open(EXAMPLES / f"recipe-{name}.json", "r", encoding="utf-8") as fh:
            d = json.load(fh)
        _BASE_CACHE[name] = (str(d["code_version"]), {str(k): str(v) for k, v in d["knobs"].items()})
    cv, kn = _BASE_CACHE[name]
    return cv, dict(kn)


def known_knobs() -> set:
    from .quality import FEATURE_KNOBS
    out = set(FEATURE_KNOBS)
    for b in BASE_RECIPES:
        if b != "custom":
            out |= set(base_recipe(b)[1])
    return out


# Plausible values of every numeric knob the quality features read: (low, high, must be a whole number). They are
# several times wider than anything in the fitted data (ffsim/quality.py FEATURES); a value outside is a typo or an
# attack, and a non-finite one would make the fit raise. Comma / colon lists are bounded element-wise.
KNOB_BOUNDS: Dict[str, Tuple[float, float, bool]] = {
    "FF_DEPTH": (1, 64, True),
    "FF_ASPECT_RATIO": (8, 1024, False),
    "FF_MLP_MULT": (0.5, 16, False),
    "FF_TOTAL_BATCH": (1024, 16777216, True),
    "FF_TIME_TARGET": (10, 7200, False),
    "FF_COOLDOWN_FRAC": (0, 1, False),
    "FF_LEAKY_RELU2": (0, 1, False),
    "FF_SOFTCAP": (0, 200, False),
    "FF_SOFTCAP_A": (0, 200, False),
    "FF_SOFTCAP_B": (-50, 50, False),
    "FF_WD_SCHED": (0, 10, False),
    "FF_MOM_PEAK": (0, 1, False),
    "FF_MUON_BETA2": (0, 1, False),
    "FF_SCALAR_BETA2": (0, 1, False),
    "FF_ROPE_BASE": (1, 1e9, False),
    "FF_QK_GAIN": (0, 20, False),
    "FF_WARMUP": (0, 100000, False),
    "FF_EMB_LR": (1e-9, 100, False),
    "FF_UNEMB_LR": (1e-9, 100, False),
    "FF_SCALAR_LR": (1e-9, 100, False),
    "FF_MATRIX_LR_SCALE": (1e-9, 100, False),
    "FF_ADAMW_LR_SCALE": (1e-9, 100, False),
    "FF_WD": (0, 10, False),
    "FF_KEY_OFFSET_RAD": (0, 10, False),
    "FF_ACC_IN_GRAPH": (0, 1, False),
    "FF_MTP": (0, 16, False),
    "FF_PTP_W": (0, 10, False),
    "FF_MOM_RAMP": (0, 100000, False),
}
LIST_KNOB_BOUNDS: Dict[str, Tuple[float, float]] = {
    "FF_ACCUM_SCHED": (0, 64),
    "FF_ACCUM_LR": (0, 100),
    "FF_MTP_PHASES": (0, 1),
    "FF_MTP_W": (0, 100),
}
_NUM_SPLIT = re.compile(r"[,:;\s]+")


def _num(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except (ValueError, OverflowError):
        return None


def knob_value_error(name: str, value: Any) -> Optional[str]:
    """Why ``value`` is not acceptable for knob ``name`` (None when it is). Any number in any knob must be finite
    (inf, nan and 1e400 are refused everywhere); the knobs in ``KNOB_BOUNDS`` / ``LIST_KNOB_BOUNDS`` must also be
    in their plausible range. The message never repeats the value."""
    s = _fmt_value(value) if not isinstance(value, str) else value.strip()
    toks = [t for t in _NUM_SPLIT.split(s) if t]
    for t in toks:
        f = _num(t)
        if f is not None and not math.isfinite(f):
            return f"$.recipe.{name} holds a number that is not finite"
    if name in KNOB_BOUNDS:
        lo, hi, whole = KNOB_BOUNDS[name]
        f = _num(s)
        if f is None:
            return f"$.recipe.{name} must be a number"
        if not (lo <= f <= hi):
            return f"$.recipe.{name} is outside its plausible range [{lo:g}, {hi:g}]"
        if whole and not f.is_integer():
            return f"$.recipe.{name} must be a whole number"
    elif name in LIST_KNOB_BOUNDS:
        lo, hi = LIST_KNOB_BOUNDS[name]
        for t in toks:
            f = _num(t)
            if f is not None and not (lo <= f <= hi):
                return f"$.recipe.{name} has an entry outside its plausible range [{lo:g}, {hi:g}]"
    elif name == "FF_SEED":
        seed = _finite_int(s)
        if seed is None or not (0 <= seed <= 2147483647):
            return "$.recipe.FF_SEED must be a whole number from 0 to 2147483647"
    return None


def record_features(r: RunRecord) -> Any:
    """The scaled knob feature vector the quality model builds for record ``r`` (record convention)."""
    from .quality import _effective_knobs, canonical_version, code_defaults_for, knob_features, record_knobs
    version = canonical_version(r.code_version, r.tags)
    return knob_features(_effective_knobs(record_knobs(r, version), code_defaults_for(version)))


def record_problem(r: RunRecord) -> Optional[str]:
    """Why a contributed RunRecord must not enter a fit (None when it is fine): non-finite or implausible score or
    steps, a knob outside ``KNOB_BOUNDS``, or a feature vector that is not finite. Never raises."""
    try:
        if r.bpb_2m is not None and not (math.isfinite(float(r.bpb_2m)) and 0.5 <= float(r.bpb_2m) <= 2.5):
            return "bpb_2m is not a finite value in [0.5, 2.5]"
        if r.official_bpb is not None and not (math.isfinite(float(r.official_bpb))
                                               and 0.5 <= float(r.official_bpb) <= 2.5):
            return "official_bpb is not a finite value in [0.5, 2.5]"
        if r.steps is None or isinstance(r.steps, bool) or not (1 <= int(r.steps) <= 200000)                 or float(r.steps) != int(r.steps):
            return "steps is not a whole number in [1, 200000]"
        if r.time_target is not None and not math.isfinite(float(r.time_target)):
            return "time_target is not finite"
        if r.code_version is not None and str(r.code_version) not in KNOWN_LINEAGES:
            return "code_version is not a known lineage"
        for k, v in (r.knobs or {}).items():
            err = knob_value_error(str(k), v)
            if err:
                return err.replace("$.recipe.", "knob ")
        import numpy as np
        x = record_features(r)
        if not bool(np.all(np.isfinite(x))):
            return "its knob feature vector is not finite"
    except Exception as e:  # noqa: BLE001 - anything unexpected is a reason to keep the record out of the fit
        return f"it cannot be read ({type(e).__name__})"
    return None


def contrib_chip(hardware: str, base: str) -> str:
    return f"contrib-{hardware}-{base}"


def anchor_shift(quality: Any, r: RunRecord) -> float:
    """The app's anchor shift for an anchored base recipe (``BASE_ANCHORS``: measured rehearsal minus the model's
    raw prediction for that base), applied while the record's (hardware, base) chip effect has no data yet; 0.0
    otherwise. Once the chip has data its fixed effect has learned the shift, so adding it again would count it
    twice."""
    base = next((t[5:] for t in r.tags if t.startswith("base:")), "custom")
    if base not in BASE_ANCHORS or quality is None or r.chip in getattr(quality, "chips", {}):
        return 0.0
    bpb, steps, seed = BASE_ANCHORS[base]
    cv, knobs = base_recipe(base)
    with open(EXAMPLES / f"recipe-{base}.json", "r", encoding="utf-8") as fh:
        chip = str(json.load(fh).get("chip") or "C")
    return float(bpb - quality.predict(knobs, steps, cv, chip, seed)[0])


def predict_contrib(quality: Any, r: RunRecord) -> Tuple[float, float]:
    """(mean, sd) of the model for a contributed record, with ``anchor_shift`` applied."""
    mean, sd = quality.predict_record(r)
    return float(mean) + anchor_shift(quality, r), float(sd)


def to_run_record(rec: Dict[str, Any], digest: str, offset_mean: float) -> RunRecord:
    hw = rec["hardware"]
    base = rec.get("base_recipe", "custom")
    chip = contrib_chip(hw, base)
    knobs: Dict[str, str] = {}
    code_version = rec.get("code_version")
    if base != "custom":
        cv, knobs = base_recipe(base)
        code_version = code_version or cv
    knobs.update(rec.get("recipe") or {})
    if rec.get("seed") is not None:
        knobs["FF_SEED"] = str(int(rec["seed"]))
    tags = [CONTRIB_SOURCE, f"hardware:{hw}", f"base:{base}", f"contrib_hash:{digest}"]
    if rec.get("contributor"):
        tags.append(f"contributor:{rec['contributor']}")
    bpb_2m: Optional[float] = None
    if rec.get("chip_val_bpb") is not None and int(rec.get("chip_eval_tokens", EVAL_TOKENS_2M)) == EVAL_TOKENS_2M:
        bpb_2m = float(rec["chip_val_bpb"])
    elif rec.get("official_val_bpb") is not None:
        bpb_2m = float(rec["official_val_bpb"]) - float(offset_mean)
        tags.append("bpb_imputed_from_official")
    elif rec.get("chip_val_bpb") is not None:
        tags.append(f"eval_tokens:{int(rec['chip_eval_tokens'])}")
    steps = int(rec["steps"])
    return RunRecord(
        run_id=f"{chip}:contrib-{digest[:12]}", chip=chip, run=f"contrib-{digest[:12]}", sources=[CONTRIB_SOURCE],
        date_utc=rec.get("date"), code_version=code_version or "custom", seed=rec.get("seed"), steps=steps,
        bpb_2m=bpb_2m, official_bpb=rec.get("official_val_bpb"), submission=None, is_screen=steps < MIN_FIT_STEPS,
        in_window=None, time_target=float(rec["time_budget_s"]), knobs=knobs, knobs_complete=True, tags=tags,
        notes="contributed run")


def usable_for_fit(r: RunRecord) -> bool:
    return (not r.is_screen) and r.bpb_2m is not None and r.steps is not None and r.steps >= MIN_FIT_STEPS


# --------------------------------------------------------------------------------------------- validation
@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    record: Optional[Dict[str, Any]] = None
    content_hash: Optional[str] = None
    run_record: Optional[RunRecord] = None
    prediction: Optional[Dict[str, float]] = None
    usable_for_fit: bool = False
    secret_found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings), "flags": list(self.flags),
                "record": self.record, "content_hash": self.content_hash,
                "run_record": dataclasses.asdict(self.run_record) if self.run_record else None,
                "prediction": self.prediction, "usable_for_fit": self.usable_for_fit,
                "secret_found": self.secret_found}


def validate_record(raw: Any, quality: Any = None, existing_hashes: Optional[Iterable[str]] = None,
                    builtin: Optional[Sequence[RunRecord]] = None, offset_mean: Optional[float] = None,
                    issue_author: Optional[str] = None,
                    contributor_counts: Optional[Dict[str, int]] = None) -> ValidationResult:
    """Check one submitted record. ``quality`` (a fitted QualityModel) enables the outlier guard and the finite-
    prediction check; ``existing_hashes`` the duplicate check; ``builtin`` the "already in the shipped data"
    warning; ``issue_author`` (the GitHub login that opened the issue) ties ``contributor`` to that account;
    ``contributor_counts`` (lower-cased handle or "" -> records already in the data) the volume flag. Never raises
    for any JSON input: an unexpected failure becomes an error."""
    try:
        return _validate_record(raw, quality, existing_hashes, builtin, offset_mean, issue_author, contributor_counts)
    except Exception as e:  # noqa: BLE001 - a hostile record must produce a readable answer, not a traceback
        return ValidationResult(False, [f"the record could not be checked ({type(e).__name__}); please report this"])


def _validate_record(raw: Any, quality: Any, existing_hashes: Optional[Iterable[str]],
                     builtin: Optional[Sequence[RunRecord]], offset_mean: Optional[float],
                     issue_author: Optional[str], contributor_counts: Optional[Dict[str, int]]) -> ValidationResult:
    if offset_mean is None:
        from .offset import OffsetModel
        offset_mean = OffsetModel().mean
    if not isinstance(raw, dict):
        return ValidationResult(False, ["the record must be a JSON object {...}"])
    if _json_depth(raw) > MAX_JSON_DEPTH:
        return ValidationResult(False, [f"the record nests deeper than {MAX_JSON_DEPTH} levels"])
    try:
        size = len(json.dumps(raw, allow_nan=True))
    except (TypeError, ValueError, RecursionError):
        return ValidationResult(False, ["the record is not JSON-serialisable"])
    if size > MAX_RECORD_CHARS:
        return ValidationResult(False, [f"the record is {size:,} characters; the limit is {MAX_RECORD_CHARS:,}"])
    secrets = scan_secrets(raw)
    if secrets:
        # Only the scan errors: schema errors could repeat a key that holds the secret.
        return ValidationResult(False, secrets, secret_found=True)
    errors = schema_errors(raw, SCHEMA)
    if errors:
        return ValidationResult(False, errors)
    rec = normalize_record(raw)
    warnings: List[str] = []
    flags: List[str] = []
    recipe = rec.get("recipe", {})
    for k, v in recipe.items():
        if any(ord(c) < 32 for c in v):
            errors.append(f"$.recipe.{k} contains control characters")
        err = knob_value_error(k, v)
        if err:
            errors.append(err)
    if rec["hardware"] == "other" and not rec.get("hardware_other"):
        errors.append("$.hardware_other is required when hardware is \"other\"")
    if "FF_SEED" in recipe and rec.get("seed") is not None:
        fs = _finite_int(recipe["FF_SEED"])
        if fs is not None and fs != int(rec["seed"]):
            errors.append(f"$.seed ({rec['seed']}) and $.recipe.FF_SEED ({fs}) disagree")
    if rec.get("chip_eval_tokens") is not None and rec.get("chip_val_bpb") is None:
        errors.append("$.chip_eval_tokens is given without $.chip_val_bpb")
    if rec.get("step_seconds") is not None and rec["step_seconds"] * rec["steps"] > 1.25 * rec["time_budget_s"]:
        errors.append(f"$.steps x $.step_seconds = {rec['step_seconds'] * rec['steps']:,.0f} s is more than the "
                      f"{rec['time_budget_s']:g} s time budget: one of the three is wrong")
    if issue_author and rec.get("contributor") and str(rec["contributor"]).lower() != str(issue_author).lower():
        errors.append("$.contributor must be your own GitHub login (the account that opened this issue), or left "
                      "out to stay anonymous")
    if errors:
        return ValidationResult(False, errors, record=rec)

    for k in ("chip_val_bpb", "official_val_bpb"):
        v = rec.get(k)
        if v is not None and not (0.85 <= v <= 1.40):
            warnings.append(f"{k} = {v} is far from every run the simulator knows (0.95-1.15): please double-check")
    if rec.get("chip_val_bpb") is not None and rec.get("official_val_bpb") is not None:
        off = rec["official_val_bpb"] - rec["chip_val_bpb"]
        if rec.get("chip_eval_tokens") == EVAL_TOKENS_2M and not (-0.005 <= off <= 0.03):
            warnings.append(f"official - local = {off:+.4f}; the team's 2M-token rehearsals ran +0.0053 to +0.0079")
    if abs(float(rec["time_budget_s"]) - 1800.0) > 1e-9:
        warnings.append(f"time budget {rec['time_budget_s']:g} s is not the challenge's 1800 s: the steps -> bpb "
                        "curve is fitted near 1,800 s")
    if rec["steps"] < MIN_FIT_STEPS:
        warnings.append(f"{rec['steps']} steps is a short screen (< {MIN_FIT_STEPS}): stored, not used for fitting")
    if rec.get("chip_val_bpb") is not None and rec.get("chip_eval_tokens") != EVAL_TOKENS_2M:
        warnings.append("local val_bpb on a different eval length than the first 2,097,152 tokens: "
                        + ("the official score is used instead (imputed)" if rec.get("official_val_bpb") is not None
                           else "stored, not used for fitting (the model's target is the 2M-token rehearsal)"))
    if rec.get("chip_val_bpb") is None:
        flags.append("bpb_imputed_from_official")
        warnings.append(f"no local val_bpb: the fit uses official - {offset_mean:+.5f} (the simulator's offset)")
    known = known_knobs()
    unknown = sorted(k for k in recipe if k not in known)
    if unknown:
        warnings.append("knobs no published recipe reads (stored; the simulator ignores them): " + ", ".join(unknown[:20])
                        + (" ..." if len(unknown) > 20 else ""))
    if rec.get("base_recipe", "custom") == "custom" and rec.get("code_version") not in (None, "custom"):
        flags.append("claims_team_lineage")
        warnings.append(f"a custom recipe that names the team's code lineage {rec['code_version']}: its era effect is "
                        "shared with the team's runs, so the reviewer should confirm the run used that code")
    if contributor_counts is not None:
        n_prev = contributor_counts.get(str(rec.get("contributor") or "").lower(), 0)
        if n_prev >= MAX_FIT_PER_CONTRIBUTOR:
            who = "handle" if rec.get("contributor") else "anonymous bucket"
            flags.append("contributor_cap")
            warnings.append(f"{n_prev} records from this {who} are already in the data: only the first "
                            f"{MAX_FIT_PER_CONTRIBUTOR} enter a fit; this one is stored")

    digest = content_hash(rec)
    if existing_hashes is not None and digest in set(existing_hashes):
        return ValidationResult(False, [f"duplicate: a record with the same measurement is already in the dataset "
                                        f"(content hash {digest[:12]})"], warnings, flags, rec, digest)
    rr = to_run_record(rec, digest, offset_mean)
    problem = record_problem(rr)
    if problem:
        return ValidationResult(False, [f"the simulator cannot use this record: {problem}"], warnings, flags, rec,
                                digest)
    if builtin and rr.bpb_2m is not None:
        for b in builtin:
            if b.steps == rr.steps and b.bpb_2m is not None and abs(float(b.bpb_2m) - rr.bpb_2m) < 5e-7:
                flags.append("matches_builtin")
                warnings.append(f"same steps and val_bpb as the shipped run {b.run_id}: if it is that run, it is "
                                "already in the data")
                break
    prediction = None
    if quality is not None and rr.bpb_2m is not None:
        try:
            mean, sd = predict_contrib(quality, rr)
        except Exception as e:  # noqa: BLE001
            return ValidationResult(False, [f"the current model cannot predict this record ({type(e).__name__})"],
                                    warnings, flags, rec, digest)
        if not (math.isfinite(mean) and math.isfinite(sd)):
            return ValidationResult(False, ["the current model's prediction for this record is not finite: a knob "
                                            "value is far outside anything it can read"], warnings, flags, rec, digest)
        z = (rr.bpb_2m - mean) / sd if sd > 0 else 0.0
        prediction = {"predicted_bpb_2m": float(mean), "sd": float(sd), "observed_bpb_2m": float(rr.bpb_2m),
                      "residual": float(rr.bpb_2m - mean), "z": float(z)}
        if abs(z) > OUTLIER_Z:
            flags.append("outlier")
            warnings.append(f"outlier vs the current model: observed {rr.bpb_2m:.5f}, predicted {mean:.5f} "
                            f"+- {sd:.5f} (z = {z:+.1f}). Flagged for the reviewer, not rejected: check the "
                            "numbers and the recipe; if they are right, this run teaches the model the most")
        try:
            far = quality.extrapolated_features(rr.knobs, _record_defaults_for(rr))
            base = rec.get("base_recipe", "custom")
            if base != "custom":      # only what the contributor's changes add beyond the published base itself
                bcv, bknobs = base_recipe(base)
                from .quality import code_defaults_for
                own = set(quality.extrapolated_features(bknobs, code_defaults_for(bcv)))
                far = [f for f in far if f not in own]
        except Exception:  # noqa: BLE001 - advisory only
            far = []
        if far:
            flags.append("extrapolates")
            warnings.append("knob features outside the range of the fitted runs (the model extrapolates there): "
                            + ", ".join(far[:10]) + (" ..." if len(far) > 10 else ""))
    return ValidationResult(True, [], warnings, flags, rec, digest, rr, prediction, usable_for_fit(rr))


def _record_defaults_for(r: RunRecord) -> Dict[str, str]:
    from .quality import canonical_version, code_defaults_for
    return code_defaults_for(canonical_version(r.code_version, r.tags))


def _json_depth(v: Any) -> int:
    """Nesting depth of a parsed JSON value, iteratively (no recursion limit); stops counting past the limit."""
    best, stack = 0, [(v, 1)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, (dict, list)):
            best = max(best, d)
            if d > MAX_JSON_DEPTH:
                return d
            stack.extend((c, d + 1) for c in (x.values() if isinstance(x, dict) else x))
    return best


# --------------------------------------------------------------------------------------------- data context
def _norm_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else b""


def file_sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(_norm_bytes(path)).hexdigest() if path.exists() else None


def _warn(msg: str) -> None:
    print(f"ffsim.contrib: {msg}", file=sys.stderr)


def load_contrib(path: Path = DEFAULT_CONTRIB,
                 skipped: Optional[List[str]] = None) -> List[Tuple[RunRecord, Dict[str, Any]]]:
    """(RunRecord, submission meta) per usable line of runs-contrib.jsonl; [] when the file does not exist.

    Every reader (the app, the CLI, refit, the workflows) comes through here, so this is where a bad merged line is
    contained: unreadable lines, repeated content hashes (two pull requests with the same run that were both merged)
    and records ``record_problem`` refuses (non-finite or implausible numbers, a non-finite feature vector) are
    skipped with a note on stderr (and appended to ``skipped`` when given). Never raises for bad content."""
    out: List[Tuple[RunRecord, Dict[str, Any]]] = []
    p = Path(path)
    if not p.exists():
        return out
    seen: set = set()
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            why: Optional[str] = None
            h: Optional[str] = None
            try:
                d = json.loads(line)
                if not isinstance(d, dict):
                    raise ValueError("not an object")
                r = RunRecord.from_dict(d)
                meta = d.get("contrib") if isinstance(d.get("contrib"), dict) else {}
                h = meta.get("content_hash") if isinstance(meta.get("content_hash"), str) else None
            except Exception as e:  # noqa: BLE001 - one bad line must never take the readers down
                why = f"unreadable ({type(e).__name__})"
            else:
                if h and h in seen:
                    why = f"duplicate content hash {h[:12]}"
                else:
                    why = record_problem(r)
            if why:
                msg = f"{p.name} line {n} skipped: {why}"
                if skipped is not None:
                    skipped.append(msg)
                _warn(msg)
                continue
            if h:
                seen.add(h)
            out.append((r, meta))
    return out


def _bucket(meta: Dict[str, Any], r: RunRecord) -> str:
    sub = meta.get("submission") if isinstance(meta.get("submission"), dict) else {}
    c = sub.get("contributor") or next((t[12:] for t in r.tags if t.startswith("contributor:")), "")
    return str(c).lower()


def contributor_counts(rows: Sequence[Tuple[RunRecord, Dict[str, Any]]]) -> Dict[str, int]:
    """Lower-cased contributor handle ("" = anonymous) -> records in ``rows``."""
    out: Dict[str, int] = {}
    for r, meta in rows:
        b = _bucket(meta, r)
        out[b] = out.get(b, 0) + 1
    return out


def contrib_fit_records(rows: Sequence[Tuple[RunRecord, Dict[str, Any]]],
                        cap: int = MAX_FIT_PER_CONTRIBUTOR) -> List[RunRecord]:
    """The contributed records that enter a fit: the first ``cap`` (file order) of each contributor handle, and of
    the anonymous bucket, so one account cannot outweigh everyone else; the rest stay stored."""
    seen: Dict[str, int] = {}
    out: List[RunRecord] = []
    for r, meta in rows:
        b = _bucket(meta, r)
        seen[b] = seen.get(b, 0) + 1
        if seen[b] <= cap:
            out.append(r)
    return out


def _load_pairs(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        pairs = json.load(fh)
    return pairs.get("pairs", pairs) if isinstance(pairs, dict) else pairs


def fit_quality_safely(builtin: Sequence[RunRecord], contrib: Sequence[RunRecord],
                       pairs: Optional[List[Dict[str, Any]]]) -> Tuple[Any, List[str]]:
    """(QualityModel on built-in + contributed records, notes). If the fit with the contributions raises, the
    model falls back to the built-in records (and says so) instead of taking the caller down; (None, notes) when
    even that fails (fewer than two fittable runs)."""
    from .quality import QualityModel
    notes: List[str] = []
    if contrib:
        try:
            return QualityModel().fit(list(builtin) + list(contrib), pairs), notes
        except Exception as e:  # noqa: BLE001
            notes.append(f"fit with {len(contrib)} contributed runs failed ({type(e).__name__}): built-in data only")
            _warn(notes[-1])
    try:
        return QualityModel().fit(list(builtin), pairs), notes
    except Exception as e:  # noqa: BLE001
        notes.append(f"quality fit failed ({type(e).__name__})")
        return None, notes


@dataclass
class Context:
    builtin: List[RunRecord]
    contrib: List[Tuple[RunRecord, Dict[str, Any]]]
    pairs: List[Dict[str, Any]]
    offset_mean: float
    uploads: Optional[Path] = None
    _quality: Any = None

    @property
    def contrib_records(self) -> List[RunRecord]:
        """The contributed records a fit uses (``contrib_fit_records``: capped per contributor)."""
        return contrib_fit_records(self.contrib)

    @property
    def hashes(self) -> set:
        return {m.get("content_hash") for _, m in self.contrib if m.get("content_hash")}

    def quality(self) -> Any:
        """QualityModel on built-in + contributed records (fitted once, milliseconds); built-in only when the fit
        with contributions fails; None when there are fewer than two fittable runs. Never raises."""
        if self._quality is None:
            self._quality, _ = fit_quality_safely(self.builtin, self.contrib_records, self.pairs)
        return self._quality


def load_context(runs: Path = DEFAULT_RUNS, contrib: Path = DEFAULT_CONTRIB, pairs: Path = DEFAULT_PAIRS,
                 uploads: Path = DEFAULT_UPLOADS) -> Context:
    from .dataset import load_runs
    from .offset import OffsetModel
    builtin = load_runs(str(runs)) if Path(runs).exists() else []
    off = OffsetModel()
    if Path(uploads).exists():
        off = off.fit(str(uploads)) or off
    return Context(builtin, load_contrib(Path(contrib)), _load_pairs(Path(pairs)), float(off.mean),
                   Path(uploads) if Path(uploads).exists() else None)


def contrib_line(res: ValidationResult) -> str:
    assert res.ok and res.run_record is not None and res.record is not None
    d = dataclasses.asdict(res.run_record)
    d["contrib"] = {"submission": res.record, "content_hash": res.content_hash, "flags": list(res.flags),
                    "prediction_at_ingest": res.prediction}
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)


def ingest_records(raws: Sequence[Any], ctx: Context, contrib_path: Path = DEFAULT_CONTRIB) -> List[ValidationResult]:
    """Validate each record and append the valid, new ones to ``contrib_path``. Duplicates inside the batch are
    caught too. Returns one result per input."""
    hashes = set(ctx.hashes)
    quality = ctx.quality()
    out: List[ValidationResult] = []
    lines: List[str] = []
    for raw in raws:
        res = validate_record(raw, quality=quality, existing_hashes=hashes, builtin=ctx.builtin,
                              offset_mean=ctx.offset_mean, contributor_counts=contributor_counts(ctx.contrib))
        if res.ok:
            try:
                line = contrib_line(res)
            except (ValueError, TypeError) as e:
                res = ValidationResult(False, [f"the record cannot be stored ({type(e).__name__})"], res.warnings,
                                       res.flags, res.record, res.content_hash)
            else:
                hashes.add(res.content_hash)
                lines.append(line)
                ctx.contrib.append((res.run_record, {"content_hash": res.content_hash, "submission": res.record}))
        out.append(res)
    if lines:
        p = Path(contrib_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(line + "\n")
        ctx._quality = None
    return out


# --------------------------------------------------------------------------------------------- stats / refit
def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def model_guard(ctx: Context, quality_full: Any, quality_builtin: Any = None) -> Dict[str, Any]:
    """How much the contributions move the model on the TEAM's data: the built-in runs' in-sample MAE and the
    validation pairs' MAE, fitted without and with the contributions, and the largest move of a published base
    recipe's prediction (chip C, seed 73, 2,300 and 2,400 steps). ``ok`` is False when either MAE gets worse by more
    than ``GUARD_MAX_WORSE`` bpb or a base prediction moves by more than ``GUARD_MAX_BASE_SHIFT`` (``refit --guard``
    then writes nothing): a handful of coordinated, plausible-looking records that pull the shared knob
    coefficients show up here."""
    from .quality import QualityModel
    out: Dict[str, Any] = {"max_worse": GUARD_MAX_WORSE, "ok": True}
    if quality_full is None or not ctx.builtin:
        return out
    q_b = quality_builtin if quality_builtin is not None else QualityModel().fit(ctx.builtin, ctx.pairs)
    rows = [r for r in ctx.builtin if r.run_id in q_b.records_by_id]

    def insample(q: Any) -> Optional[float]:
        return _mean([abs(float(r.bpb_2m) - q.predict_record(r)[0]) for r in rows])

    def pairs_mae(q: Any) -> Optional[float]:
        if not ctx.pairs:
            return None
        try:
            return q.pair_validation(ctx.pairs).get("mae")
        except Exception:  # noqa: BLE001
            return None

    same = quality_full is q_b or getattr(quality_full, "n", None) == getattr(q_b, "n", None)
    ib = insample(q_b)
    ia = ib if same else insample(quality_full)
    pb = pairs_mae(q_b)
    pa = pb if same else pairs_mae(quality_full)
    shift = 0.0
    if not same:
        for name in BASE_RECIPES:
            if name == "custom":
                continue
            cv, knobs = base_recipe(name)
            for steps in (2300.0, 2400.0):
                a = quality_full.predict(knobs, steps, cv, "C", 73)[0]
                b = q_b.predict(knobs, steps, cv, "C", 73)[0]
                shift = max(shift, abs(a - b))
    out.update({"builtin_insample_mae_before": ib, "builtin_insample_mae_after": ia,
                "pair_mae_before": pb, "pair_mae_after": pa, "base_prediction_shift": shift,
                "max_base_shift": GUARD_MAX_BASE_SHIFT})
    worse = [a - b for a, b in ((ia, ib), (pa, pb)) if a is not None and b is not None]
    out["worst_change"] = max(worse) if worse else 0.0
    out["ok"] = all(w <= GUARD_MAX_WORSE for w in worse) and shift <= GUARD_MAX_BASE_SHIFT
    return out


def compute_stats(ctx: Context, quality_full: Any = None, loo: bool = True,
                  data_sha: Optional[str] = None) -> Dict[str, Any]:
    """contrib/stats.json: counts, contributors, hardware, held-out error on contributed runs, and the guard.

    mae_before    the model fitted on the built-in data only, predicting every usable contributed run (with the
                  base recipe's anchor shift, ``anchor_shift``, as the app applies it: without it a K82s4-based
                  run would carry the model's known ~0.004 bpb K82s4 bias).
    mae_after_loo leave-one-out: for each usable contributed run, the model fitted on the built-in data plus every
                  OTHER contribution predicts it (only with >= LOO_MIN usable contributions; else null).
    guard         ``model_guard``: the built-in in-sample and pair-validation MAE without / with contributions.
    """
    from .quality import QualityModel
    recs = ctx.contrib
    hw: Dict[str, int] = {}
    bases: Dict[str, int] = {}
    handles: set = set()
    n_anon = 0
    n_off = 0
    n_imp = 0
    n_outlier = 0
    offsets: List[float] = []
    for r, meta in recs:
        sub = meta.get("submission") or {}
        h = sub.get("hardware") or next((t[9:] for t in r.tags if t.startswith("hardware:")), "?")
        hw[h] = hw.get(h, 0) + 1
        b = sub.get("base_recipe") or next((t[5:] for t in r.tags if t.startswith("base:")), "?")
        bases[b] = bases.get(b, 0) + 1
        c = sub.get("contributor")
        if c:
            handles.add(str(c).lower())
        else:
            n_anon += 1
        if r.official_bpb is not None:
            n_off += 1
        if "bpb_imputed_from_official" in r.tags:
            n_imp += 1
        if "outlier" in (meta.get("flags") or []):
            n_outlier += 1
        if (sub.get("chip_val_bpb") is not None and sub.get("official_val_bpb") is not None
                and sub.get("chip_eval_tokens", EVAL_TOKENS_2M) == EVAL_TOKENS_2M):
            offsets.append(float(sub["official_val_bpb"]) - float(sub["chip_val_bpb"]))
    q_full = quality_full if quality_full is not None else ctx.quality()
    usable = [r for r, _ in recs if usable_for_fit(r) and q_full is not None and r.run_id in q_full.records_by_id]
    mae_before = mae_after = None
    n_scored_before = 0
    q_b = None
    if ctx.builtin:
        q_b = QualityModel().fit(ctx.builtin, ctx.pairs)
        builtin_n, builtin_sigma = q_b.n, q_b.sigma
        if usable:
            errs = [abs(float(r.bpb_2m) - predict_contrib(q_b, r)[0]) for r in usable]
            mae_before, n_scored_before = sum(errs) / len(errs), len(errs)
    else:
        builtin_n, builtin_sigma = 0, None
    if loo and len(usable) >= LOO_MIN and q_full is not None:
        errs = [abs(float(r.bpb_2m) - predict_contrib(q_full.refit_without([r.run_id]), r)[0]) for r in usable]
        mae_after = sum(errs) / len(errs)
    guard = model_guard(ctx, q_full, q_b) if q_b is not None else {"max_worse": GUARD_MAX_WORSE, "ok": True}

    def rnd(x: Optional[float]) -> Optional[float]:
        return None if x is None else float(f"{x:.6g}")

    return {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(recs),
        "n_contributors": len(handles),
        "n_anonymous_records": n_anon,
        "hardware": dict(sorted(hw.items())),
        "base_recipes": dict(sorted(bases.items())),
        "n_used_in_fit": len(usable),
        "n_with_official": n_off,
        "n_imputed_from_official": n_imp,
        "n_flagged_outlier": n_outlier,
        "contrib_offset_mean": rnd(sum(offsets) / len(offsets)) if offsets else None,
        "mae_before": rnd(mae_before),
        "mae_after_loo": rnd(mae_after),
        "n_scored": n_scored_before,
        "loo_min_records": LOO_MIN,
        "max_fit_per_contributor": MAX_FIT_PER_CONTRIBUTOR,
        "builtin": {"n_records": len(ctx.builtin), "n_quality_fit": int(builtin_n),
                    "quality_sigma": rnd(builtin_sigma)},
        "guard": {k: (rnd(v) if isinstance(v, float) else v) for k, v in sorted(guard.items())},
        "quality_fit_n": int(q_full.n) if q_full is not None else 0,
        "data_sha256": data_sha,
        "model": "ffsim/model/params.json",
        "note": ("mae_before: built-in-only model on contributed runs, with the app's anchor shift for an anchored "
                 "base recipe (K82s4); mae_after_loo: leave-one-out over contributed runs with built-in data plus "
                 f"the other contributions (null below {LOO_MIN} usable runs); each (hardware, base recipe) has its "
                 "own fixed effect, so part of the before/after gap is that effect being learned. guard: built-in "
                 "in-sample and pair-validation MAE without/with the contributions (refit --guard refuses a change "
                 f"worse than {GUARD_MAX_WORSE}). Units: val_bpb on the first 2,097,152 public tokens. "
                 "params.json is a published snapshot of the fit; the app and the CLI refit from the data."),
    }


def stats_line(stats: Dict[str, Any]) -> str:
    n, c = stats.get("n_records", 0), stats.get("n_contributors", 0)
    if not n:
        return ("Contributed runs so far: **0**. Be the first: the simulator refits on every merged submission "
                "(see [CONTRIBUTING.md](CONTRIBUTING.md)).")
    hw = ", ".join(f"{k} {v}" for k, v in stats.get("hardware", {}).items())
    s = (f"Contributed runs so far: **{n}** from **{c}** named contributor{'s' if c != 1 else ''} ({hw}); "
         f"{stats.get('n_used_in_fit', 0)} used in the fit.")
    if stats.get("mae_before") is not None:
        s += f" Error on contributed runs: built-in model {stats['mae_before']:.5f} bpb"
        s += (f", after learning from the others {stats['mae_after_loo']:.5f} bpb (leave-one-out)."
              if stats.get("mae_after_loo") is not None else f" (leave-one-out needs {LOO_MIN} usable runs).")
    return s


def update_readme(readme: Path, stats: Dict[str, Any]) -> bool:
    """Replace the text between README_START and README_END with ``stats_line``. True when the file changed."""
    p = Path(readme)
    if not p.exists():
        return False
    with open(p, "r", encoding="utf-8", newline="") as fh:      # keep the file's own line endings
        text = fh.read()
    i, j = text.find(README_START), text.find(README_END)
    if i < 0 or j < i:
        return False
    new = text[:i + len(README_START)] + stats_line(stats) + text[j:]
    if new == text:
        return False
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)
    return True


def _write_json(path: Path, obj: Any) -> bool:
    text = json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return True


def refit(runs: Path = DEFAULT_RUNS, contrib: Path = DEFAULT_CONTRIB, pairs: Path = DEFAULT_PAIRS,
          uploads: Path = DEFAULT_UPLOADS, params_out: Optional[Path] = DEFAULT_PARAMS,
          stats_out: Optional[Path] = DEFAULT_STATS, readme: Optional[Path] = None,
          space_contrib: Optional[Path] = None, loo: bool = True, guard: bool = False) -> Dict[str, Any]:
    """Fit the three models on built-in + contributed data, export params.json and stats.json. With ``guard``,
    nothing is written when ``model_guard`` fails (``out["guard_failed"]``)."""
    from .offset import OffsetModel
    from .params import export_params, write_params
    from .quality import QualityModel
    from .steptime import StepTimeModel
    ctx = load_context(Path(runs), Path(contrib), Path(pairs), Path(uploads))
    records = ctx.builtin + ctx.contrib_records
    st = StepTimeModel()
    st = st.fit(records) or st
    q = QualityModel().fit(records, ctx.pairs)
    off = OffsetModel()
    if ctx.uploads is not None:
        off = off.fit(str(ctx.uploads)) or off
    meta = {"runs_sha256": file_sha256(Path(runs)), "contrib_sha256": file_sha256(Path(contrib)),
            "n_builtin_records": len(ctx.builtin), "n_contrib_records": len(ctx.contrib), "quality_fit_n": q.n}
    data_sha = hashlib.sha256((str(meta["runs_sha256"]) + str(meta["contrib_sha256"])).encode()).hexdigest()
    out: Dict[str, Any] = {"params_changed": False, "stats_changed": False, "readme_changed": False,
                           "space_synced": False, "guard_failed": False}
    stats = compute_stats(ctx, q, loo=loo, data_sha=data_sha)
    out["stats"] = stats
    out["quality_fit_n"] = q.n
    if guard and not stats["guard"].get("ok", True):
        out["guard_failed"] = True
        return out
    if params_out is not None:
        out["params_changed"] = write_params(export_params(st, q, off, meta), Path(params_out))
    if stats_out is not None:
        out["stats_changed"] = _write_json(Path(stats_out), stats)
    if readme is not None:
        out["readme_changed"] = update_readme(Path(readme), stats)
    if space_contrib is not None and Path(contrib).exists():
        sp = Path(space_contrib)
        if sp.parent.parent.exists():
            sp.parent.mkdir(parents=True, exist_ok=True)
            if _norm_bytes(sp) != _norm_bytes(Path(contrib)):
                shutil.copyfile(contrib, sp)
                out["space_synced"] = True
    return out


# --------------------------------------------------------------------------------------------- issue round trip
RECORD_HEADING = "Run record (JSON)"
CONSENT_HEADING = "Consent"
_HEADING = re.compile(r"^###[ \t]+(.*?)[ \t]*$")
_TICKED = re.compile(r"^\s*-[ \t]*\[[xX]\]")


def _sections(body: str) -> List[Tuple[Optional[str], List[str]]]:
    """(heading or None for the text before the first heading, lines) in order: one linear pass over the lines."""
    secs: List[Tuple[Optional[str], List[str]]] = [(None, [])]
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            secs.append((m.group(1), []))
        else:
            secs[-1][1].append(line)
    return secs


def _fenced(lines: List[str]) -> Optional[Tuple[int, int]]:
    """(first, last+1) line indices of the first ``` fenced block's content; an unclosed fence runs to the end.
    Linear: the first opener, then the first line that is only a fence."""
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("```")), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if lines[j].strip() == "```"), len(lines))
    return start + 1, end


def _parse_issue(body: Optional[str]) -> Tuple[str, Optional[bool], str]:
    """(record text, consent state or None, every other piece of free text) of an issue body: one linear pass."""
    if not body or not str(body).strip():
        raise ValueError("the issue body is empty: paste the run record JSON into the form")
    body = str(body)
    if len(body) > MAX_ISSUE_CHARS:
        raise ValueError("the issue body is too long")
    secs = _sections(body)
    consent: Optional[bool] = None
    rec_lines: Optional[List[str]] = None
    free: List[str] = []
    for heading, lines in secs:
        if heading == RECORD_HEADING and rec_lines is None:
            rec_lines = lines
        elif heading == CONSENT_HEADING:
            consent = any(_TICKED.match(ln) for ln in lines)
        else:
            free += [ln for ln in lines if ln.strip() != "_No response_"]
    if rec_lines is None:                 # no form layout: the record is the body's first fenced block, or the body
        rec_lines, free = body.splitlines(), []
        span = _fenced(rec_lines)
        if span:
            free = rec_lines[:span[0] - 1] + rec_lines[span[1] + 1:]
    span = _fenced(rec_lines)
    inner = rec_lines[span[0]:span[1]] if span else rec_lines
    # The form wraps the textarea in its own ```json fence: a pasted fenced block arrives fenced twice.
    while inner and inner[0].strip().startswith("```"):
        inner = inner[1:]
    while inner and inner[-1].strip().startswith("```"):
        inner = inner[:-1]
    return "\n".join(inner).strip(), consent, "\n".join(free).strip()


def _loads_record(text: str) -> Any:
    if text in ("", "_No response_"):
        raise ValueError("no run record JSON found in the issue")
    if max((text.count("{"), text.count("["))) > 400:
        raise ValueError("the run record is nested or repeated too deeply")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        i, j = text.find("{"), text.rfind("}")
        if 0 <= i < j and (i, j) != (0, len(text) - 1):
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"the run record is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno})") from None
    except RecursionError:
        raise ValueError("the run record is nested too deeply") from None


def extract_issue_record(body: Optional[str]) -> Tuple[Any, Optional[bool]]:
    """(parsed JSON, consent checkbox state or None when the body has no Consent section) from an issue body
    written by the run-submission form or by ``issue_body``. Raises ValueError with a readable message."""
    text, consent, _ = _parse_issue(body)
    return _loads_record(text), consent


def issue_free_text(body: Optional[str]) -> str:
    """Everything in an issue body except the record and the consent box (the "Anything else" textarea, text
    around a template-less record): scanned for secrets like the record, since it is just as public."""
    return _parse_issue(body)[2]


def _record_json(rec: Dict[str, Any], compact: bool = False) -> str:
    if compact:
        return json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(rec, indent=1, ensure_ascii=False)


def _issue_title(rec: Dict[str, Any]) -> str:
    score = rec.get("chip_val_bpb", rec.get("official_val_bpb"))
    hw = rec.get("hardware", "?")
    base = rec.get("base_recipe", "custom")
    s = f"{float(score):.5f}" if isinstance(score, (int, float)) else "?"
    return f"[run] {hw} {base} val_bpb {s} @ {rec.get('steps', '?')} steps"


def issue_body(rec: Dict[str, Any]) -> str:
    """The body the run-submission form produces (and ``extract_issue_record`` reads)."""
    return ("### Run record (JSON)\n\n```json\n" + _record_json(rec) + "\n```\n\n### Consent\n\n- [X] " + CONSENT_TEXT
            + "\n")


def _url(params: List[Tuple[str, str]], repo: str) -> str:
    return f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def build_issue_url(rec: Dict[str, Any], repo: str = GITHUB_REPO, max_len: int = ISSUE_URL_MAX) -> str:
    """A pre-filled 'new issue' URL for the run-submission form. Issue forms are pre-filled by FIELD ID (the JSON
    textarea's id is ``record``), not by ``body``, which GitHub applies only to template-less issues: see
    ``build_plain_issue_url`` for that fallback. Pretty JSON when it fits, compact JSON otherwise; ValueError when
    even that is longer than ``max_len`` (paste the JSON into the form instead)."""
    for compact in (False, True):
        url = _url([("template", ISSUE_TEMPLATE), ("title", _issue_title(rec)),
                    (ISSUE_FIELD, _record_json(rec, compact))], repo)
        if len(url) <= max_len:
            return url
    raise ValueError(f"the record makes a {len(url):,}-character URL (limit {max_len:,}): copy the JSON into the "
                     "issue form by hand")


def build_plain_issue_url(rec: Dict[str, Any], repo: str = GITHUB_REPO, max_len: int = ISSUE_URL_MAX) -> str:
    """Fallback: a template-less issue with ``title`` + ``body`` (+ the run-submission label, applied when the
    opener may label). The workflow accepts it because the title starts with [run]."""
    body = issue_body(rec)
    url = _url([("title", _issue_title(rec)), ("labels", "run-submission"), ("body", body)], repo)
    if len(url) > max_len:
        compact = ("### Run record (JSON)\n\n```json\n" + _record_json(rec, True) + "\n```\n\n### Consent\n\n- [X] "
                   + CONSENT_TEXT + "\n")
        url = _url([("title", _issue_title(rec)), ("labels", "run-submission"), ("body", compact)], repo)
    if len(url) > max_len:
        raise ValueError(f"the record makes a {len(url):,}-character URL (limit {max_len:,})")
    return url


_MD_SPECIAL = re.compile(r"([\\\[\]()!*_|~#])")


def _md(s: Any, n: int = 300) -> str:
    """Untrusted text made inert for a GitHub comment: Markdown punctuation escaped (no links, images, emphasis or
    table breaks), no code-span breaks, no @mentions, no HTML, one line, bounded."""
    t = str(s).replace("\r", " ").replace("\n", " ")
    t = t if len(t) <= n else t[:n] + "..."
    t = _MD_SPECIAL.sub(r"\\\1", t.replace("`", "'"))
    return t.replace("@", "@​").replace("<", "&lt;").replace(">", "&gt;").replace("://", ":​//")


def _code(s: Any, n: int = 60) -> str:
    """Text for inside a `code span` in a table cell: no backticks, pipes or newlines (backslashes are literal)."""
    t = str(s).replace("`", "'").replace("|", "/").replace("\r", " ").replace("\n", " ")
    return t if len(t) <= n else t[:n] + "..."


SECRET_NOTICE = ("**The issue text was replaced by this check** because it looked like it contained a secret or "
                 "personal data (the kind is named below). Issues are public the moment they are opened: if it was a "
                 "real key or token, revoke or rotate it now. The original stays in the issue's edit history until "
                 "a maintainer deletes that revision. Open a new issue (or edit this one) with the record only.")


def format_comment(res: ValidationResult, consent: Optional[bool] = True) -> str:
    lines: List[str] = ["<!-- ffsim-contrib-check -->"]
    if res.ok and consent is not False:
        rec = res.record or {}
        lines += ["**Thanks: this run record is valid.** A maintainer reviews it here; when they add the "
                  f"`{APPROVAL_LABEL}` label, a pull request adding it to "
                  "`research/sim-data/contrib/runs-contrib.jsonl` is opened. Once that is merged, the simulator is "
                  "refitted on it. Editing the issue after approval withdraws the approval.", "",
                  "| field | value |", "|---|---|"]
        for k in ("hardware", "base_recipe", "time_budget_s", "steps", "seed", "chip_val_bpb", "official_val_bpb"):
            if rec.get(k) is not None:
                lines.append(f"| {k} | `{_code(rec[k])}` |")
        lines.append(f"| recipe knobs | {len(rec.get('recipe') or {})} |")
        lines.append(f"| content hash | `{(res.content_hash or '')[:12]}` |")
        lines.append(f"| used for fitting | {'yes' if res.usable_for_fit else 'no (stored only)'} |")
        if res.prediction:
            p = res.prediction
            lines += ["", f"Current model: predicted {p['predicted_bpb_2m']:.5f} +- {p['sd']:.5f}, observed "
                          f"{p['observed_bpb_2m']:.5f} (z = {p['z']:+.1f})."]
    else:
        if res.secret_found:
            lines += [SECRET_NOTICE, ""]
        lines += ["**This run record cannot be accepted yet.** Edit the issue to fix it; the check reruns on every "
                  "edit.", ""]
        errs = list(res.errors)
        if consent is False:
            errs.insert(0, "the consent box is not ticked")
        lines += [f"- {_md(e)}" for e in errs]
    if res.warnings:
        lines += ["", "**Notes:**"] + [f"- {_md(w)}" for w in res.warnings]
    if res.flags:
        lines += ["", "Flags for the reviewer: " + ", ".join(f"`{_code(f, 40)}`" for f in res.flags)]
    lines += ["", f"<sub>Checked by `python -m ffsim contrib validate` (schema v{SCHEMA_VERSION}). What is shared and "
                  "how review works: CONTRIBUTING.md.</sub>"]
    return "\n".join(lines) + "\n"


def pr_body(res: ValidationResult, issue_number: Optional[int]) -> str:
    rec = res.record or {}
    ref = f"Closes #{int(issue_number)}.\n\n" if issue_number else ""
    p = res.prediction
    pred = (f"Model before this record: predicted {p['predicted_bpb_2m']:.5f} +- {p['sd']:.5f}, observed "
            f"{p['observed_bpb_2m']:.5f} (z = {p['z']:+.1f}).\n\n") if p else ""
    flags = ", ".join(f"`{_code(f, 40)}`" for f in res.flags) or "none"
    return (f"{ref}Adds one contributed run (content hash `{(res.content_hash or '')[:12]}`, hardware "
            f"`{_code(rec.get('hardware', '?'), 40)}`, base `{_code(rec.get('base_recipe', '?'), 20)}`, "
            f"{int(rec.get('steps') or 0)} steps).\n\n{pred}Flags: {flags}.\n\n"
            "Reviewer checklist:\n\n- [ ] numbers plausible for the hardware and recipe\n"
            "- [ ] no personal data or host details in free-text fields\n"
            "- [ ] if flagged `outlier`, `extrapolates`, `shifts_builtin_fit` or `contributor_cap`: the reason is "
            "understood\n"
            "- [ ] CI was run on this pull request (pull requests opened with the workflow token do not start it: "
            "close and reopen this one)\n\n"
            "**Merging changes the published model.** After the merge, `.github/workflows/refit.yml` refits the "
            "simulator and opens a pull request with the refreshed `ffsim/model/params.json` and "
            "`contrib/stats.json`; it refuses (and fails) if the contributions worsen the model's error on the "
            "team's own runs.\n")


# --------------------------------------------------------------------------------------------- CLI
def _read_json_file(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _gh_output(path: Optional[str], **kv: Any) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in kv.items():
            fh.write(f"{k}={v}\n")


def _ctx_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runs", default=str(DEFAULT_RUNS), help="built-in runs.jsonl")
    p.add_argument("--contrib", default=str(DEFAULT_CONTRIB), help="contributed runs (jsonl)")
    p.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    p.add_argument("--uploads", default=str(DEFAULT_UPLOADS))


def _ctx(a: argparse.Namespace) -> Context:
    return load_context(Path(a.runs), Path(a.contrib), Path(a.pairs), Path(a.uploads))


def _guard_with(ctx: Context, res: ValidationResult) -> Dict[str, Any]:
    """``model_guard`` with ``res``'s record added to the contributions (the check job's preview of the refit)."""
    from .quality import QualityModel
    rows = list(ctx.contrib) + [(res.run_record, {"content_hash": res.content_hash, "submission": res.record})]
    trial = Context(ctx.builtin, rows, ctx.pairs, ctx.offset_mean, ctx.uploads)
    q_full = QualityModel().fit(ctx.builtin + trial.contrib_records, ctx.pairs)
    return model_guard(trial, q_full)


def _issue_inputs(a: argparse.Namespace) -> Tuple[List[Any], List[str], Optional[bool], Optional[int], Optional[str],
                                                   bool]:
    """(records, errors, consent, issue number, issue author login, secret found) from --issue-event/--issue-body."""
    raws: List[Any] = []
    errors: List[str] = []
    consent: Optional[bool] = None
    number: Optional[int] = None
    author: Optional[str] = None
    secret = False
    try:
        if a.issue_event:
            ev = _read_json_file(a.issue_event)
            issue = ev.get("issue") if isinstance(ev, dict) and isinstance(ev.get("issue"), dict) else {}
            n = issue.get("number")
            number = n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None
            user = issue.get("user") if isinstance(issue.get("user"), dict) else {}
            login = user.get("login")
            author = login if isinstance(login, str) and re.match(r"^[A-Za-z0-9-]{1,39}(\[bot\])?$", login) else None
            body = issue.get("body")
        else:
            with open(a.issue_body, "r", encoding="utf-8") as fh:
                body = fh.read()
        text, consent, free = _parse_issue(body)
        kind = scan_text(free) if free else None
        if kind:
            secret = True
            errors.append(f"the issue text outside the record looks like it contains {kind}: remove it (issues are "
                          "public)")
        else:
            raws.append(_loads_record(text))
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:  # noqa: BLE001 - hostile input must still produce a comment
        errors.append(f"the issue could not be read ({type(e).__name__})")
    return raws, errors, consent, number, author, secret


def cmd_validate(a: argparse.Namespace) -> int:
    issue_number: Optional[int] = None
    issue_author: Optional[str] = None
    consent: Optional[bool] = None
    raws: List[Any] = []
    pre_errors: List[str] = []
    pre_secret = False
    if a.issue_event or a.issue_body:
        raws, pre_errors, consent, issue_number, issue_author, pre_secret = _issue_inputs(a)
    raws += [_read_json_file(f) for f in a.files]
    if not raws and not pre_errors:
        print("nothing to validate: give record files, --issue-event or --issue-body", file=sys.stderr)
        return 2
    ctx = _ctx(a)
    quality = None if a.no_model else ctx.quality()
    counts = contributor_counts(ctx.contrib)
    results = [ValidationResult(False, pre_errors, secret_found=pre_secret)] if pre_errors else []
    results += [validate_record(r, quality=quality, existing_hashes=ctx.hashes, builtin=ctx.builtin,
                                offset_mean=ctx.offset_mean, issue_author=issue_author, contributor_counts=counts)
                for r in raws]
    first = results[0]
    line: Optional[str] = None
    if first.ok:
        try:
            line = contrib_line(first)
        except (ValueError, TypeError) as e:
            first = results[0] = ValidationResult(False, [f"the record cannot be stored ({type(e).__name__})"],
                                                  first.warnings, first.flags, first.record, first.content_hash)
    if first.ok and a.guard and first.usable_for_fit and not a.no_model:
        try:
            g = _guard_with(ctx, first)
        except Exception as e:  # noqa: BLE001
            g = {"ok": False, "error": type(e).__name__}
        if not g.get("ok", True):
            first.flags.append("shifts_builtin_fit")
            first.warnings.append("adding this record moves the model's error on the team's own runs by more than "
                                  f"{GUARD_MAX_WORSE} bpb: the reviewer should look closely (the refit would refuse "
                                  "it)")
    all_ok = all(r.ok for r in results) and consent is not False
    for i, r in enumerate(results):
        tag = "VALID" if (r.ok and consent is not False) else "INVALID"
        print(f"[{i}] {tag} hash={(r.content_hash or '-')[:12]} usable_for_fit={r.usable_for_fit}")
        for e in r.errors:
            print(f"    error: {e}")
        if consent is False:
            print("    error: the consent box is not ticked")
        for w in r.warnings:
            print(f"    note: {w}")
        if r.flags:
            print(f"    flags: {', '.join(r.flags)}")
        if r.prediction:
            p = r.prediction
            print(f"    model: predicted {p['predicted_bpb_2m']:.5f} +- {p['sd']:.5f}, observed "
                  f"{p['observed_bpb_2m']:.5f}, z {p['z']:+.2f}")
    if a.out_dir:
        od = Path(a.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "comment.md").write_text(format_comment(first, consent), encoding="utf-8")
        (od / "result.json").write_text(json.dumps(first.to_dict(), indent=1, default=str) + "\n", encoding="utf-8")
        if first.ok and consent is not False and line is not None:
            (od / "record.json").write_text(json.dumps(first.record, indent=1) + "\n", encoding="utf-8")
            (od / "pr-body.md").write_text(pr_body(first, issue_number), encoding="utf-8")
            with open(od / "contrib-line.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
    digest = first.content_hash or ""
    _gh_output(a.github_output, valid="true" if all_ok else "false",
               hash=digest[:12] if re.fullmatch(r"[0-9a-f]{64}", digest) else "",
               issue_number=issue_number or "", secret="true" if any(r.secret_found for r in results) else "false")
    if a.json:
        print(json.dumps([r.to_dict() for r in results], indent=1, default=str))
    return 0 if (all_ok or a.always_zero) else 1


def cmd_ingest(a: argparse.Namespace) -> int:
    ctx = _ctx(a)
    raws = [_read_json_file(f) for f in a.files]
    res = ingest_records(raws, ctx, Path(a.contrib))
    n_ok = sum(r.ok for r in res)
    for f, r in zip(a.files, res):
        print(f"{f}: {'appended ' + (r.content_hash or '')[:12] if r.ok else 'REJECTED'}")
        for e in r.errors:
            print(f"    error: {e}")
        for w in r.warnings:
            print(f"    note: {w}")
    print(f"ingest: {n_ok}/{len(res)} appended to {a.contrib}")
    return 0 if n_ok == len(res) else 1


def cmd_refit(a: argparse.Namespace) -> int:
    out = refit(Path(a.runs), Path(a.contrib), Path(a.pairs), Path(a.uploads),
                None if a.params_out == "none" else Path(a.params_out),
                None if a.stats_out == "none" else Path(a.stats_out),
                Path(a.readme) if a.readme else None,
                SPACE_CONTRIB if a.sync_space else None, loo=not a.no_loo, guard=a.guard)
    s = out["stats"]
    print(f"refit: quality fit on {out['quality_fit_n']} full runs ({s['builtin']['n_quality_fit']} built-in + "
          f"{s['n_used_in_fit']} contributed); {s['n_records']} contributed records from {s['n_contributors']} "
          f"named contributors")
    print(f"  mae_before {s['mae_before']}  mae_after_loo {s['mae_after_loo']}")
    g = s.get("guard", {})
    print(f"  guard: built-in in-sample MAE {g.get('builtin_insample_mae_before')} -> "
          f"{g.get('builtin_insample_mae_after')}, pair MAE {g.get('pair_mae_before')} -> {g.get('pair_mae_after')}"
          f" (max worsening {GUARD_MAX_WORSE}), base-recipe shift {g.get('base_prediction_shift')} (max "
          f"{GUARD_MAX_BASE_SHIFT}): {'ok' if g.get('ok', True) else 'FAILED'}")
    if out.get("guard_failed"):
        print("  refit refused: the contributions worsen the model on the team's own runs; nothing was written",
              file=sys.stderr)
        return 3
    changed = [k for k in ("params_changed", "stats_changed", "readme_changed", "space_synced") if out[k]]
    print("  changed: " + (", ".join(changed) or "nothing"))
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    ctx = _ctx(a)
    s = compute_stats(ctx, loo=not a.no_loo)
    if a.out:
        _write_json(Path(a.out), s)
    print(json.dumps(s, indent=1) if a.json else stats_line(s))
    return 0


def cmd_schema(a: argparse.Namespace) -> int:
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(schema_json())
        print(f"wrote {p}")
    else:
        sys.stdout.write(schema_json())
    return 0


def cmd_issue_url(a: argparse.Namespace) -> int:
    raw = _read_json_file(a.file)
    res = validate_record(raw)
    if not res.ok:
        for e in res.errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    print(build_plain_issue_url(res.record) if a.plain else build_issue_url(res.record))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ffsim contrib",
                                 description="community run contributions: validate, ingest, refit, stats")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check record JSON files or a GitHub issue")
    v.add_argument("files", nargs="*", help="record JSON files ('-' = stdin)")
    v.add_argument("--issue-event", help="GitHub event JSON (GITHUB_EVENT_PATH): reads issue.body and issue.user")
    v.add_argument("--issue-body", help="a file holding an issue body (markdown)")
    v.add_argument("--out-dir", help="write comment.md, result.json and (when valid) record.json, pr-body.md and "
                                     "contrib-line.jsonl (the exact line ingest would append)")
    v.add_argument("--github-output", help="append valid=, hash=, issue_number=, secret= to this file ($GITHUB_OUTPUT)")
    v.add_argument("--always-zero", action="store_true", help="exit 0 even when invalid (CI reads --github-output)")
    v.add_argument("--no-model", action="store_true", help="skip the outlier guard (no model fit)")
    v.add_argument("--guard", action="store_true", help="also preview the refit guard with this record added")
    v.add_argument("--json", action="store_true", help="also print the results as JSON")
    _ctx_args(v)
    v.set_defaults(fn=cmd_validate)
    i = sub.add_parser("ingest", help="validate and append records to the contributed-runs file")
    i.add_argument("files", nargs="+")
    _ctx_args(i)
    i.set_defaults(fn=cmd_ingest)
    r = sub.add_parser("refit", help="refit on built-in + contributed runs; write params.json and stats.json")
    _ctx_args(r)
    r.add_argument("--params-out", default=str(DEFAULT_PARAMS), help="'none' to skip")
    r.add_argument("--stats-out", default=str(DEFAULT_STATS), help="'none' to skip")
    r.add_argument("--readme", default=None, help="README to update between the contrib-stats markers")
    r.add_argument("--sync-space", action="store_true", help="copy the contributed runs into space/research/sim-data")
    r.add_argument("--no-loo", action="store_true")
    r.add_argument("--guard", action="store_true",
                   help=f"write nothing and exit 3 when the contributions worsen the built-in in-sample or "
                        f"pair-validation MAE by more than {GUARD_MAX_WORSE} bpb")
    r.set_defaults(fn=cmd_refit)
    s = sub.add_parser("stats", help="print contribution statistics")
    _ctx_args(s)
    s.add_argument("--out", default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-loo", action="store_true")
    s.set_defaults(fn=cmd_stats)
    sc = sub.add_parser("schema", help="print (or --out) the record JSON Schema")
    sc.add_argument("--out", default=None)
    sc.set_defaults(fn=cmd_schema)
    u = sub.add_parser("issue-url", help="print a pre-filled GitHub issue URL for a record file")
    u.add_argument("file")
    u.add_argument("--plain", action="store_true", help="template-less issue with title + body")
    u.set_defaults(fn=cmd_issue_url)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
