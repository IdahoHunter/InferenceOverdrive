#!/usr/bin/env python3
"""
Inference server manager behind the /server page.

Owns the inference backend process (llama.cpp's llama-server or vLLM's
`vllm serve`) on 127.0.0.1:8080 behind nginx, its parameters, and the model
library under /models. Settings persist in /state/server.json; the first run
seeds them from the container env (.env) and the defaults in seed_config().

The container starts with no backend unless one is asked for (boot_plan):
AUTOSTART=1 in .env starts the backend/model named there, else the boot
default saved on /server, else nothing — the pages and this API run either way.

API (nginx maps /server/api/* to /api/*). When SERVER_ADMIN_TOKEN is set,
every non-GET request needs `Authorization: Bearer <token>`.
  GET    /api/state                     backend status, config, versions
  GET    /api/health                    200 whenever this API answers (a running model is optional)
  GET    /api/vllm/envs                 vLLM installs: the image's + upgrades under /state/envs, the selected one
  POST   /api/vllm/install              {version}: install vLLM into /state/envs/vllm-<version>, check it, select it
  POST   /api/vllm/install/cancel       stop that install (the half-made env is removed)
  PUT    /api/vllm/env                  {name}: use that env from the next vLLM start (null = the image's: rollback)
  DELETE /api/vllm/envs/<name>          remove an env that isn't selected
  POST   /api/manager/restart           re-exec this manager (picks up a new manager.py); the backend keeps
                                        running and is adopted from /state/backend.json
  PUT    /api/boot, DELETE /api/boot    save / clear the backend + settings started with the container
  GET|PUT|DELETE /api/hf/token          Hugging Face token status / save (validated) / remove
  GET    /api/templates                 saved chat templates (/state/templates/*.jinja)
  GET|PUT|DELETE /api/templates/<name>  read / save (syntax-checked) / delete one
  POST   /api/templates/check           Jinja2 syntax check + render of a sample conversation
  POST   /api/templates/from-model      the selected model's own template {backend, values}
  GET    /api/schema                    parameter form per backend (curated fields + form-editor changes)
  PUT    /api/forms, DELETE /api/forms?backend=  save / reset a backend's form layout (gear on /server)
  POST   /api/boot/resume               clear the failed-boot count so autostart runs again
  GET    /api/updates                   last daily check for newer llama.cpp / vLLM releases
  POST   /api/updates/check             run that check now
  POST   /api/smoke                     probe the running backend: tool call parsed? reasoning split out?
  GET    /api/catalog?backend=          every flag parsed from the backend's --help
  GET    /api/logs?after=N              backend output after sequence number N
  PUT    /api/config                    save config without restarting
  POST   /api/apply                     save config, (re)start the selected backend (refused when a flag
                                        is gone from the installed version's --help, see validate_config)
  POST   /api/start, /api/stop          start/stop the selected backend
  GET    /api/models                    local GGUF files and HF model folders
  DELETE /api/models?path=              delete a model file/folder under /models
  GET    /api/hf/search?q=&task=&format=&sort=  repos of an owner, or a search (task/format/sort filters)
  GET    /api/hf/files?repo=&revision=  files (with sizes) in a Hugging Face repo + quant plan for this GPU
  GET    /api/downloads                 download jobs with progress
  POST   /api/downloads                 {repo, files?, revision?, layout: gguf|hf}
  DELETE /api/downloads/<id>            cancel a running download; remove a finished one from the list

Stdlib only, plus huggingface_hub (from the vLLM venv) for downloads.
"""

import argparse
import glob
import hmac
import json
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
# Hugging Face hub cache = /models itself (HF_HUB_CACHE, set in the Dockerfile): repos land in
# /models/models--<owner>--<name>, one copy shared by /server downloads, `vllm serve owner/name`
# and huggingface-cli. llama.cpp's own downloads (LLAMA_CACHE) go to /models too.
HUB_CACHE = os.environ.get("HF_HUB_CACHE", MODELS_DIR)
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, ".huggingface"))
# Older /server downloads used plain repo folders here; still listed and loadable.
HF_DIR = os.path.join(MODELS_DIR, "hf")
STATE_DIR = os.environ.get("STATE_DIR", "/state")
CONFIG_PATH = os.path.join(STATE_DIR, "server.json")
# Jinja chat templates edited on /server (--chat-template-file / --chat-template point here)
TEMPLATES_DIR = os.path.join(STATE_DIR, "templates")
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8080  # must match the upstream in docker/nginx.conf
LLAMA_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
VLLM_BIN = os.environ.get("VLLM_BIN", "vllm")
ADMIN_TOKEN = os.environ.get("SERVER_ADMIN_TOKEN", "")
BACKENDS = ("llama.cpp", "vllm")
# Flags that would move the backend off the address nginx proxies to.
RESERVED_FLAGS = {"--host", "--port", "--api-prefix", "--path", "--uds"}


# ── parameter schema ──────────────────────────────────────────────────────
def field(group, flag, label, kind, help="", options=None, neg=None, placeholder=None):
    return {"group": group, "flag": flag, "label": label, "type": kind, "help": help,
            "options": options, "neg": neg, "placeholder": placeholder}


LLAMA_KV = ["", "f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]
LLAMA_SCHEMA = [
    field("Model", "--model", "Target model", "gguf", "GGUF under /models."),
    field("Model", "--spec-draft-model", "Draft model", "gguf", "DFlash2 / draft GGUF for speculative decoding."),
    field("Model", "--mmproj", "Vision projector", "gguf", "mmproj GGUF for image input."),
    field("Model", "--alias", "Served model name", "str", "Name reported by /v1/models."),
    field("Model", "--mmproj-offload", "Projector on GPU", "bool", neg="--no-mmproj-offload"),
    field("Speculative", "--spec-type", "Speculative type", "select",
          options=["", "none", "draft-dflash", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dspark",
                   "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"]),
    field("Speculative", "--spec-draft-n-max", "Draft tokens per round", "int"),
    field("Speculative", "--spec-draft-ngl", "Draft GPU layers", "int"),
    field("Speculative", "--spec-draft-type-k", "Draft KV type K", "select", options=LLAMA_KV),
    field("Speculative", "--spec-draft-type-v", "Draft KV type V", "select", options=LLAMA_KV),
    field("Context & memory", "--ctx-size", "Context size", "int"),
    field("Context & memory", "--n-gpu-layers", "GPU layers", "int"),
    field("Context & memory", "--cache-type-k", "KV cache type K", "select", options=LLAMA_KV),
    field("Context & memory", "--cache-type-v", "KV cache type V", "select", options=LLAMA_KV),
    field("Context & memory", "--flash-attn", "Flash attention", "select", options=["", "on", "off", "auto"]),
    field("Context & memory", "--fit", "Auto-fit to VRAM", "select", options=["", "on", "off"]),
    field("Throughput", "--parallel", "Parallel slots", "int"),
    field("Throughput", "--batch-size", "Batch size", "int"),
    field("Throughput", "--ubatch-size", "Micro-batch size", "int"),
    field("Throughput", "--threads", "CPU threads", "int"),
    field("Sampling defaults", "--temp", "Temperature", "float"),
    field("Sampling defaults", "--top-p", "Top-p", "float"),
    field("Sampling defaults", "--top-k", "Top-k", "int"),
    field("Sampling defaults", "--min-p", "Min-p", "float"),
    field("Sampling defaults", "--presence-penalty", "Presence penalty", "float"),
    field("Sampling defaults", "--repeat-penalty", "Repeat penalty", "float"),
    field("Chat", "--jinja", "Jinja chat templates", "bool", neg="--no-jinja"),
    field("Chat", "--chat-template-file", "Chat template", "template",
          "A template saved on /server; empty = the one in the model's GGUF. Needs Jinja on."),
    field("Chat", "--reasoning-effort", "Reasoning effort", "select", options=["", "default", "low", "medium", "high", "xhigh"]),
    field("Chat", "--reasoning-preserve", "Preserve reasoning", "bool", neg="--no-reasoning-preserve"),
    field("Server", "--slots", "/slots endpoint", "bool", "The monitor's slot lanes use it.", neg="--no-slots"),
    field("Server", "--ui", "Built-in web UI", "bool", "llama.cpp's own chat page, at /index.html. Chat normally goes "
          "through Open WebUI (the Chat link).", neg="--no-ui"),
    # "_gpus" isn't a flag: it becomes CUDA_VISIBLE_DEVICES for the backend process (see gpu_env)
    field("Multi-GPU", "_gpus", "GPUs", "gpus", "Cards this backend may use. None ticked = all detected cards."),
    field("Multi-GPU", "--split-mode", "Split mode", "select", "layer: layers + KV spread over cards (default); "
          "row/tensor: split each matrix (needs fast links); none: one card (--main-gpu).",
          options=["", "layer", "row", "tensor", "none"]),
    field("Multi-GPU", "--tensor-split", "Tensor split", "str", "Share per card, e.g. 24,24 or 3,1.", placeholder="even"),
    field("Multi-GPU", "--main-gpu", "Main GPU", "int", "Card for split-mode none, or for intermediate results with row."),
    field("Multi-GPU", "--spec-draft-device", "Draft device", "str", "e.g. CUDA1 to keep the draft off the main card.",
          placeholder="follows --device"),
    field("Multi-GPU", "--mmproj-device", "Projector device", "str", "e.g. CUDA1.", placeholder="follows --device"),
]

VLLM_SCHEMA = [
    field("Model", "model", "Model", "hfmodel",
          "Hugging Face repo id (owner/name). Loads from the hub cache in /models; vLLM downloads it there if missing."),
    field("Model", "--served-model-name", "Served model name", "str", "Defaults to the folder or repo name."),
    field("Model", "--quantization", "Quantization", "select",
          options=["", "awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors", "fp8", "bitsandbytes", "gguf"]),
    field("Model", "--dtype", "Dtype", "select", options=["", "auto", "half", "float16", "bfloat16"]),
    field("Model", "--tokenizer", "Tokenizer", "str", "Repo id or path; needed for GGUF models."),
    field("Model", "--trust-remote-code", "Trust remote code", "bool"),
    field("Memory", "--max-model-len", "Context length", "int"),
    field("Memory", "--gpu-memory-utilization", "GPU memory fraction", "float",
          "Share of total VRAM vLLM claims for weights + KV cache. Windows itself holds ~1-2 GB."),
    field("Memory", "--kv-cache-dtype", "KV cache dtype", "select", options=["", "auto", "fp8", "fp8_e4m3", "fp8_e5m2"]),
    field("Memory", "--cpu-offload-gb", "CPU offload (GB)", "float"),
    field("Scheduling", "--max-num-seqs", "Max concurrent sequences", "int"),
    field("Scheduling", "--max-num-batched-tokens", "Max batched tokens", "int"),
    field("Scheduling", "--enable-prefix-caching", "Prefix caching", "bool", neg="--no-enable-prefix-caching"),
    field("Scheduling", "--enable-chunked-prefill", "Chunked prefill", "bool", neg="--no-enable-chunked-prefill"),
    field("Scheduling", "--enforce-eager", "Enforce eager", "bool", "Skips CUDA graph capture: faster start, slower decode."),
    field("Multi-GPU", "_gpus", "GPUs", "gpus", "Cards this backend may use. None ticked = all detected cards."),
    field("Multi-GPU", "--tensor-parallel-size", "Tensor parallel size", "int",
          "Split every layer over N cards (attention heads must divide by N)."),
    field("Multi-GPU", "--pipeline-parallel-size", "Pipeline parallel size", "int",
          "Split the layer stack over N cards; for cards too unequal for tensor parallel."),
    field("Speculative", "--speculative-config", "Speculative config", "json",
          placeholder='{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'),
    # options filled from vLLM's own parser registries at /api/schema time (see vllm_parsers)
    field("Chat & tools", "--reasoning-parser", "Reasoning parser", "select", options=[""]),
    field("Chat & tools", "--enable-auto-tool-choice", "Auto tool choice", "bool"),
    field("Chat & tools", "--tool-call-parser", "Tool call parser", "select", options=[""]),
    field("Chat & tools", "--chat-template", "Chat template", "template",
          "A template saved on /server; empty = the model's tokenizer template."),
    field("Chat & tools", "--generation-config", "Generation config", "select", options=["", "auto", "vllm"]),
    field("Multimodal", "--limit-mm-per-prompt", "Media per prompt", "json", placeholder='{"image": 1, "video": 0}'),
]
SCHEMAS = {"llama.cpp": LLAMA_SCHEMA, "vllm": VLLM_SCHEMA}
MODEL_FLAGS = {"llama.cpp": "--model", "vllm": "model"}
FIELD_TYPES = ("text", "int", "float", "bool", "select", "json")
FLAG_RE = r"-{1,2}[A-Za-z0-9][\w.-]*"


# ── form layout (the gear on /server) ─────────────────────────────────────
# cfg["forms"][backend] = {"order": [{flag, group}], "custom": {flag: field def}, "removed": [base flags]}.
# The layout only changes what the page shows: argv is built from every curated field plus every custom
# definition (all_fields), so a value saved in a boot default never silently drops out. The page clears
# a field's value when it is taken off the form.
def clean_form(backend, form, previous=None):
    """Validate a layout posted by the page; None resets to the curated form."""
    if not form or form.get("order") is None:
        return None
    base = {f["flag"] for f in SCHEMAS[backend]}
    defs = form.get("custom") or {}
    custom = dict((previous or {}).get("custom") or {})  # keep defs a boot default may still carry values for
    order, seen = [], set()
    for e in form["order"]:
        flag = e.get("flag") if isinstance(e, dict) else None
        if not isinstance(flag, str) or flag in seen:
            continue
        group = str(e.get("group") or "").strip()[:40] or "Custom"
        if flag not in base:
            d = defs.get(flag) or custom.get(flag)
            if not d:
                raise ValueError(f"{flag}: no field definition")
            if not re.fullmatch(FLAG_RE, flag) or flag in RESERVED_FLAGS:
                raise ValueError(f"{flag} can't be a form field")
            kind = d.get("type")
            if kind not in FIELD_TYPES:
                raise ValueError(f"{flag}: type must be one of {', '.join(FIELD_TYPES)}")
            options = [str(o) for o in d.get("options") or []] if kind == "select" else None
            if kind == "select" and not options:
                raise ValueError(f"{flag}: a select needs options")
            neg = d.get("neg") if kind == "bool" and re.fullmatch(FLAG_RE, d.get("neg") or "") else None
            custom[flag] = {"label": str(d.get("label") or flag).strip()[:60], "type": kind,
                            "help": str(d.get("help") or "")[:300], "options": options, "neg": neg}
        seen.add(flag)
        order.append({"flag": flag, "group": group})
    if MODEL_FLAGS[backend] not in seen:
        raise ValueError("the model field can't be taken off the form")
    return {"order": order, "custom": custom, "removed": [f["flag"] for f in SCHEMAS[backend] if f["flag"] not in seen]}


def _custom_field(group, flag, d):
    return {**field(group, flag, d["label"], d["type"], d.get("help", ""), d.get("options"), d.get("neg")), "custom": True}


def form_schema(backend, form=None):
    """The fields the page shows, in order. Curated fields added in a newer manager than the saved
    layout appear after their curated neighbour."""
    if not form:
        return [dict(f) for f in SCHEMAS[backend]]
    base = {f["flag"]: f for f in SCHEMAS[backend]}
    out = []
    for e in form["order"]:
        if e["flag"] in base:
            out.append({**base[e["flag"]], "group": e["group"]})
        elif e["flag"] in form.get("custom", {}):
            out.append(_custom_field(e["group"], e["flag"], form["custom"][e["flag"]]))
    shown, removed = {f["flag"] for f in out}, set(form.get("removed") or [])
    for i, f in enumerate(SCHEMAS[backend]):
        if f["flag"] in shown or f["flag"] in removed:
            continue
        before = {g["flag"] for g in SCHEMAS[backend][:i]}
        at = max((k for k, g in enumerate(out) if g["flag"] in before), default=len(out) - 1)
        out.insert(at + 1, dict(f))
        shown.add(f["flag"])
    return out


def all_fields(backend, form=None):
    """Every field argv can carry: the curated ones plus the form editor's custom ones."""
    fields = [dict(f) for f in SCHEMAS[backend]]
    for flag, d in ((form or {}).get("custom") or {}).items():
        if flag not in {f["flag"] for f in SCHEMAS[backend]}:
            fields.append(_custom_field("Custom", flag, d))
    return fields


# ── config ────────────────────────────────────────────────────────────────
def _env(name, default=None):
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _num(name, default, conv=int):
    try:
        return conv(_env(name, default))
    except (TypeError, ValueError):
        return None


def _rel_model(path):
    return os.path.relpath(path, MODELS_DIR) if path and path.startswith(MODELS_DIR + "/") else path


def _flag(name):
    return (_env(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _model_env(path_var, file_var):
    """A model named in .env: a full /models path (TARGET_GGUF) or a file under /models (TARGET_FILE)."""
    path = _env(path_var) or (os.path.join(MODELS_DIR, _env(file_var)) if _env(file_var) else None)
    return _rel_model(path) if path else None


def seed_config():
    """Settings from the env (.env), else the RTX 3090 defaults below. No model unless .env names one: the
    container starts with no backend; see boot_plan for what does start one."""
    kv_k, kv_v = _env("CACHE_TYPE_K", "q8_0"), _env("CACHE_TYPE_V", "q8_0")
    llama = {
        "--model": _model_env("TARGET_GGUF", "TARGET_FILE"),
        "--spec-draft-model": _model_env("DRAFT_GGUF", "DRAFT_FILE"),
        "--mmproj": _model_env("MMPROJ_GGUF", "MMPROJ_FILE"),
        "--mmproj-offload": _env("MMPROJ_OFFLOAD", "1") == "1",
        "--alias": _env("ALIAS"),
        "--spec-type": _env("SPEC_TYPE", "draft-dflash"),
        "--spec-draft-n-max": _num("SPEC_DRAFT_N_MAX", 4),
        "--spec-draft-ngl": _num("DRAFT_GPU_LAYERS", 99),
        "--spec-draft-type-k": kv_k,
        "--spec-draft-type-v": kv_v,
        "--ctx-size": _num("CTX_SIZE", 32768),
        "--n-gpu-layers": _num("N_GPU_LAYERS", 99),
        "--cache-type-k": kv_k,
        "--cache-type-v": kv_v,
        "--flash-attn": _env("FLASH_ATTN", "on"),
        "--fit": _env("FIT", "off"),
        "--parallel": _num("PARALLEL", 1),
        "--batch-size": _num("BATCH", 2048),
        "--ubatch-size": _num("UBATCH", 256),
        "--threads": _num("THREADS", 8),
        "--temp": _num("TEMPERATURE", 1.0, float),
        "--top-p": _num("TOP_P", 0.95, float),
        "--top-k": _num("TOP_K", 20),
        "--jinja": _env("JINJA", "1") == "1",
        "--reasoning-effort": _env("REASONING_EFFORT", "xhigh"),
        "--reasoning-preserve": _env("REASONING_PRESERVE", "1") == "1",
        "--slots": _env("SLOTS", "1") == "1",
        "--ui": _env("LLAMA_WEBUI", "0") == "1",  # chat goes through Open WebUI
    }
    vllm = {
        "model": _env("VLLM_MODEL", ""),
        "--max-model-len": _num("VLLM_MAX_MODEL_LEN", None),
        "--gpu-memory-utilization": _num("VLLM_GPU_MEMORY_UTILIZATION", 0.88, float),
        "--enable-prefix-caching": True,
    }
    return {
        "backend": _env("BACKEND", "llama.cpp"),
        "llama.cpp": {"values": llama, "extra": []},
        "vllm": {"values": vllm, "extra": []},
        # page behaviour: filter model pickers by role/compatibility; re-tune params when the model changes
        "ui": {"smart_filter": True, "auto_tune": True, "update_check": True, "smoke_test": True},
        # backend + settings started when the container starts (set on /server); None = start nothing
        "boot": None,
        # form editor layouts per backend (see clean_form); absent = the curated form
        "forms": {},
    }


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = seed_config()
        save_config(cfg)
        return cfg
    seeded = seed_config()
    cfg.pop("autostart", None)  # replaced by "boot": nothing starts unless a boot default or AUTOSTART says so
    cfg.setdefault("forms", {})
    for key in ("backend", "ui", "boot", *BACKENDS):
        cfg.setdefault(key, seeded[key])
    for key, value in seeded["ui"].items():
        cfg["ui"].setdefault(key, value)
    # llama.cpp's built-in UI became opt-in (chat goes through Open WebUI): settings saved before
    # the toggle existed, including a boot default, get it off rather than the backend's "on".
    for conf in (cfg.get("llama.cpp"), ((cfg.get("boot") or {}).get("config") if (cfg.get("boot") or {}).get("backend") == "llama.cpp" else None)):
        if conf is not None:
            conf.setdefault("values", {}).setdefault("--ui", False)
    return cfg


def boot_plan(cfg):
    """What the container starts: .env with AUTOSTART=1 > boot default saved on /server > nothing.
    Returns (source, backend, {"values", "extra"}) or (None, None, None)."""
    if _flag("AUTOSTART"):
        seeded = seed_config()
        return "env", seeded["backend"], seeded[seeded["backend"]]
    boot = cfg.get("boot")
    if boot and boot.get("backend") in BACKENDS and boot.get("config"):
        return "default", boot["backend"], boot["config"]
    return None, None, None


# ── Hugging Face token ────────────────────────────────────────────────────
# Saved from /server into /state (git-ignored, mode 600). huggingface_hub, and so vLLM and the
# downloader, read it through HF_TOKEN_PATH; an HF_TOKEN in .env takes precedence over it.
HF_TOKEN_PATH = os.environ.setdefault("HF_TOKEN_PATH", os.path.join(STATE_DIR, "hf_token"))
_hf_users = {}


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "env"
    try:
        with open(HF_TOKEN_PATH) as f:
            token = f.read().strip()
        return (token, "saved") if token else (None, None)
    except OSError:
        return None, None


def hf_headers():
    token, _ = hf_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_whoami(token):
    """Hugging Face account name for a token; ValueError if Hugging Face rejects it."""
    if token in _hf_users:
        return _hf_users[token]
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            name = json.load(r).get("name")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError("Hugging Face rejected this token (invalid or revoked)")
        raise
    _hf_users[token] = name
    return name


def hf_token_status(check=True):
    token, source = hf_token()
    status = {"source": source, "hint": f"hf_…{token[-4:]}" if token else None, "user": None, "path": HF_TOKEN_PATH}
    if token and check:
        try:
            status["user"] = hf_whoami(token)
        except Exception as e:
            status["error"] = str(e)
    return status


def save_hf_token(token):
    token = (token or "").strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", token):
        raise ValueError("that doesn't look like a Hugging Face token (hf_…)")
    user = hf_whoami(token)
    os.makedirs(os.path.dirname(HF_TOKEN_PATH), exist_ok=True)
    fd = os.open(HF_TOKEN_PATH + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.replace(HF_TOKEN_PATH + ".tmp", HF_TOKEN_PATH)
    return user


def save_config(cfg):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def models_path(rel, must_exist=True):
    """Resolve a /models-relative path, refusing anything that escapes /models.

    Returns the path as named (hub-cache snapshot files are symlinks into blobs/ and keep their
    .gguf names this way); containment is checked on the fully resolved target."""
    named = os.path.normpath(os.path.join(MODELS_DIR, rel))
    full = os.path.realpath(named)
    root = os.path.realpath(MODELS_DIR)
    if full == root or not full.startswith(root + os.sep) or not named.startswith(MODELS_DIR.rstrip("/") + "/"):
        raise ValueError(f"path outside {MODELS_DIR}: {rel}")
    if must_exist and not os.path.exists(full):
        raise ValueError(f"not found: {rel}")
    return named


def build_argv(backend, cfg):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    values = cfg.get(backend, {}).get("values", {})
    if backend == "llama.cpp":
        if not values.get("--model"):
            raise ValueError("pick a target model")
        argv = [LLAMA_BIN, "--host", BACKEND_HOST, "--port", str(BACKEND_PORT), "--metrics"]
    else:
        model = (values.get("model") or "").strip()
        if not model:
            raise ValueError("pick a vLLM model")
        # a repo id resolves through the hub cache in /models; legacy hf/<owner>--<name> folders still work
        local = os.path.join(MODELS_DIR, model)
        is_repo = re.fullmatch(r"[\w.-]+/[\w.-]+", model) and not model.startswith("hf/")
        target = model if is_repo else models_path(model) if os.path.exists(local) else None
        if not target:
            raise ValueError(f"{model!r} is neither a Hugging Face repo id nor a folder under {MODELS_DIR}")
        argv = [vllm_bin(), "serve", target, "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)]
        if not values.get("--served-model-name"):
            argv += ["--served-model-name", os.path.basename(model.rstrip("/")).replace("--", "/", 1)
                     if model.startswith("hf/") else model]

    for f in all_fields(backend, (cfg.get("forms") or {}).get(backend)):
        flag, kind = f["flag"], f["type"]
        if not flag.startswith("-"):
            continue
        value = values.get(flag)
        if value is None or value == "":
            continue
        if kind == "bool":
            if value:
                argv.append(flag)
            elif f["neg"]:
                argv.append(f["neg"])
        elif kind == "gguf":
            argv += [flag, models_path(value)]
        elif kind == "template":
            path = template_path(value)
            if not os.path.exists(path):
                raise ValueError(f"chat template {value!r} not found in {TEMPLATES_DIR}")
            argv += [flag, path]
        elif kind == "json":
            try:
                json.loads(value)
            except ValueError as e:
                raise ValueError(f"{flag}: invalid JSON ({e})")
            argv += [flag, value]
        else:
            argv += [flag, str(value)]

    for extra in cfg.get(backend, {}).get("extra", []):
        if not extra.get("enabled", True):
            continue
        flag = (extra.get("flag") or "").strip()
        if not flag:
            continue
        if not re.fullmatch(r"-{1,2}[A-Za-z0-9][\w.-]*", flag):
            raise ValueError(f"not a flag: {flag!r}")
        if flag in RESERVED_FLAGS:
            raise ValueError(f"{flag} is managed by the server (nginx proxies to {BACKEND_HOST}:{BACKEND_PORT})")
        argv.append(flag)
        value = extra.get("value")
        if value not in (None, ""):
            argv.append(str(value))
    return argv


def hf_env(base=None):
    """Point every Hugging Face / llama.cpp download at /models (the Dockerfile sets the same)."""
    env = dict(base or os.environ)
    env.setdefault("HF_HUB_CACHE", HUB_CACHE)
    env.setdefault("HF_HOME", HF_HOME)
    env.setdefault("LLAMA_CACHE", MODELS_DIR)
    return env


def hub_repo_dir(repo):
    return os.path.join(HUB_CACHE, "models--" + repo.replace("/", "--"))


def hub_snapshot(repo):
    """Local snapshot folder of a cached repo (refs/main, else the newest snapshot), or None."""
    root = hub_repo_dir(repo)
    try:
        with open(os.path.join(root, "refs", "main")) as f:
            snap = os.path.join(root, "snapshots", f.read().strip())
            if os.path.isdir(snap):
                return snap
    except OSError:
        pass
    snaps = sorted(glob.glob(os.path.join(root, "snapshots", "*")), key=os.path.getmtime)
    return snaps[-1] if snaps else None


def gpu_env(cfg, backend):
    """CUDA_VISIBLE_DEVICES from the backend's GPU picker; empty = every card."""
    sel = cfg.get(backend, {}).get("values", {}).get("_gpus") or []
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(int(i)) for i in sorted(sel))} if sel else {}


def backend_env(backend, extra=None):
    env = hf_env()
    # number CUDA devices like nvidia-smi does, so the GPU picker's indices mean the same card
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.update(extra or {})
    if backend == "vllm":
        venv = vllm_env()
        if venv:  # an upgraded vLLM from /state/envs: its own python / console scripts first
            env["VIRTUAL_ENV"] = venv
            env["PATH"] = os.path.join(venv, "bin") + ":" + env.get("PATH", os.environ.get("PATH", ""))
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        env.setdefault("DO_NOT_TRACK", "1")
        # torch.compile artifacts and FlashInfer/Triton JIT kernels take minutes to build the first
        # time; keep them on the /state volume so container rebuilds don't redo it.
        cache = os.path.join(STATE_DIR, "cache")
        env.setdefault("VLLM_CACHE_ROOT", os.path.join(cache, "vllm"))
        env.setdefault("FLASHINFER_WORKSPACE_BASE", cache)
        env.setdefault("TRITON_CACHE_DIR", os.path.join(cache, "triton"))
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
    return env


# ── backend process ───────────────────────────────────────────────────────
# The running engine survives a manager restart (POST /api/manager/restart re-executes this file
# in place with os.execv, so the PID and the engine's parent stay the same): it is recorded in
# BACKEND_RECORD while it runs and writes its output to BACKEND_LOG instead of a pipe, and the
# new manager adopts it from there instead of running the boot plan.
BACKEND_RECORD = os.path.join(STATE_DIR, "backend.json")
BACKEND_LOG = os.path.join(STATE_DIR, "logs", "backend.log")
MANAGER_STARTED_AT = time.time()
ADOPT_LOG_BYTES = 512 * 1024  # how much of the engine's log an adopting manager reloads


def proc_starttime(pid):
    """Start time of a live, non-zombie process (clock ticks since boot, /proc/<pid>/stat field 22);
    None if there is none. pid + start time identify a process across PID reuse."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] in ("Z", "X") else int(fields[19])


def reap_inherited(keep=()):
    """After an os.execv restart, children the previous manager had in flight (a --help probe, the
    parser registry check, hw-metrics from the entrypoint) are still ours but nothing waits for them:
    wait for each in a thread so they don't linger as zombies. Call before spawning anything."""
    me = str(os.getpid())
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in keep:
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = f.read().rsplit(")", 1)[1].split()[1]
        except (OSError, IndexError):
            continue
        if ppid == me:
            threading.Thread(target=lambda pid=int(d): _waitpid_quiet(pid), daemon=True).start()


def _waitpid_quiet(pid):
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


class AdoptedProc:
    """The Popen-like handle (pid, poll, wait, returncode) for an engine the manager adopted after an
    os.execv restart. It is still this process's child, so waitpid reaps it and gets its exit code."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            if self.returncode is None:
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                    if pid:
                        self.returncode = os.waitstatus_to_exitcode(status)
                except ChildProcessError:  # not our child after all: all we can tell is that it's gone
                    if proc_starttime(self.pid) is None:
                        self.returncode = -1
            return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(f"pid {self.pid}", timeout)
            time.sleep(0.1)
        return self.returncode


class Backend:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.kind = None
        self.argv = None
        self.state = "stopped"  # stopped | starting | running | stopping | failed
        self.intent = "stopped"
        self.started_at = self.ready_at = None
        self.exit_code = None
        self.error = None
        self.logs = deque(maxlen=5000)
        self.seq = 0
        self.log_id = uuid.uuid4().hex[:8]  # changes with each manager (re)start, so the page reloads its log view
        self.listener = None  # called with ("ready"|"failed", error) for the current process (boot guard)
        self.meta = {}  # kept in BACKEND_RECORD for whoever adopts this process (e.g. which boot plan started it)

    def log(self, line, echo=True, tag=None):
        with self.lock:
            self.seq += 1
            self.logs.append((self.seq, time.time(), line))
        if echo:
            print(f"[{tag}] {line}" if tag else line, flush=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind, argv, extra_env=None, listener=None, meta=None):
        with self.lock:
            if self.alive():
                raise RuntimeError("backend already running")
            self.kind, self.argv, self.listener, self.meta = kind, argv, listener, dict(meta or {})
            self.state, self.intent = "starting", "running"
            self.started_at, self.ready_at = time.time(), None
            self.exit_code = self.error = None
            prefix = "".join(f"{k}={v} " for k, v in (extra_env or {}).items())
            env = backend_env(kind, extra_env)
            env.setdefault("PYTHONUNBUFFERED", "1")  # vLLM's output goes to a file now, not a pipe: no block buffering
            try:
                os.makedirs(os.path.dirname(BACKEND_LOG), exist_ok=True)
                try:
                    os.unlink(BACKEND_LOG)  # a fresh file (inode), so a tailer of the last run never reads this one
                except FileNotFoundError:
                    pass
                with open(BACKEND_LOG, "wb") as out:
                    # the command line goes in the file too, so an adopting manager shows it
                    out.write(("$ " + prefix + shlex.join(argv) + "\n").encode())
                    out.flush()
                    self.proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                 env=env, start_new_session=True)
            except OSError as e:
                self.log("$ " + prefix + shlex.join(argv), tag="server")
                self.state, self.error = "failed", str(e)
                self.log(f"failed to start: {e}", tag="server")
                self._notify("failed", str(e))
                return
            proc = self.proc
            self._save_record()
        threading.Thread(target=self._tail, args=(proc,), daemon=True).start()
        threading.Thread(target=self._watch, args=(proc,), daemon=True).start()

    def adopt(self, listener=None):
        """Manager start: take over the engine BACKEND_RECORD names if it is still the same live process
        (pid + start time match). Returns the record, or None (nothing to adopt; a stale record is removed)."""
        rec = _read_json(BACKEND_RECORD)
        if not rec:
            return None
        pid = rec.get("pid")
        if not isinstance(pid, int) or rec.get("kind") not in BACKENDS or proc_starttime(pid) != rec.get("starttime"):
            self.log(f"not adopting the {rec.get('kind')} recorded in {BACKEND_RECORD} (pid {pid}): "
                     "that process is gone", tag="server")
            self._clear_record()
            return None
        with self.lock:
            self.proc = AdoptedProc(pid)
            self.kind, self.argv, self.listener, self.meta = rec["kind"], rec.get("argv"), listener, rec.get("meta") or {}
            self.started_at, self.ready_at = rec.get("started_at"), rec.get("ready_at")
            # still loading → the /health polling in _watch takes it to running, as for a fresh start
            self.state, self.intent = ("running" if self.ready_at else "starting"), "running"
            self.exit_code = self.error = None
            proc = self.proc
        try:
            offset = max(0, os.path.getsize(BACKEND_LOG) - ADOPT_LOG_BYTES)
        except OSError:
            offset = 0
        self.log(f"adopted the running {self.kind} (pid {pid}, {self.state}) after a manager restart", tag="server")
        threading.Thread(target=self._tail, args=(proc, offset, True), daemon=True).start()
        threading.Thread(target=self._watch, args=(proc,), daemon=True).start()
        return rec

    def _save_record(self):
        proc = self.proc
        rec = {"kind": self.kind, "pid": proc.pid, "starttime": proc_starttime(proc.pid), "argv": self.argv,
               "started_at": self.started_at, "ready_at": self.ready_at, "log": BACKEND_LOG, "meta": self.meta}
        try:
            with open(BACKEND_RECORD + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(BACKEND_RECORD + ".tmp", BACKEND_RECORD)
        except OSError as e:
            self.log(f"cannot write {BACKEND_RECORD}: {e} (a manager restart won't find this backend)", tag="server")

    @staticmethod
    def _clear_record():
        try:
            os.remove(BACKEND_RECORD)
        except OSError:
            pass

    def _tail(self, proc, offset=0, replay=False):
        """Engine output → the log buffer, following BACKEND_LOG like `tail -f` until the process is gone.
        \\r (progress bars) ends a line too, as the text-mode pipe this replaces did."""
        try:
            f = open(BACKEND_LOG, "rb")
        except OSError as e:
            self.log(f"cannot read {BACKEND_LOG}: {e}", tag="server")
            return
        with f:
            # an adopting manager reloads what is already in the file: into the page's log, but not
            # printed to `docker logs` a second time
            replay_end = os.fstat(f.fileno()).st_size if replay else 0
            f.seek(offset)
            if offset:
                f.readline()  # started mid-file: skip the partial line
            buf, done, pos = b"", False, f.tell()  # pos: file offset where buf starts
            while True:
                chunk = f.read(65536)
                if chunk:
                    buf += chunk
                    cut = len(buf) - 1 if buf.endswith(b"\r") else len(buf)  # \r\n may be split across reads
                    data, start = buf[:cut], 0
                    for sep in re.finditer(rb"\r\n|\r|\n", data):
                        self.log(data[start:sep.start()].decode(errors="replace"), echo=pos + sep.end() > replay_end)
                        start = sep.end()
                    pos += start
                    buf = data[start:] + buf[cut:]
                    continue
                if done:
                    break
                # one more read after the process is gone, for what it wrote just before exiting
                done = proc.poll() is not None or self.proc is not proc
                if not done:
                    time.sleep(0.25)
            if buf.strip(b"\r"):
                self.log(buf.strip(b"\r").decode(errors="replace"))

    def _watch(self, proc):
        while proc.poll() is None:
            if self.state == "starting" and backend_healthy():
                with self.lock:
                    ready = self.proc is proc and self.state == "starting"
                    if ready:
                        self.state, self.ready_at = "running", time.time()
                        self.log(f"{self.kind} ready after {self.ready_at - self.started_at:.0f}s", tag="server")
                        self._save_record()
                if ready:
                    self._notify("ready")
            time.sleep(1)
        with self.lock:
            if self.proc is not proc:
                return
            self._clear_record()
            self.exit_code = proc.returncode
            failed = self.intent != "stopped"
            if not failed:
                self.state = "stopped"
            else:
                self.state, self.error = "failed", f"{self.kind} exited with code {proc.returncode}"
                self.log(self.error, tag="server")
        if failed:
            self._notify("failed", self.error)

    def _notify(self, event, error=None):
        listener, self.listener = self.listener, None  # one outcome per start
        if listener:
            try:
                listener(event, error)
            except Exception as e:  # never let bookkeeping take the watcher down
                self.log(f"listener error: {e}", tag="server")

    def stop(self, timeout=60):
        with self.lock:
            self.intent = "stopped"
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.state = "stopped"
                return
            self.state = "stopping"
            self.log(f"stopping {self.kind} (pid {proc.pid})", tag="server")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.log("did not exit in time, killing", tag="server")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass
        with self.lock:
            self.state, self.exit_code = "stopped", proc.returncode
            if self.proc is proc:
                self._clear_record()
        # vLLM's engine core can outlive the API process for a moment; let its VRAM go first.
        for _ in range(20):
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.5)

    def snapshot(self):
        with self.lock:
            now = time.time()
            return {
                "kind": self.kind, "state": self.state, "intent": self.intent,
                "pid": self.proc.pid if self.alive() else None,
                "argv": self.argv, "exit_code": self.exit_code, "error": self.error,
                "started_at": self.started_at, "ready_at": self.ready_at,
                "uptime": now - self.ready_at if self.state == "running" and self.ready_at else None,
                "log_seq": self.seq,
            }

    def logs_after(self, after, limit=1500):
        with self.lock:
            lines = [l for l in self.logs if l[0] > after]
        return lines[-limit:]


def backend_healthy():
    try:
        with urllib.request.urlopen(f"http://{BACKEND_HOST}:{BACKEND_PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# ── --help catalog ────────────────────────────────────────────────────────
_catalog_cache = {}


def parse_help(text):
    """Turn llama.cpp's or argparse's --help text into [{flag, aliases, metavar, choices, help, group}]."""
    out, cur, group = [], None, ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        m = re.fullmatch(r"-{3,}\s*(.+?)\s*-{3,}", stripped)  # llama.cpp: "----- common params -----"
        if m or (indent == 0 and stripped.endswith(":") and not stripped.startswith("-")):  # argparse: "ModelConfig:"
            group, cur = (m.group(1) if m else stripped[:-1]), None
            continue
        if stripped.startswith("-") and indent < 8:
            # llama.cpp pads aliases ("-t,    --threads N"); the help text starts after the next 2+ space gap
            pieces = re.split(r"\s{2,}", re.sub(r",\s+", ", ", stripped), maxsplit=1)
            spec, desc = pieces[0], (pieces[1] if len(pieces) > 1 else "")
            tokens = spec.split()
            is_flag = [bool(re.fullmatch(r"-{1,2}\w[\w.-]*,?", t)) for t in tokens]
            flags = [t.rstrip(",") for t, f in zip(tokens, is_flag) if f]
            # keeps comma choice lists ("none,draft-simple,...") in one piece; argparse repeats
            # the metavar after every alias ("--a X, --b X"), so dedupe
            metavar = " ".join(dict.fromkeys(t.rstrip(",") for t, f in zip(tokens, is_flag) if not f))
            longs = [f for f in flags if f.startswith("--") and not f.startswith("--no-")]
            if not flags:
                continue
            choices = None
            m = re.fullmatch(r"[\[{]?([\w.-]+(?:[|,][\w.-]+)+)[\]}]?", metavar)
            if m:
                choices = re.split(r"[|,]", m.group(1))
            cur = {"flag": (longs or flags)[0], "aliases": [f for f in flags if f != (longs or flags)[0]],
                   "metavar": metavar, "choices": choices, "help": desc.strip(), "group": group}
            out.append(cur)
        elif cur is not None and indent >= 8:
            cur["help"] = (cur["help"] + " " + stripped).strip()
    for c in out:  # llama.cpp spells some choice lists out in the text: "allowed values: f32, f16, ..."
        m = not c["choices"] and re.search(r"(?i)allowed values:\s*([\w.-]+(?:,\s*[\w.-]+)+)", c["help"])
        if m:
            c["choices"] = re.split(r",\s*", m.group(1))
    return out


CATALOG_FORMAT = 2  # bump when parse_help changes, so cached catalogs are re-read
_catalog_lock = threading.Lock()
_catalog_loading = set()


def _catalog_file(backend):
    ver = re.sub(r"[^\w.-]+", "_", versions().get(backend) or "unknown")
    return os.path.join(STATE_DIR, "cache", f"catalog{CATALOG_FORMAT}-{backend}-{ver}.json")


def catalog(backend, block=True):
    """Every flag in the installed backend's --help. Cached in memory and in /state/cache per version
    (vLLM's --help imports torch: ~20 s). block=False returns None at once while it loads."""
    if backend in _catalog_cache:
        return _catalog_cache[backend]
    path = _catalog_file(backend)
    try:
        with open(path) as f:
            flags = json.load(f)
        if flags:
            _catalog_cache[backend] = flags
            return flags
    except (OSError, ValueError):
        pass
    if not block:
        if backend not in _catalog_loading:
            _catalog_loading.add(backend)
            threading.Thread(target=lambda: (catalog(backend), _catalog_loading.discard(backend)), daemon=True).start()
        return None
    with _catalog_lock:
        if backend in _catalog_cache:
            return _catalog_cache[backend]
        argv = [LLAMA_BIN, "--help"] if backend == "llama.cpp" else [vllm_bin(), "serve", "--help=all"]
        res = subprocess.run(argv, capture_output=True, text=True, timeout=180, env=backend_env(backend))
        flags = parse_help(res.stdout or res.stderr)
        if flags:
            _catalog_cache[backend] = flags
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(flags, f)
            except OSError:
                pass
        return flags


# a statement that the flag itself is deprecated — "(DEPRECATED)", "is deprecated and will be removed", ...
# — not a flag that merely mentions deprecated things ("--allow-deprecated-quantization")
DEPRECATED_RE = re.compile(r"(?i)[(\[]deprecated[)\]]|\b(?:is|are|now|will be|has been)\s+deprecated\b|^deprecated\b")
BLOCKING = ("removed", "bad-value")


def validate_config(backend, cfg, block=True):
    """Check the flags these settings pass against the installed version's --help:
    removed (not listed: renamed or dropped — the backend would refuse to start), deprecated (still
    works, going away), bad-value (not one of the flag's listed choices). Returns a list of issues,
    [] when --help couldn't be parsed (no grounds to judge), None while it loads with block=False."""
    flags = catalog(backend, block)
    if flags is None:
        return None
    if len(flags) < 20:
        return []
    idx = {}
    for c in flags:
        for name in (c["flag"], *c["aliases"]):
            idx.setdefault(name, c)
    conf = cfg.get(backend, {})
    values, issues = conf.get("values", {}), []

    def check(flag, value, source):
        c = idx.get(flag)
        if c is None:
            issues.append({"flag": flag, "level": "removed", "source": source,
                           "msg": f"{flag} isn't in this {backend} version's --help — renamed or removed"})
            return
        if DEPRECATED_RE.search(c["help"]):
            note = next((s for s in re.split(r"(?<=[.!?])\s+", c["help"]) if DEPRECATED_RE.search(s)), c["help"])
            note = re.sub(r"\s*\(env: [^)]*\)", "", note)
            issues.append({"flag": flag, "level": "deprecated", "source": source, "msg": f"{flag}: {note[:220]}"})
        if c.get("choices") and value not in (None, "") and str(value) not in c["choices"]:
            issues.append({"flag": flag, "level": "bad-value", "source": source,
                           "msg": f"{flag} {value}: not one of {', '.join(c['choices'])}"})

    for f in all_fields(backend, (cfg.get("forms") or {}).get(backend)):
        flag, value = f["flag"], values.get(f["flag"])
        if not flag.startswith("-") or value is None or value == "":
            continue
        if f["type"] == "bool":
            if value or f.get("neg"):
                check(flag if value else f["neg"], None, "form")
        elif f["type"] not in ("gguf", "template", "json"):
            check(flag, value, "form")
        else:
            check(flag, None, "form")
    for x in conf.get("extra", []):
        flag = (x.get("flag") or "").strip()
        if x.get("enabled", True) and flag:
            check(flag, x.get("value"), "extra")
    return issues


def blocking_issues(issues):
    bad = [i for i in issues or [] if i["level"] in BLOCKING]
    return "; ".join(i["msg"] for i in bad) if bad else None


# ── update check ──────────────────────────────────────────────────────────
# Once a day (operational toggle ui.update_check on /server) ask GitHub for llama.cpp's latest release
# and PyPI for vLLM's; nothing is installed — the page shows what to set in .env and rebuild.
UPDATES_PATH = os.path.join(STATE_DIR, "updates.json")
UPDATE_EVERY = 24 * 3600


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "inference-overdrive-server",
                                               "Accept": "application/vnd.github+json, application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _ver_tuple(v):
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.?post(\d+))?", v or "")
    return tuple(int(x or 0) for x in m.groups()) if m else None


def read_updates():
    try:
        with open(UPDATES_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def check_updates():
    v, out = versions(), {"checked_at": time.time()}
    try:
        rel = _get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        tag = rel["tag_name"]
        info = {"latest": tag, "published": rel.get("published_at"), "url": rel.get("html_url"),
                "how": f"set LLAMA_CPP_REF={tag} in .env, then: docker compose build llm-server && docker compose up -d"}
        commit = (v.get("llama.cpp") or "").rpartition(" · ")[2]
        if re.fullmatch(r"[0-9a-f]{7,40}", commit):
            cmp = _get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/compare/{commit}...{tag}")
            info.update(installed=commit, behind=cmp.get("ahead_by", 0),
                        available=cmp.get("status") in ("ahead", "diverged") and cmp.get("ahead_by", 0) > 0)
        else:
            info.update(installed=None, available=None)
        out["llama.cpp"] = info
    except Exception as e:
        out["llama.cpp"] = {"error": str(e)[:200]}
    try:
        latest = _get_json("https://pypi.org/pypi/vllm/json")["info"]["version"]
        have, new = _ver_tuple(v.get("vllm")), _ver_tuple(latest)
        out["vllm"] = {"latest": latest, "installed": v.get("vllm"), "url": f"https://github.com/vllm-project/vllm/releases/tag/v{latest}",
                       "available": bool(have and new and new > have),
                       "how": "Upgrade in the vLLM card's gear menu (installs into state/envs, no rebuild), or set "
                              f"VLLM_VERSION={latest} in .env, then: docker compose build llm-server && docker compose up -d"}
    except Exception as e:
        out["vllm"] = {"error": str(e)[:200]}
    try:
        tmp = UPDATES_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, UPDATES_PATH)
    except OSError:
        pass
    return out


def update_loop(app):
    time.sleep(60)  # stay out of the way of the boot model's load
    while True:
        if app.cfg.get("ui", {}).get("update_check", True) and time.time() - read_updates().get("checked_at", 0) >= UPDATE_EVERY:
            check_updates()
        time.sleep(3600)


_versions = {}


def versions():
    if "llama.cpp" not in _versions:
        try:
            out = subprocess.run([LLAMA_BIN, "--version"], capture_output=True, text=True, timeout=30)
            text = out.stdout + out.stderr
            # "version: 0.7.0-dev (build 1, commit f74e2fa)" or older "version: 1234 (abcdef0)"
            m = re.search(r"version:\s*(\S+)\s*\((?:build \d+, commit )?([0-9a-f]+)\)", text)
            _versions["llama.cpp"] = f"{m.group(1)} · {m.group(2)}" if m else None
        except Exception:
            _versions["llama.cpp"] = None
    if "vllm" not in _versions:  # dropped when another vLLM env is selected
        _versions["vllm"] = env_vllm_version(vllm_env())
    return _versions


# ── vLLM environments: upgrade without a rebuild ──────────────────────────
# The image's vLLM lives in /opt/venv. An upgrade from /server installs another vLLM into its own
# venv under /state/envs/vllm-<version> (a full, separate install: torch and all, ~15 GB, so the
# image's copy stays untouched for rollback), checks it, and selects it in /state/vllm_env.json.
# Every vLLM call (serve, --help, parser registry, version) goes through the selected env.
IMAGE_VENV = os.environ.get("IMAGE_VENV", "/opt/venv")
VLLM_ENVS_DIR = os.path.join(STATE_DIR, "envs")
VLLM_ENV_SELECT = os.path.join(STATE_DIR, "vllm_env.json")
# same indexes as docker/Dockerfile: the CUDA 13.0 torch build and FlashInfer's prebuilt kernels
TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
FLASHINFER_INDEX_URL = os.environ.get("FLASHINFER_INDEX_URL", "https://flashinfer.ai/whl/")
VLLM_ENV_MIN_FREE = 20 * 1024**3
_vllm_env = {"loaded": False, "dir": None}


def vllm_env():
    """The selected vLLM env's directory, or None for the image's own vLLM. A selection whose env
    is gone or incomplete falls back to the image's copy."""
    if not _vllm_env["loaded"]:
        name = (_read_json(VLLM_ENV_SELECT) or {}).get("active")
        d = os.path.join(VLLM_ENVS_DIR, name) if name else None
        if d and not _env_ok(d):
            print(f"[server] vLLM env {name} is missing or failed its check; using the image's vLLM", flush=True)
            d = None
        _vllm_env.update(loaded=True, dir=d)
    return _vllm_env["dir"]


def _env_ok(d):
    return os.path.isfile(os.path.join(d, "bin", "vllm")) and (_read_json(os.path.join(d, ".io-env.json")) or {}).get("ok")


def vllm_bin():
    d = vllm_env()
    return os.path.join(d, "bin", "vllm") if d else VLLM_BIN


def vllm_python():
    d = vllm_env()
    return os.path.join(d, "bin", "python") if d else sys.executable


def env_vllm_version(d):
    """vLLM version in env d (None = the image's, which is this manager's own venv)."""
    try:
        if not d:
            from importlib.metadata import version
            return version("vllm")
        out = subprocess.run([os.path.join(d, "bin", "python"), "-c",
                              "import importlib.metadata as m; print(m.version('vllm'))"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or None
    except Exception:
        return None


def select_vllm_env(name):
    """Use env `name` (None = the image's vLLM) from the next vLLM start on."""
    d = os.path.join(VLLM_ENVS_DIR, name) if name else None
    if d and not _env_ok(d):
        raise ValueError(f"{name} is not an installed, checked vLLM env")
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(VLLM_ENV_SELECT + ".tmp", "w") as f:
        json.dump({"active": name, "at": time.time()}, f)
    os.replace(VLLM_ENV_SELECT + ".tmp", VLLM_ENV_SELECT)
    _vllm_env.update(loaded=True, dir=d)
    # everything derived from the installed vLLM
    _versions.pop("vllm", None)
    _catalog_cache.pop("vllm", None)
    _parsers.clear()
    threading.Thread(target=vllm_parsers, daemon=True).start()


class VllmEnvs:
    """Install / check / select / remove vLLM envs under /state/envs (one install at a time)."""
    STEPS = ("venv", "pip", "vllm", "flashinfer", "check")

    def __init__(self):
        self.lock = threading.Lock()
        self.job = None  # {version, name, step, status: installing|done|failed|cancelled, error, tail, ...}
        self.proc = None

    def running(self):
        return bool(self.job and self.job["status"] == "installing")

    def list(self):
        active = vllm_env()
        envs = [{"name": None, "version": env_vllm_version(None), "label": "image", "active": active is None, "ok": True}]
        try:
            names = sorted(os.listdir(VLLM_ENVS_DIR))
        except OSError:
            names = []
        for n in names:
            d = os.path.join(VLLM_ENVS_DIR, n)
            if not os.path.isdir(d) or (self.running() and self.job["name"] == n):
                continue
            meta = _read_json(os.path.join(d, ".io-env.json")) or {}
            envs.append({"name": n, "version": meta.get("vllm"), "label": n, "active": active == d, "ok": bool(meta.get("ok")),
                         "torch": meta.get("torch"), "cuda": meta.get("cuda"), "flashinfer": meta.get("flashinfer"),
                         "installed_at": meta.get("installed_at"), "error": meta.get("error")})
        job = {k: (list(v) if isinstance(v, deque) else v) for k, v in self.job.items()} if self.job else None
        return {"envs": envs, "job": job, "dir": VLLM_ENVS_DIR}

    def install(self, version):
        version = (version or "").strip()
        if not re.fullmatch(r"\d+\.\d+(\.\d+)?((\.post|rc|\.dev)\d+)?", version):
            raise ValueError("version must look like 0.32.0")
        with self.lock:
            if self.running():
                raise RuntimeError(f"already installing vLLM {self.job['version']}")
            name = f"vllm-{version}"
            d = os.path.join(VLLM_ENVS_DIR, name)
            if _env_ok(d):
                select_vllm_env(name)
                self.job = {"version": version, "name": name, "status": "done", "step": "check", "error": None,
                            "tail": deque([f"{name} is already installed and checked; selected it"], maxlen=40),
                            "started": time.time(), "finished": time.time(), "switched": True}
                return self.list()
            os.makedirs(VLLM_ENVS_DIR, exist_ok=True)
            free = shutil.disk_usage(VLLM_ENVS_DIR).free
            if free < VLLM_ENV_MIN_FREE:
                raise RuntimeError(f"only {free / 1024**3:.0f} GB free under {VLLM_ENVS_DIR}; a vLLM env needs about "
                                   f"{VLLM_ENV_MIN_FREE / 1024**3:.0f} GB")
            self.job = {"version": version, "name": name, "status": "installing", "step": None, "error": None,
                        "tail": deque(maxlen=40), "started": time.time(), "finished": None, "switched": False}
        threading.Thread(target=self._run, args=(self.job, d), daemon=True).start()
        return self.list()

    def _step(self, job, name, argv, log, env=None, timeout=None):
        job["step"] = name
        job["tail"].append(f"$ {shlex.join(argv)}")
        log.write(f"\n$ {shlex.join(argv)}\n")
        log.flush()
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     text=True, errors="replace", env=env, start_new_session=True)
        deadline = time.time() + timeout if timeout else None
        for line in self.proc.stdout:
            log.write(line)
            line = line.rstrip()
            if line:
                job["tail"].append(line)
            if deadline and time.time() > deadline:
                os.killpg(self.proc.pid, signal.SIGKILL)
        code = self.proc.wait()
        if job["status"] == "cancelled":
            raise RuntimeError("cancelled")
        if code != 0:
            raise RuntimeError(f"{name} step failed (exit code {code})")

    def _run(self, job, d):
        os.makedirs(VLLM_ENVS_DIR, exist_ok=True)
        log_path = d + ".log"
        meta_path = os.path.join(d, ".io-env.json")
        py, pip = os.path.join(d, "bin", "python"), [os.path.join(d, "bin", "python"), "-m", "pip"]
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1", PYTHONUNBUFFERED="1",
                   VIRTUAL_ENV=d, PATH=os.path.join(d, "bin") + ":" + os.environ.get("PATH", ""))
        env.pop("PYTHONHOME", None)
        try:
            with open(log_path, "w") as log:
                shutil.rmtree(d, ignore_errors=True)  # a half-finished earlier attempt
                # the system python (not this venv's) as the base, like /opt/venv itself
                base = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
                self._step(job, "venv", [base, "-m", "venv", d], log, env)
                self._step(job, "pip", [*pip, "install", "--upgrade", "pip"], log, env, 600)
                self._step(job, "vllm", [*pip, "install", "--extra-index-url", TORCH_INDEX_URL,
                                         "--extra-index-url", FLASHINFER_INDEX_URL, f"vllm=={job['version']}"], log, env, 3600)
                # FlashInfer's prebuilt kernels must match the flashinfer-python vLLM pulled in; without
                # them it JIT-compiles on first use (slower first load, still works)
                fi = subprocess.run([py, "-c", "import importlib.metadata as m; print(m.version('flashinfer-python'))"],
                                    capture_output=True, text=True, env=env).stdout.strip()
                if fi:
                    try:
                        self._step(job, "flashinfer", [*pip, "install", "--extra-index-url", FLASHINFER_INDEX_URL,
                                                       f"flashinfer-cubin=={fi}"], log, env, 1800)
                    except RuntimeError as e:
                        if job["status"] == "cancelled":
                            raise
                        job["tail"].append(f"warning: no flashinfer-cubin {fi} ({e}); kernels will JIT-compile")
                check = ("import json, torch, vllm, importlib.metadata as m\n"
                         "ok = torch.cuda.is_available()\n"
                         "print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,"
                         " 'gpu': ok and torch.cuda.get_device_name(0), 'gpus': torch.cuda.device_count() if ok else 0}))\n"
                         "assert ok, 'torch cannot see a GPU: its CUDA build may be too new for this driver'")
                self._step(job, "check", [py, "-c", check], log, {**backend_env("vllm"), "PATH": env["PATH"], "VIRTUAL_ENV": d}, 600)
                info = json.loads(next(l for l in reversed(job["tail"]) if l.startswith("{")))
                # the engine's own CLI and the flag catalog the settings check relies on
                res = subprocess.run([os.path.join(d, "bin", "vllm"), "serve", "--help=all"], capture_output=True, text=True,
                                     timeout=300, env={**backend_env("vllm"), "PATH": env["PATH"], "VIRTUAL_ENV": d})
                nflags = len(parse_help(res.stdout or res.stderr))
                job["tail"].append(f"vllm serve --help: {nflags} flags")
                if nflags < 20:
                    raise RuntimeError(f"`vllm serve --help` failed (exit {res.returncode}): {(res.stderr or res.stdout)[-300:]}")
                if job["status"] == "cancelled":
                    raise RuntimeError("cancelled")
                with open(meta_path, "w") as f:
                    json.dump({**info, "flashinfer": fi or None, "flags": nflags, "ok": True, "installed_at": time.time()}, f)
            select_vllm_env(job["name"])
            job.update(status="done", switched=True, finished=time.time())
            job["tail"].append(f"vLLM {info['vllm']} (torch {info['torch']}, CUDA {info['cuda']}) installed, checked and "
                               "selected; it is used from the next vLLM start")
        except Exception as e:
            cancelled = job["status"] == "cancelled"
            job.update(status="cancelled" if cancelled else "failed", error=None if cancelled else str(e)[:400],
                       finished=time.time())
            if cancelled:
                shutil.rmtree(d, ignore_errors=True)
            else:
                try:
                    os.makedirs(d, exist_ok=True)
                    with open(meta_path, "w") as f:
                        json.dump({"ok": False, "error": str(e)[:400], "vllm": job["version"]}, f)
                except OSError:
                    pass
        finally:
            self.proc = None

    def cancel(self):
        if not self.running():
            raise RuntimeError("no install is running")
        self.job["status"] = "cancelled"
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)

    def remove(self, name):
        d = os.path.join(VLLM_ENVS_DIR, name or "")
        if not name or os.path.dirname(os.path.normpath(d)) != os.path.normpath(VLLM_ENVS_DIR) or not os.path.isdir(d):
            raise KeyError(name)
        if vllm_env() == d:
            raise RuntimeError(f"{name} is in use; switch to another vLLM first")
        if self.running() and self.job["name"] == name:
            raise RuntimeError(f"{name} is being installed")
        shutil.rmtree(d)
        try:
            os.remove(d + ".log")
        except OSError:
            pass
        return self.list()


# ── model library ─────────────────────────────────────────────────────────
def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def list_models(cfg):
    in_use = set()
    for backend in BACKENDS:
        values = cfg.get(backend, {}).get("values", {})
        for f in SCHEMAS[backend]:
            if f["type"] in ("gguf", "hfmodel") and values.get(f["flag"]):
                in_use.add(values[f["flag"]].rstrip("/"))
    gguf = []
    for root, dirs, files in os.walk(MODELS_DIR):
        rel_root = os.path.relpath(root, MODELS_DIR)
        # skip HF bookkeeping at the cache root: blobs/ is hf_xet's shared, hash-named store
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (rel_root == "." and d in ("hf", "blobs"))]
        if rel_root.startswith("models--") and os.sep not in rel_root:
            dirs[:] = [d for d in dirs if d == "snapshots"]  # hub cache repo: GGUFs show up via snapshots/, not blobs/
        for name in files:
            if not name.lower().endswith(".gguf"):
                continue
            rel = os.path.normpath(os.path.join(rel_root, name))
            full = os.path.join(root, name)
            st = os.stat(full)
            meta = gguf_info(full)
            gguf.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime, "role": meta.get("role", "model"),
                         "meta": {k: meta.get(k) for k in ("arch", "name", "basename", "base_repo", "quant", "size_label",
                                                           "ctx_train", "embd", "projection_dim", "has_vision", "error")},
                         "in_use": rel in in_use})
    def hf_entry(path, repo, value, folder, size, kind):
        hf_cfg = (_read_json(os.path.join(folder, "config.json")) if folder else None) or {}
        return {"path": path, "repo": repo, "value": value, "kind": kind, "size": size,
                "complete": bool(hf_cfg), "in_use": value in in_use,
                "meta": {"model_type": hf_cfg.get("model_type"), "archs": hf_cfg.get("architectures"),
                         "vision": "vision_config" in hf_cfg,
                         "quant": (hf_cfg.get("quantization_config") or {}).get("quant_method")}}

    hf = []
    # hub cache repos (/models/models--owner--name): vLLM loads these by repo id
    if os.path.isdir(HUB_CACHE):
        for name in sorted(os.listdir(HUB_CACHE)):
            if not name.startswith("models--") or not os.path.isdir(os.path.join(HUB_CACHE, name)):
                continue
            repo = name[len("models--"):].replace("--", "/", 1)
            hf.append(hf_entry(os.path.relpath(os.path.join(HUB_CACHE, name), MODELS_DIR), repo, repo,
                               hub_snapshot(repo), _dir_size(os.path.join(HUB_CACHE, name, "blobs")), "hub"))
    # legacy plain folders from older /server downloads: loaded by path
    if os.path.isdir(HF_DIR):
        for name in sorted(os.listdir(HF_DIR)):
            full = os.path.join(HF_DIR, name)
            if os.path.isdir(full) and not name.startswith("."):
                hf.append(hf_entry(f"hf/{name}", name.replace("--", "/", 1), f"hf/{name}", full, _dir_size(full), "folder"))
    usage = shutil.disk_usage(MODELS_DIR)
    return {"gguf": sorted(gguf, key=lambda m: m["path"]), "hf": hf,
            "disk": {"total": usage.total, "used": usage.used, "free": usage.free}}


def delete_model(rel, cfg, backend):
    full = models_path(rel)
    name = os.path.basename(full.rstrip("/"))
    repo = name[len("models--"):].replace("--", "/", 1) if name.startswith("models--") else None
    snap = backend.snapshot()
    if snap["state"] in ("starting", "running") and snap["argv"] and any(
            full in a or (repo and a == repo) for a in snap["argv"]):
        raise ValueError(f"{rel} is loaded by the running backend; stop or switch it first")
    if os.path.islink(full):
        # a file inside a hub-cache snapshot: snapshot link -> repo blob (itself a link into the
        # cache-wide shared store with hf_xet) -> data. Drop the whole chain or no space is freed.
        chain, p = [], full
        while os.path.islink(p) and len(chain) < 4:
            chain.append(p)
            p = os.path.normpath(os.path.join(os.path.dirname(p), os.readlink(p)))
        for link in chain[1:]:
            os.remove(link)
        os.remove(full)
        if os.path.isfile(p) and not _shared_blob_in_use(p):
            os.remove(p)
    elif os.path.isdir(full):
        # a hub-cache repo: its blobs/ may only be links into the shared store; free what no other repo uses
        targets = {os.path.realpath(os.path.join(full, "blobs", b)) for b in os.listdir(os.path.join(full, "blobs"))} \
            if repo and os.path.isdir(os.path.join(full, "blobs")) else set()
        shutil.rmtree(full)
        for t in targets:
            if t.startswith(os.path.join(os.path.realpath(HUB_CACHE), "blobs") + os.sep) and os.path.isfile(t) \
                    and not _shared_blob_in_use(t):
                os.remove(t)
        if os.path.isdir(HF_DIR) and not os.listdir(HF_DIR):
            os.rmdir(HF_DIR)  # last legacy folder gone
    else:
        os.remove(full)


def _shared_blob_in_use(path):
    """Does any cached repo still link to this file in the shared blob store?"""
    path = os.path.realpath(path)
    for repo_blobs in glob.glob(os.path.join(HUB_CACHE, "models--*", "blobs")):
        for b in os.listdir(repo_blobs):
            if os.path.realpath(os.path.join(repo_blobs, b)) == path:
                return True
    return False


# ── model metadata and suggested settings ─────────────────────────────────
# Authority order for every suggestion: the model's own metadata (GGUF header,
# config.json / generation_config.json), then measured VRAM, then the
# FAMILY_RULES fallback table. Anything still unknown is left at the backend default.
GiB = 1 << 30
# Windows desktop + other apps on the same GPU; CUDA context + compute buffers.
DESKTOP_RESERVE = float(os.environ.get("VRAM_RESERVE_GB", "1.5")) * GiB
COMPUTE_RESERVE = 1.5 * GiB
KV_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
            "auto": 2, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}
GGUF_FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
                   11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
                   17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
                   23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
                   29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# (regex over model_type / architecture / name, fallback sampling, vLLM reasoning parser, vLLM tool parser).
# First match wins. Sampling here is only used when the model ships none; parser names are checked
# against the installed vLLM's registries before use.
FAMILY_RULES = [
    (r"qwen3.*coder", {"temp": 0.7, "top_p": 0.8, "top_k": 20}, "qwen3", "qwen3_coder"),
    (r"qwen3[._]?(5|6|7|8|next)|qwen35|qwen3_5", {"temp": 1.0, "top_p": 0.95, "top_k": 20}, "qwen3", "qwen3_coder"),
    (r"qwen3", {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}, "qwen3", "hermes"),
    (r"qwen2", {"temp": 0.7, "top_p": 0.8, "top_k": 20}, None, "hermes"),
    (r"gpt[-_]?oss", {"temp": 1.0, "top_p": 1.0}, "openai_gptoss", "openai"),
    (r"deepseek.?r1", {"temp": 0.6, "top_p": 0.95}, "deepseek_r1", "deepseek_v3"),
    (r"deepseek", {"temp": 0.6, "top_p": 0.95}, "deepseek_v3", "deepseek_v31"),
    (r"gemma.?4", {"temp": 1.0, "top_p": 0.95, "top_k": 64}, "gemma4", "gemma4"),
    (r"gemma", {"temp": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0}, None, "pythonic"),
    (r"nemotron_h|nemotron.?3", None, "nemotron_v3", None),
    (r"llama.?4", {"temp": 0.6, "top_p": 0.9}, None, "llama4_pythonic"),
    (r"llama", {"temp": 0.6, "top_p": 0.9}, None, "llama3_json"),
    (r"magistral", {"temp": 0.7, "top_p": 0.95}, "mistral", "mistral"),
    (r"mistral|ministral|devstral", {"temp": 0.15}, None, "mistral"),
    (r"glm.?4\.?7|glm47", None, "glm47", "glm47"),
    (r"glm", None, "glm45", "glm45"),
    (r"kimi", None, "kimi_k2", "kimi_k2"),
    (r"minimax", None, "minimax_m2", "minimax_m2"),
    (r"granite", None, "granite", "granite4"),
    (r"phi.?4.*mini", None, None, "phi4_mini_json"),
    (r"olmo", None, "olmo3", "olmo3"),
    (r"seed.?oss", None, "seed_oss", "seed_oss"),
    (r"ernie", None, "ernie45", "ernie45"),
]
SAMPLING_FLAGS = {"temp": "--temp", "temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k",
                  "min_p": "--min-p", "presence_penalty": "--presence-penalty",
                  "repeat_penalty": "--repeat-penalty", "repetition_penalty": "--repeat-penalty"}


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_gguf_meta(path, keep_template=False):
    """Key/values of a GGUF header. Large arrays (the vocab) are skipped over, not loaded; the chat
    template is reduced to True unless keep_template."""
    meta = {}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rs():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                item, n = rd("<I"), rd("<Q")
                if item == 8:
                    if n > 64:
                        for _ in range(n):
                            f.seek(rd("<Q"), 1)
                        return None
                    return [rs() for _ in range(n)]
                size = struct.calcsize(_GGUF_SCALARS[item])
                if n > 64:
                    f.seek(size * n, 1)
                    return None
                data = f.read(size * n)
                return [struct.unpack_from(_GGUF_SCALARS[item], data, i * size)[0] for i in range(n)]
            return rd(_GGUF_SCALARS[t])

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        rd("<I"), rd("<Q")
        for _ in range(rd("<Q")):
            key = rs()
            value = val(rd("<I"))
            if value is not None:
                meta[key] = bool(value) if key == "tokenizer.chat_template" and not keep_template else value
    return meta


def gguf_tensor_names(path):
    """Tensor names from a GGUF (the table after the header's key/values; no weights are read)."""
    names = []
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def skip_val(t):
            if t == 8:
                f.seek(rd("<Q"), 1)
            elif t == 9:
                item, n = rd("<I"), rd("<Q")
                if item == 8:
                    for _ in range(n):
                        f.seek(rd("<Q"), 1)
                else:
                    f.seek(struct.calcsize(_GGUF_SCALARS[item]) * n, 1)
            else:
                f.seek(struct.calcsize(_GGUF_SCALARS[t]), 1)

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        rd("<I")
        n_tensors, n_kv = rd("<Q"), rd("<Q")
        for _ in range(n_kv):
            f.seek(rd("<Q"), 1)
            skip_val(rd("<I"))
        for _ in range(n_tensors):
            names.append(f.read(rd("<Q")).decode("utf-8", "replace"))
            f.seek(rd("<I") * 8 + 4 + 8, 1)  # dims, type, offset
    return names


_meta_cache = {}


def gguf_info(full):
    """Normalized facts about a GGUF: role, pairing keys, shape for KV math, shipped sampling."""
    st = os.stat(full)
    key = (full, st.st_mtime, st.st_size)
    if key in _meta_cache:
        return _meta_cache[key]
    try:
        m = read_gguf_meta(full)
    except Exception as e:
        low = os.path.basename(full).lower()  # unreadable or split shard: fall back to the filename
        info = {"error": str(e), "role": "mmproj" if "mmproj" in low else "draft" if "draft" in low or "dflash" in low else "model"}
        _meta_cache[key] = info
        return info
    arch = m.get("general.architecture", "")

    def a(k):
        return m.get(f"{arch}.{k}")

    tags = m.get("general.tags") or []
    gtype = m.get("general.type", "model")
    role = ("mmproj" if gtype == "mmproj" or arch == "clip"
            else "draft" if arch in ("dflash", "eagle", "eagle3") or "draft-model" in tags else "model")
    base = m.get("general.base_model.0.repo_url") or m.get("general.source.url") or ""
    base_repo = re.sub(r"^https?://huggingface\.co/", "", base).strip("/") or None
    heads, heads_kv, embd = a("attention.head_count"), a("attention.head_count_kv"), a("embedding_length")
    if isinstance(heads_kv, list):  # per-layer counts
        heads_kv = max(heads_kv) if heads_kv else None
    if isinstance(heads, list):
        heads = max(heads) if heads else None
    key_len = a("attention.key_length") or (embd // heads if embd and heads else None)
    layers, interval = a("block_count"), a("full_attention_interval")
    kv_list = a("attention.head_count_kv")
    if layers and interval:
        attn_layers = layers // interval
    elif isinstance(kv_list, list) and kv_list:  # per-layer counts: 0 = no KV (SSM / MLP-only block)
        attn_layers = sum(1 for x in kv_list if x)
    else:
        attn_layers = layers
    # recurrent (SSM / gated delta-net) state per sequence: llama.cpp keeps conv + ssm state in f32
    conv, inner, dstate, groups = a("ssm.conv_kernel"), a("ssm.inner_size"), a("ssm.state_size"), a("ssm.group_count") or 1
    rec_layers = (layers - attn_layers) if (layers and conv and inner and dstate) else 0
    state_per_slot = rec_layers * ((conv - 1) * (inner + 2 * groups * dstate) + dstate * inner) * 4 if rec_layers else 0
    info = {
        "arch": arch, "type": gtype, "role": role, "tags": tags,
        "name": m.get("general.name"), "basename": m.get("general.basename"), "base_repo": base_repo,
        "size_label": m.get("general.size_label"), "quant": GGUF_FILE_TYPES.get(m.get("general.file_type")),
        "ctx_train": a("context_length"), "layers": layers, "embd": embd,
        # hybrid models (linear-attention / SSM layers) only keep KV on their attention layers
        "attn_layers": attn_layers, "rec_layers": rec_layers, "state_per_slot": state_per_slot,
        "repo_url": m.get("general.repo_url"),
        "kv_heads": heads_kv, "key_len": key_len, "val_len": a("attention.value_length") or key_len,
        "sliding_window": a("attention.sliding_window"), "mtp_layers": a("nextn_predict_layers"),
        "chat_template": bool(m.get("tokenizer.chat_template")),
        "projection_dim": m.get("clip.vision.projection_dim"), "has_vision": bool(m.get("clip.has_vision_encoder")),
        "sampling": {k.rsplit(".", 1)[1]: v for k, v in m.items() if k.startswith("general.sampling.")},
    }
    _meta_cache[key] = info
    return info


def compatible(target, other):
    """True/False when a draft or projector provably does / doesn't belong to the target; None if unknown."""
    if not target or not other or "error" in target or "error" in other:
        return None
    if target.get("base_repo") and other.get("base_repo"):
        return target["base_repo"].lower() == other["base_repo"].lower()
    if target.get("basename") and other.get("basename") and target["basename"].lower() == other["basename"].lower():
        return True
    width = other.get("projection_dim") if other.get("role") == "mmproj" else other.get("embd")
    return (width == target.get("embd")) if width and target.get("embd") else None


# Tool-call syntax in a chat template → the vLLM parser that reads it back. Checked in order (the
# Qwen3-Coder XML format also uses <tool_call>); used when the family table names no tool parser.
TEMPLATE_TOOL_FORMATS = [
    (("<function=", "<parameter="), (), "qwen3_coder"),
    (("[TOOL_CALLS]",), (), "mistral"),
    (("<tool_call>",), ("<arg_key>", "<function="), "hermes"),
]


def template_tool_parser(template):
    """(parser, what was seen) for the tool-call format a chat template writes, or (None, None)."""
    for needles, absent, parser in TEMPLATE_TOOL_FORMATS:
        if template and all(n in template for n in needles) and not any(a in template for a in absent):
            return parser, " + ".join(needles)
    return None, None


def family_rule(*texts):
    hay = " ".join(t for t in texts if t).lower()
    return next((r for r in FAMILY_RULES if re.search(r[0], hay)), None)


def gpu_list():
    """Detected CUDA cards from the hw-metrics sampler: [{index, name, total, used}] (bytes), measured?"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8083/hw", timeout=2) as r:
            gpus = [{"index": g.get("index", i), "name": g.get("name") or f"GPU {i}",
                     "total": (g.get("vram_total") or 0) * 2**20, "used": (g.get("vram_used") or 0) * 2**20}
                    for i, g in enumerate(json.load(r).get("gpus") or [])]
            if gpus:
                return gpus, True
    except Exception:
        pass
    return [{"index": 0, "name": "GPU 0", "total": 24 * GiB, "used": 0}], False


def selected_gpus(values):
    """The cards the GPU picker allows (all when nothing is ticked)."""
    gpus, measured = gpu_list()
    sel = set(values.get("_gpus") or [])
    use = [g for g in gpus if g["index"] in sel] if sel else gpus
    return (use or gpus), measured


def gpu_names(use):
    names = sorted({g["name"].replace("NVIDIA ", "").replace("GeForce ", "") for g in use})
    return f"{len(use)}× " + " + ".join(names) if len(use) > 1 else names[0]


def fit_ctx(budget, per_token, ctx_train, floor=4096):
    """Largest power-of-two context (>= floor, <= trained) whose KV cache fits the budget.

    Picks within 90% of the fit: the budget is an estimate (desktop use moves, buffers grow a bit
    with context), so a context that only just fits isn't chosen."""
    limit = int(budget // per_token) if per_token > 0 else (ctx_train or 32768)
    if ctx_train:
        limit = min(limit, ctx_train)
    safe = int(limit * 0.9) if not ctx_train or limit < ctx_train else limit
    if safe < floor:
        return None, limit
    ctx = floor
    while ctx * 2 <= safe:
        ctx *= 2
    return ctx, limit


def hf_json(repo, filename):
    """A small JSON file from a Hugging Face repo (with the HF token for gated repos); None if absent."""
    headers = hf_headers()
    try:
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{filename}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except Exception:
        return None


def hf_weights_size(repo):
    headers = hf_headers()
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            sib = json.load(r).get("siblings", [])
        return sum(s.get("size") or 0 for s in sib if s["rfilename"].endswith(".safetensors")) or None
    except Exception:
        return None


_parsers = {}


def vllm_parsers():
    """Reasoning / tool parser names registered in the installed vLLM (cached; ~15 s the first time)."""
    if not _parsers:
        code = ("import json\nfrom vllm.reasoning import ReasoningParserManager as R\n"
                "from vllm.tool_parsers import ToolParserManager as T\n"
                "def names(m):\n"
                "    for a in ('list_registered', 'get_all_names'):\n"
                "        if hasattr(m, a): return sorted(getattr(m, a)())\n"
                "    return sorted(next(v for k, v in vars(m).items() if isinstance(v, dict) and v))\n"
                "print(json.dumps({'reasoning': names(R), 'tool': names(T)}))")
        try:
            out = subprocess.run([vllm_python(), "-c", code], capture_output=True, text=True, timeout=180,
                                 env=backend_env("vllm"))
            _parsers.update(json.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:
            pass
    return _parsers


def _sampling_values(sampling, flags_ok=True):
    out = {}
    for k, v in (sampling or {}).items():
        flag = SAMPLING_FLAGS.get(k)
        if flag and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[flag] = round(float(v), 4) if flag != "--top-k" else int(v)
    return out


# ── auto-tune helpers: model card, base model, recurrent state, GPU format, MTP ──
def readme_text(local, repo):
    """A repo's model card (README.md): the downloaded copy, else the Hub."""
    if local and os.path.exists(os.path.join(local, "README.md")):
        with open(os.path.join(local, "README.md"), errors="replace") as f:
            return f.read()
    return hf_text(repo, "README.md") if repo and re.fullmatch(r"[\w.-]+/[\w.-]+", repo) else None


def card_meta(text):
    """A model card's YAML front matter, flat keys only (`key: value` and `key:` + `- item` lists)."""
    m = re.match(r"---\s*\n(.*?)\n---", text or "", re.S)
    out, key = {}, None
    for line in m.group(1).splitlines() if m else []:
        kv = re.match(r"([\w-]+):\s*(.*)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip().strip("'\"")
            out[key] = val or []
        elif key and isinstance(out.get(key), list) and re.match(r"\s*-\s+", line):
            out[key].append(re.sub(r"^\s*-\s+", "", line).strip().strip("'\""))
    return out


QUANT_NAME_RE = re.compile(r"(?i)awq|gptq|fp8|fp4|int4|int8|w4a16|w8a8|nvfp4|mxfp4|gguf|quant|bnb|[48]bit|exl2|mlx")


def base_model_of(text, repo, quantized=False):
    """The model a quantized repo was made from (model card front matter), when its own files may be
    stripped down: sampling defaults, serve commands and the family name then come from there."""
    meta = card_meta(text)
    base = meta.get("base_model")
    base = base[0] if isinstance(base, list) and base else base
    if not isinstance(base, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", base) or base.lower() == (repo or "").lower():
        return None
    is_quant = quantized or meta.get("base_model_relation") == "quantized" or bool(QUANT_NAME_RE.search(repo or ""))
    return base if is_quant else None


CARD_CMDS = {"vllm": r"\bvllm\s+serve\b", "llama.cpp": r"\bllama-(?:server|cli)\b"}
# never taken from a card: this server owns them, or they name files/prompts on the author's machine
CARD_SKIP = {
    "vllm": {"--host", "--port", "--served-model-name", "--model", "--download-dir", "--api-key", "--uds", "--uvicorn-log-level"},
    "llama.cpp": {"--host", "--port", "-m", "--model", "-hf", "--hf-repo", "-hff", "--hf-file", "-mu", "--model-url",
                  "--api-key", "-p", "--prompt", "-cnv", "--conversation", "-no-cnv", "--no-conversation", "-sys",
                  "--system-prompt", "-f", "--file", "-n", "--n-predict", "--log-file", "-mm", "--mmproj", "-md", "--model-draft"},
}
# sizing, hardware and trust: auto-tune works these out for this machine, so a card's values come unticked
CARD_UNTICKED = {
    "--max-model-len", "--gpu-memory-utilization", "--kv-cache-memory-bytes", "--tensor-parallel-size", "-tp",
    "--pipeline-parallel-size", "-pp", "--data-parallel-size", "-dp", "--trust-remote-code", "--cpu-offload-gb",
    "--swap-space", "--enforce-eager", "--chat-template", "--max-num-batched-tokens",
    "--ctx-size", "-c", "--n-gpu-layers", "-ngl", "--gpu-layers", "--threads", "-t", "--tensor-split", "-ts",
    "--main-gpu", "-mg", "--split-mode", "-sm", "--device", "-dev", "--parallel", "-np", "--batch-size", "-b",
    "--ubatch-size", "-ub", "--cache-type-k", "-ctk", "--cache-type-v", "-ctv", "--chat-template-file",
}
FLAG_TOKEN_RE = re.compile(r"--?[A-Za-z][\w.-]*(=.*)?")


def flag_index(flags):
    idx = {}
    for c in flags or []:
        for name in (c["flag"], *c["aliases"]):
            idx.setdefault(name, c)
    return idx


def card_commands(text, backend, idx):
    """The flags of every `vllm serve` / `llama-server` command in a model card's code blocks (or a
    shell script): [[(flag, value)]]. A flag the --help catalog lists without a metavar takes no value."""
    if not text:
        return []
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    body = re.sub(r"\\[ \t]*\n", " ", "\n".join(blocks) if blocks else text)
    cmds = []
    for line in body.splitlines():
        m = re.search(CARD_CMDS[backend], line)
        if not m:
            continue
        try:
            toks = shlex.split(line[m.end():], comments=True)
        except ValueError:
            continue
        pairs, i = [], 0
        while i < len(toks):
            tok = toks[i]
            if tok in ("|", "||", "&&", ";", "&", ">", ">>", "2>&1"):
                break
            if FLAG_TOKEN_RE.fullmatch(tok):
                flag, eq, val = tok.partition("=")
                c = idx.get(flag)
                takes = (c["metavar"] != "") if c else True
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if not eq and takes and nxt is not None and (not FLAG_TOKEN_RE.fullmatch(nxt) or re.fullmatch(r"-\d[\d.]*", nxt)) \
                        and nxt not in ("|", "||", "&&", ";", "&"):
                    val, i = nxt, i + 1
                pairs.append((flag, val if (eq or val) else None))
            i += 1
        if pairs:
            cmds.append(pairs)
    return cmds


def card_suggestions(backend, sources, chosen):
    """Flags from model cards / repo scripts, as [{source, flags: [{flag, value, orig, tick, note}]}].
    `chosen` is what auto-tune already set: a card never silently overrides those (unticked, noted)."""
    idx = flag_index(catalog(backend))
    groups, seen = [], set()
    for source, text in sources:
        rows = []
        for cmd in card_commands(text, backend, idx):
            for flag, value in cmd:
                c = idx.get(flag)
                canon = c["flag"] if c else flag
                if {flag, canon} & (CARD_SKIP[backend] | RESERVED_FLAGS) or (value and "$" in value):
                    continue
                if c and not c["metavar"]:  # switch: "--no-x" is the off form of --x
                    value = not (flag.startswith("--no-") and canon != flag)
                key = (canon, json.dumps(value))
                if key in seen:
                    continue
                seen.add(key)
                tick, note = True, None
                if idx and not c:
                    tick, note = False, f"not in this {backend} version's --help"
                elif c and c.get("choices") and isinstance(value, str) and value not in c["choices"]:
                    tick, note = False, f"not one of {', '.join(c['choices'][:8])}"
                elif {flag, canon} & CARD_UNTICKED:
                    tick, note = False, "sized for the card author's hardware — auto-tune sizes this for yours"
                elif canon in chosen and chosen[canon] is not None:
                    if json.dumps(chosen[canon]) == json.dumps(value) or str(chosen[canon]) == str(value):
                        continue  # auto-tune already says the same
                    tick, note = False, f"auto-tune chose {chosen[canon]}"
                rows.append({"flag": canon, "orig": flag, "value": value, "tick": tick, "note": note})
        if rows:
            groups.append({"source": source, "flags": rows})
    return groups


def repo_scripts(local):
    """Shell scripts a downloaded repo ships (examples/, scripts/): often a tested serve command."""
    out = []
    for p in sorted(glob.glob(os.path.join(local, "*.sh")) + glob.glob(os.path.join(local, "*", "*.sh")))[:6] if local else []:
        try:
            with open(p, errors="replace") as f:
                out.append((os.path.relpath(p, local), f.read()))
        except OSError:
            pass
    return out


def recurrent_layout(tc):
    """(KV-cache layers, recurrent layers, recurrent state bytes per sequence). Hybrid models (Mamba,
    linear attention) keep a fixed-size state per running sequence instead of KV for those layers."""
    kinds, pattern = tc.get("layers_block_type") or tc.get("layer_types"), tc.get("hybrid_override_pattern")
    layers = tc.get("num_hidden_layers") or len(kinds or pattern or [])
    if isinstance(kinds, list) and kinds:
        attn = sum(1 for k in kinds if "attention" in str(k) and "linear" not in str(k))
        rec = sum(1 for k in kinds if "mamba" in str(k) or "linear" in str(k))
    elif isinstance(pattern, str) and pattern:  # Nemotron-H: M = Mamba, * = attention, E/- = MoE/MLP
        attn, rec = pattern.count("*"), pattern.count("M")
    elif tc.get("full_attention_interval"):
        attn = layers // tc["full_attention_interval"]
        rec = layers - attn
    elif tc.get("attn_layer_period"):  # Jamba
        attn = layers // tc["attn_layer_period"]
        rec = layers - attn
    else:
        return layers, 0, 0
    if not rec:
        return attn, 0, 0
    ssm_bytes = 4 if str(tc.get("mamba_ssm_cache_dtype", "")).endswith("32") else 2
    if tc.get("linear_num_value_heads"):  # gated DeltaNet (Qwen3-Next)
        kh, kd = tc.get("linear_num_key_heads", 0), tc.get("linear_key_head_dim", 0)
        vh, vd = tc["linear_num_value_heads"], tc.get("linear_value_head_dim", 0)
        conv = (tc.get("linear_conv_kernel_dim", 4) - 1) * (2 * kh * kd + vh * vd) * 2
        ssm = vh * kd * vd * ssm_bytes
    else:
        heads = tc.get("mamba_num_heads") or tc.get("mamba_n_heads")
        hdim = tc.get("mamba_head_dim") or tc.get("mamba_d_head")
        state = tc.get("ssm_state_size") or tc.get("mamba_d_state") or tc.get("state_size") or 128
        groups = tc.get("n_groups") or tc.get("mamba_n_groups") or 1
        kernel = tc.get("conv_kernel") or tc.get("mamba_d_conv") or 4
        if heads and hdim:  # Mamba2
            inner = heads * hdim
            conv = (kernel - 1) * (inner + 2 * groups * state) * 2
        else:  # Mamba1: d_inner = expand × hidden
            inner = (tc.get("mamba_expand") or tc.get("expand") or 2) * (tc.get("hidden_size") or 0)
            conv = (kernel - 1) * inner * 2
        ssm = inner * state * ssm_bytes
    return attn, rec, rec * (conv + ssm)


def recurrent_seqs(budget, per_seq, share=0.5, cap=256):
    """--max-num-seqs for a hybrid model: vLLM needs one recurrent-state block per running sequence and
    refuses to start when the default (256) is more than fit. Give states at most `share` of the KV
    budget, rounded down to a power of two."""
    n = int(max(budget, 0) * share // per_seq) if per_seq else cap
    p = 1
    while p * 2 <= min(max(n, 1), cap):
        p *= 2
    return p


_gpu_caps = {}


def gpu_caps():
    """{index: (major, minor)} CUDA compute capability per card (nvidia-smi; cached)."""
    if not _gpu_caps:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15).stdout
            for line in out.splitlines():
                i, cap = [x.strip() for x in line.split(",")]
                _gpu_caps[int(i)] = tuple(int(x) for x in cap.split("."))
        except Exception:
            pass
    return _gpu_caps


def weight_format(cfg):
    """'fp8' / 'nvfp4' / 'mxfp4' when a checkpoint stores float low-precision weights, else None."""
    q = (cfg or {}).get("quantization_config") or {}
    method, algo = str(q.get("quant_method", "")).lower(), str(q.get("quant_algo", "")).upper()
    if method in ("fp8", "fbgemm_fp8") or algo.startswith("FP8"):
        return "fp8"
    if method == "mxfp4" or "MXFP4" in algo:
        return "mxfp4"
    if "NVFP4" in algo or method == "modelopt_fp4":
        return "nvfp4"
    if method == "compressed-tensors":
        for g in (q.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            if w.get("type") == "float":
                return {8: "fp8", 4: "nvfp4"}.get(w.get("num_bits"))
    return None


NATIVE_FORMAT = {"fp8": ((8, 9), "Ada / Hopper (sm_89+)"), "nvfp4": ((10, 0), "Blackwell (sm_100+)"),
                 "mxfp4": ((10, 0), "Blackwell (sm_100+)")}


def gpu_format_note(cfg, indexes=None):
    """A warning when the checkpoint's weight format has no native math on the selected cards."""
    fmt, caps = weight_format(cfg), gpu_caps()
    if not fmt or not caps:
        return None
    low = min((caps[i] for i in (indexes or caps) if i in caps), default=None)
    need, label = NATIVE_FORMAT[fmt]
    if not low or low >= need:
        return None
    return (f"⚠ {fmt.upper()} weights, but this GPU (sm_{low[0]}{low[1]}) has no native {'FP8' if fmt == 'fp8' else 'FP4'} "
            f"math — that needs {label}. vLLM falls back to a weight-only kernel (Marlin) where it has one: it loads "
            "but runs slower, and some layouts don't load at all. An AWQ / GPTQ / INT4 build is usually the better pick here.")


def safetensor_names(local):
    """Tensor names in a downloaded checkpoint (index file, else the safetensors headers)."""
    for idx in glob.glob(os.path.join(local, "*.safetensors.index.json")):
        return list((_read_json(idx) or {}).get("weight_map", {}))
    names = []
    for p in glob.glob(os.path.join(local, "*.safetensors")):
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                names += [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        except (OSError, ValueError, struct.error):
            pass
    return names


# model types whose MTP head vLLM knows without a config key (Qwen3-Next ships an `mtp.*` block)
MTP_MODEL_TYPES = {"qwen3_next": 1}


def mtp_suggestion(tc, local, model_type):
    """(--speculative-config JSON or None, note) for a model with built-in MTP layers. vLLM 0.31 maps
    every model-specific MTP method (deepseek_mtp, qwen3_next_mtp, nemotron_h_mtp, ...) onto "mtp"."""
    n = tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers") or MTP_MODEL_TYPES.get(model_type, 0)
    if not n:
        return None, None
    layers = tc.get("num_hidden_layers") or 0
    if local:
        names = safetensor_names(local)

        def is_mtp(name):
            hit = re.search(r"\.layers\.(\d+)\.", name)
            return "mtp" in name or "nextn" in name or bool(hit and int(hit.group(1)) >= layers)
        if names and not any(is_mtp(x) for x in names):
            return None, (f"Config declares {n} MTP layer(s), but this checkpoint ships no MTP weights "
                          "(quantized builds often drop them) — no speculative decoding.")
    tokens = 1 if "deepseek" in model_type else 2
    cfg = json.dumps({"method": "mtp", "num_speculative_tokens": tokens})
    return cfg, (f"{n} built-in MTP layer(s){'' if local else ' (weights not checked: not downloaded yet)'} → "
                 f"--speculative-config mtp, {tokens} token(s) ahead: faster decoding for a little VRAM. "
                 "If the monitor shows poor acceptance, clear it.")


SOURCES_PATH = os.path.join(MODELS_DIR, ".sources.json")
_sources_lock = threading.Lock()
_repo_files = {}


def read_sources():
    try:
        with open(SOURCES_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_sources(repo, revision, files, guessed=False):
    """Remember which Hugging Face repo flat files in /models came from (models/.sources.json)."""
    with _sources_lock:
        s = read_sources()
        for name in files:
            s[name] = {"repo": repo, "revision": revision, **({"guessed": True} if guessed else {})}
        try:
            with open(SOURCES_PATH + ".tmp", "w") as f:
                json.dump(s, f, indent=1)
            os.replace(SOURCES_PATH + ".tmp", SOURCES_PATH)
        except OSError:
            pass


def hf_repo_files(repo):
    """File names in a Hub repo (cached); None if the repo doesn't exist or can't be reached."""
    if repo not in _repo_files:
        try:
            req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers=hf_headers())
            with urllib.request.urlopen(req, timeout=10) as r:
                _repo_files[repo] = [s["rfilename"] for s in json.load(r).get("siblings", [])]
        except Exception:
            _repo_files[repo] = None
    return _repo_files[repo]


def gguf_source_repo(rel, t):
    """(repo, how) for a GGUF under /models: the download record, else a guess from the header —
    general.repo_url's owner + the model name + "-GGUF" — kept only if that repo has this very file."""
    src = read_sources().get(rel)
    if src:
        return src["repo"], "guessed from the GGUF header" if src.get("guessed") else "recorded at download"
    owner = re.sub(r"^https?://huggingface\.co/", "", t.get("repo_url") or "").strip("/")
    if not owner:
        return None, None
    fname = os.path.basename(rel)
    stem = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", fname)
    q = QUANT_RE.search(stem)
    names = [stem[:q.start()] if q else stem, (t.get("name") or "").replace(" ", "-"),
             "-".join(x for x in (t.get("basename"), t.get("size_label")) if x)]
    candidates = [owner] if "/" in owner else [f"{owner}/{n.strip('-')}-GGUF" for n in dict.fromkeys(names) if n]
    for repo in candidates:
        files = hf_repo_files(repo)
        if files and any(os.path.basename(f) == fname for f in files):
            record_sources(repo, None, [rel], guessed=True)
            return repo, "guessed from the GGUF header"
    return None, None


def suggest_llama(values):
    target_rel = values.get("--model")
    if not target_rel:
        raise ValueError("pick a target model first")
    t = gguf_info(models_path(target_rel))
    if "error" in t:
        raise ValueError(f"can't read {target_rel}: {t['error']}")
    out, notes = {}, []
    if t["role"] != "model":
        notes.append(f"⚠ {target_rel} looks like a {t['role']}, not a target model.")

    out["--alias"] = t.get("name") or t.get("basename")
    out["--jinja"] = True if t["chat_template"] else None

    # sampling: GGUF header > base repo's generation_config.json > family default
    sampling, src = t["sampling"], "GGUF metadata"
    if not sampling and t.get("base_repo"):
        sampling, src = hf_json(t["base_repo"], "generation_config.json") or {}, f"{t['base_repo']} generation_config.json"
    rule = family_rule(t["arch"], t.get("name"), t.get("base_repo"))
    if not _sampling_values(sampling) and rule and rule[1]:
        sampling, src = rule[1], f"{rule[0]!r} family default"
    samp = _sampling_values(sampling)
    for flag in ("--temp", "--top-p", "--top-k", "--min-p", "--presence-penalty", "--repeat-penalty"):
        out[flag] = samp.get(flag)
    if samp:
        notes.append("Sampling from " + src + ": " + ", ".join(f"{k} {v}" for k, v in samp.items()))

    # companions: draft + projector that provably belong to this target
    others = []
    for root, dirs, files in os.walk(MODELS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "hf"]
        others += [os.path.join(root, n) for n in files if n.lower().endswith(".gguf")]
    rel = lambda p: os.path.relpath(p, MODELS_DIR)
    drafts = [rel(p) for p in others if gguf_info(p).get("role") == "draft" and compatible(t, gguf_info(p))]
    projs = [rel(p) for p in others if gguf_info(p).get("role") == "mmproj" and compatible(t, gguf_info(p))]
    keep = lambda flag, pool: values.get(flag) if values.get(flag) in pool else (pool[0] if pool else None)
    draft = keep("--spec-draft-model", drafts)
    out["--spec-draft-model"] = draft
    if draft:
        d = gguf_info(models_path(draft))
        spec = {"dflash": "draft-dflash", "eagle3": "draft-eagle3", "eagle": "draft-eagle3"}.get(d.get("arch"), "draft-simple")
        out["--spec-type"] = spec
        # DFlash drafts a block per pass; 4 beats the card's 7 on 24 GB (see README)
        out["--spec-draft-n-max"] = 4 if spec == "draft-dflash" else values.get("--spec-draft-n-max")
        out["--spec-draft-ngl"] = 99
        notes.append(f"Draft {draft} ({d.get('arch')}) pairs with this target → --spec-type {spec}.")
    else:
        out["--spec-type"] = None
        out["--spec-draft-n-max"] = out["--spec-draft-ngl"] = out["--spec-draft-type-k"] = out["--spec-draft-type-v"] = None
        if t.get("mtp_layers"):
            spec_flag = flag_index(catalog("llama.cpp")).get("--spec-type") or {}
            try:
                has_mtp = any("nextn" in n or "mtp" in n for n in gguf_tensor_names(models_path(target_rel)))
            except Exception:
                has_mtp = False
            if not has_mtp:
                notes.append(f"Header declares {t['mtp_layers']} MTP layer(s), but this GGUF ships no MTP tensors — no draft-mtp.")
            elif "draft-mtp" not in (spec_flag.get("choices") or []):
                notes.append(f"This model has {t['mtp_layers']} MTP layer(s), but this llama.cpp build has no --spec-type draft-mtp.")
            else:
                out["--spec-type"] = "draft-mtp"
                notes.append(f"No matching draft model, but {t['mtp_layers']} built-in MTP layer(s) ship in this GGUF → "
                             "--spec-type draft-mtp (speculation from the model's own head, no extra file). "
                             "If the monitor shows poor acceptance, set it back to none.")
    proj = keep("--mmproj", projs)
    out["--mmproj"] = proj
    out["--mmproj-offload"] = True if proj else None
    if proj:
        notes.append(f"Vision projector {proj} matches this target.")

    # memory: weights of everything on the GPU, then the biggest power-of-two context whose KV fits
    kv_k = values.get("--cache-type-k") or "q8_0"
    kv_v = values.get("--cache-type-v") or "q8_0"
    out["--cache-type-k"], out["--cache-type-v"] = kv_k, kv_v
    out["--flash-attn"] = "on"  # quantized V cache needs flash attention
    if draft:
        out["--spec-draft-type-k"], out["--spec-draft-type-v"] = kv_k, kv_v
    use, measured = selected_gpus(values)
    total = sum(g["total"] for g in use)
    # the desktop sits on one card; every extra card also holds its own CUDA context + buffers
    reserve = DESKTOP_RESERVE + COMPUTE_RESERVE + 0.5 * GiB * (len(use) - 1)
    weights = os.path.getsize(models_path(target_rel))
    weights += os.path.getsize(models_path(draft)) if draft else 0
    weights += os.path.getsize(models_path(proj)) if proj else 0
    per_tok = (t.get("attn_layers") or 0) * (t.get("kv_heads") or 0) * (
        (t.get("key_len") or 0) * KV_BYTES.get(kv_k, 2) + (t.get("val_len") or 0) * KV_BYTES.get(kv_v, 2))
    # hybrid: every server slot holds its own recurrent state (auto = -1 is counted as 4 slots)
    par = values.get("--parallel")
    slots = par if isinstance(par, int) and par > 0 else 4
    states = slots * (t.get("state_per_slot") or 0)
    budget = total - reserve - weights - states
    ctx, limit = fit_ctx(budget, per_tok, t.get("ctx_train"))
    hybrid = f" ({t['attn_layers']} of {t['layers']} layers keep KV)" if t.get("attn_layers") != t.get("layers") else ""
    notes.append(f"VRAM {total / GiB:.1f} GB on {gpu_names(use)}{'' if measured else ' (assumed)'} − "
                 f"{reserve / GiB:.1f} desktop+compute − {weights / GiB:.1f} weights = {max(budget, 0) / GiB:.1f} GB for KV; "
                 f"{per_tok / 1024:.0f} KiB/token at {kv_k}/{kv_v}{hybrid}.")
    if states:
        notes.append(f"Recurrent state: {t['state_per_slot'] / 2**20:.0f} MiB per slot ({t['rec_layers']} linear-attention / SSM layers) × "
                     f"{slots} slot{'s' if slots > 1 else ''}{' (--parallel auto, counted as 4)' if slots == 4 and not (isinstance(par, int) and par > 0) else ''} "
                     f"= {states / 2**20:.0f} MiB, taken off the KV budget above.")
    if len(use) > 1:
        # layer split spreads weights *and* KV over the cards, so the pooled budget above holds;
        # weight each card by what it can actually give (the first one also carries the desktop)
        usable = [g["total"] - (DESKTOP_RESERVE if i == 0 else 0) for i, g in enumerate(use)]
        out["--split-mode"] = "layer"
        out["--tensor-split"] = ",".join(f"{u / GiB:.0f}" for u in usable)
        out["--main-gpu"] = None
        notes.append(f"{len(use)} cards: --split-mode layer, --tensor-split {out['--tensor-split']} (usable GB per card). "
                     "row/tensor split only pays off with NVLink-class links.")
    else:
        out["--split-mode"] = out["--tensor-split"] = out["--main-gpu"] = None
    if ctx:
        out["--ctx-size"], out["--n-gpu-layers"] = ctx, 99
        notes.append(f"Fits ~{limit:,} tokens → --ctx-size {ctx:,}" + (f" (trained {t['ctx_train']:,})" if t.get("ctx_train") else "") + ", all layers on GPU.")
    else:
        # partial offload: keep an 8K context on the GPU and give the rest of the budget to layers
        room = total - reserve - states - per_tok * 8192 - (weights - os.path.getsize(models_path(target_rel)))
        frac = max(0.0, min(1.0, room / os.path.getsize(models_path(target_rel))))
        out["--ctx-size"], out["--n-gpu-layers"] = 8192, int((t.get("layers") or 0) * frac)
        notes.append(f"Too big for full offload: ~{frac:.0%} of {t.get('layers')} layers fit with an 8K context (rest on CPU, much slower).")
    # model card: the GGUF's own repo when it sits in the hub cache, else the base model's card
    full = models_path(target_rel)
    hub = re.search(r"/models--([^/]+?)--([^/]+)/snapshots/[^/]+/", full)
    sources = []
    if hub:
        snap = full[:hub.end()]
        own = readme_text(snap, f"{hub.group(1)}/{hub.group(2)}")
        if own:
            sources.append((f"{hub.group(1)}/{hub.group(2)} model card", own))
    if not hub:
        src_repo, how = gguf_source_repo(target_rel, t)
        own = readme_text(None, src_repo) if src_repo else None
        if own:
            sources.append((f"{src_repo} model card ({how})", own))
    if not any(card_commands(s[1], "llama.cpp", {}) for s in sources) and t.get("base_repo"):
        base = readme_text(None, t["base_repo"])
        if base:
            sources.append((f"{t['base_repo']} model card (base model)", base))
    card = card_suggestions("llama.cpp", sources, out) if sources else []
    if card:
        notes.append("The model card has a llama.cpp command: its flags are listed below — tick what to use.")
    return {"values": out, "notes": notes, "model": t.get("name") or target_rel, "card": card}


def suggest_vllm(values):
    model = (values.get("model") or "").strip().rstrip("/")
    if not model:
        raise ValueError("pick a model first")
    # cached repo in /models (hub cache) > legacy hf/ folder > Hugging Face API
    local = hub_snapshot(model) if re.fullmatch(r"[\w.-]+/[\w.-]+", model) and not model.startswith("hf/") else None
    if not local and os.path.isdir(os.path.join(MODELS_DIR, model)):
        local = os.path.join(MODELS_DIR, model)
    if local and os.path.exists(os.path.join(local, "config.json")):
        cfg, gen = _read_json(os.path.join(local, "config.json")), _read_json(os.path.join(local, "generation_config.json"))
        weights = sum(os.path.getsize(os.path.join(r, n)) for r, _, fs in os.walk(local) for n in fs if n.endswith(".safetensors"))
        repo = os.path.basename(model).replace("--", "/", 1) if model.startswith("hf/") else model
    else:
        cfg, gen, weights, repo = hf_json(model, "config.json"), hf_json(model, "generation_config.json"), hf_weights_size(model), model
    if not cfg:
        raise ValueError(f"no config.json for {model} (not downloaded, or a gated repo and no HF token is set)")
    tc = cfg.get("text_config") or cfg
    model_type = tc.get("model_type") or cfg.get("model_type") or ""
    out, notes = {"--served-model-name": repo}, []

    layers = tc.get("num_hidden_layers") or len(tc.get("layers_block_type") or tc.get("layer_types")
                                                 or tc.get("hybrid_override_pattern") or [])
    heads = tc.get("num_attention_heads") or 0
    kv_heads = tc.get("num_key_value_heads") or heads
    head_dim = tc.get("head_dim") or ((tc.get("hidden_size") or 0) // heads if heads else 0)
    attn_layers, rec_layers, state_per_seq = recurrent_layout(tc)
    max_pos = tc.get("max_position_embeddings") or cfg.get("max_position_embeddings")

    card_text = readme_text(local, repo)
    base = base_model_of(card_text, repo, quantized=bool(cfg.get("quantization_config")))
    if base:
        notes.append(f"Base model {base} (from the model card): its card, sampling and family fill in what this build lacks.")
    quant = (cfg.get("quantization_config") or {}).get("quant_method")
    out["--quantization"] = None  # vLLM reads quantization_config itself (and picks Marlin kernels on Ampere)
    out["--dtype"] = "half" if quant in ("awq", "awq_marlin") else None
    if quant:
        notes.append(f"Quantized ({quant}); vLLM auto-detects it" + (", --dtype half as vLLM recommends for AWQ." if out["--dtype"] else "."))
    elif weights and weights > 20 * GiB:
        notes.append(f"Unquantized weights are {weights / GiB:.1f} GB; look for an AWQ / GPTQ / FP8 build of {repo} to fit 24 GB.")

    use, measured = selected_gpus(values)
    fmt_note = gpu_format_note(cfg, [g.get("index") for g in use if g.get("index") is not None])
    if fmt_note:
        notes.append(fmt_note)
    # tensor parallel: the largest power of two <= cards that divides the attention heads.
    # Weights and KV heads shard over it; every card gets the same slice, so the smallest bounds it.
    tp = 1
    while tp * 2 <= len(use) and heads and heads % (tp * 2) == 0:
        tp *= 2
    out["--tensor-parallel-size"] = tp if tp > 1 else None
    out["--pipeline-parallel-size"] = None
    total = min(g["total"] for g in use[:tp])
    kv_shards = min(tp, kv_heads) if kv_heads else 1  # fewer KV heads than cards: vLLM replicates them
    if tp > 1:
        notes.append(f"{len(use)} cards → --tensor-parallel-size {tp}: each card holds 1/{tp} of the weights "
                     f"and KV; sized by the smallest ({total / GiB:.1f} GB).")
    elif len(use) > 1:
        notes.append(f"{len(use)} cards, but {heads} attention heads don't split evenly; "
                     "try --pipeline-parallel-size instead.")
    weights_per_card = (weights or 0) / tp
    per_tok16 = attn_layers * kv_heads * head_dim * 2 * KV_BYTES["auto"] / kv_shards
    # Try the cheapest change first; take the first combo that fits a 16K context, else the roomiest.
    util0 = values.get("--gpu-memory-utilization") or 0.88
    combos = [(util0, "auto"), (util0, "fp8"), (max(util0, 0.92), "fp8"), (max(util0, 0.95), "fp8")]
    tried = []
    for util, kv_dtype in combos:
        budget = total * util - weights_per_card - COMPUTE_RESERVE
        # hybrid: running sequences hold recurrent state too; cap them, then fit KV in what's left
        seqs = recurrent_seqs(budget, state_per_seq / tp) if state_per_seq and budget > 0 else None
        per_tok = per_tok16 * KV_BYTES[kv_dtype] / 2
        ctx, limit = fit_ctx(budget - (seqs or 0) * state_per_seq / tp, per_tok, max_pos)
        tried.append((ctx or 0, util, kv_dtype, budget, per_tok, limit, seqs))
        if (ctx or 0) >= 16384:
            break
    ctx, util, kv_dtype, budget, per_tok, limit, seqs = next((c for c in tried if c[0] >= 16384), max(tried))
    out["--gpu-memory-utilization"] = util
    out["--kv-cache-dtype"] = kv_dtype if kv_dtype != "auto" else None
    out["--max-model-len"] = ctx or None
    if kv_dtype == "fp8":
        notes.append("fp8 KV cache: half the bytes per token, so twice the context.")
    if util > util0:
        notes.append(f"Raised --gpu-memory-utilization to {util} to make room; close other GPU apps (browsers, games) first.")
    hybrid = f" ({attn_layers} of {layers} layers keep KV)" if attn_layers != layers else ""
    if seqs:
        out["--max-num-seqs"] = seqs
        notes.append(f"Hybrid model: {rec_layers} recurrent (Mamba / linear-attention) layers hold "
                     f"{state_per_seq / 2**20:.0f} MiB of state per running sequence; vLLM refuses to start when "
                     f"--max-num-seqs (default 256) exceeds the states that fit → --max-num-seqs {seqs}.")
    if weights:
        notes.append(f"VRAM {total / GiB:.1f} GB{' per card' if tp > 1 else ''} on {gpu_names(use[:tp])}"
                     f"{'' if measured else ' (assumed)'} × {util} − {weights_per_card / GiB:.1f} weights − "
                     f"{COMPUTE_RESERVE / GiB:.1f} activations = {max(budget, 0) / GiB:.1f} GB for KV; "
                     f"{per_tok / 1024:.0f} KiB/token{' per card' if tp > 1 else ''}{hybrid}.")
    notes.append(f"Fits ~{max(limit, 0):,} tokens → --max-model-len {ctx:,}" if ctx else
                 "⚠ Weights leave no room for KV cache at this --gpu-memory-utilization; use a smaller/quantized model.")

    if "vision_config" in cfg:
        # vLLM profiles the largest allowed media input at startup; skipping video saves a lot of VRAM
        out["--limit-mm-per-prompt"] = '{"image": 2, "video": 0}'
        notes.append("Vision model: images on, video off (vLLM reserves memory for the largest allowed input).")
    else:
        out["--limit-mm-per-prompt"] = None

    rule = family_rule(model_type, " ".join(cfg.get("architectures") or []), repo, base)
    reg = vllm_parsers()
    reasoning = rule[2] if rule else None
    tool, seen = (rule[3] if rule else None), None
    if not tool:  # e.g. Nemotron 3 writes Qwen3-Coder style <function=…> calls
        try:
            tool, seen = template_tool_parser(model_template("vllm", values)["content"])
        except ValueError:
            pass
    if reg:
        reasoning = reasoning if reasoning in reg.get("reasoning", []) else None
        tool = tool if tool in reg.get("tool", []) else None
    out["--reasoning-parser"], out["--tool-call-parser"] = reasoning, tool
    out["--enable-auto-tool-choice"] = True if tool else None
    if rule or tool:
        notes.append(f"{model_type or repo}: reasoning parser {reasoning or '—'}, tool parser {tool or '—'}"
                     f"{f' (from the chat template: {seen})' if seen and tool else ''}.")

    out["--generation-config"] = None  # "auto": vLLM applies the repo's generation_config.json defaults
    samp = {k: gen[k] for k in ("temperature", "top_p", "top_k", "min_p") if gen and k in gen}
    extra_card = []
    if samp:
        notes.append("Sampling defaults from generation_config.json (vLLM applies them): " + ", ".join(f"{k} {v}" for k, v in samp.items()))
    elif base:
        bgen = hf_json(base, "generation_config.json") or {}
        bsamp = {k: bgen[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in bgen}
        if bsamp:
            notes.append(f"This build ships no sampling defaults; {base} does: " + ", ".join(f"{k} {v}" for k, v in bsamp.items()))
            extra_card.append((f"{base} generation_config.json (base model)",
                               f"```\nvllm serve x --override-generation-config '{json.dumps(bsamp)}'\n```"))
    if cfg.get("auto_map"):
        notes.append("⚠ Ships custom modeling code (auto_map): it only loads with --trust-remote-code — enable it only if you trust this repo.")
    spec, spec_note = mtp_suggestion(tc, local, model_type)
    if spec:
        out["--speculative-config"] = spec
    if spec_note:
        notes.append(spec_note)

    # model card: tested serve commands (this repo's card and scripts, else the base model's card)
    sources = [(f"{repo} model card", card_text)] if card_text else []
    sources += [(f"{repo}/{name}", text) for name, text in repo_scripts(local)]
    if base and not any(card_commands(s[1], "vllm", {}) for s in sources):
        btext = readme_text(None, base)
        if btext:
            sources.append((f"{base} model card (base model)", btext))
    card = card_suggestions("vllm", sources + extra_card, out)
    if card:
        notes.append("The model card / repo scripts have vLLM commands: their flags are listed below — tick what to use.")
    return {"values": out, "notes": notes, "model": repo, "card": card}


def suggest(backend, values):
    return suggest_llama(values) if backend == "llama.cpp" else suggest_vllm(values)


# ── chat templates ────────────────────────────────────────────────────────
TEMPLATE_SAMPLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "What is 2+2?"},
]


def template_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]{0,63}", name or ""):
        raise ValueError("template names: letters, digits, . _ - (up to 64)")
    return os.path.join(TEMPLATES_DIR, name + ".jinja")


def list_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        return []
    out = []
    for n in sorted(os.listdir(TEMPLATES_DIR)):
        if n.endswith(".jinja"):
            st = os.stat(os.path.join(TEMPLATES_DIR, n))
            out.append({"name": n[:-6], "size": st.st_size, "mtime": st.st_mtime})
    return out


def read_template(name):
    try:
        with open(template_path(name)) as f:
            return f.read()
    except FileNotFoundError:
        raise KeyError(name)


def template_users(name, cfg):
    """Where a template is selected: the editable config of either backend and the boot default."""
    users = []
    for backend in BACKENDS:
        flag = "--chat-template-file" if backend == "llama.cpp" else "--chat-template"
        if cfg.get(backend, {}).get("values", {}).get(flag) == name:
            users.append(f"{backend} settings")
    boot = cfg.get("boot") or {}
    if boot and name in (boot.get("config", {}).get("values", {}).get(f) for f in ("--chat-template-file", "--chat-template")):
        users.append("boot default")
    return users


def check_template(content):
    """Parse with Jinja2 and render a short sample conversation, the way HF tokenizers / vLLM do.
    llama.cpp uses its own Jinja subset (minja): a template that passes here may still need tweaks there."""
    import datetime
    from jinja2 import TemplateSyntaxError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(message):
        raise ValueError(message)

    env.globals.update(raise_exception=raise_exception,
                       strftime_now=lambda fmt: datetime.datetime.now().strftime(fmt))
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, separators=separators, sort_keys=sort_keys)
    try:
        tpl = env.from_string(content)
    except TemplateSyntaxError as e:
        return {"ok": False, "syntax_ok": False, "error": e.message, "line": e.lineno}
    try:
        out = tpl.render(messages=TEMPLATE_SAMPLE, add_generation_prompt=True, bos_token="<s>", eos_token="</s>",
                         tools=None, enable_thinking=True)
    except Exception as e:
        return {"ok": False, "syntax_ok": True, "error": f"renders with an error: {e}", "line": None}
    return {"ok": True, "syntax_ok": True, "preview": out[:6000]}


def save_template(name, content):
    if not (content or "").strip():
        raise ValueError("template is empty")
    if len(content) > 512_000:
        raise ValueError("template too large")
    result = check_template(content)
    if not result["syntax_ok"]:
        raise ValueError(f"Jinja syntax error on line {result['line']}: {result['error']}")
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    path = template_path(name)
    with open(path + ".tmp", "w") as f:
        f.write(content)
    os.replace(path + ".tmp", path)
    return result  # a render error (e.g. a template that refuses the sample) is reported, not fatal


def hf_text(repo, filename):
    try:
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{filename}", headers=hf_headers())
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def _tokenizer_template(tok_cfg):
    t = (tok_cfg or {}).get("chat_template")
    if isinstance(t, list):  # [{"name": "default", "template": ...}, {"name": "tool_use", ...}]
        t = next((x.get("template") for x in t if x.get("name") == "default"), t[0].get("template") if t else None)
    return t if isinstance(t, str) else None


def model_template(backend, values):
    """The chat template the selected model ships, to start editing from."""
    if backend == "llama.cpp":
        model = values.get("--model")
        if not model:
            raise ValueError("pick a target model first")
        t = read_gguf_meta(models_path(model), keep_template=True).get("tokenizer.chat_template")
        if not isinstance(t, str):
            raise ValueError(f"{model} has no chat template in its GGUF header")
        return {"content": t, "source": f"{model} (GGUF header)"}
    model = (values.get("model") or "").strip().rstrip("/")
    if not model:
        raise ValueError("pick a model first")
    local = hub_snapshot(model) if re.fullmatch(r"[\w.-]+/[\w.-]+", model) and not model.startswith("hf/") else None
    if not local and os.path.isdir(os.path.join(MODELS_DIR, model)):
        local = os.path.join(MODELS_DIR, model)
    if local:
        jinja_file = os.path.join(local, "chat_template.jinja")
        if os.path.exists(jinja_file):
            with open(jinja_file) as f:
                return {"content": f.read(), "source": f"{model}/chat_template.jinja"}
        t = _tokenizer_template(_read_json(os.path.join(local, "tokenizer_config.json")))
        if t:
            return {"content": t, "source": f"{model}/tokenizer_config.json"}
    else:
        t = hf_text(model, "chat_template.jinja")
        if t:
            return {"content": t, "source": f"{model}/chat_template.jinja (Hugging Face)"}
        t = _tokenizer_template(hf_json(model, "tokenizer_config.json"))
        if t:
            return {"content": t, "source": f"{model}/tokenizer_config.json (Hugging Face)"}
    raise ValueError(f"no chat template found for {model}")


# ── Hugging Face downloads ────────────────────────────────────────────────
def hf_files(repo, revision=None):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision or None, files_metadata=True)
    files = sorted(({"name": s.rfilename, "size": s.size} for s in info.siblings), key=lambda f: f["name"])
    plan = repo_plan(files)
    if plan.get("kind") == "safetensors":
        plan["gpu_note"] = gpu_format_note(hf_json(info.id, "config.json"))
    return {"repo": info.id, "revision": info.sha, "gated": bool(getattr(info, "gated", False)),
            "pipeline": getattr(info, "pipeline_tag", None), "files": files, "plan": plan}


# ── Hugging Face browsing: repo search, quant grouping, VRAM-based recommendation ──
# quant tag in a GGUF name: Q4_K_M, UD-Q4_K_XL, IQ4_XS, Q8_0, BF16, MXFP4_MOE, …
QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[0-9A-Z]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4|TQ\d_\d))(?=$|[-_.])")
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
KV_HEADROOM = 2 * GiB  # room left for a working context's KV cache when judging "fits"


HF_SORTS = ("downloads", "likes", "trendingScore", "lastModified", "createdAt")


def hf_search(q, task=None, fmt=None, sort="downloads", limit=40):
    """Repos for an owner name (author=) or, failing that / for free text, a search.

    task = HF pipeline_tag (text-generation, image-to-image, ...), fmt = a library/format tag
    (gguf, safetensors, awq, ...); with either set, an empty q browses the whole hub."""
    q, task, fmt = (q or "").strip(), (task or "").strip(), (fmt or "").strip()
    sort = sort if sort in HF_SORTS else "downloads"
    for v in (task, fmt):
        if v and not re.fullmatch(r"[\w.-]+", v):
            raise ValueError(f"bad filter {v!r}")
    if not (q or task or fmt):
        raise ValueError("type an owner, owner/repo, or search words — or pick a task/format filter")
    base = {"sort": sort, "direction": "-1", "limit": str(limit),
            "expand[]": ["downloads", "likes", "lastModified", "gated", "tags", "pipeline_tag", "library_name"]}
    if task:
        base["pipeline_tag"] = task
    if fmt:
        base["filter"] = fmt

    def query(**params):
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({**base, **params}, doseq=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=hf_headers()), timeout=20) as r:
            return json.load(r)

    if not q:
        mode, rows = "browse", query()
    else:
        mode, rows = "author", (query(author=q) if re.fullmatch(r"[\w.-]+", q) else [])
        if not rows:
            mode, rows = "search", query(search=q)
    out = []
    for m in rows:
        tags = m.get("tags") or []
        base_model = next((t.split(":", 1)[1] for t in tags if t.startswith("base_model:") and t.count(":") == 1), None)
        formats = [t for t in ("gguf", "awq", "gptq", "fp8", "mlx", "exl2", "compressed-tensors") if t in tags]
        out.append({"id": m["id"], "downloads": m.get("downloads"), "likes": m.get("likes"),
                    "updated": m.get("lastModified"), "gated": bool(m.get("gated")), "pipeline": m.get("pipeline_tag"),
                    "formats": formats, "base_model": base_model,
                    "quantized": any(t.startswith("base_model:quantized:") for t in tags)})
    return {"mode": mode, "query": q, "task": task, "format": fmt, "sort": sort, "repos": out}


def gguf_groups(files):
    """GGUF files grouped into downloadable units: one per quant (split parts summed), by role."""
    groups = {}
    for f in files:
        name = f["name"]
        if not name.lower().endswith(".gguf"):
            continue
        stem = SPLIT_RE.sub("", os.path.basename(name)[:-5])
        low = stem.lower()
        kind = ("mmproj" if "mmproj" in low else "other" if "imatrix" in low
                else "draft" if low.startswith("mtp") or "/mtp" in name.lower() or any(k in low for k in ("dflash", "draft", "eagle"))
                else "model")
        m = QUANT_RE.search(stem)
        quant = m.group(1).upper() if m else stem
        g = groups.setdefault((kind, os.path.dirname(name), quant),
                              {"kind": kind, "quant": quant, "dir": os.path.dirname(name), "files": [], "size": 0})
        g["files"].append(name)
        g["size"] += f["size"] or 0
    return sorted(groups.values(), key=lambda g: (["model", "draft", "mmproj", "other"].index(g["kind"]), -g["size"]))


def repo_plan(files):
    """What fits on the detected GPUs: per-quant fit labels + the recommended GGUF quant, or for a
    safetensors repo (vLLM) whether the weights fit at all."""
    gpus, measured = gpu_list()
    total = sum(g["total"] for g in gpus)
    hard = total - DESKTOP_RESERVE - COMPUTE_RESERVE  # weights + nothing else
    groups = gguf_groups(files)
    vram = {"total": total, "gpus": gpu_names(gpus), "measured": measured}
    if groups:
        # leave room for the smallest vision projector and draft the repo ships, since you'd load them too
        mm = [g["size"] for g in groups if g["kind"] == "mmproj"]
        dr = [g["size"] for g in groups if g["kind"] == "draft"]
        mm_room, draft_room = (min(mm) if mm else 0), (min(dr) if dr else 0)
        budget = hard - KV_HEADROOM - mm_room - draft_room
        for g in groups:
            if g["kind"] == "model":
                g["fit"] = "fits" if g["size"] <= budget else "tight" if g["size"] <= hard else "offload"
        fitting = [g for g in groups if g.get("fit") == "fits"]
        rec = max(fitting, key=lambda g: g["size"]) if fitting else None
        if rec:
            rec["recommended"] = True
        extras = " and ".join(x for x in ("the vision projector" if mm_room else "", "the draft model" if draft_room else "") if x)
        why = (f"{rec['quant']}: the largest quant that fits {vram['gpus']} ({total / GiB:.0f} GB) with ~"
               f"{KV_HEADROOM / GiB:.0f} GB left for context{' plus ' + extras if extras else ''}."
               if rec else f"No quant fits {total / GiB:.0f} GB with room for context; the smallest still needs partial CPU offload.")
        return {"kind": "gguf", "groups": groups, "vram": {**vram, "budget": budget}, "why": why}
    weights = sum(f["size"] or 0 for f in files if f["name"].endswith(".safetensors"))
    if weights:
        # vLLM: weights + activations + a working KV cache inside ~90% of a card (tensor parallel splits weights)
        card = min(g["total"] for g in gpus)
        per_card = card * 0.9 - COMPUTE_RESERVE - KV_HEADROOM
        # "tight": fits once auto-tune raises --gpu-memory-utilization to 0.95 and uses fp8 KV (~0.5 GB of KV)
        tight_card = card * 0.95 - COMPUTE_RESERVE - 0.5 * GiB
        cards = len(gpus)
        fit = ("fits" if weights <= per_card else "tight" if weights <= tight_card
               else "tp" if cards > 1 and weights / cards <= per_card else "too-big")
        w = f"{weights / GiB:.1f} GB of weights"
        why = {"fits": f"{w} fits one {vram['gpus'].split(' + ')[0]} under vLLM.",
               "tight": f"{w} fits only just: auto-tune will raise --gpu-memory-utilization and use an fp8 KV cache, "
                        "leaving a short context (~16K).",
               "tp": f"{w} needs tensor parallel over {cards} cards.",
               "too-big": f"{w} won't fit {total / GiB:.0f} GB — pick a smaller model or a lower-bit build "
                          "(or a GGUF quant with llama.cpp, which can offload layers to the CPU)."}[fit]
        return {"kind": "safetensors", "weights": weights, "fit": fit, "vram": vram, "why": why}
    return {"kind": "other", "vram": vram, "why": None}


class Downloads:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}

    def start(self, repo, files, revision, layout):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
            raise ValueError("repo must look like owner/name")
        if layout not in ("gguf", "hf"):
            raise ValueError("layout must be gguf or hf")
        listing = hf_files(repo, revision)
        sizes = {f["name"]: f["size"] or 0 for f in listing["files"]}
        if files:
            unknown = [f for f in files if f not in sizes]
            if unknown:
                raise ValueError(f"not in {repo}: {', '.join(unknown)}")
        else:
            files = list(sizes)
        # gguf: the picked files land flat in /models; hf: the repo goes into the shared hub cache in
        # /models (models--owner--name), the same copy `vllm serve owner/name` and huggingface-cli use
        dest = MODELS_DIR if layout == "gguf" else hub_repo_dir(repo)
        job = {"id": uuid.uuid4().hex[:10], "repo": repo, "revision": listing["revision"], "layout": layout,
               "dest": os.path.relpath(dest, MODELS_DIR), "files": [{"name": f, "size": sizes[f]} for f in files],
               "total": sum(sizes[f] for f in files), "done": 0, "status": "downloading", "error": None,
               "started": time.time(), "finished": None, "tail": deque(maxlen=20)}
        # a branch name (not the resolved sha) so the hub cache records refs/<branch> for this repo
        spec = {"repo": repo, "revision": revision or "main", "layout": layout, "dest": dest, "files": files}
        env = hf_env(dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1"))
        job["proc"] = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--download", json.dumps(spec)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                       start_new_session=True)
        with self.lock:
            self.jobs[job["id"]] = job
        threading.Thread(target=self._run, args=(job, dest), daemon=True).start()
        return self.public(job)

    def _progress(self, job, dest):
        if job["layout"] == "hf":
            # hub cache: content (finished and *.incomplete) lives in blobs/ under hash names
            job["done"] = min(_dir_size(os.path.join(dest, "blobs")), job["total"])
            return
        done = 0
        for f in job["files"]:
            final = os.path.join(dest, f["name"])
            if os.path.exists(final):
                done += os.path.getsize(final)
                continue
            pattern = os.path.join(dest, ".cache", "huggingface", "download", glob.escape(f["name"]) + ".*.incomplete")
            done += sum(os.path.getsize(p) for p in glob.glob(pattern) if os.path.exists(p))
        job["done"] = min(done, job["total"]) if job["total"] else done

    def _run(self, job, dest):
        proc = job["proc"]
        reader = threading.Thread(target=lambda: [job["tail"].append(l.rstrip()) for l in proc.stdout], daemon=True)
        reader.start()
        while proc.poll() is None:
            self._progress(job, dest)
            time.sleep(1)
        reader.join(2)
        self._progress(job, dest)
        job["finished"] = time.time()
        if job["status"] == "cancelled":
            return
        if proc.returncode == 0:
            job["status"], job["done"] = "done", job["total"]
            if job["layout"] == "gguf":  # flat files lose their repo; auto-tune reads its model card
                record_sources(job["repo"], job["revision"], [f["name"] for f in job["files"]])
        else:
            job["status"] = "failed"
            job["error"] = next((l for l in reversed(job["tail"]) if l.strip()), f"exit code {proc.returncode}")

    def cancel(self, job_id):
        """Stop a running download; a finished one (done / failed / cancelled) is dropped from the list."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job["proc"].poll() is not None and job["finished"]:
                del self.jobs[job_id]
                return "removed"
        if job["proc"].poll() is None:
            job["status"] = "cancelled"
            os.killpg(job["proc"].pid, signal.SIGTERM)
        return "cancelled"

    @staticmethod
    def public(job):
        return {k: (list(v) if isinstance(v, deque) else v) for k, v in job.items() if k != "proc"}

    def list(self):
        with self.lock:
            jobs = list(self.jobs.values())
        return [self.public(j) for j in sorted(jobs, key=lambda j: -j["started"])]


def download_main(spec_json):
    """Child process for one download job, so it can be cancelled with a signal."""
    from huggingface_hub import snapshot_download
    spec = json.loads(spec_json)
    print(f"downloading {spec['repo']}@{spec['revision']}: {len(spec['files'])} files -> {spec['dest']}", flush=True)
    patterns = [glob.escape(f) for f in spec["files"]]
    if spec["layout"] == "hf":
        path = snapshot_download(spec["repo"], revision=spec["revision"], cache_dir=HUB_CACHE,
                                 allow_patterns=patterns, max_workers=8)
    else:
        os.makedirs(spec["dest"], exist_ok=True)
        path = snapshot_download(spec["repo"], revision=spec["revision"], local_dir=spec["dest"],
                                 allow_patterns=patterns, max_workers=8)
    print(f"done: {path}", flush=True)


# ── HTTP API ──────────────────────────────────────────────────────────────
# ── smoke test: does the loaded model's tool calling / reasoning actually come back parsed? ──
SMOKE_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
RAW_TOOL_RE = re.compile(r"<tool_call>|<function=|\[TOOL_CALLS\]|<\|tool_call|<｜tool▁call|\"name\"\s*:\s*\"get_weather\"")


def _backend_json(path, body=None, timeout=240):
    req = urllib.request.Request(f"http://{BACKEND_HOST}:{BACKEND_PORT}{path}", headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def smoke_test():
    """Two small requests against the running backend: one that should come back as a parsed tool call,
    one that should have its reasoning split out. Each check is (status, detail) with status
    ok / fail (parser missing or wrong) / warn (inconclusive) / off (not enabled) / na."""
    model = _backend_json("/v1/models", timeout=15)["data"][0]["id"]
    res = {"model": model}
    try:
        choice = _backend_json("/v1/chat/completions", {
            "model": model, "temperature": 0, "max_tokens": 2048, "tools": [SMOKE_TOOL], "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Use the get_weather tool to check the weather in Boise."}]})["choices"][0]
        msg, calls = choice["message"], choice["message"].get("tool_calls") or []
        content = msg.get("content") or ""
        if calls:
            fn = calls[0]["function"]
            res["tools"] = ("ok", f"{fn['name']}({fn.get('arguments')}) came back in tool_calls")
        elif RAW_TOOL_RE.search(content):
            res["tools"] = ("fail", "the tool call came back as plain text — the tool-call parser is missing or wrong for this model")
        elif choice.get("finish_reason") == "length":
            res["tools"] = ("warn", "ran out of tokens before answering (long thinking?) — inconclusive")
        else:
            res["tools"] = ("warn", "the model answered without calling the tool — inconclusive")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:300]
        res["tools"] = (("off", "tool calling isn't enabled: " + err) if re.search(r"(?i)tool|jinja", err)
                        else ("fail", f"HTTP {e.code}: {err}"))
    except Exception as e:
        res["tools"] = ("fail", str(e)[:200])
    try:
        msg = _backend_json("/v1/chat/completions", {
            "model": model, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": "What is 17 × 23? Think it through, then answer."}]})["choices"][0]["message"]
        reasoning, content = msg.get("reasoning_content") or msg.get("reasoning"), msg.get("content") or ""
        if reasoning:
            res["reasoning"] = ("ok", f"reasoning split out ({len(reasoning):,} chars)")
        elif "<think>" in content or "</think>" in content:
            res["reasoning"] = ("fail", "<think> text left in the answer — the reasoning parser is missing or wrong")
        else:
            res["reasoning"] = ("na", "no reasoning in the reply (not a reasoning model, or thinking is off)")
    except Exception as e:
        res["reasoning"] = ("fail", str(e)[:200])
    return res


BOOT_RECORD = os.path.join(STATE_DIR, "boot_attempts.json")
BOOT_MAX_FAILS = 2


def read_boot_record():
    try:
        with open(BOOT_RECORD) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_boot_record(rec):
    try:
        with open(BOOT_RECORD + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(BOOT_RECORD + ".tmp", BOOT_RECORD)
    except OSError:
        pass


class App:
    def __init__(self):
        self.cfg = load_config()
        self.backend = Backend()
        self.downloads = Downloads()
        self.vllm_envs = VllmEnvs()
        self.apply_lock = threading.Lock()
        self.smoke = None  # last smoke_test result for the running backend

    def run_smoke(self):
        kind, pid = self.backend.kind, self.backend.proc.pid if self.backend.proc else None
        self.smoke = {"running": True, "at": time.time(), "backend": kind}
        try:
            res = smoke_test()
        except Exception as e:
            res = {"error": str(e)[:200]}
        if self.backend.kind == kind and (self.backend.proc.pid if self.backend.proc else None) == pid:
            self.smoke = {**res, "at": time.time(), "backend": kind}
            for name in ("tools", "reasoning"):
                if name in res:
                    self.backend.log(f"smoke test · {name}: {res[name][0]} — {res[name][1]}", tag="server")

    def start_selected(self, listener=None, meta=None):
        kind = self.cfg["backend"]
        try:
            argv = build_argv(kind, self.cfg)
            issues = validate_config(kind, self.cfg)
            blocked = blocking_issues(issues)
            if blocked:
                raise ValueError(blocked)
        except ValueError as e:
            with self.backend.lock:
                self.backend.kind, self.backend.state, self.backend.error = kind, "failed", str(e)
            self.backend.log(f"cannot start {kind}: {e}", tag="server")
            if listener:
                listener("failed", str(e))
            return
        for i in issues:
            self.backend.log(f"warning: {i['msg']}", tag="server")
        self.smoke = None
        self.backend.start(kind, argv, gpu_env(self.cfg, kind), self._with_smoke(listener), meta)

    def _with_smoke(self, listener=None):
        def outcome(event, error=None):
            if listener:
                listener(event, error)
            if event == "ready" and self.cfg.get("ui", {}).get("smoke_test", True):
                threading.Thread(target=self.run_smoke, daemon=True).start()
        return outcome

    @staticmethod
    def _boot_outcome(backend):
        """Boot guard: the attempt counted as failed by boot_now is cleared once the backend is ready."""
        def outcome(event, error=None):
            if event == "ready":
                write_boot_record({"fails": 0, "at": time.time(), "backend": backend})
            else:
                write_boot_record({**read_boot_record(), "error": (error or "failed")[:300]})
        return outcome

    def adopt(self):
        """Manager start: keep the engine the previous manager left running (see Backend.adopt). A boot
        model still loading keeps its boot guard, and the smoke test runs once it is ready as usual."""
        rec = self.backend.adopt()
        if not rec:
            return False
        boot = (rec.get("meta") or {}).get("boot")
        self.boot_source, self.boot_skipped = boot, None
        if self.backend.state == "starting":
            self.backend.listener = self._with_smoke(self._boot_outcome(rec["kind"]) if boot else None)
        return True

    def restart_manager(self):
        """Re-execute this file in place (POST /api/manager/restart): picks up a new manager.py without
        touching the engine. os.execv keeps the PID, so the engine stays our child and is adopted from
        BACKEND_RECORD by the new process. Refused while a start/stop or a download is in flight
        (a download's output pipe would not survive the exec)."""
        if any(j["proc"].poll() is None for j in list(self.downloads.jobs.values())):
            raise RuntimeError("a download is running; restart the manager once it finishes or is cancelled")
        if self.vllm_envs.running():
            raise RuntimeError("a vLLM upgrade is installing; restart the manager once it finishes or is cancelled")
        if not self.apply_lock.acquire(blocking=False):  # held until the exec: no start/apply in between
            raise RuntimeError("a backend restart is in progress")
        if self.backend.state == "stopping":
            self.apply_lock.release()
            raise RuntimeError("the backend is stopping; try again once it has stopped")

        def run():
            time.sleep(0.5)  # let the HTTP reply go out first
            with self.backend.lock:  # not mid-start: the record of a just-started engine is written
                self.backend.log("restarting the manager; the backend keeps running", tag="server")
                sys.stdout.flush()
                sys.stderr.flush()
                os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])
        threading.Thread(target=run, daemon=True).start()

    def apply(self, new_cfg):
        """Validate, save, then restart in the background so the HTTP call returns at once."""
        if new_cfg.get("backend") not in BACKENDS:
            raise ValueError("backend must be llama.cpp or vllm")
        build_argv(new_cfg["backend"], new_cfg)
        blocked = blocking_issues(validate_config(new_cfg["backend"], {**new_cfg, "forms": self.cfg.get("forms", {})}))
        if blocked:
            raise ValueError(blocked)
        if not self.apply_lock.acquire(blocking=False):
            raise RuntimeError("a restart is already in progress")
        self.cfg = self.merge_page_config(new_cfg)
        save_config(self.cfg)

        def run():
            try:
                self.backend.stop()
                self.start_selected()
            finally:
                self.apply_lock.release()
        threading.Thread(target=run, daemon=True).start()

    def merge_page_config(self, new_cfg):
        """The page posts its whole draft; the boot default and UI prefs have their own endpoints,
        so keep the server's copies rather than a possibly stale one from the browser."""
        merged = dict(new_cfg)
        merged["boot"] = self.cfg.get("boot")
        merged["ui"] = self.cfg.get("ui", {})
        merged["forms"] = self.cfg.get("forms", {})
        return merged

    def boot_now(self):
        """Container start: run what boot_plan picks (and show it as the page's config).

        Guarded so a boot default that can't come up doesn't keep failing: it is checked against the
        installed version's --help first, and an attempt counts as failed until the backend reports
        ready — after BOOT_MAX_FAILS in a row (crashes, or the container killed mid-load) autostart
        pauses until the page resumes it or a new boot default is saved."""
        source, backend, conf = boot_plan(self.cfg)
        self.boot_source, self.boot_skipped = source, None
        if not source:
            self.backend.log("no boot model: the container is up with no backend; start one from /server, "
                             "or save a boot default there", tag="server")
            return
        self.cfg["backend"] = backend
        self.cfg[backend] = json.loads(json.dumps(conf))
        save_config(self.cfg)
        where = ".env (AUTOSTART=1)" if source == "env" else "the boot default saved on /server"
        rec = read_boot_record()
        if rec.get("fails", 0) >= BOOT_MAX_FAILS:
            self.boot_skipped = (f"autostart paused: the boot model failed to come up {rec['fails']} times in a row"
                                 f"{' (' + rec['error'] + ')' if rec.get('error') else ''}. Its settings are loaded — "
                                 "press Start to try it, fix them, or resume autostart.")
        else:
            blocked = blocking_issues(validate_config(backend, self.cfg))
            if blocked:
                self.boot_skipped = f"autostart skipped — the boot settings don't match this {backend} version: {blocked}"
        if self.boot_skipped:
            self.backend.log(self.boot_skipped, tag="server")
            return
        write_boot_record({"fails": rec.get("fails", 0) + 1, "at": time.time(), "backend": backend,
                           "error": "the container stopped before it was ready"})
        self.backend.log(f"boot: starting {backend} from {where}", tag="server")
        self.start_selected(self._boot_outcome(backend), meta={"boot": source})

    def state(self):
        snap = self.backend.snapshot()
        return {"backend": snap, "config": self.cfg, "versions": versions(),
                "auth_required": bool(ADMIN_TOKEN), "busy": self.apply_lock.locked(),
                "boot": {"source": getattr(self, "boot_source", None), "autostart_env": _flag("AUTOSTART"),
                         "skipped": getattr(self, "boot_skipped", None), "fails": read_boot_record().get("fails", 0)},
                "updates": read_updates(), "smoke": self.smoke,
                "webui_port": int(_env("WEBUI_PORT", "3000")),  # Open WebUI, for the pages' Chat link
                "models_dir": MODELS_DIR, "endpoint": f"http://{BACKEND_HOST}:{BACKEND_PORT}",
                "manager": {"pid": os.getpid(), "started_at": MANAGER_STARTED_AT},
                # running vLLM started from another env than the selected one → restart it to switch
                "vllm_envs": {**self.vllm_envs.list(),
                              "restart_needed": snap["kind"] == "vllm" and snap["state"] in ("running", "starting")
                              and bool(snap["argv"]) and snap["argv"][0] != vllm_bin()}}


class Handler(BaseHTTPRequestHandler):
    app = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def authorized(self):
        if not ADMIN_TOKEN:
            return True
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {ADMIN_TOKEN}".encode())

    def route(self, method):
        url = urlparse(self.path)
        path = re.sub(r"^/server", "", url.path).rstrip("/")
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        app = self.app
        if method != "GET" and not self.authorized():
            return self.send_json(401, {"error": "admin token required"})
        try:
            if method == "GET" and path == "/api/state":
                return self.send_json(200, app.state())
            if method == "GET" and path == "/api/vllm/envs":
                return self.send_json(200, app.vllm_envs.list())
            if method == "POST" and path == "/api/vllm/install":
                return self.send_json(202, app.vllm_envs.install(self.body().get("version")))
            if method == "POST" and path == "/api/vllm/install/cancel":
                app.vllm_envs.cancel()
                return self.send_json(200, app.vllm_envs.list())
            if method == "PUT" and path == "/api/vllm/env":  # select / roll back; null = the image's vLLM
                if app.vllm_envs.running() and app.vllm_envs.job["name"] == self.body().get("name"):
                    raise RuntimeError("that env is still installing")
                select_vllm_env(self.body().get("name") or None)
                return self.send_json(200, app.vllm_envs.list())
            m = re.fullmatch(r"/api/vllm/envs/([\w.+-]+)", path)
            if method == "DELETE" and m:
                return self.send_json(200, app.vllm_envs.remove(m.group(1)))
            if method == "POST" and path == "/api/manager/restart":
                app.restart_manager()
                return self.send_json(202, {"ok": True, "pid": os.getpid()})
            if method == "GET" and path == "/api/health":
                # the container is healthy when its pages and this API answer; a model is optional
                snap = app.backend.snapshot()
                return self.send_json(200, {"ok": True, "backend": snap["kind"], "state": snap["state"]})
            if method == "PUT" and path == "/api/boot":
                b = self.body()
                backend, conf = b.get("backend"), b.get("config") or {}
                if backend not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                conf = {"values": conf.get("values", {}), "extra": conf.get("extra", [])}
                check = {backend: conf, "forms": app.cfg.get("forms", {})}
                build_argv(backend, check)  # refuse a boot default that can't start
                blocked = blocking_issues(validate_config(backend, check))
                if blocked:
                    raise ValueError(blocked)
                app.cfg["boot"] = {"backend": backend, "config": conf, "saved_at": time.time()}
                save_config(app.cfg)
                write_boot_record({})  # a new boot default gets fresh attempts
                app.boot_skipped = None
                return self.send_json(200, app.cfg["boot"])
            if method == "DELETE" and path == "/api/boot":
                app.cfg["boot"] = None
                save_config(app.cfg)
                write_boot_record({})
                app.boot_skipped = None
                return self.send_json(200, {"ok": True})
            if method == "POST" and path == "/api/boot/resume":
                write_boot_record({})
                app.boot_skipped = None
                return self.send_json(200, {"ok": True})
            if method == "POST" and path == "/api/smoke":
                if app.backend.state != "running":
                    raise ValueError("start a model first")
                if (app.smoke or {}).get("running"):
                    raise RuntimeError("a smoke test is already running")
                threading.Thread(target=app.run_smoke, daemon=True).start()
                return self.send_json(202, {"ok": True})
            if method == "GET" and path == "/api/updates":
                return self.send_json(200, read_updates())
            if method == "POST" and path == "/api/updates/check":
                return self.send_json(200, check_updates())
            if method == "PUT" and path == "/api/forms":
                b = self.body()
                backend = b.get("backend")
                if backend not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                forms = app.cfg.setdefault("forms", {})
                forms[backend] = clean_form(backend, b.get("form"), forms.get(backend))
                save_config(app.cfg)
                return self.send_json(200, forms[backend] or {})
            if method == "DELETE" and path == "/api/forms":
                backend = q.get("backend")
                if backend not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                # keep custom definitions: a boot default may still carry values for them
                old = app.cfg.setdefault("forms", {}).get(backend) or {}
                app.cfg["forms"][backend] = ({"order": [{"flag": f["flag"], "group": f["group"]} for f in SCHEMAS[backend]],
                                              "custom": old["custom"], "removed": []} if old.get("custom") else None)
                save_config(app.cfg)
                return self.send_json(200, {"ok": True})
            if method == "GET" and path == "/api/templates":
                return self.send_json(200, list_templates())
            if method == "POST" and path == "/api/templates/check":
                return self.send_json(200, check_template(self.body().get("content") or ""))
            if method == "POST" and path == "/api/templates/from-model":
                b = self.body()
                if b.get("backend") not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                return self.send_json(200, model_template(b["backend"], b.get("values") or {}))
            m = re.fullmatch(r"/api/templates/([^/]+)", path)
            if m and method == "GET":
                return self.send_json(200, {"name": m.group(1), "content": read_template(m.group(1))})
            if m and method == "PUT":
                result = save_template(m.group(1), self.body().get("content") or "")
                return self.send_json(200, {"name": m.group(1), **result})
            if m and method == "DELETE":
                users = template_users(m.group(1), app.cfg)
                if users:
                    raise ValueError(f"{m.group(1)} is selected in: {', '.join(users)}; pick another template there first")
                os.remove(template_path(m.group(1)))
                return self.send_json(200, {"ok": True})
            if method == "GET" and path == "/api/hf/token":
                return self.send_json(200, hf_token_status())
            if method == "PUT" and path == "/api/hf/token":
                user = save_hf_token(self.body().get("token"))
                return self.send_json(200, {**hf_token_status(check=False), "user": user})
            if method == "DELETE" and path == "/api/hf/token":
                if os.path.exists(HF_TOKEN_PATH):
                    os.remove(HF_TOKEN_PATH)
                return self.send_json(200, hf_token_status(check=False))
            if method == "GET" and path == "/api/schema":
                forms = app.cfg.get("forms") or {}
                schemas = {be: form_schema(be, forms.get(be)) for be in BACKENDS}
                reg = _parsers  # filled in the background at startup; the page refetches until present
                for f in schemas["vllm"]:
                    if f["flag"] == "--reasoning-parser":
                        f["options"] = [""] + reg.get("reasoning", [])
                    elif f["flag"] == "--tool-call-parser":
                        f["options"] = [""] + reg.get("tool", [])
                return self.send_json(200, {**schemas, "_parsers_ready": bool(reg)})
            if method == "POST" and path == "/api/suggest":
                b = self.body()
                backend = b.get("backend")
                if backend not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                return self.send_json(200, suggest(backend, b.get("values") or {}))
            if method == "PUT" and path == "/api/ui":
                b = self.body()
                app.cfg.setdefault("ui", {}).update({k: bool(v) for k, v in b.items() if k in ("smart_filter", "auto_tune", "update_check", "smoke_test")})
                save_config(app.cfg)
                return self.send_json(200, app.cfg["ui"])
            if method == "GET" and path == "/api/catalog":
                return self.send_json(200, catalog(q.get("backend", app.cfg["backend"])))
            if method == "GET" and path == "/api/logs":
                lines = app.backend.logs_after(int(q.get("after", 0)))
                return self.send_json(200, {"lines": lines, "seq": app.backend.seq, "id": app.backend.log_id})
            if method == "PUT" and path == "/api/config":
                cfg = self.body()
                if cfg.get("backend") not in BACKENDS:
                    raise ValueError("backend must be llama.cpp or vllm")
                app.cfg = app.merge_page_config(cfg)
                save_config(app.cfg)
                return self.send_json(200, {"ok": True})
            if method == "POST" and path == "/api/apply":
                app.apply(self.body())
                return self.send_json(202, {"ok": True})
            if method == "POST" and path == "/api/start":
                if app.backend.alive():
                    raise RuntimeError("already running")
                app.start_selected()
                return self.send_json(202, {"ok": True})
            if method == "POST" and path == "/api/stop":
                threading.Thread(target=app.backend.stop, daemon=True).start()
                return self.send_json(202, {"ok": True})
            if method == "POST" and path == "/api/preview":
                cfg = {**self.body(), "forms": app.cfg.get("forms", {})}
                argv = build_argv(cfg.get("backend"), cfg)
                # issues: None while --help is still being read (the page asks again)
                return self.send_json(200, {"argv": argv, "env": gpu_env(cfg, cfg.get("backend")),
                                            "issues": validate_config(cfg["backend"], cfg, block=False)})
            if method == "GET" and path == "/api/models":
                return self.send_json(200, list_models(app.cfg))
            if method == "DELETE" and path == "/api/models":
                delete_model(q.get("path", ""), app.cfg, app.backend)
                return self.send_json(200, {"ok": True})
            if method == "GET" and path == "/api/hf/search":
                return self.send_json(200, hf_search(q.get("q", ""), q.get("task"), q.get("format"),
                                                    q.get("sort") or "downloads"))
            if method == "GET" and path == "/api/hf/files":
                return self.send_json(200, hf_files(q.get("repo", ""), q.get("revision")))
            if method == "GET" and path == "/api/downloads":
                return self.send_json(200, app.downloads.list())
            if method == "POST" and path == "/api/downloads":
                b = self.body()
                job = app.downloads.start(b.get("repo", "").strip(), b.get("files") or None,
                                          b.get("revision") or None, b.get("layout", "gguf"))
                return self.send_json(202, job)
            m = re.fullmatch(r"/api/downloads/(\w+)", path)
            if method == "DELETE" and m:
                return self.send_json(200, {"ok": True, "result": app.downloads.cancel(m.group(1))})
            return self.send_json(404, {"error": f"no route {method} {url.path}"})
        except KeyError as e:
            return self.send_json(404, {"error": f"not found: {e}"})
        except (ValueError, RuntimeError) as e:
            return self.send_json(400 if isinstance(e, ValueError) else 409, {"error": str(e)})
        except Exception as e:  # huggingface_hub errors, subprocess timeouts, ...
            return self.send_json(502, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")


def main():
    parser = argparse.ArgumentParser(description="Inference server manager")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--download", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.download:
        return download_main(args.download)

    reap_inherited(keep={(_read_json(BACKEND_RECORD) or {}).get("pid")})  # the engine itself: Backend.adopt
    app = App()
    Handler.app = app

    def shutdown(signum, frame):
        app.backend.stop(timeout=25)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"[server] manager on {args.host}:{args.port}, config {CONFIG_PATH}, backend {app.cfg['backend']}", flush=True)
    threading.Thread(target=vllm_parsers, daemon=True).start()  # warm the parser registry for /api/schema
    threading.Thread(target=update_loop, args=(app,), daemon=True).start()
    if not app.adopt():  # an engine left running by a manager restart, else the boot plan
        app.boot_now()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
