# Inference Overdrive

**One Docker image for Windows + WSL 2 that runs local LLMs with llama.cpp _or_ vLLM, managed entirely from your browser.**

Build it once, open `http://127.0.0.1:8888/`, then browse Hugging Face, download a model, let auto-tune size it to your GPU, and click *Apply & restart*. No command-line flags to remember, no Python environments on Windows, no second container per engine. Chat runs through Open WebUI. Everything else (switching engines, editing every flag, watching tokens/s and VRAM) happens on two pages served by the same container.

- **Two engines, one image.** llama.cpp (GGUF, CUDA 13.4.2, built from source for your GPU) and vLLM 0.31.0 (safetensors, AWQ/GPTQ/FP8, continuous batching). Switch with one click; the OpenAI API URL stays the same.
- **Made for WSL 2 + Docker Desktop.** GPU passthrough through `gpus: all`, pinned memory re-enabled for vLLM under WSL, VRAM reserved for the Windows desktop, and every path and port works from both Windows and the WSL shell.
- **A browser control panel** (`/server`) for every engine setting, model downloads with quant recommendations, auto-tune from model metadata, chat-template editing, boot defaults and a smoke test that proves tool calling works.
- **A live monitor** (`/monitor`) showing tokens/s, speculative-decoding acceptance, slots, a request log, and GPU/CPU/RAM telemetry.
- **Open WebUI** for chat with whichever engine is running, started with the stack.

---

## Contents

- [What's in the image](#whats-in-the-image)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Pages, ports and URLs](#pages-ports-and-urls)
- [How it fits together](#how-it-fits-together)
- [Features](#features)
  - [Switch engines and edit every setting](#switch-engines-and-edit-every-setting)
  - [Auto-tune](#auto-tune)
  - [Smoke test](#smoke-test)
  - [Settings checked against the installed engine](#settings-checked-against-the-installed-engine)
  - [Models: browse, download, recommend, delete](#models-browse-download-recommend-delete)
  - [Chat templates](#chat-templates)
  - [What starts with the container](#what-starts-with-the-container)
  - [Update check](#update-check)
  - [Monitor](#monitor)
  - [Chat with Open WebUI](#chat-with-open-webui)
  - [OpenAI-compatible API](#openai-compatible-api)
  - [Vision](#vision)
  - [Multiple GPUs](#multiple-gpus)
  - [Access control](#access-control)
- [WSL 2 notes](#wsl-2-notes)
- [Configuration (`.env`)](#configuration-env)
- [Where your data lives](#where-your-data-lives)
- [Updating llama.cpp or vLLM](#updating-llamacpp-or-vllm)
- [Reference profile: Qwen3.8-27B + DFlash 2 on an RTX 3090](#reference-profile-qwen38-27b--dflash-2-on-an-rtx-3090)
- [Troubleshooting](#troubleshooting)
- [Make targets and scripts](#make-targets-and-scripts)
- [Repository layout](#repository-layout)
- [Tested on, and known limits](#tested-on-and-known-limits)

---

## What's in the image

| Piece | What it is |
| --- | --- |
| **llama.cpp** | `llama-server` built from `master` (or any tag/commit via `LLAMA_CPP_REF`) with CMake against **CUDA 13.4.2**, for your GPU architecture (`CUDA_ARCH`, default `86` = RTX 30xx), with **NCCL** for multi-GPU. Includes DFlash 2 speculative decoding ([PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)). |
| **vLLM** | **0.31.0** prebuilt wheel (`VLLM_VERSION`) with the CUDA 13.0 PyTorch build, **FlashInfer 0.7.0.post1** and the CUDA compiler packages it needs for its just-in-time kernels. Lives in `/opt/venv`. |
| **Server manager** | `python/Server/manager.py`, the container's main process. It starts and stops the chosen engine, owns its settings, downloads models, runs auto-tune and serves the `/server` API. |
| **Control page** | `python/Server/index.html`, served at `/server/`. |
| **Monitor** | `python/Dashboard/monitor.html` at `/monitor/`, fed by the `hw-metrics.py` sampler at `/monitor/hw`. |
| **nginx** | The single front door on port **8888**: the engine's API, `/server/`, `/monitor/`, and a redirect from `/` to `/server/`. Streams responses unbuffered. |
| **Open WebUI** | Separate container (`ghcr.io/open-webui/open-webui:main`) on port **3000**, pointed at the engine through Docker's network. |

Both engines listen on the same internal address (`127.0.0.1:8080`), so the API, Open WebUI and the monitor don't care which one is running. Only one runs at a time, since each wants most of the GPU.

The built image is about 25 GB, most of it vLLM, PyTorch and CUDA.

## Requirements

- **Windows 11** (or Windows 10 21H2+) with **WSL 2** and an Ubuntu distribution.
- **Docker Desktop** with the WSL 2 engine and GPU support (Settings → Resources → WSL integration on for your distro).
- An **NVIDIA GPU** and a Windows driver that reports **CUDA 13.4 or newer** in `nvidia-smi.exe`. It was verified with driver 617.42. Inside WSL you never install a Linux driver; Docker Desktop injects the Windows one.
- **Disk:** about 25 GB for the image plus your models (a 27B model in 4-bit is about 17 GB).
- **GPU architecture:** the default build targets RTX 30xx (`CUDA_ARCH=86`). For RTX 40xx set `89`, for RTX 50xx `120`, and for mixed cards a list like `86;89`, all **before** the first build.

## Quick start

Run these from your WSL shell, in a clone of this repository:

```bash
cp .env.example .env
```

Optional: everything has a working default; edit `.env` to change ports, set tokens or pick your GPU architecture.

```bash
./scripts/check-gpu.sh
```

Checks the NVIDIA driver from WSL and from inside a CUDA container.

```bash
docker compose up --build -d
```

Builds the image (the **first build is long**: it compiles llama.cpp for your GPU and installs vLLM) and starts the engine container plus Open WebUI.

```bash
./scripts/health.sh
```

Confirms the pages and the manager are up. The container is healthy **without a model**; you load one from the browser.

Then open **http://127.0.0.1:8888/** (it redirects to `/server/`) and:

1. **Download a model.** In *Download from Hugging Face*, type an owner (`unsloth`), a repo (`unsloth/Qwen3.8-27B-GGUF`) or search words. Pick a repo; the recommended quant for your GPU is preselected. Click download.
2. **Pick the engine and the model.** Click the **llama.cpp** card for GGUF files or the **vLLM** card for Hugging Face repos, then choose the model. Auto-tune fills in context size, sampling, draft/projector, parsers and memory settings, and shows why.
3. **Apply & restart.** Watch it load in *Backend log*; the smoke test then checks tool calling and reasoning.
4. **Chat.** Use the **Chat ↗** link (Open WebUI at http://127.0.0.1:3000), or call the API at `http://127.0.0.1:8888/v1`.
5. Optional: **Set current as boot default** so this model loads every time the container starts.

## Pages, ports and URLs

| URL | What |
| --- | --- |
| `http://127.0.0.1:8888/` | Redirects to `/server/` |
| `http://127.0.0.1:8888/server/` | Control page: engines, settings, models, downloads, logs |
| `http://127.0.0.1:8888/monitor/` | Live monitor: tokens/s, speculation, slots, requests, hardware |
| `http://127.0.0.1:8888/v1/...` | OpenAI-compatible API of the running engine (`/v1/models`, `/v1/chat/completions`, …) |
| `http://127.0.0.1:8888/server/api/...` | Manager API (state, apply, models, downloads, …) |
| `http://127.0.0.1:3000/` | Open WebUI |

The ports come from `.env`: `LLM_PORT` (default 8888; the older name `LLAMA_PORT` still works) and `WEBUI_PORT` (default 3000). Docker Desktop forwards them to Windows, so the same URLs work in a Windows browser.

## How it fits together

```text
 Windows browser / your apps
        │ :8888                                   │ :3000
        ▼                                         ▼
 ┌─ inference-overdrive-llm-server ──────────┐   ┌─ inference-overdrive-open-webui ─┐
 │ nginx :8888                               │   │ Open WebUI                       │
 │  ├─ /           → 302 /server/            │◄──┤  http://llm-server:8888/v1       │
 │  ├─ /server/    → control page            │   └──────────────────────────────────┘
 │  ├─ /server/api → manager :8090           │
 │  ├─ /monitor/   → monitor page            │
 │  ├─ /monitor/hw → hw-metrics :8083        │
 │  └─ everything else → engine :8080        │
 │                                           │
 │ manager.py ── starts one of ──┐           │
 │   llama-server   or   vllm serve (:8080)  │
 │                                           │
 │ /models  ← ./models (GGUFs + HF cache)    │
 │ /state   ← ./state  (settings, token, …)  │
 └───────────────────────────────────────────┘
          GPU(s) via Docker Desktop (gpus: all)
```

The Compose project is named `inference-overdrive`. That fixes the network (`inference-overdrive_default`) and the Open WebUI volume name whatever the folder is called.

## Features

### Switch engines and edit every setting

The top of `/server` has one card per engine, showing its installed version and an update badge, plus a **Backend** panel with what's running, its state, uptime, PID and VRAM. The buttons are **Apply & restart**, **Save** (takes effect on the next start), **Start**, **Stop** and **Revert** (discards unsaved edits).

- **Parameters form.** A curated form per engine, grouped into sections (Model, Speculative, Context & memory, Throughput, Sampling defaults, Chat, Server, Multi-GPU for llama.cpp; Model, Memory, Scheduling, Speculative, Chat & tools, Multimodal, Multi-GPU for vLLM). An empty field means the engine's own default. On/off flags have three states: *default* / *on* / *off*.
- **Full details on hover.** Hovering a field shows that flag's complete `--help` entry, read from the installed engine: aliases, value format, choices, default, and the `LLAMA_ARG_*` environment variable. It updates by itself when the engine is upgraded.
- **Every other flag.** *Additional flags* takes any flag, with **Browse all flags**: a searchable list of everything in `llama-server --help` or `vllm serve --help=all` (250–320 flags).
- **Form editor** (gear on an engine card → *Edit form*). Search any flag from `--help`, choose its label, input type, section (or a new one) and position, then **Add to form**. On every field, ↑ ↓ reorder, a menu moves it to another section, and ✕ takes it off. Sections move too. Layouts are saved per engine; *Reset to default form* brings the curated form back. A flag already under *Additional flags* moves into its new field.
- **Command preview** shows the exact command line (and `CUDA_VISIBLE_DEVICES`) that *Apply* will run.
- **Backend log** streams the engine's output live, with a filter, follow and clear.
- **Smart model filter** (on by default). The target picker lists only main models; the draft and projector pickers list only files that belong to the chosen target, read from GGUF metadata. Turn it off to see everything.
- **Gear menu** on each engine card: *Edit form*, *Smoke test after load*, *Daily update check*, *Check for updates now*.
- The flags `--host`, `--port`, `--api-prefix` and `--path` are reserved; nginx owns the routing.

### Auto-tune

When you pick a model (or click **Suggest settings**), auto-tune reads the model's own metadata and fills the form. It lists every change (*old → new*) with a reason, and **Undo** reverts all of them.

**Shared by both engines**

- **Memory sizing.** Context length is the largest power of two whose KV cache fits: VRAM − `VRAM_RESERVE_GB` (Windows desktop, default 1.5 GB) − compute buffers − weights. It's picked within 90% of the fit so a context that only just fits isn't chosen. Only real attention layers count on hybrid models.
- **Model card.** `vllm serve` / `llama-server` commands found in the repo's README and shipped scripts (`examples/*.sh`, `scripts/*.sh`) are listed as tick boxes; **Use ticked flags** puts them in the form or *Additional flags*. Flags the installed engine doesn't know, or values outside its choices, can't be ticked. Hardware-sizing flags (`--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `-c`, `-ngl`, …) and anything auto-tune already decided differently start unticked.
- **Base model.** For a quantized repo (model card `base_model` with `base_model_relation: quantized`, or a quant-looking name), the original model fills the gaps: its family for parser choice, its model card's commands, and its sampling defaults.
- **Hybrid models** (Mamba / linear attention: Nemotron-H, Qwen3-Next, Qwen3.5/3.8, Jamba, Granite 4). Auto-tune works out the per-sequence recurrent state from the config.
  - **vLLM:** caps `--max-num-seqs` so those states fit. vLLM otherwise refuses to start with `max_num_seqs (256) exceeds available Mamba cache blocks`. Nemotron-3.5-Lightning on 24 GB gets 32.
  - **llama.cpp:** reserves each server slot's state before sizing the context. Qwen3.8-27B uses 153 MiB per slot.
- **Multi-token prediction (MTP).** Models with built-in MTP layers get speculative decoding from their own head, but only when the downloaded weights really contain it (quantized builds often drop it).

**llama.cpp (GGUF)**

- Reads the GGUF header: role (target / draft / projector), pairing (`general.base_model.*`, projector width = target width), shipped sampling (`general.sampling.*`), trained context and attention shape.
- **Pairs a draft model** automatically: DFlash 2 → `--spec-type draft-dflash`, EAGLE-3 → `draft-eagle3`, otherwise `draft-simple`. With no draft but MTP tensors in the GGUF (`blk.N.nextn.*`), it picks `--spec-type draft-mtp`.
- **Pairs the vision projector** (`--mmproj`) belonging to the target.
- Sampling comes from the GGUF header, then the base repo's `generation_config.json`, then a family default (Qwen3, Gemma, Llama, Mistral, gpt-oss, DeepSeek-R1, …).
- If the weights don't fit, it falls back to partial GPU offload with an 8K context and the share of layers that fits.
- **Finds the GGUF's own model card.** Flat downloads are recorded in `models/.sources.json`. For older files the repo is guessed from the header (`general.repo_url` owner + model name + `-GGUF`) and used only if that repo really contains the file.

**vLLM (Hugging Face repos)**

- Reads `config.json`, `generation_config.json` and file sizes, locally or through the Hub API.
- Sizes `--max-model-len`, trying fp8 KV cache and then raising `--gpu-memory-utilization` up to 0.95 before giving up. Sets `--dtype half` for AWQ, and images on / video off for vision models (vLLM reserves memory for the largest allowed input).
- **Reasoning and tool-call parsers.**
  - **Per-family table** first, checked against the parsers this vLLM actually has.
  - **The model's chat template** when the table has no tool parser: `<function=…><parameter=…>` → `qwen3_coder`, `[TOOL_CALLS]` → `mistral`, JSON in `<tool_call>` → `hermes`.
  - **Example:** Nemotron 3 models get `nemotron_v3` + `qwen3_coder` this way.
  - Turns on `--enable-auto-tool-choice` when a tool parser is set.
- **GPU format check.** FP8 / NVFP4 / MXFP4 weights on a card without native support get a warning. An RTX 3090 is sm_86: FP8 needs sm_89+, FP4 needs sm_100+, and vLLM falls back to slower kernels or refuses some layouts.
- **MTP:** `--speculative-config {"method": "mtp", …}` for DeepSeek-V3, GLM-4.5, MiMo, Qwen3-Next and other MTP models.
- Flags custom modeling code (`auto_map`) and never turns on `--trust-remote-code` by itself.

### Smoke test

When a model finishes loading (gear menu → *Smoke test after load*, on by default), or when you click **Run** in the Backend panel, the manager sends the model two small requests:

- **tools:** a `get_weather` tool call. ✓ means it came back as a parsed `tool_calls` entry. ✗ means it came back as raw text, so the tool-call parser is missing or wrong. *off* means tool calling isn't enabled.
- **reasoning:** a small arithmetic question. ✓ means the reasoning was split out of the answer. ✗ means `<think>` text was left in the reply, so the reasoning parser is wrong. *–* means the model doesn't reason.

*?* means inconclusive: the model answered without the tool, or ran out of tokens. Results also go to the Backend log.

### Settings checked against the installed engine

Every flag the settings pass is checked against the installed engine's own `--help` (cached per version in `state/cache/`). Problems are listed above the form and marked on the fields:

- **won't start:** the flag isn't in this version's `--help` (renamed or removed). Blocks *Apply*, *Start* and saving a boot default.
- **invalid value:** not one of the flag's listed choices. Also blocks.
- **deprecated:** still works; it's only a warning.

If `--help` can't be parsed, nothing is blocked.

### Models: browse, download, recommend, delete

- **One folder for both engines.** `./models` is mounted at `/models` **and** is the Hugging Face hub cache (`HF_HUB_CACHE=/models`). A repo downloaded once (`models/models--owner--name`) is exactly what `vllm serve owner/name` loads. llama.cpp's own downloads (`LLAMA_CACHE`) land there too. Repo files are deduplicated by the shared blob store.
- **Browse Hugging Face.** Type an owner to list their repos, search words to search, or `owner/repo` to open one. Each repo shows formats (GGUF, AWQ, GPTQ, FP8, …), task, downloads, likes and date.
- **Filters:**
  - **Task:** text generation, image-text-to-text, text-to-image, image-to-image, speech recognition, text-to-speech, embeddings, …
  - **Format:** GGUF, safetensors, AWQ, GPTQ, FP8, compressed-tensors, bitsandbytes, MLX, EXL2.
  - **Sort:** downloads, trending, likes, recently updated, newest.

  With a filter set and no search text, they browse all of Hugging Face.
- **Quant recommendation (GGUF).** A table of every quant (split parts summed) marked *fits* / *tight* / *CPU offload* for your detected GPUs. The largest quant that fits with about 2 GB of context, the repo's vision projector and its draft is **recommended** and preselected.
- **Safetensors verdict (vLLM).** *fits* / *tight* / *multi-GPU* / *too big*, plus one-click searches for AWQ / GPTQ / FP8 builds and the GPU format warning.
- **Two download layouts:** picked files flat into `models/` (typical for GGUF), or the whole repo into the hub cache (for vLLM). Progress and cancel are in the download list.
- **Delete** from the *Models* panel. Deleting a cached GGUF frees its blob; deleting a repo removes its cache folder. The disk bar shows space used.
- **Hugging Face token** for gated or private repos: click **HF token** in the download panel and paste it. It's checked with Hugging Face and saved to `state/hf_token` (owner-only permissions). The page only ever shows your username, never the token. `HF_TOKEN` in `.env` takes precedence.

### Chat templates

Both engines have a **Chat template** picker; empty means the model's own template.

- **Customize** / **New** opens an editor seeded with the model's own template (from the GGUF header, or the repo's `chat_template.jinja` / `tokenizer_config.json`). It checks Jinja syntax and renders a sample conversation as you type.
- **Save & use** stores it in `state/templates/<name>.jinja`. It's passed as `--chat-template-file` (llama.cpp, with `--jinja`) or `--chat-template` (vLLM).
- The check uses Python Jinja2, which is what vLLM uses. llama.cpp's template engine is a subset (for example, a `set` inside an `{% if %}` doesn't escape the block), so test on llama.cpp after bigger edits.

### What starts with the container

- **Nothing, by default.** The container is healthy as soon as `/monitor` and `/server` answer; loading a model is your choice.
- **Boot default:** *Set current as boot default* (Backend panel) saves the engine and settings on screen to start on every container start. *Clear* removes it.
- **`AUTOSTART=1` in `.env`** instead starts `BACKEND` with the model named in `.env` (`TARGET_FILE` / `DRAFT_FILE` / `MMPROJ_FILE` for llama.cpp, `VLLM_MODEL` for vLLM). It wins over the boot default.
- **Guarded boot.** The boot settings are first checked against the installed engine's `--help`; a removed flag or invalid value skips the autostart, with the reason shown on the page. A boot attempt counts as failed until the engine reports ready. After **2 failed boots in a row** (crashes, or the container killed mid-load), autostart pauses until you click **Resume autostart** or save a new boot default. The settings are still loaded, so **Start** can try them by hand.

### Update check

Once a day (gear menu → *Daily update check*, on by default) the manager asks GitHub for llama.cpp's latest release, including how many commits it's ahead of your build, and PyPI for vLLM's latest version. An **update** badge on the engine card says what to set in `.env` before rebuilding. **Check for updates now** runs it on demand; results are kept in `state/updates.json`. Nothing is installed automatically; see [Updating](#updating-llamacpp-or-vllm).

### Monitor

`http://127.0.0.1:8888/monitor/` works with both engines:

- **Throughput:** live tokens/s gauge, prompt and generation rates.
- **Speculative decoding:** acceptance per draft position. It's labelled *DFlash2 speculation* when a DFlash draft is active.
- **Slots:** what each server slot is doing.
- **Requests:** a live request log.
- **System** and **Telemetry · last 2 min:** per-GPU utilization, VRAM, power vs. limit, temperature, fan, GPU and memory clocks, PCIe generation and width, plus CPU, RAM and the engine process's own RSS/CPU. There's one tab per GPU on multi-GPU machines.

vLLM's Prometheus metrics are mapped onto the same panels. Set `HW_METRICS=0` in `.env` to skip the hardware sampler.

### Chat with Open WebUI

Open WebUI starts with the stack and reaches the engine at `http://llm-server:8888/v1` (Docker's internal network), whichever engine runs. The **Chat ↗** link on both pages opens it on the right host and port. After switching engines, reload Open WebUI and pick the new model.

- Sign-in is off for local use (`WEBUI_AUTH=false`): anyone who can reach port 3000 can chat.
- Chats and settings live in the `inference-overdrive_open-webui-data` volume.
- llama.cpp's own chat page is **off** by default. Turn on *Built-in web UI* (Server section on `/server`) to get it at `http://127.0.0.1:8888/index.html`.

### OpenAI-compatible API

Use any OpenAI client with `base_url = http://127.0.0.1:8888/v1`. The model name is the served name shown in `/v1/models`: `--alias` on llama.cpp, `--served-model-name` on vLLM (defaults to the repo id).

```bash
curl http://127.0.0.1:8888/v1/models
```

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B-DFlash2", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}'
```

Streaming, tool calls (with `--jinja` on llama.cpp, or a tool parser on vLLM) and reasoning output work as each engine supports them. nginx passes streams through unbuffered.

### Vision

- **llama.cpp:** pick the model's projector GGUF (`mmproj-*.gguf`) under *Vision projector*; auto-tune does it for you when the projector is downloaded. *Projector on GPU* keeps it in VRAM; turn it off to save about 1 GB at some speed cost. `/props` reports `modalities.vision: true` when it's loaded.
- **vLLM:** vision models load as they are; auto-tune limits each prompt to 2 images and no video to keep startup memory down.

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B-DFlash2", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

### Multiple GPUs

Every card is passed into the container and listed under the title on `/server`, with a tab per card on the monitor. Indices follow PCI order everywhere (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), matching `nvidia-smi`. The *Multi-GPU* section is hidden when only one card is found.

- **Choose cards per engine:** *Multi-GPU → GPUs* sets `CUDA_VISIBLE_DEVICES` for that engine (none ticked = all).
- **llama.cpp:** *Split mode* (`layer` spreads layers and KV over the cards; `row` / `tensor` split each matrix and want NVLink-class links; `none` with *Main GPU*), *Tensor split*, and *Draft device* / *Projector device* (for example, keep the draft on the second card). Built with NCCL.
- **vLLM:** *Tensor parallel size* (the attention heads must divide by it) or *Pipeline parallel size* for unequal cards.
- **Auto-tune counts the selected cards.** For llama.cpp it pools their VRAM with `--split-mode layer` and a `--tensor-split` weighted by each card's usable VRAM. For vLLM it picks the largest power-of-two tensor parallel size that divides the heads, and sizes the context by the smallest card.
- **Mixed GPU generations:** build with every architecture, for example `CUDA_ARCH=86;89`.

### Access control

- `SERVER_ADMIN_TOKEN` in `.env` makes every change (`POST` / `PUT` / `DELETE` on `/server/api`) require `Authorization: Bearer <token>`. The page asks once and remembers it in your browser. **Without it, anyone who can reach port 8888 can restart the engine or delete models.**
- Open WebUI has sign-in off (see above).
- The Hugging Face token is stored owner-only in `state/` and never shown back in full.

## WSL 2 notes

- **GPU access** is `gpus: all` in `docker-compose.yml`. Docker Desktop injects the Windows driver (`libcuda`, `nvidia-smi`) into the container, so nothing NVIDIA is installed in WSL itself. Bind-mounting only `/usr/lib/wsl/lib/libcuda.so.1` is **not** enough: the real driver lives under `/usr/lib/wsl/drivers/`, and the shim alone fails with `CUDA driver version is insufficient for CUDA runtime version`.
- **The driver must report CUDA 13.4+** (`nvidia-smi.exe` on Windows). Older drivers fail the same way.
- **Pinned memory for vLLM.** vLLM turns pinned host memory off under WSL 2; `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (the default here) turns it back on for faster host↔GPU copies.
- **The Windows desktop uses VRAM too.** Auto-tune leaves `VRAM_RESERVE_GB` (default 1.5) for it. Raise it if you run games or GPU-heavy apps alongside.
- **`/dev/shm` is 8 GB** (`shm_size`). vLLM moves tensors between its processes through it, and multi-GPU NCCL falls back to it without NVLink.
- **Keep models on the Linux side.** `./models` lives in the WSL filesystem; loading tens of GB from `/mnt/c` is much slower.
- **Ports** published by Docker Desktop are reachable from Windows at `127.0.0.1`, so the pages work in a Windows browser.
- **Compile caches** (vLLM's `torch.compile`, FlashInfer and Triton) are kept in `state/cache/`, so the second start of a model is much faster than the first.

## Configuration (`.env`)

Copy `.env.example` to `.env`. Every value is optional.

| Variable | Default | What it does |
| --- | --- | --- |
| `LLM_PORT` | `8888` | Host port for the API, `/server` and `/monitor` (`LLAMA_PORT` is the old name) |
| `WEBUI_PORT` | `3000` | Host port for Open WebUI |
| `OPENAI_API_KEY` | `sk-local` | Key Open WebUI sends to the engine |
| `SERVER_ADMIN_TOKEN` | empty | Require this bearer token for every change on `/server` |
| `HF_TOKEN` | empty | Hugging Face token (overrides the one saved on `/server`) |
| `VRAM_RESERVE_GB` | `1.5` | VRAM auto-tune leaves for Windows and other apps |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | `1` | Pinned host memory for vLLM under WSL 2 |
| `HW_METRICS` | `1` | `0` skips the hardware sampler behind the monitor |
| `AUTOSTART` | `0` | `1` starts `BACKEND` with the model below on every container start |
| `BACKEND` | `llama.cpp` | Engine for `AUTOSTART`: `llama.cpp` or `vllm` |
| `VLLM_MODEL` | empty | vLLM model (repo id) for `AUTOSTART` |
| `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION` | empty, `0.88` | vLLM memory settings for the first run / `AUTOSTART` |
| `TARGET_FILE`, `DRAFT_FILE`, `MMPROJ_FILE` | Qwen3.8 files | llama.cpp model, draft and projector under `models/` for the first run / `AUTOSTART` |
| `MMPROJ_OFFLOAD` | `1` | Projector on the GPU |
| `CTX_SIZE`, `SPEC_DRAFT_N_MAX`, `CACHE_TYPE_K`, `CACHE_TYPE_V`, `N_GPU_LAYERS`, `DRAFT_GPU_LAYERS`, `PARALLEL`, `UBATCH`, `BATCH`, `THREADS` | RTX 3090 profile | llama.cpp settings for the first run / `AUTOSTART` |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | `1.0`, `0.95`, `20` | Sampling defaults for the first run / `AUTOSTART` |
| `ALIAS` | `Qwen3.8-27B-DFlash2` | Model name llama.cpp reports in `/v1/models` |
| `LLAMA_WEBUI` | `0` | llama.cpp's built-in chat page on the first run (`1` = on) |
| **Build arguments** | | *(change, then rebuild)* |
| `LLAMA_CPP_REF` | `master` | llama.cpp branch, tag or commit to build |
| `VLLM_VERSION` | `0.31.0` | vLLM wheel version |
| `CUDA_ARCH` | `86` | GPU architecture(s): `86` RTX 30xx, `89` RTX 40xx, `120` RTX 50xx, or `86;89` |
| `GGML_CUDA_NCCL` | `ON` | NCCL for multi-GPU llama.cpp (`OFF` saves ~300 MB on one card) |
| `BUILD_JOBS` | `8` | Parallel compile jobs; lower it if the build runs out of memory |

The engine settings in `.env` only **seed** the first start (and `AUTOSTART`). After that, `/server` owns them in `state/server.json`; delete that file to reseed from `.env`.

## Where your data lives

| Path | Contents | Survives rebuilds |
| --- | --- | --- |
| `models/` | GGUF files, the Hugging Face hub cache (`models--owner--name`), `models/.sources.json` (which repo each flat GGUF came from), Hugging Face housekeeping in `models/.huggingface/` | yes (bind mount) |
| `state/server.json` | All `/server` settings: both engines, boot default, form layouts, page preferences | yes |
| `state/hf_token` | Hugging Face token (owner-only) | yes |
| `state/templates/` | Saved chat templates | yes |
| `state/cache/` | `--help` catalogs per engine version, vLLM / FlashInfer / Triton compile caches | yes |
| `state/updates.json`, `state/boot_attempts.json` | Last update check; boot-guard counter | yes |
| volume `inference-overdrive_open-webui-data` | Open WebUI chats, users, settings | yes (`docker compose down` keeps it; `down -v` deletes it) |

`models/`, `state/` and `.env` are git-ignored, and `.dockerignore` only sends `docker/` and `python/` to the build, so models and tokens never end up in the image.

## Updating llama.cpp or vLLM

1. Set the new version in `.env`: `LLAMA_CPP_REF=<tag or commit>` and/or `VLLM_VERSION=<version>`. The update badge on `/server` tells you the value.
2. Rebuild and restart:

   ```bash
   docker compose build llm-server && docker compose up -d
   ```

3. On the next start the settings check compares your saved settings with the new engine's `--help`. Renamed or removed flags are flagged before anything breaks, and the boot guard keeps a broken boot default from looping.

For vLLM, check that the new version still matches the CUDA 13.0 PyTorch build (`TORCH_INDEX_URL` in `docker/Dockerfile`) before rebuilding.

## Reference profile: Qwen3.8-27B + DFlash 2 on an RTX 3090

The image grew out of this setup, and the first-run defaults still describe it. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) runs with the [DFlash 2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) draft model. DFlash 2 is not a standalone LLM: it drafts a block of tokens in one pass and the 27B target verifies them, so output is identical (lossless) while decoding is faster.

| Piece | Repo / file | Size |
| --- | --- | --- |
| Target | `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16.5 GB |
| Draft | `incoai/Qwen3.8-27B-DFlash2-GGUF` → `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.14 GB |
| Vision projector | `unsloth/Qwen3.8-27B-GGUF` → `mmproj-F16.gguf` | 0.93 GB |

- All three fit on a 24 GB card with a 32K `q8_0` KV cache. Auto-tune finds about 64K, since only 16 of the 65 layers keep KV.
- `--spec-draft-n-max` is **4**, not the model card's 7. On 24 GB consumer cards, width 7 spends VRAM on draft state and is often slower.
- Sampling follows Qwen3.8: temperature 1.0, top-p 0.95, top-k 20, reasoning effort `xhigh`.
- Image turns don't get a DFlash speed-up (the drafter can't follow the vision span); text turns do.

**vLLM on the same card.** vLLM loads Hugging Face safetensors, not these GGUFs, and reserves `--gpu-memory-utilization` × VRAM up front. For a 27B model use a 4-bit AWQ/GPTQ repo (about 16 GB). DFlash 2 is llama.cpp-only; vLLM has its own `--speculative-config` methods (MTP, EAGLE, n-gram).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `CUDA driver version is insufficient for CUDA runtime version` | Update the Windows NVIDIA driver until `nvidia-smi.exe` reports CUDA 13.4+; don't mount `libcuda` by hand (see [WSL 2 notes](#wsl-2-notes)). |
| Open WebUI says *No models available* | Open WebUI keeps its connection in its volume after the first start. Set it to `http://llm-server:8888/v1` under *Admin → Settings → Connections*. Also check that a model is loaded on `/server`. |
| vLLM: `max_num_seqs (256) exceeds available Mamba cache blocks` | A hybrid model. Run *Suggest settings* (auto-tune caps `--max-num-seqs`), or set *Max concurrent sequences* lower by hand. |
| vLLM: `Using default Mamba SSU config. Performance might be sub-optimal!` | Harmless. vLLM ships tuned kernel settings only for datacenter GPUs; GeForce cards use generic ones at a small speed cost. |
| `TensorFloat32 tensor cores … available but not enabled` | Harmless. A generic PyTorch hint; bf16/fp16 models don't use float32 matrix multiplies. |
| Smoke test ✗ on tools | The tool-call parser doesn't match the model. Run *Suggest settings* or pick the parser that matches its chat template. |
| *won't start* above the form after an upgrade | A flag was renamed or removed in the new engine version. Hover it for details, fix or remove it, then Apply. |
| Autostart paused after failed boots | Fix the settings, then *Resume autostart* (or save a new boot default). |
| `docker compose up --build` fails pulling a base image with a digest / checksum mismatch | A Docker Desktop download glitch. Restart Docker Desktop and retry. |
| Out of memory while building | Lower `BUILD_JOBS` in `.env` (for example `4`). |
| Slow first load of a vLLM model | Normal: it compiles kernels once. They're cached in `state/cache/`, so later loads are faster. |

## Make targets and scripts

The make targets are shortcuts for the commands next to them. Ubuntu on WSL doesn't ship `make`; install it with `sudo apt install make`, or run the commands directly.

| Command | What it does |
| --- | --- |
| `make gpu-check` / `./scripts/check-gpu.sh` | Checks the NVIDIA driver from WSL and from a CUDA container |
| `make build` | `docker compose build llm-server` |
| `make up` | Starts the engine container and Open WebUI |
| `make down` | Stops and removes both containers (keeps models, state and the Open WebUI volume) |
| `make logs` | Follows the engine container's log |
| `make health` / `./scripts/health.sh` | Checks the manager, then the loaded model if there is one |

## Repository layout

```text
docker/Dockerfile                 # CUDA 13.4.2: llama.cpp CMake build + vLLM wheel, nginx, pages
docker/entrypoint.sh              # starts hw-metrics + nginx, then runs the server manager
docker/nginx.conf                 # :8888 front door: / redirect, /server/, /monitor/, engine proxy
docker-compose.yml                # llm-server + open-webui (project "inference-overdrive")
python/Server/manager.py          # engine process, settings, auto-tune, downloads, API (/server/api/)
python/Server/index.html          # control page (/server/)
python/Dashboard/monitor.html     # monitor (/monitor/)
python/Dashboard/hw-metrics.py    # GPU / CPU / RAM sampler (/monitor/hw)
scripts/check-gpu.sh              # host GPU / driver check
scripts/health.sh                 # container + model health
.env.example                      # every setting, documented
models/  state/                   # your data (git-ignored, created on first start)
```

## Tested on, and known limits

- **Tested on:** Windows 11, WSL 2 with Ubuntu 26.04, Docker Desktop (engine 29.8), an RTX 3090 Ti (24 GB) on driver 617.42.
- **Models exercised end to end:** Qwen3.8-27B + DFlash 2 + projector (llama.cpp), Qwen3-0.6B and NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 (vLLM, with tool calling and reasoning passing the smoke test).
- **Multi-GPU** paths have only been checked against simulated card lists (one GPU here). Without NVLink, NCCL goes through PCIe and host memory.
- **Images with DFlash enabled** haven't been re-tested since llama.cpp's [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) merged. If image requests fail, set *Draft model* to none for vision work.
- **The `CUDA_ARCH` default** targets RTX 30xx; other cards need it set before building.
