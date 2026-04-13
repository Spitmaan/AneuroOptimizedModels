# Phase 8 — ANeurologic Demo Server

**Goal:** Branded test server where users can interact with ANeurologic AI models via web UI and CLI.
**Date:** 2026-04-13
**Hardware:** Jetson Orin Nano 8 GB
**URL:** https://lab.aneurologics.com/

---

## Architecture

```
Internet → Cloudflare Tunnel (lab.aneurologics.com) → nginx:80 → Open WebUI:8080 → Ollama:11434
```

All models served through Ollama. Open WebUI provides chat UI with login, conversation history, and image upload. All models visible to all authenticated users (`BYPASS_MODEL_ACCESS_CONTROL=true`).

## Branded Models

| Aneurologic Name | Base Model | Category | Speed | Multimodal | Status |
|-----------------|-----------|----------|------:|:----------:|--------|
| **Aneuro Spark** | Qwen2.5-0.5B Q4_K_M | SLM | ~94 t/s GPU | No | DEPLOYED |
| **Aneuro Synapse** | LFM2.5-1.2B IQ4_XS | SLM | ~8 t/s CPU | No | DEPLOYED |
| **Aneuro Cortex** | Llama-3.2-1B IQ4_XS | SLM | ~11 t/s CPU | No | DEPLOYED |
| **Aneuro Vision** | Gemma 4 E2B Q4_K_M (Ollama) | VLM | ~10 t/s CPU | Image | DEPLOYED |
| Aneuro World | V-JEPA 2.1 | World Model | 74 FPS | Video | PLANNED (Gradio) |
| Aneuro Spike | Spikformer | SNN Vision | 61.7 FPS | Image | PLANNED (Gradio) |

All models respond with ANeurologic identity. System prompts enforce branding.

**Note on speeds:** Aneuro Spark is the only model using GPU (small enough at 397 MB to fit in IOVM alongside Open WebUI). Cortex and Synapse run on CPU due to IOVM constraints. Vision runs CPU-only due to 4.65B total params.

## VLM/VLA Assessment

### What was tested across all phases

| Model | Phase | Type | Speed | Servable via Ollama? | Status |
|-------|------:|------|------:|:-------------------:|--------|
| **Gemma 4 E2B** | 7 | VLM (img+audio) | ~10 t/s CPU | Yes (Ollama registry) | DEPLOYED as Aneuro Vision |
| LFM2-VL-450M | 1 | VLM (img) | 42.1 t/s | No (transformers only) | Not servable |
| LFM2-VL-1.6B | 1 | VLM (img) | 24.8 t/s | No (transformers only) | Not servable |
| VILA 1.5-3B | 1 | VLM (img) | 12.4 t/s | Needs nano_llm container | Container deleted |
| Qwen2-VL-7B | 1 | VLM (img) | 0.56 t/s | Too slow, OOM | Not viable |
| Pi-0.5 (OpenPi) | 1 | VLA (robot) | 23.7 Hz | No (TensorRT custom) | Container deleted |

**Only Gemma 4 E2B is servable through Ollama+Open WebUI** with native image upload support. The LFM-VL and VILA models require specialized containers (transformers, nano_llm) that don't integrate with the Ollama/Open WebUI chat stack. Pi-0.5 (VLA) requires a custom TensorRT pipeline.

## Deployed Components

### Ollama + Models
- Ollama v0.20.6 on Jetson
- 3 SLM models from local GGUFs with branded Modelfiles (custom system prompts + chat templates)
- 1 VLM model pulled from Ollama registry (`gemma4:e2b`, 7.2 GB, includes multimodal projector)
- Aneuro Vision aliased from `gemma4:e2b` with ANeurologic system prompt

### Open WebUI
- Docker: `ghcr.io/open-webui/open-webui:main` (v0.8.12)
- Port 8080 via `--network host`
- Admin: `admin@aneurologic.com` / `aneuro2026`
- Key env vars:
  - `WEBUI_NAME='ANeurologic AI'`
  - `BYPASS_MODEL_ACCESS_CONTROL=true` (all users see all models)
  - `OLLAMA_BASE_URL=http://localhost:11434`
- Branding patches applied via `patch_webui.sh` (removes "(Open WebUI)" suffix, replaces favicons/logos, replaces text in frontend JS)

### Nginx
- Config: `/etc/nginx/sites-available/aneurologic`
- Routes: `/` → Open WebUI, `/demo/` → Gradio (future)
- WebSocket support, no-cache headers for static files

### Cloudflare Tunnel
- Hostname: `lab.aneurologics.com` → `http://localhost:8080`
- Configured in Cloudflare Zero Trust dashboard (token-based tunnel)

## Key Technical Decisions

### Model visibility fix
Open WebUI v0.8.12 hides Ollama models from non-admin users by default. The `get_filtered_models()` function in `utils/models.py` checks `BYPASS_MODEL_ACCESS_CONTROL` env var. Setting it to `true` in docker run makes all models visible to all authenticated users. No code patches or DB manipulations needed.

### Branding
The "(Open WebUI)" suffix is appended by `env.py` line 130: `WEBUI_NAME += ' (Open WebUI)'`. Patched by `patch_webui.sh` on every container restart. Frontend text replacement also done via sed on all JS/HTML build files.

### Memory constraints
Only Aneuro Spark (397 MB) fits in GPU IOVM alongside Open WebUI. All other models run CPU-only. Gemma 4's 4.65B total params (262K vocab + PLE) make GPU offload impossible at any quantization.

## Files

```
aneurologic-server/   (GitHub: Spitmaan/aneurologic-server)
├── anctl                    # Model swap controller (for manual switching)
├── assets/
│   ├── ANEUROLOGIC_ICO.png  # Logo
│   └── aneuro_favicon.svg   # SVG favicon
├── modelfiles/
│   ├── Modelfile.cortex     # Llama-3.2-1B (SLM)
│   ├── Modelfile.spark      # Qwen-0.5B (SLM)
│   ├── Modelfile.synapse    # LFM-1.2B (SLM)
│   └── Modelfile.vision     # Gemma 4 E2B (VLM) — Ollama alias
├── patch_webui.sh           # Applies branding to Open WebUI container
├── save_image.sh            # Docker image snapshot utility
├── setup.sh                 # One-command full deployment
└── start_vision.sh          # Legacy llama-server launcher (before Ollama)
```

## Access

- **Web UI:** https://lab.aneurologics.com/ (Cloudflare tunnel)
- **CLI:** `ollama run aneuro-cortex "your question"` on Jetson
- **API:** `curl https://lab.aneurologics.com/api/chat/completions` (requires auth token)

## Remaining (Future)

- Gradio app for V-JEPA (Aneuro World) + Spikformer (Aneuro Spike) specialized demos
- Pre-built test example presets in Open WebUI
- CLI documentation for external users
