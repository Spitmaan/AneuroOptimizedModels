# Phase 8 — ANeurologic Demo Server

**Goal:** Branded test server where users can interact with ANeurologic AI models via web UI and CLI.
**Date:** 2026-04-13
**Hardware:** Jetson Orin Nano 8 GB

---

## Architecture

```
Internet → Cloudflare Tunnel → nginx:80 → Open WebUI:8080 → Ollama:11434
                                                              (SLM on GPU)
                                                           or llama-server:11434
                                                              (VLM on CPU)
```

Model swap controller (`anctl`) manages which backend is active. Only one runs at a time (8 GB memory constraint).

## Branded Models

| Aneurologic Name | Base Model | Category | Speed | Status |
|-----------------|-----------|----------|------:|--------|
| **Aneuro Cortex** | Llama-3.2-1B IQ4_XS | SLM | ~60 t/s GPU | DEPLOYED |
| **Aneuro Synapse** | LFM2.5-1.2B IQ4_XS | SLM | ~65 t/s GPU | DEPLOYED |
| **Aneuro Spark** | Qwen2.5-0.5B Q4_K_M | SLM | ~94 t/s GPU | DEPLOYED |
| **Aneuro Vision** | Gemma 4 E2B + mmproj | VLM | ~11 t/s CPU | DEPLOYED |
| Aneuro World | V-JEPA 2.1 | World Model | 74 FPS | PLANNED (Gradio) |
| Aneuro Spike | Spikformer | SNN Vision | 61.7 FPS | PLANNED (Gradio) |

All models respond with ANeurologic identity when asked "Who are you?"

## Deployed Components

### Stage 1: Ollama + SLM Models (COMPLETE)
- Ollama v0.20.6 installed on Jetson
- 3 Modelfiles with branded system prompts + correct chat templates (Llama-3.1 / Qwen)
- Models registered: `aneuro-cortex`, `aneuro-synapse`, `aneuro-spark`
- GPU offload working, ctx=2048

### Stage 2: Open WebUI (COMPLETE)
- Docker: `ghcr.io/open-webui/open-webui:main`
- Port 8080, connected to Ollama at localhost:11434
- Admin: `admin@aneurologic.com` / `aneuro2026`
- Conversation persistence via SQLite at `/data/open-webui/`
- Branding: `WEBUI_NAME="ANeurologic AI"`

### Stage 3: Aneuro Vision (COMPLETE)
- Startup script: `/home/spitman/aneurologic-server/start_vision.sh`
- Gemma 4 E2B IQ4_XS + mmproj, CPU-only, t=4, port 11434
- Requires stopping Ollama first (memory constraint)
- Image upload → description workflow proven in Phase 7

### Stage 4: Model Swap Controller (COMPLETE)
- `/usr/local/bin/anctl` — symlink to `/home/spitman/aneurologic-server/anctl`
- Commands: `anctl load cortex|synapse|spark|vision`, `anctl status`, `anctl unload`
- Handles mutual exclusion between Ollama and llama-server

### Stage 6: Nginx (COMPLETE)
- Config: `/etc/nginx/sites-available/aneurologic`
- Routes `/` → Open WebUI, `/demo/` → Gradio (future)
- WebSocket support for streaming responses

## Access

- **Web UI:** http://localhost:8080 (local) or via Cloudflare tunnel (external)
- **CLI:** `ollama run aneuro-cortex "your question"` or `curl` against API
- **API:** `http://localhost:8080/api/chat/completions` (OpenAI-compatible, requires auth token)

## Files on Jetson

```
/home/spitman/aneurologic-server/
├── modelfiles/
│   ├── Modelfile.cortex    (Llama-3.2-1B branded)
│   ├── Modelfile.synapse   (LFM2.5-1.2B branded)
│   └── Modelfile.spark     (Qwen2.5-0.5B branded)
├── anctl                   (model swap controller)
└── start_vision.sh         (Aneuro Vision launcher)
```

## Remaining (Future)

- Stage 5: Gradio app for V-JEPA + Spikformer specialized demos
- Stage 7: CLI documentation
- Stage 8: Pre-built test example presets in Open WebUI
