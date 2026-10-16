# Ekphra FLUX.2 Comic Studio

Ekphra turns prose into an editable comic storyboard, then sends approved
individual panels to a separately running ComfyUI instance. The FastAPI
environment does not load model weights, Diffusers, PyTorch, or GPU runtimes.

## Local services

1. Start LM Studio's OpenAI-compatible server and configure `.env` from
   `.env.example`. `OPENAI_BASE_URL` and `OPENAI_CHAT_MODEL` drive storyboard
   creation.
2. Start ComfyUI separately at `COMFYUI_BASE_URL` (default
   `http://127.0.0.1:8188`). ComfyUI owns FLUX.2, Krea, custom nodes, and GPU
   inference.
3. Configure `renderer_profiles.json` as described in
   [COMFYUI_PROFILES.md](COMFYUI_PROFILES.md). Placeholder profiles intentionally
   cannot render until their workflow exports and real node bindings are added.

## App startup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python api.py
```

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run dev -- --host 127.0.0.1
```

The workflow is deliberately two-stage: create a storyboard, inspect/edit its
JSON, then choose a renderer profile and render. Vision QA is disabled by
default and never triggers automatic rerenders.
