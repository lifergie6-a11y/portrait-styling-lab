# Portrait Styling Lab

**100% SFW** modular open-source portrait makeup & hairstyle editing pipeline for fast early validation and foundation model A/B selection (`FireRed-Image-Edit 1.1`, `Qwen-Image-2.1`, `Qwen-Image-Edit-2511`, `FLUX 3 Image Klein`).

Designed to be **completely decoupled from Colab notebooks**: all business logic lives in `src/portrait_styling/`, while Colab is only a thin GPU runner or a remote `colabllm` endpoint.

---

## Architecture (4-Stage Minimal Pipeline)

```text
[Input] Portrait Photo + Structured Expert Diagnosis JSON
   │
   ▼
1. Preprocess & Face Anchor (src/portrait_styling/preprocess.py)
   - Face-aware 3:4 / 1:1 framing with generous headroom & shoulder margin
   - Extracts ArcFace (InsightFace buffalo_l) 512-d identity embedding
   │
   ▼
2. Visual Prompt Translator (src/portrait_styling/prompt_translator.py)
   - Converts expert diagnosis fields (face shape, facial thirds, skin undertone, hair & makeup plan)
     into concrete, SFW visual editing instructions
   - Supports zero-cost deterministic rule templates + optional OpenAI-compatible LLM refinement
   │
   ▼
3. Foundation Model Inference (src/portrait_styling/generators/)
   - Mode A (Standalone Colab / GPU via diffusers):
       * FireRedTeam/FireRed-Image-Edit-1.1  (Apache 2.0, 20B)
       * Qwen/Qwen-Image-2.1 & Turbo         (7B DiT + Qwen3-VL)
       * Qwen/Qwen-Image-Edit-2511           (Apache 2.0, 20B)
   - Mode B (Local Mac/PC -> Remote Colab via colabllm CLI):
       * colabllm:qwen-image  (Qwen-Image-2.1 on G4/A100/H100)
       * colabllm:flux-klein  (FLUX 3 Image Klein on G4/L4/H100)
   - Best-of-N multi-seed sampling in a single editing pass (avoids multi-pass VAE drift)
   │
   ▼
4. Auto-Scoring & Blind A/B HTML Dashboard (evaluator.py + report.py)
   - Hard ID Preservation Gate: InsightFace ArcFace cosine similarity (default >= 0.65)
   - Optional VLM Rule-Adherence Scorer (any OpenAI-compatible vision model, e.g. Qwen-VL / colabllm qwen-flash)
   - Generates a self-contained static HTML dashboard with Blind A/B toggle & expert rating JSON export
```

---

## Supported Models (`configs/models.yaml`)

| Alias | Backend | Model ID | License | Recommended Colab GPU | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `firered-1.1` | `diffusers` | `FireRedTeam/FireRed-Image-Edit-1.1` | Apache 2.0 | G4 (96G BF16) / A100 / L4 (4-bit NF4) | Primary candidate for Asian portrait ID & makeup aesthetics |
| `qwen-2.1` | `diffusers` | `Qwen/Qwen-Image-2.1` | Qwen Research | G4 (96G BF16) / A100-40G (offload/FP8) | Native multi-ref editing + strong Chinese prompt adherence |
| `qwen-2.1-turbo` | `diffusers` | `Qwen/Qwen-Image-2.1-Turbo` | Qwen Research | G4 / A100 | 8-step fast iteration variant |
| `qwen-edit-2511` | `diffusers` | `Qwen/Qwen-Image-Edit-2511` | Apache 2.0 | G4 / A100 / L4 (4-bit NF4) | Apache 2.0 multi-image editing with low drift |
| `colabllm:qwen-image` | `colabllm` | `Qwen/Qwen-Image-2.1` | Apache 2.0 / Research | G4 / A100 / H100 (via `colabllm`) | Run from local laptop against remote Colab runtime |
| `colabllm:flux-klein` | `colabllm` | `black-forest-labs/flux-3-image-klein` | FLUX Kommunity | L4 (`fp8r`) / G4 (`bf16`) | Fast 16-step realistic skin texture reference |

---

## Quick Start

### Option A: Standalone Google Colab (Recommended for `FireRed-1.1` vs `Qwen-2.1` A/B)

1. Open [`notebooks/colab_ab_benchmark.ipynb`](notebooks/colab_ab_benchmark.ipynb) in Google Colab.
2. **(Zero-Waste Tip)** First attach a **free CPU runtime** to pre-download weights into your Google Drive cache (`scripts/colab_cache_models.py`), then switch to **G4 / A100 / L4 GPU**.
3. Put 10–30 consented test portraits and a `manifest.json` in `MyDrive/portrait-styling-lab/dataset/` (see `scripts/create_sample_dataset.py`).
4. Run the benchmark cell:
   ```bash
   pstyle benchmark \
     --manifest /content/drive/MyDrive/portrait-styling-lab/dataset/manifest.json \
     --models firered-1.1,qwen-2.1 \
     --num-seeds 2 \
     --output-dir /content/drive/MyDrive/portrait-styling-lab/runs/ab_v0
   ```
5. Download `ab_v0/report.html` and open it locally in any browser (or send the static HTML to styling experts for blind evaluation).

### Option B: Local Machine + `colabllm` Remote Acceleration

```bash
git clone https://github.com/lifergie6-a11y/portrait-styling-lab.git
cd portrait-styling-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[eval]"

# 1. Start your Colab runtime via colabllm (e.g., Qwen-Image-2.1 on G4 or A100)
colabllm qwen-image up

# 2. Initialize a dataset manifest from a local folder of selfies
python3 scripts/create_sample_dataset.py --images-dir ./my_test_selfies --out ./dataset/manifest.json

# 3. Run single-image smoke test
pstyle run-one \
  --image ./my_test_selfies/sample01.jpg \
  --diagnosis configs/sample_diagnosis.json \
  --model colabllm:qwen-image \
  --num-seeds 2 \
  --output-dir ./outputs/smoke

# 4. Run full A/B benchmark & generate static HTML comparison report
pstyle benchmark \
  --manifest ./dataset/manifest.json \
  --models colabllm:qwen-image,colabllm:flux-klein \
  --num-seeds 2 \
  --output-dir ./outputs/ab_v0
```

---

## SFW & Google Colab ToS Compliance Guardrails

To keep your Google Colab account safe during early R&D:

1. **Strictly SFW & No Face-Swapping Tools**: This repo only uses official SFW foundation editing models (`FireRed-Image-Edit`, `Qwen-Image`, `FLUX`) and standard face-recognition embeddings (`InsightFace buffalo_l`) for identity preservation scoring. It never bundles deepfake/face-swap tools (`DeepFaceLab`, `Roop`, `FaceFusion`, `inswapper`) or NSFW checkpoints/LoRAs.
2. **Offline Batch Research Only (No Public Web Services on Colab)**: Google Colab prohibits serving public web apps / third-party APIs or unauthorized reverse tunnels on managed runtimes. Always run batch evaluation in Colab and share the generated **static `report.html`** with styling experts rather than exposing a live Colab Gradio/FastAPI URL to external users.
3. **Consent & Privacy**: Never commit real user selfies to this public repository (`.gitignore` excludes `dataset/`, `outputs/`, `*.jpg`, `*.png`). Use consented test portraits or licensed stock portraits for benchmarking.
