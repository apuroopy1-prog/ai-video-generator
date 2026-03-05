# AI Content Generator

> **Text-to-Image, Text-to-Video, and Text-to-Speech in a single pipeline** — runs on Apple Silicon (MPS), NVIDIA GPU (CUDA), or free Google Colab T4.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS_+_CUDA-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)](https://gradio.app)
[![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?logo=google-colab)](https://colab.research.google.com)

---

## What It Does

Generate rich media from text prompts — images, videos, and narrated audio — all from one Gradio interface. Designed to run free on Google Colab (T4 GPU) or locally on Mac M1/M2/M3 with Apple MPS acceleration.

---

## Capabilities

| Mode | Model | Speed | Quality |
|---|---|---|---|
| **Text-to-Image** | SDXL-Turbo (Stability AI) | ~2s (MPS) | High |
| **Text-to-Video** | ModelScope text-to-video | ~30s (T4) | Medium |
| **Text-to-Speech** | Bark (Suno) | ~10s | High quality |
| **Text-to-Speech** | Edge TTS (Microsoft) | ~1s | Fast, free |
| **Combined** | Video + Audio narration | — | Full pipeline |

---

## Architecture

```
Text Prompt
     │
     ├──▶ ImageGenerator  ──▶ SDXL-Turbo (stabilityai/sdxl-turbo)
     │                         AutoPipelineForText2Image
     │                         Apple MPS / CUDA / CPU
     │
     ├──▶ VideoGenerator  ──▶ ModelScope (damo-vilab/text-to-video-ms-1.7b)
     │                         TextToVideoSDPipeline
     │
     ├──▶ AudioGenerator  ──▶ Bark (suno/bark) — high quality
     │                    ──▶ Edge TTS          — fast, free
     │
     └──▶ Pipeline        ──▶ Combined: video + audio → final output
```

---

## Quick Start

### Option 1: Google Colab (Recommended — Free T4 GPU)

1. Open [`notebooks/AI_Content_Generator_MVP.ipynb`](notebooks/AI_Content_Generator_MVP.ipynb)
2. Upload to [Google Colab](https://colab.research.google.com)
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells → Gradio public URL is generated automatically

### Option 2: Local (Mac M1/M2/M3)

```bash
git clone https://github.com/apuroopy1-prog/ai-video-generator.git
cd ai-video-generator

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python app.py
# Open http://localhost:7860
```

---

## Project Structure

```
ai-content-generator/
├── app.py                          # Main Gradio web UI
├── quick_start.py                  # Minimal Python API example
├── requirements.txt
├── notebooks/
│   └── AI_Content_Generator_MVP.ipynb   # Colab notebook
├── src/
│   ├── image_generator.py          # SDXL-Turbo text-to-image
│   ├── video_generator.py          # ModelScope text-to-video
│   ├── audio_generator.py          # Bark + Edge TTS
│   └── pipeline.py                 # Combined video+audio pipeline
├── configs/                        # Model config files
├── huggingface-space/              # HuggingFace Spaces deployment
└── outputs/                        # Generated files (gitignored)
```

---

## Python API

```python
from src.image_generator import ImageGenerator
from src.video_generator import VideoGenerator
from src.audio_generator import AudioGenerator

# Generate image
img_gen = ImageGenerator()
image = img_gen.generate("A futuristic city at sunset, cyberpunk style")

# Generate video
vid_gen = VideoGenerator()
video = vid_gen.generate("A robot walking through a forest")

# Generate speech
audio_gen = AudioGenerator()
audio = audio_gen.generate("Welcome to the future of AI content creation")
```

---

## Hardware Requirements

| Hardware | Image | Video | Audio |
|---|---|---|---|
| **Apple M1/M2/M3** (MPS) | ✅ Fast | ✅ Slow | ✅ |
| **NVIDIA GPU** (CUDA) | ✅ Fast | ✅ Fast | ✅ |
| **Google Colab T4** | ✅ Fast | ✅ Fast | ✅ |
| **CPU only** | ✅ Very slow | ⚠️ Very slow | ✅ |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Text-to-Image** | Diffusers, SDXL-Turbo, AutoPipelineForText2Image |
| **Text-to-Video** | ModelScope, TextToVideoSDPipeline |
| **Text-to-Speech** | Bark (suno), Edge TTS |
| **UI** | Gradio |
| **Deep Learning** | PyTorch (MPS + CUDA) |
| **Models** | HuggingFace Hub |

---

## Built By

**Apuroop Yarabarla** — AI/ML Engineer & AI Product Owner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-apuroopyarabarla-0077B5?logo=linkedin)](https://linkedin.com/in/apuroopyarabarla)
[![GitHub](https://img.shields.io/badge/GitHub-apuroopy1--prog-181717?logo=github)](https://github.com/apuroopy1-prog)
