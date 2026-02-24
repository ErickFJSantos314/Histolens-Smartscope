# HistoLens SmartScope AI

HistoLens SmartScope AI is a layered, cloud-edge pathology assistant for hands-free microscope workflows, combining speech interaction, intent-aware AI routing, remote medical vision inference, and spoken clinical feedback.

## Demo

<img width="974" height="482" alt="image" src="https://github.com/user-attachments/assets/a6e530ad-8301-47ba-b5f3-737bbe106358" />


## Architecture Overview

The system is organized in five layers for clarity, scalability, and cost control:

1. **Capture Layer**
   - `histolens.py`: real-time camera capture (original product concept, connected to microscope camera)
   - `microscopyo.py`: digital twin simulation over `.svs` whole-slide images (engineering fallback when hardware is unavailable)

2. **Cognitive Layer (`maestro.py`)**
   - Gemini routes each utterance into `basic_chat`, `image_interaction`, `technical_doubt`, or `heavy_diagnostic`.

3. **Vision Layer (`medgemma.py` + remote notebook API)**
   - MedGemma 1.5 4B runs remotely (Kaggle/Colab + Ngrok).
   - Local client sends image payloads and receives morphology findings.
   - Kaggle notebook used for remote API: [HistoLens SmartScope MedGemma API](https://www.kaggle.com/code/erickfjsantos/histolens-smartscope-medgemma-processing-api)

4. **Synthesis Layer (`maestro.py`)**
   - Gemini converts findings into concise clinician-facing responses suitable for TTS.

5. **I/O Layer (`stt.py`, `tts.py`)**
   - STT with Google medical models
   - TTS with Google Cloud Text-to-Speech

## Validation Strategy (Hackathon Engineering)

`histolens.py` is the original real-time product vision (camera-linked microscope workflow).  
`microscopyo.py` and the remote Kaggle/Colab notebook were engineering alternatives used to validate the full end-to-end product under constraints such as limited hardware and lack of laboratory equipment.

## Repository Structure

- `histolens.py` — real-time camera client (`cv2.VideoCapture`)
- `microscopyo.py` — digital twin microscope simulator over `.svs`
- `maestro.py` — routing and orchestration
- `medgemma.py` — remote MedGemma API client
- `stt.py` — microphone capture and transcription
- `tts.py` — speech synthesis
- `run_histolens.bat` — one-click Windows launcher (real-time mode)
- `run_histolens.sh` — one-click Linux/macOS launcher (real-time mode)

## Prerequisites

- Python 3.10+ (3.11+ recommended)
- Conda (Miniconda/Anaconda)
- Google Cloud credentials with Speech + TTS enabled
- Active remote vision endpoint (`/analyze`) via Kaggle/Colab + Ngrok
- OpenSlide runtime (required only for `microscopyo.py`)

### OpenSlide on Windows (important)

Download OpenSlide Windows binaries from https://openslide.org/, extract them, and add the `bin` folder to your system `PATH`.

## Environment Variables

Required:

- `GOOGLE_API_KEY`
- `REMOTE_VISION_API_URL`
- `GOOGLE_APPLICATION_CREDENTIALS`

Optional:

- `REMOTE_VISION_API_TOKEN`
- `HF_TOKEN`
- `MEDGEMMA_VISION_MAX_SIDE`
- `MEDGEMMA_VISION_MAX_TOKENS`
- `HISTOLENS_CAMERA_INDEX`
- `HISTOLENS_FRAME_WIDTH`
- `HISTOLENS_FRAME_HEIGHT`
- `HISTOLENS_SVS_PATH` (for `microscopyo.py`)

> Backward compatibility: `COLAB_API_URL` and `COLAB_API_TOKEN` are still accepted.

## Installation

```bash
git clone <your-repo-url>
cd HistoLens
conda create -n histolens python=3.10 -y
conda activate histolens
pip install -r requirements.txt
```

## Create `.env`

Create a `.env` file in the root directory and add the following variables:

```bash
copy .env.example .env
```

(On Linux/macOS: `cp .env.example .env`)

```env
GOOGLE_API_KEY=your_google_api_key
REMOTE_VISION_API_URL=https://your-ngrok-url.ngrok-free.dev/analyze
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/service-account.json

# Optional
REMOTE_VISION_API_TOKEN=
HF_TOKEN=
MEDGEMMA_VISION_MAX_SIDE=512
MEDGEMMA_VISION_MAX_TOKENS=128
HISTOLENS_CAMERA_INDEX=0
HISTOLENS_FRAME_WIDTH=1280
HISTOLENS_FRAME_HEIGHT=720
HISTOLENS_SVS_PATH=archives/your-slide-file.svs
```

## Running

> Note: Make sure to run the MedGemma Kaggle notebook first to get your active Ngrok URL, then update your `.env` file before launching the local client.

### Real-time mode (original concept)

- Windows: `run_histolens.bat`
- Linux/macOS:

```bash
chmod +x run_histolens.sh
./run_histolens.sh
```

### Digital twin mode (`microscopyo.py`)

If `archives/` has no `.svs` files, add your own slide and set the path:

```bash
set HISTOLENS_SVS_PATH=archives\\your-slide-file.svs
python microscopyo.py
```

(`export HISTOLENS_SVS_PATH=...` on Linux/macOS)

Controls in `microscopyo.py`:

- `W/A/S/D` move viewport
- `Q/E` zoom out/in
- `R` start/stop recording
- `X` exit

Controls in `histolens.py`:

- `R` start/stop recording
- `X` exit

## Hackathon Positioning

HistoLens demonstrates a practical **Cloud-Edge AI strategy** for pathology:

- hardware-agnostic deployment path (digital twin to real camera)
- remote GPU inference with local lightweight client
- deterministic intent routing for cost control
- clinician-friendly audio output for hands-free usage

## Meet the Team

- [Erick](https://www.linkedin.com/in/erick-francisco-28a756274/)
- [David](https://www.linkedin.com/in/david-antonio-29807434a/)
- [Samara](https://www.linkedin.com/in/samara-souza-szp/)
- [Maria](https://www.linkedin.com/in/maria-clara-paix%C3%A3o-2061362a0/)
- [Nayara](https://www.linkedin.com/in/nayara-lorena-ramos-santos-73b904274/)
