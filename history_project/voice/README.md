# Voice workflow: Qwen3-TTS (text-to-speech) + Seed-VC (voice-to-voice)

Facts and measurements verified on 2026-10-19. See also the [FINDINGS entry](../../docs/FINDINGS.md).

The project needs one consistent, **fully synthetic** voice for the narrator character, produced two ways:

1. **Text-to-speech (TTS):** Qwen3-TTS reads a written script in the narrator voice (voice clone).
2. **Voice-to-voice (V2V):** Seed-VC converts a human performance recording to the narrator voice, for lines that need acted timing or emphasis.

Only use reference voices you have the rights to: a voice you designed yourself, or a synthetic voice whose license allows this use. Do not clone real people. See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## Hardware used

| Box | Use |
|---|---|
| Windows GPU box, 12 GB (RTX 4070 Super) | all measurements below |
| Linux GPU box | optional; same steps with Linux venv paths |

Placeholders used below: `<TTS_DIR>` (Qwen3-TTS venv + models), `<SEEDVC_DIR>` (Seed-VC clone + venv), `<PROJECT_DIR>` (your scripts, inputs and outputs). Put real paths in a gitignored `config.local.yaml` (template: [config.example.yaml](config.example.yaml)) or a gitignored `.env`. Audio files (`*.wav`, `*.mp3`, `*.m4a`, `*.flac`) are gitignored.

---

## 1. Text-to-speech: Qwen3-TTS

**License:** Apache-2.0 (code and the models below).

### Environment
- Python 3.12, `torch 2.8.0+cu128`, `transformers 4.57.3`, `qwen-tts 0.1.1`
- Attention: `sdpa` (flash-attn is not needed and was not installed)

```bash
cd <TTS_DIR>
python3.12 -m venv venv            # Windows: py -3.12 -m venv venv
# activate the venv, then:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install qwen-tts==0.1.1 transformers==4.57.3 soundfile numpy
```

### Models (download into `<TTS_DIR>/models/`)

| Model | Role |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning (`generate_voice_clone`) |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | design a new voice from a text description (`generate_voice_design`, `instruct=`) |
| `Qwen/Qwen3-TTS-Tokenizer-12Hz` | speech tokenizer |

The scripts set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so download the models first.

### Workflow
1. **Get a reference voice.** Either:
   - **design one** with the VoiceDesign model: `model.generate_voice_design(text=..., instruct="calm, warm adult narrator, medium pace", language="English")`, and keep the best take; or
   - **use a reference clip of a synthetic voice you have rights to.**
2. **Trim the reference** to about 15–20 s, cutting at a quiet point so no word is cut off. Use a mono WAV.
3. **Clone** with the Base model's `generate_voice_clone`:
   - **x-vector-only mode** (`QWEN_TTS_XVEC_ONLY=1`, i.e. `x_vector_only_mode=True`) uses only the speaker embedding and needs no transcript.
   - **ICL / full mode** (`QWEN_TTS_XVEC_ONLY=0` plus `QWEN_TTS_REF_TEXT` set to the exact transcript of the clip) matches the reference voice and delivery more closely.
4. **Test the clone** with [`scripts/tts_clone_test.py`](scripts/tts_clone_test.py) on one short passage.
5. **Read a full script** with [`scripts/tts_script_runner.py`](scripts/tts_script_runner.py).

### Script runner behaviour (`scripts/tts_script_runner.py`)
- Splits the text file on blank lines and runs **one generation per paragraph**.
- Uses a **fixed seed per paragraph** (`BASE_SEED + paragraph index`). Reruns are reproducible, and one paragraph can be redone without changing the others.
- Builds the **voice-clone prompt once** (`create_voice_clone_prompt`) and reuses it for every paragraph.
- Joins paragraphs with **0.4 s of silence** and writes a **24 kHz** 16-bit WAV.
- **Sanity check:** each paragraph must come out at **1.6 to 4.0 words per second**. A take outside that range (usually truncated or run-on) is regenerated **once** with a new seed (`BASE_SEED + 1000 + index`) and the better take is kept. A paragraph still out of range is flagged `retried-STILL-SUSPECT` so you can listen to it.

```bash
export QWEN_TTS_MODEL_DIR=<TTS_DIR>/models/Qwen3-TTS-12Hz-1.7B-Base
export QWEN_TTS_REF_WAV=<PROJECT_DIR>/inputs/voice_ref.wav
export QWEN_TTS_XVEC_ONLY=1            # or 0 together with QWEN_TTS_REF_TEXT="exact transcript"
python scripts/tts_clone_test.py <PROJECT_DIR>/outputs/clone_test.wav
python scripts/tts_script_runner.py <PROJECT_DIR>/scripts/episode_01.txt <PROJECT_DIR>/outputs/episode_01.wav
```

### Measured (Windows GPU box, 12 GB, RTX 4070 Super, 1.7B-Base, bf16, sdpa)
| Metric | Value |
|---|---|
| Model load | 2.6 s |
| Generation speed | ~1.36–1.46× realtime (generation time / audio length; e.g. 85 s to generate 62.6 s of audio) |
| Peak VRAM | ~4.2–4.3 GB |

### Prosody and delivery limits (checked in the qwen-tts 0.1.1 source)
- The **Base clone path has no `instruct`, speed or emotion parameters.** Only the instruct-trained models (VoiceDesign, CustomVoice) accept `instruct`, and CustomVoice only offers its preset speakers.
- Levers that do work for a cloned voice:
  - **Reference clip style:** the delivery follows the reference. Keep separate reference clips (for example one calm, one excited) and pick one per scene.
  - **Punctuation:** commas, full stops, ellipses and question marks shape pauses and intonation.
  - **Sampling:** pass `temperature` / `subtalker_temperature` as generate kwargs (default 0.9 each), generate a few takes and pick the best.
  - **Post time-stretch:** 0.9–1.15× (pitch-preserving) to adjust pace.
- **Fine-tuning:** `finetuning/sft_12hz.py` in the Qwen3-TTS repo turns a voice into a `custom_voice` speaker. Whether `instruct` then works on that speaker is **unverified**.

---

## 2. Voice-to-voice: Seed-VC

**Repo:** https://github.com/Plachtaa/seed-vc, pinned to commit `51383efd921027683c89e5348211d93ff12ac2a8`. The repo was **archived on 2025-04-20** and gets no further updates.
**License:** **GPL-3.0** for both code and weights. Keep it in its own folder and venv, and check the GPL obligations before distributing anything built on it.

### Install (own venv)
- Python 3.12.10, `torch==2.8.0+cu128`, `torchaudio==2.8.0+cu128`, `torchvision==0.23.0` (instead of the torch lines in `requirements.txt`).
- Everything else from `requirements.txt` as pinned, including `numpy 1.26.4` and `transformers 4.46.3`.
- **Windows only:** `resemblyzer` depends on `webrtcvad`, which does not build there. Install `webrtcvad-wheels` instead, and install `resemblyzer` with `--no-deps`.
- **v2 only:** needs ffmpeg. `pip install imageio-ffmpeg` inside the venv and put its binary on PATH.
- The total download is about **8 GB**: ≈3.9 GB of packages plus ≈4.2 GB of models, which download on the first run.

```bash
git clone https://github.com/Plachtaa/seed-vc <SEEDVC_DIR>
cd <SEEDVC_DIR> && git checkout 51383efd921027683c89e5348211d93ff12ac2a8
python3.12 -m venv venv     # Windows: py -3.12 -m venv venv
# activate, then:
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
grep -viE '^(torch|torchvision|torchaudio|resemblyzer)\b' requirements.txt > requirements.rest.txt
pip install -r requirements.rest.txt
pip install webrtcvad-wheels && pip install resemblyzer --no-deps   # Windows
# Linux: pip install resemblyzer
pip install imageio-ffmpeg                                          # v2 only
```

### Default: v1 offline inference
Argument names were checked against `inference.py` at the pinned commit.

```bash
python inference.py \
  --source <PROJECT_DIR>/inputs/performance.wav \
  --target <PROJECT_DIR>/inputs/voice_ref.wav \
  --output <PROJECT_DIR>/outputs/v2v \
  --diffusion-steps 30 --length-adjust 1.0 --inference-cfg-rate 0.7 \
  --f0-condition False --fp16 True
```

- `--source` is the human performance to convert. `--target` is a clip of the narrator voice, such as a Qwen3-TTS output.
- `--f0-condition False` selects the speech model rather than the singing model.

### Measured (Windows GPU box, 12 GB)
| Variant | Speed | Extra VRAM |
|---|---|---|
| **v1 (default)** | RTF ~0.14 (6.2 s for 43 s of audio) | ~2.1 GB |
| v2 (`inference_v2.py`) | slower: 25–34 s for the same clip | ~3.6–3.8 GB |

The project owner **chose v1** over v2 in a listening test.

### Recording tips for the source performance
- Record **loud and close to the mic**, at **44.1 kHz or higher**. A quiet 16 kHz source clearly limits the output quality.
- **Watch for clipping**: keep peaks below 0 dBFS.

---

## 3. Planned (not yet tested): Applio for production V2V

[Applio](https://github.com/IAHispano/Applio) is a maintained RVC fork (MIT license, v3.7.0). The plan is to train an RVC model on **30–60 min of TTS-generated narrator audio** (Qwen3-TTS output) and use it for production voice-to-voice. **Status: planned, not tested.**

---

## Files
- [`scripts/tts_clone_test.py`](scripts/tts_clone_test.py): one-shot clone test (x-vector-only or ICL).
- [`scripts/tts_script_runner.py`](scripts/tts_script_runner.py): reads a script paragraph by paragraph, with fixed seeds and a words/sec check.
- [`config.example.yaml`](config.example.yaml): placeholder config; copy it to `config.local.yaml` (gitignored).
