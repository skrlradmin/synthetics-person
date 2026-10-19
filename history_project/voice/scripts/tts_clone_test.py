r"""Qwen3-TTS 1.7B Base: one-shot voice-clone test from a reference clip.

Use it to check a new reference clip before running a whole script with tts_script_runner.py.

Modes:
  x-vector-only (QWEN_TTS_XVEC_ONLY=1, default, or when no transcript is given):
      only the speaker embedding of the reference is used. No transcript needed.
  ICL / full mode (QWEN_TTS_XVEC_ONLY=0 and QWEN_TTS_REF_TEXT set):
      the reference audio plus its exact transcript are used as an in-context prompt.
      Usually a closer match to the reference voice and delivery.

Configuration (environment variables; keep real values in a gitignored .env or config.local.yaml):
  QWEN_TTS_MODEL_DIR, QWEN_TTS_REF_WAV, QWEN_TTS_REF_TEXT, QWEN_TTS_XVEC_ONLY, QWEN_TTS_LANGUAGE
  QWEN_TTS_SEED        default 1234

Usage:
  python tts_clone_test.py [<PROJECT_DIR>/outputs/clone_test.wav] ["text to speak"]
"""
import gc
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

TTS_DIR = os.environ.get("TTS_DIR", "<TTS_DIR>")
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", os.path.join(TTS_DIR, "models", "Qwen3-TTS-12Hz-1.7B-Base"))
REF_WAV = os.environ.get("QWEN_TTS_REF_WAV", os.path.join(TTS_DIR, "inputs", "voice_ref.wav"))
REF_TEXT = os.environ.get("QWEN_TTS_REF_TEXT") or None
LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")
SEED = int(os.environ.get("QWEN_TTS_SEED", "1234"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TTS_DIR, "outputs", "clone_test.wav")
TEXT = sys.argv[2] if len(sys.argv) > 2 else (
    "Some historians say the Great Pyramid was a tomb. Others think it was something stranger. "
    "Let's look at what the evidence actually tells us, and what it doesn't."
)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    xvec = os.environ.get("QWEN_TTS_XVEC_ONLY", "1") == "1" or not REF_TEXT
    print(f"torch {torch.__version__} attn=sdpa mode={'x-vector-only' if xvec else 'ICL'}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t1 = time.perf_counter()
    if xvec:
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language=LANGUAGE, ref_audio=REF_WAV, x_vector_only_mode=True
        )
    else:
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language=LANGUAGE, ref_audio=REF_WAV, ref_text=REF_TEXT
        )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t1

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    sf.write(OUT, wavs[0], sr)
    dur = len(wavs[0]) / sr
    print(f"load_time_s={load_s:.2f}")
    print(f"gen_time_s={gen_s:.2f} audio_duration_s={dur:.2f} sr={sr} realtime_factor={gen_s / dur:.2f}")
    print(f"peak_vram_allocated_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"saved={OUT}")
    del model, wavs
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
