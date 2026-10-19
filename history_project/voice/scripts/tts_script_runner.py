r"""Qwen3-TTS 1.7B Base: read a narration script paragraph by paragraph in a cloned voice and
join the takes into one 24 kHz WAV.

- Paragraphs are split on blank lines; one generation per paragraph.
- Each paragraph gets a fixed seed (BASE_SEED + index), so a rerun reproduces the same takes.
- The voice-clone prompt is built once from the reference clip and reused for every paragraph.
- Paragraphs are joined with GAP_S seconds of silence.
- Sanity check: words/sec per paragraph must fall in [WPS_MIN, WPS_MAX]; otherwise the paragraph
  is regenerated once with a new seed and the better of the two takes is kept.

Configuration (environment variables; keep real values in a gitignored .env or config.local.yaml):
  QWEN_TTS_MODEL_DIR   path to Qwen3-TTS-12Hz-1.7B-Base         (default <TTS_DIR>/models/Qwen3-TTS-12Hz-1.7B-Base)
  QWEN_TTS_REF_WAV     reference clip of a synthetic voice you have rights to (~15-20 s, mono WAV)
  QWEN_TTS_REF_TEXT    exact transcript of the reference clip (optional; enables ICL / full mode)
  QWEN_TTS_XVEC_ONLY   "1" = x-vector-only mode (default when no transcript), "0" = ICL mode
  QWEN_TTS_LANGUAGE    default "English"
  QWEN_TTS_BASE_SEED   default 1234

Usage:
  python tts_script_runner.py <PROJECT_DIR>/scripts/episode_01.txt <PROJECT_DIR>/outputs/episode_01.wav
"""
import gc
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

TTS_DIR = os.environ.get("TTS_DIR", "<TTS_DIR>")
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", os.path.join(TTS_DIR, "models", "Qwen3-TTS-12Hz-1.7B-Base"))
REF_WAV = os.environ.get("QWEN_TTS_REF_WAV", os.path.join(TTS_DIR, "inputs", "voice_ref.wav"))
REF_TEXT = os.environ.get("QWEN_TTS_REF_TEXT") or None
XVEC_ONLY = os.environ.get("QWEN_TTS_XVEC_ONLY", "1") == "1" or REF_TEXT is None
LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")
BASE_SEED = int(os.environ.get("QWEN_TTS_BASE_SEED", "1234"))
RETRY_SEED_OFFSET = 1000
GAP_S = 0.4
WPS_MIN, WPS_MAX = 1.6, 4.0
WPS_TARGET = 2.6  # used only to pick the better take when both are out of range


def split_paragraphs(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    return [p for p in paras if p]


def generate(model, prompt, text, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t = time.perf_counter()
    wavs, sr = model.generate_voice_clone(text=text, language=LANGUAGE, voice_clone_prompt=prompt)
    torch.cuda.synchronize()
    return np.asarray(wavs[0], dtype=np.float32), sr, time.perf_counter() - t


def words_per_sec(words, wav, sr):
    dur = len(wav) / sr
    return dur, (words / dur if dur else 0.0)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: tts_script_runner.py <script.txt> <out.wav>")
    txt_path, out_path = sys.argv[1], sys.argv[2]
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    paras = split_paragraphs(txt_path)
    mode = "x-vector-only" if XVEC_ONLY else "ICL"
    print(f"paragraphs={len(paras)} mode={mode} attn=sdpa")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    torch.cuda.synchronize()
    print(f"load_time_s={time.perf_counter() - t0:.2f}")

    # Build the voice prompt once and reuse it for every paragraph.
    if XVEC_ONLY:
        prompt = model.create_voice_clone_prompt(ref_audio=REF_WAV, x_vector_only_mode=True)
    else:
        prompt = model.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=REF_TEXT)

    pieces, sr_out, total_gen = [], None, 0.0
    for i, text in enumerate(paras, 1):
        words = len(text.split())
        seed = BASE_SEED + i
        wav, sr, g = generate(model, prompt, text, seed)
        total_gen += g
        dur, wps = words_per_sec(words, wav, sr)
        status = "ok"
        if not (WPS_MIN <= wps <= WPS_MAX):
            print(f"para {i}: SUSPECT dur={dur:.2f}s words={words} wps={wps:.2f} seed={seed} -> retry once")
            seed2 = BASE_SEED + RETRY_SEED_OFFSET + i
            wav2, sr, g2 = generate(model, prompt, text, seed2)
            total_gen += g2
            dur2, wps2 = words_per_sec(words, wav2, sr)
            ok2 = WPS_MIN <= wps2 <= WPS_MAX
            if ok2 or abs(wps2 - WPS_TARGET) < abs(wps - WPS_TARGET):
                wav, dur, wps, g, seed = wav2, dur2, wps2, g2, seed2
            status = "retried-ok" if ok2 else "retried-STILL-SUSPECT"
        sr_out = sr_out or sr
        print(f"para {i}: words={words} gen_s={g:.2f} audio_s={dur:.2f} wps={wps:.2f} seed={seed} {status}")
        pieces.append(wav)
        if i < len(paras):
            pieces.append(np.zeros(int(GAP_S * sr_out), dtype=np.float32))

    out = np.concatenate(pieces)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, out, sr_out, subtype="PCM_16")
    audio_s = len(out) / sr_out
    print(f"total_gen_s={total_gen:.2f} total_audio_s={audio_s:.2f} sr={sr_out} "
          f"realtime_factor={total_gen / audio_s:.2f}")
    print(f"peak_vram_allocated_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"saved={out_path}")
    del model, prompt
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
