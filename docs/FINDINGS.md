# Findings log

## 2026-10-16: Environment gap check
- **Goal:** confirm the baseline stack and find what's missing for identity work.
- **Stack:** 32 GB GPU, ComfyUI v0.39.0-47 (06c67806), Python 3.14, torch 2.14.1+cu130, CUDA 13.2. A local 120B VLM (llama.cpp, about 14.8 GB of VRAM) runs alongside ComfyUI, so peak VRAM during generation is about 32 GB.
- **Base model:** Qwen-Image 2.1 int8 "convrot" ComfyUI repack: 32 transformer blocks, hidden size 4096, 64 latent channels, int8 weights with fp32 per-row scales. The file metadata doesn't record its source.
  - sha256 `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d`
- **VAE:** qwen_image_2.1_vae_bf16, sha256 `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9`
- **Text encoder:** qwen3vl_8b int8 repack.
- **Character LoRA (private, not in repo):** trained with ai-toolkit 0.13.25, 2000 steps, base qwen_image_2.
  - sha256 `58f6429bc8b6cbda87bac83a47aabd3cbc5ff9bdec1a8bd95baa7df4c011ba52`
- **Speed:** 1088x1408, 25 steps, euler/simple, cfg 1.0 takes 8.0–9.3 s per image with models resident.
- **Gaps:** no Qwen-Image-Edit weights (the core Edit nodes exist); no face-embedding tool (insightface absent). Qwen-Image 2.1 is research-license only.
- **Verdict:** works for t2i. Edit weights needed for reference-based identity.
- **Next:** download Edit-2511 and run identity arm B.

## 2026-10-16: char01 canonical reference candidates
- **Goal:** first frontal canonical reference for a new fictional character (char01), with no LoRA.
- **Config:** Qwen-Image 2.1 int8 (hash above), no LoRA, 1088x1408, 25 steps, cfg 1.0, euler/simple, seeds 1101–1108. Prompt: `templates/canon_prompt.txt` / `canon_negative.txt`.
- **Results:** 8/8 succeeded. 10.5 s for the first image, then 8.0 s each, about 66 s in total. No out-of-memory errors with the VLM resident.
- **Observations:** prompt adherence is consistent (dark wavy shoulder-length hair, gray tee, plain gray background, frontal neutral pose). Faces are similar across seeds but not identical, so a base model alone does not lock identity.
- **Verdict:** works.
- **Next:** pick a canonical seed, then build a multi-view dataset for the char01 LoRA.

## 2026-10-16: Identity consistency, arm A (LoRA only): IN PROGRESS
- **Config:** base model above + private character LoRA at 0.8, 20 varied SFW smartphone scenes (`scripts/idexp.py`), seeds 9001–9020, 1088x1408, 25 steps.
- **Results so far:** 20/20 generated at 8.0–8.5 s each. VLM-judge scoring is pending; the first attempt returned no parsable score.

## 2026-10-16: Identity consistency, arm B (LoRA + Qwen-Image-Edit reference): BLOCKED, then approved
- Blocked because no Edit weights were installed. Needed: qwen_image_edit_2511_int8_convrot (20.5 GB), qwen_2.5_vl_7b_fp8_scaled (9.38 GB), qwen_image_vae (0.25 GB). The download is now approved and running.
- Note: the 2.1 LoRA probably won't load on Edit-2511, which uses a different base architecture.
