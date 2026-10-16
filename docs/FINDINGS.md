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

## 2026-10-16: Identity consistency, arm A (LoRA only): generation done, scoring INCONCLUSIVE
- **Config:** base model above + private character LoRA at 0.8, 20 varied SFW smartphone scenes (`scripts/idexp.py`), seeds 9001–9020, 1088x1408, 25 steps, cfg 1.0, euler/simple.
- **Results:** 20/20 generated at 8.0–8.5 s each.
- **VLM judge:** a local 120B VLM rated "same person" on 1–10 against a baseline reference. The 20 scenes scored 9–10 (mean about 9.3). However, the **negative control** (a completely different character, char01 s1101) also scored **10**. The judge does not discriminate identity, so the scores are not valid.
  - The first judge run returned no scores because max_tokens=400 truncated the model's reasoning; fixed by raising it to 2000.
- **Verdict:** generation works; identity measurement fails.
- **Next:** use a face-embedding metric (ArcFace/insightface, not installed yet) or human review. Further VLM scoring is SKIPPED because the VLM server was shut down on 2026-10-16 to free VRAM.

## 2026-10-16: Qwen-Image-Edit-2511 installed
- Downloaded from Comfy-Org/Qwen-Image-Edit_ComfyUI and Comfy-Org/Qwen-Image_ComfyUI. Every sha256 matches the Hugging Face LFS oid:
  - qwen_image_edit_2511_int8_convrot.safetensors (20.5 GB): `11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829`
  - qwen_2.5_vl_7b_fp8_scaled.safetensors (9.38 GB): `cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4`
  - qwen_image_vae.safetensors (0.25 GB): `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f`
- License: Apache-2.0 (Qwen-Image-Edit-2511, Qwen2.5-VL).
- The local VLM server (about 15 GB VRAM) was stopped to free memory; VRAM dropped from 31.2 GB to 16.1 GB.

## 2026-10-16: char01 identity preservation with Qwen-Image-Edit-2511 (no LoRA)
- **Goal:** can Edit-2511 produce new views of char01 from one canonical image while keeping the identity?
- **Config:** reference = char01 canon (candidate s1103, sha256 `e0e7f62e92b35be724854b92a8c6c7ee6ccede158372f0264f43227a24c646e4`); Edit-2511 int8 (hash above); no LoRA; 1088x1408, 25 steps, cfg 4.0, shift 3.1, euler/simple; seeds 2101–2104. Prompts: `templates/view_tests.yaml`; script: `scripts/edit_views.py`.
- **Speed:** 43.6–51.4 s per image (cfg>1 means two passes; the first image includes model load).
- **three_quarter:** pose correct in 4/4. Overall likeness and hair are good, and nose, lips and jaw are close. Drift: eyes turn brown instead of hazel-green; the beauty mark and freckles mostly disappear; skin is slightly smoother and softer (mild beautification); no aging. Verdict: partial.
- **profile:** a true 90° profile in 4/4. Nose and jaw are plausible, but this can't be verified against a frontal-only reference; no beauty mark visible. Verdict: partial (plausible).
- **full_body:** 4/4 full-body, same outfit plus jeans. The face is too small to judge. This is a **plausible extrapolation, not verified body preservation**. Verdict: plausible.
- **Next:** strengthen eye-colour and marking cues in the prompt, try a higher-resolution reference or face crop, and add a face-embedding metric.

## 2026-10-16: Identity consistency, arm B (Qwen-Image-Edit-2511 from a reference, no LoRA) vs arm A
- **Config B:** Edit-2511 int8 (hash above); reference = one baseline character image (a mirror selfie); same 20 scenes and seeds 9001–9020; 1088x1408, 25 steps, cfg 4.0, shift 3.1; no LoRA (the 2.1 LoRA is incompatible with the Edit base). Script: `scripts/idexp_edit.py`.
- **Speed:** A takes 8.0–8.5 s per image; B takes 43.6–44.6 s per image, about 5.4× slower.
- **Scoring:** the automatic judge was SKIPPED/PENDING (VLM server stopped, and the earlier judge failed its negative control). The read below is a visual review only.
- **Arm A (LoRA):** face identity is stable across all 20 scenes. Drift: the body is rendered curvier than specified despite negatives, she reads somewhat older than intended, and there is a strong "posed portrait" look.
- **Arm B (Edit):** scene and prompt adherence is more varied and more candid. But identity is looser: face shape and age vary, and some outputs look like a different person. B also copies the reference composition (a phone or mirror-selfie pose shows up in about 9/20 scenes). It is over-exposed and over-sharpened with blown highlights and saturated colour, and has occasional skin artifacts.
- **Verdict:** A is better for identity (works), B is partial. Edit-from-one-reference doesn't replace a LoRA for identity, but it's useful for making new views (see the char01 test).
- **Next:** use a neutral frontal reference (not a selfie) for Edit, try cfg 2.5–3 or fewer steps to reduce over-processing, consider a Lightning LoRA for speed, and add an ArcFace similarity metric.
