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

## 2026-10-16: char01 view-test evaluation (round 2 images) + reference-conditioning proof
- **Proof the reference reached conditioning (all 12 images):** shown both in the workflow graph embedded in each PNG and in the ComfyUI `/history` execution records (12/12 `success`).
  - Node 4 `LoadImage` (canon copy; is_changed sha256 e0e7f62e… = canon) feeds node 5 `FluxKontextImageScale` [4,0].
  - Node 5 feeds `image1` of node 6 (positive) and node 13 (negative), both `TextEncodeQwenImageEditPlus` with `vae` [3,0].
  - Node 9 `KSampler` takes positive [6,0] and negative [13,0], on an empty latent (node 8) with denoise 1.0.
  - In ComfyUI's source, the node VAE-encodes image1 at about 1 MP into `reference_latents` and passes it to the VL text encoder at 384² as a "Picture 1" vision token.
  - The reference is also in the negative conditioning (the template default).
- **Config:** Edit-2511 int8 (11b5af5a…), Qwen2.5-VL-7B fp8 (cb5636d8…), qwen_image_vae (a70580f0…); no LoRA; 1088x1408, 25 steps, cfg 4.0, shift 3.1, euler/simple, seeds 2101–2104.
- **Profiles:** 2/4 face image right (2101, 2103) and 2/4 face image left (2102, 2104), all about 75–85° rather than a true 90°. Nose projection (straight, rounded and slightly upturned tip), lips and hairline are consistent across both directions, and all four are plausible. Chin and jaw definition varies (softer in 2101/2103).
- **Full body:** about 7.5 heads tall, shoulders slightly wider than the hips, normal torso, legs slightly elongated (a catalog-style bias), relaxed A-pose, hands fine. The face is too small for identity checks. Plausible, not verified.
- **Artifacts:** a crinkled or wavy synthetic texture on the T-shirt in most edit outputs; skin smoothing and lost freckles and beauty mark in the three-quarter and profile views; eyes drift to brown in the three-quarter views; no anatomy errors.
- **Reports:** per-image CSV/JSON from `scripts/eval_report.py`. The similarity column is pending (no embedder installed).
- **Reliable scores:** none yet. The VLM judge failed its negative control and is invalid. A face-embedding cosine (ArcFace) would be the first reliable metric.

## 2026-10-16: Round 3: controlled six-view test (Edit-2511, no LoRA)
- **Config:** reference = canon (unchanged, sha256 e0e7f62e…); same models as above; 1088x1408 (the canon's resolution), 25 steps, cfg 4.0, shift 3.1, euler/simple.
  - **Seed policy:** one fixed seed (3101) for all six views. Only the angle/expression phrase varies (`templates/round3_views.yaml`).
- **Speed:** 43.3, 43.6, 43.7, 43.8, 43.7 and 43.7 s.
- **Results (visual review):**
  - frontal_neutral is close to an exact reconstruction (eyes, freckles and features kept).
  - frontal_smile keeps identity well under the expression change.
  - left45 turned about 30–35° in the **wrong** direction.
  - right45 only turned about 10–15°.
  - left90 turned about 70–75° toward image left; right90 only about 55–60°, also toward image left.
  - So **"her left/right" wording is not followed** and turn angles fall short. Larger turns bring back the shirt texture artifact and mild hair haloing. Eye colour stays hazel this round (the prompt names it explicitly).
- **Verdict:** partial. Identity holds best near frontal. Angle control by text alone is unreliable.
- **Next (top 3 setting changes):**
  1. Give the face more reference resolution: drop `FluxKontextImageScale` and pass a tight face crop of the canon as `image2` alongside the full image.
  2. Lower cfg from 4.0 to about 2.5–3.0 to reduce smoothing, synthetic texture and over-processing.
  3. Use camera-relative wording ("facing image left/right", "nose pointing toward the left edge of the frame"), or an explicit pose/angle control (e.g. a multiple-angles LoRA for Edit-2509/2511) instead of "her left/right". Also consider removing the reference from the negative conditioning.

## 2026-10-16: ArcFace identity scores for char01 (InsightFace, analysis only)
- **Tooling:** a separate isolated venv (not ComfyUI's Python): insightface 2.1, onnxruntime 1.31.0 (CPU provider; the GPU wheel installed but the CPU package took precedence, which is fine for about 26 images), numpy 2.5.4, opencv 5.0.
  - Model pack buffalo_l: det_10g `5838f7fe…5b91`, w600k_r50 ArcFace `4c06341c…9e43`, 2d106det `f001b856…dbf`, 1k3d68 `df5c06b8…c45cc`, genderage `4fde69b1…52fb`.
  - **License:** InsightFace pretrained models are non-commercial research only.
- **Method:** `scripts/face_scores.py`. Largest face, cosine of normalized ArcFace embeddings vs canon, detection score, box size and yaw/pitch from 3D landmarks. |yaw| > 45° is marked unreliable (profiles).
- **Controls (vs canon):**
  - canon vs itself and the s1103 candidate: 1.000.
  - Other char01 candidates (same prompt, different seeds: s1101/s1102/s1105/s1108): 0.659, 0.623, 0.634, 0.632.
  - A different character (two baseline images): 0.203 and 0.369.
- **Results (cosine; yaw):**
  - three_quarter 2101–2104: 0.677 (−31°), 0.655 (+32°), 0.686 (+29°), 0.674 (−38°).
  - profile 2101–2104: 0.383 (77°), 0.570 (−51°), 0.455 (63°), 0.499 (−69°). All UNRELIABLE (profiles).
  - full_body 2101–2104: 0.764, 0.775, 0.778, 0.813 (frontal; faces only about 104x143 px).
  - round3: frontal_neutral 0.910; right45 0.900 (it actually came out about 2° yaw, nearly frontal); frontal_smile 0.708; left45 0.714 (−28°); left90 0.566 (−61°, unreliable); right90 0.593 (−48°, unreliable).
- **Detection:** 0 failures across 26 images (scores 0.71–0.91).
- **Interpretation:**
  - The scale runs from about 0.2–0.37 for an unrelated person, through about 0.62–0.66 for same-prompt lookalikes, to about 0.90+ for a near-copy.
  - The three-quarter views (0.66–0.71) score at about the same level as same-prompt lookalikes, so ArcFace **cannot confirm** the identity was kept at about 30° yaw. ArcFace scores also drop with yaw, so this is partly a pose effect. No pose-matched ground truth exists to separate the two.
  - Frontal outputs score above the lookalikes: full body 0.76–0.81, smile 0.71, and neutral 0.91 (which is close to a reconstruction).
  - Arm-level verdict: **identity is confirmed only for near-frontal views.** Angled-view identity is unverified.
- **Next:** pose-binned thresholds (compare angled outputs against angled lookalike negatives), use the multi-view consensus as the reference, and re-score after the reference-crop and cfg changes.

## 2026-10-16: Round 4 / Exp 1: dual-reference face conditioning (Edit-2511 int8, no LoRA)
- **Goal:** does adding a face crop as `image2`, and/or replacing `FluxKontextImageScale`, improve identity?
- **Official workflow check:** the Comfy-Org `image_qwen_image_edit_2511_int8` template does the following:
  - scales image1 with FluxKontextImageScale and passes image2/image3 unscaled;
  - wraps both conditionings in `FluxKontextMultiReferenceLatentMethod(index_timestep_zero)`;
  - uses CFGNorm, a VAEEncode of image1 as the latent, and 40 steps at cfg 4.
  - In our ComfyUI, the 2511 checkpoint's `__index_timestep_zero__` key sets `default_ref_method=index_timestep_zero` automatically, so our graphs already use that method.
  - `TextEncodeQwenImageEditPlus` resizes **every** image itself: to 384² area for the VL encoder, and to about 1 MP (multiple of 8) for `reference_latents`. So small crops get **upscaled**.
- **Config:** canon unchanged (sha256 e0e7f62e… before and after). Face crop = an exact slice of canon at x 291–787, y 52–724 (496x672, no resampling), sha256 3ffeae58….
  - Seeds 4101–4104; 25 steps; cfg 4.0; shift 3.1; euler/simple; output 1088x1408.
  - Two views per config: frontal, and threeq ("face points toward the left edge of the image").
  - The prompt names image1 + image2 as two views of the same fictional woman (single-image wording for A/C). Script: `scripts/edit_multiref_exp.py`.
- **Measured encoder inputs:**
  - A/B image1 is 1088x1408 → 880x1184 (Kontext preset), giving VL 331x445 and ref latent 880x1184.
  - C/D image1 is 896x1168 (ImageScaleToTotalPixels at 1 MP, steps of 16), giving VL 338x441 and ref latent 896x1168.
  - image2 is 496x672, giving VL 330x447 and ref latent **880x1192** (about a 1.8× upscale inside the encoder).
- **Time and VRAM:** A and C take 43.1–44.3 s per image; B and D take 73.9–74.9 s (+70%). Peak VRAM is 30.1–31.0 GB (ComfyUI keeps a reserved pool, so this mostly reflects the cache).
- **ArcFace vs canon (mean of 4; min–max):**
  - frontal: A 0.890 (0.83–0.91), B 0.883 (0.84–0.90), C 0.879 (0.79–0.91), D 0.892 (0.86–0.92).
  - threeq: A 0.716, B 0.634, C 0.766, D 0.614, but the turn rate differs. Turned (|yaw| ≥ 20°): A 3/4 (one in the opposite direction), B 4/4, C 2/4, D 4/4.
  - Yaw-matched (turned only, |yaw| 26–42°): A 0.681, B 0.634, C 0.629, D 0.614.
  - Controls: canon self 1.000; same-prompt lookalikes 0.623–0.659; a different character 0.203.
- **Visual review** (contact sheets plus selected full-resolution images):
  - B/D keep hazel-green eyes and freckles best. They place the mark on the correct side (image right) but **multiply it into several moles**. They frame the face tighter (zoomed in), with harsher contrast and a slightly older look.
  - A/C are softer and smoother, with the mark rarely visible.
  - The crinkled shirt texture persists in all configs.
- **Verdict: no winner.**
  - Frontal identity is statistically indistinguishable across A–D (n=4).
  - At matched yaw, all configs score at or below the frontal lookalike level, so angled identity isn't confirmed for any config. The dual reference doesn't raise ArcFace and costs +70% time.
  - The dual reference **does** measurably improve angle compliance: 8/8 consistent turns for B+D versus 5/8 for A+C. It also preserves freckles and eye colour better, but adds extra moles and over-texture.
  - The ImageScaleToTotalPixels swap (C) shows no measurable benefit.
- **Next:** Exp 2–4 as planned. Ideas: feed the crop at a larger native size (or crop from a higher-resolution render), try a lower cfg to tame the over-texture, and use pose-matched lookalike controls for angled-identity scoring.
