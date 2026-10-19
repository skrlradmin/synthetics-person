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

## Round 4 / Exp 2 — cfg sweep + official-template quality arm (Edit-2511 int8)

Setup: config A = image1 (canon) only; config B = image1 + face crop as image2. Seeds 4101–4104,
views frontal + three-quarter, same prompts/dims. Sweep: 25 steps, cfg 4.0 (from Exp1) / 3.0 / 2.5.
Quality arm (official 2511 template): CFGNorm(1.0), VAE-encoded image1 start latent (880x1184), 40 steps,
at the best sweep cfg per config (A 2.5, B 3.0). Script: `scripts/edit_cfg_exp.py`.
ArcFace (buffalo_l) cosine vs canon; lookalike controls ≈0.62–0.66; "turned" = |yaw| ≥ 20° on the three-quarter view.

| Arm | Frontal cos mean (min) | 3/4 cos mean (min) | 3/4 turned | s/img |
|---|---|---|---|---|
| A 25s cfg4 | 0.890 (0.832) | 0.716 (0.650) | 3/4 | ~44 |
| A 25s cfg3 | 0.905 (0.857) | 0.804 (0.669) | 2/4 | 43.6 |
| A 25s cfg2.5 | 0.902 (0.836) | 0.832 (0.692) | 2/4 | 43.7 |
| **A q40 cfg2.5** | **0.928 (0.904)** | 0.747 (0.645) | **4/4** | 51.5 |
| B 25s cfg4 | 0.883 (0.838) | 0.634 (0.573) | 4/4 | ~74 |
| B 25s cfg3 | 0.903 (0.882) | 0.715 (0.603) | 3/4 | 74.2 |
| B 25s cfg2.5 | 0.906 (0.888) | 0.743 (0.607) | 3/4 | 74.1 |
| B q40 cfg3 | 0.925 (0.907) | 0.683 (0.580) | 4/4 | 95.8 |

Peak VRAM ≈30.3 GB, min available RAM ≈50.5 GB for all arms.

Findings:
- Lower cfg raises 3/4 cosine mostly by *not turning* (cosine of untouched frontal-ish faces); it is a prompt-adherence regression, not an identity gain.
- The official template (CFGNorm + encoded start latent + 40 steps) gives the best frontal identity of any arm and 4/4 turn compliance, while keeping turned-face cosine above every cfg4 arm. Framing varies (template follows the Kontext-resized image1; some outputs are tighter crops; one B frontal crops the hairline).
- Visual: both quality arms keep hazel eyes and freckles with natural pore texture; B keeps more freckles/marks but still tends to multiply the beauty mark; A renders it inconsistently. Shirt texture artifacts milder than cfg4. No hard-reject anatomy defects seen; B q40 s4102 3/4 is a borderline framing/crop reject.
- Turned-face cosine (0.58–0.80) is still in the lookalike band → identity across angles remains the open problem; no arm passes the hard identity criterion reliably at 3/4.
- Recommendation: config A + official template (CFGNorm, encoded latent, 40 steps) at cfg 2.5 as the new baseline; B adds ~85% time without a measured identity gain at 3/4.
- bf16/fp8 assessment (no download): bf16 Edit-2511 ≈40.9 GB, exceeds 32 GB VRAM and would need RAM offload on ~59 GB RAM (slow, risky); fp8mixed (~20.5 GB) is not expected to beat int8 convrot (community ranking BF16 > INT8 convrot > fp8 > NVFP4). Only bf16 is a plausible quality step; needs approval (>20 GB). A bf16 text encoder (~16.6 GB) is a possible minor step.

## Round 4 / Exp 3 — camera angle: prompt-only vs Multiple-Angles LoRA (Edit-2511 int8)

Base: Exp 2 winner (config A, official template: CFGNorm 1.0, VAE-encoded image1 latent 880x1184, 40 steps, cfg 2.5).
Script: `scripts/edit_angle_exp.py` (new workflow copy; production workflow untouched).
LoRA: fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA (targets Edit-2511 specifically; Apache-2.0; 295 MB; sha256 42426ded…6765; rank 16).
Compatibility with the int8 convrot build: 840/840 LoRA targets map to model weights (diffusers `transformer.*` names);
ComfyUI reports "840 patches attached" with the built-in LoraLoaderModelOnly; no custom nodes needed
(the optional ComfyUI-qwenmultiangle camera-widget node only formats the prompt; not installed).
LoRA prompt = its trained phrasing only: `<sks> {azimuth} eye-level shot medium shot`, azimuths front view /
front-left quarter view / front-right quarter view / left side view / right side view. The published workflow uses a
4-step Lightning LoRA at cfg 1 (distilled); not run (quality-first; would need another download). Card recommends strength 0.8–1.0 (start 0.9).
Prompt-only = identity prompt + frame-relative wording ("face points toward the left/right edge of the image", 45° / 90° full profile).
Yaw sign verified visually: negative = face points image-left. ArcFace unreliable beyond |yaw| ≈ 45°.

| Arm | View | Yaw per seed 4101–4104 | Correct direction | Mean cosine |
|---|---|---|---|---|
| Prompt | front | -1, -1, -1, -1 | 4/4 | 0.928 |
| Prompt | L45 | -48, 21, -32, -26 | 3/4 | 0.747 |
| Prompt | R45 | -16, 32, -1, -14 | 1/4 | 0.814 (mostly not turned) |
| Prompt | L90 | -70, -73, -60, -74 | 4/4 | 0.520* |
| Prompt | R90 | -71, 66, 72, 85 | 3/4 | 0.504* |
| LoRA 1.0 | front | -1, -1, -2, -1 | 4/4 | 0.941 |
| LoRA 1.0 | "front-left quarter" | -40, 37, -44, 52 | magnitude 4/4, side follows seed | 0.612 |
| LoRA 1.0 | "front-right quarter" | -42, 40, -43, 54 | magnitude 4/4, side follows seed | 0.607 |
| LoRA 1.0 | "left side" | -75, 77, 73, 81 | magnitude 4/4, side follows seed | 0.314* |
| LoRA 1.0 | "right side" | -74, -73, -72, 84 | magnitude 4/4, side follows seed | 0.494* |
| LoRA 0.8 (2 seeds) | front / quarter / side | -1,-1 / ±40 / -74, 29 | side follows seed (4101 left, 4102 right) | 0.946 / 0.62 / 0.57* |
(* |yaw| > 45°, ArcFace unreliable.) Time ≈51.5 s/img for all arms (LoRA adds no measurable time); peak VRAM 30.3 GB; min available RAM 50.5 GB.

Findings:
- The LoRA gives accurate rotation magnitude (quarter ≈40–54°, side ≈72–84°, real camera orbit incl. body) but **ignores left vs right**: the side is set by the seed (4101 always image-left, 4102 always image-right), so azimuth phrases are not a direction control on this build/input. Lowering to 0.8 changes nothing material.
- Identity at 45° drops with the LoRA (cos ≈0.57–0.67, inside the lookalike band 0.62–0.66) vs prompt-only (0.65–0.80) → per quality-first, the LoRA quarter views are identity hard-rejects more often.
- Prompt-only is direction-correct for image-left and for 90° profiles, but weak for image-right at 45° (bias toward frontal/left).
- Fronts: hazel eyes, freckles, natural skin kept in all arms; no anatomy hard rejects seen. Beauty-mark side is not resolvable reliably at this output scale; not scored.
- Mirroring an image-left result to get image-right is not acceptable (flips facial asymmetry and the beauty mark).
- Best angle approach for now: prompt-only on the Exp 2 template; use seed selection + ArcFace/yaw filtering for the needed side. The angle LoRA is not adopted (keep for possible future tests: elevation/low-angle, or with a frame-relative direction phrase added).

## Round 4 / Exp 4 + deliverable — LoRA pilot plan (no generation, no training)
See `docs/round4_summary.md`.
- Proposed a 24-slot curated pilot set plus 4 held-out validation images. All are sourced from Exp 2 template edits with prompt-only angles and N-seed overgeneration.
- ArcFace thresholds per yaw band are calibrated from the controls: 0.85 (<15°), 0.75 (15–35°), 0.68 (35–50°), 0.45 floor + manual review (>50°). Near-duplicate rule: cosine ≥ 0.97 and pHash ≤ 8.
- Trainer: ai-toolkit `qwen_image_2` supports 2.1 (pin a commit at or after 07abdbe); musubi-tuner does not. Train on base 2.1, not the CFG-distilled Turbo.
- License: Qwen-Image-2.1 is research-only, so a char01 LoRA trained on it inherits the non-commercial restriction. Edit-2511, the Qwen-Image VAE, the Qwen2.5-VL / Qwen3-VL upstreams and the fal angle LoRA are Apache-2.0. InsightFace models are non-commercial.

## FLUX 3 Image Klein: Step 1 investigation (2026-10-16; no install, no generation)
Facts, from primary sources: the BFL docs (docs.bfl.ml/flux_3/flux3_image_klein, release notes), the HF repos (black-forest-labs/flux-3-image-klein, -base, -comfy, -pup), the HF Diffusers blog (huggingface.co/blog/flux-3), Comfy-Org/ComfyUI commits, and the ostris/ai-toolkit repo.
- **Models.** Released 2026-10-15. Both are 5.6B DiTs:
  - **Klein**: step-distilled, 16 steps, CFG 1.
  - **Klein Base**: undistilled, 50 steps, CFG 3.5. This is the one for fine-tuning and LoRA.
  - One model covers t2i, editing and multi-reference (up to 10 refs), plus bounding-box layout. Max resolution is 4K.
  - Text encoder: Qwen3-VL-4B-Instruct. VAE: the FLUX.2 AE (128 latent channels).
  - BFL says not to use it without its prompt upsampler (flux-3-image-klein-pup, Qwen3.5-9B based, Apache-2.0).
- **ComfyUI repack sizes:**
  - Klein / Klein Base: 11.08 GB each in bf16, 5.81 GB int8_convrot, 5.26 GB nvfp4.
  - Text encoder: 8.88 GB bf16, 4.86 GB int8.
  - Upsampler: 9.75 GB int8, 6.72 GB w4a8.
  - VAE: 0.34 GB.
  - Diffusers package: fp8r 6.3 GB, nvfp4 5.3 GB.
- **ComfyUI.** Native support arrived in "Support Flux3-image" (20b4606, 2026-10-15), released in v0.41.0. **Our v0.39.0-47 (06c67806, 2026-10-12) has no FLUX 3 code.** Running it needs a ComfyUI core update.
- **Access.** All Klein repos are **gated** (license acceptance on an HF account). No HF token is configured on the GPU host, so the files can't be downloaded yet.
- **License (FLUX Kommunity License v1.0).**
  - It is non-commercial by default.
  - §2.a: "Qualifying Users" (gross annualized revenue below US$5M, affiliates included) may use outputs commercially and run the model in production only to generate those outputs.
  - Using the model "to train, fine tune, or distill other models for commercial use" is defined as not non-commercial. A commercial character LoRA therefore needs a BFL commercial license (bfl.ai/licensing), unless §2.a is read to cover it. Legal review needed.
- **Trainers:**
  - **ai-toolkit**: `arch: flux3`, merged in PR #1078 on 2026-10-15, with fixes 438595b, 12395a1 and 0947356 the same day. It trains Base only; distilled and pre-quantized packages are rejected and the toolkit quantizes itself (int8_convrot supported).
  - **Diffusers**: official LoRA scripts for t2i and edit (gradient checkpointing, cached latents, 8-bit Adam, rank 16, 1024 px).
  - **musubi-tuner, diffusion-pipe, kohya sd-scripts**: no FLUX 3 support found.
- **Memory.** Diffusers inference peak (H100) is 20.7 GiB in bf16, 16.3 GiB in fp8r and 15.3 GiB in nvfp4. A LoRA-training VRAM figure is **not published**. Estimate: 5.6B bf16 (≈11 GB) plus cached text embeddings and gradient checkpointing should fit in 32 GB; unverified.
- **Highest-quality feasible inference on 32 GB:**
  - Klein Base bf16, 50 steps, CFG 3.5, bf16 text encoder, prompt upsampler, about 1 MP to 2 MP. No offloading needed.
  - The distilled Klein bf16 (16 steps, CFG 1) as a second arm.
  - BFL publishes no sampler name for ComfyUI; the official workflows are behind the gate.
- **Blockers (stopped before Step 2):**
  1. A ComfyUI core update to v0.41.x or later (a change to the production install).
  2. HF license acceptance and a token on the GPU host.
  3. Downloads: the quality-first set (Base bf16 11.08 + Klein bf16 11.08 + TE bf16 8.88 + upsampler int8 9.75 + VAE 0.34) is **41.1 GB**. The minimum Base-only bf16 set (Base + TE + VAE) is **20.30 GB**. Both exceed the 20 GB limit.
  All three need owner approval. Step 2 (FLUX 3 vs Qwen) has not been run.

## Decision: FLUX 3 Klein skipped (2026-10-16)
By owner decision, FLUX 3 is not being pursued. No ComfyUI copy or update and no downloads. The Step 1 facts above stay for reference.

## Qwen-Image-2512 as the commercial production / LoRA backbone (report only; no downloads, installs or changes)
Sources: HF Qwen/Qwen-Image-2512 (model card and file tree), Comfy-Org/Qwen-Image_ComfyUI and Comfy-Org/Qwen-Image-Edit_ComfyUI (file trees with LFS sha256), kohya-ss/musubi-tuner docs/qwen_image.md, and ostris/ai-toolkit (commit 4d5a649 and the example configs).
- **Repo and license.** `Qwen/Qwen-Image-2512` (rev 25468b98, created 2025-12-30) is **Apache-2.0** and not gated, so commercial use is allowed. Trained LoRAs carry no upstream non-commercial restriction.
- **Architecture.** QwenImageTransformer2DModel: 60 dual-stream layers, 24 heads × 128, joint_attention_dim 3584, in 64 / out 16 channels, patch 2.
  - The config is **identical to Qwen-Image v1**.
  - Edit-2511 has the same shape plus `zero_cond_t: true` (the edit reference-timestep variant).
  - It is unrelated to Qwen-Image 2.1, which has 32 single-stream layers, a Qwen3-VL text encoder and its own VAE.
- **Shared files: we already have the text encoder and VAE.**
  - In the HF repos, the 2512 and Edit-2511 text_encoder shards (d725335e…, b1830db6…, 09c1807c…, 5dd06833…) and VAE (0c8bc8b7…) have the same sha256.
  - Locally, our `qwen_2.5_vl_7b_fp8_scaled` (cb5636d8…) and `qwen_image_vae` (a70580f0…) match the Comfy-Org files byte-for-byte.
  - **Only the 2512 DiT is missing.**
- **ComfyUI.** v0.39.0-47 supports the Qwen-Image architecture natively; the same model class already runs Edit-2511. A 2512 DiT loads with the core UNETLoader, so no custom nodes and no update are needed. (Inferred from the identical config; not load-tested, since there were no downloads.)
- **ComfyUI DiT files** (Comfy-Org/Qwen-Image_ComfyUI, split_files/diffusion_models):
  - `qwen_image_2512_bf16` 40.86 GB (sha cbf55390…)
  - `qwen_image_2512_fp8_e4m3fn` 20.43 GB (sha 5dc80554…)
  - No official int8_convrot or fp8mixed build of 2512 exists; v1 has fp8mixed, fp8_hq and nvfp4. Community GGUF, FP8 and nf4 repacks exist but are not Comfy-Org official.
- **Trainers:**
  - **ai-toolkit:** UI arch `qwen_image:2512` (added in 4d5a649, v0.7.15, 2025-12-31), with model `arch: qwen_image` and `name_or_path: Qwen/Qwen-Image-2512`. Defaults: quantize on (qfloat8, or uint4/uint3 with accuracy-recovery adapters, commits 6870ab4 and 926097a), quantize_te qfloat8, low_vram, cached text embeddings, gradient checkpointing, adamw8bit, lr 1e-4, rank 16. The example `train_lora_qwen_image_24gb.yaml` targets 24 GB. A native 1328 training resolution was added in ca72eb1. It downloads the diffusers repo (≈57.7 GB) unless given local paths.
  - **musubi-tuner (v0.3.6, 2026-09-27):** `qwen_image_train_network.py --network_module networks.lora_qwen_image`, `--model_version original`. The doc doesn't name 2512, but the architecture is identical; this is an inference, so verify by loading. It needs the **bf16** DiT and the **bf16** text encoder `qwen_2.5_vl_7b.safetensors` (16.58 GB); fp8_scaled and fp8_e4m3fn files are rejected.
    - Documented VRAM at 1024² with bs1 and gradient checkpointing: 42 GB without options, **30 GB with `--fp8_base --fp8_scaled`**, 24 GB with `+ --blocks_to_swap 16`. 64 GB system RAM is recommended for block swapping.
- **32 GB feasibility.** Training fits: musubi fp8_base/fp8_scaled at 30 GB, or more safely with blocks_to_swap 16 at 24 GB; ai-toolkit qfloat8 with low_vram at about 24 GB or less. RAM (59 GB total, ~49 GB available) is slightly under musubi's 64 GB recommendation for block swap; keep swap counts low.
- **Highest-quality config that fits:**
  - **Inference:** `qwen_image_2512_bf16` with ComfyUI's dynamic partial offload. The 40.9 GB doesn't fit in 32 GB VRAM, so it will be slower, but it is lossless. The fallback, `fp8_e4m3fn`, fits fully in VRAM.
  - Settings from the model card: 50 steps, true CFG 4.0, native resolutions 1328², 1104×1472 (3:4) or 928×1664 (9:16), with its negative prompt. The Lightning/Turbo LoRAs are distilled and are skipped under quality-first.
  - **Training:** musubi with bf16 DiT + bf16 text encoder and `--fp8_base --fp8_scaled` (or + `--blocks_to_swap 16`) at 1024–1328, or ai-toolkit `qwen_image:2512` with qfloat8.
- **Downloads needed** (VAE and fp8 text encoder already present):

  | Set | Files | Size |
  |---|---|---|
  | Inference, minimum | fp8 DiT | 20.43 GB |
  | Inference, quality-first | bf16 DiT | 40.86 GB |
  | Training (musubi) | bf16 DiT + bf16 TE | 57.44 GB |
  | Recommended single set | bf16 DiT + bf16 TE (bf16 DiT also used for inference) | **57.44 GB** |
  | Optional fp8 DiT for fast iteration | on top of the recommended set | +20.43 GB = 77.87 GB |

  Disk has ~1.2 TB free. All of these are over 20 GB, so they need approval, as does installing a trainer (new Python environment).

## Qwen-Image-2512 fp8 load test (approved download; no LoRA, no trainer)
- **File:** downloaded `qwen_image_2512_fp8_e4m3fn.safetensors` (20.43 GB) as a new file. sha256 5dc80554…876b **matches HF**.
- **Script:** `scripts/t2i_2512_loadtest.py` (new). Uses the existing `qwen_2.5_vl_7b_fp8_scaled` text encoder and `qwen_image_vae`, core nodes only (UNETLoader, CLIPLoader type qwen_image, ModelSamplingAuraFlow 3.1, KSampler euler/simple).
- **Settings:** model card's: 50 steps, cfg 4.0, 1104x1472, official negative prompt. The prompt is char01's canonical description text. Seeds 5101 and 5102.
- **Load:** succeeded on ComfyUI v0.39.0-47 with no update or custom nodes. The DiT staged as QwenImage, 19.5 GB.
- **Performance:**

  | Seed | Time | Peak VRAM | Min RAM available |
  |---|---|---|---|
  | 5101 | 83.3 s (incl. model load) | 29.7 GB | 51.0 GB |
  | 5102 | 82.8 s | 29.7 GB | 57.0 GB |

- **Visual:**
  - Strongly photographic: real pores, fine forehead lines, natural flyaway hair, hazel-green eyes, faint freckles, no waxy smoothing. No anatomy or texture defects.
  - Prompt deviations:
    - Both read older than 28 (mid-30s look).
    - Framing came out head-and-shoulders rather than the canonical standing pose.
    - The s5101 beauty mark sits on the image-left cheek (subject's right), the wrong side vs spec. s5102's is faint.
  - As expected for t2i without a LoRA, the faces are a different person from canon s1103. This confirms the need for a character LoRA.
- Canon sha256 unchanged.

## 2026-10-16: char01 LoRA pilot dataset v1 (Edit-2511, no LoRA): 8/24 slots filled, top-up needed
- **Config:** approved Edit-2511 workflow (int8 convrot, canon s1103 as image1, CFGNorm, 40 steps, cfg 2.5, 1088x1408), 200 seeds across 24 training slots + 4 validation slots (`templates/pilot_v1_slots.json`, `scripts/gen_pilot_dataset.py`). About 52 s per image, peak VRAM about 30 GB.
- **Filter:** `scripts/filter_pilot_dataset.py`. ArcFace per yaw band 0.85/0.75/0.68, >50° to manual review, direction ±15°, near-duplicate cos ≥ 0.97 and pHash ≤ 8. Thresholds unchanged.
- **Result:** 8 slots filled provisionally and 3 profiles in review; 13 unfilled. Validation: only V3 (profile, review).
  - Auto-gate pass rates: frontal 18%, three-quarter left 31%, three-quarter right 7%, half body 13%, full body 0%, camera variation 6%.
  - Rejection reasons: identity 141, wrong direction 67, near-duplicate 2.
- **Causes:**
  - Smiling expressions and warm or hard light pull identity down.
  - Three-quarter image-right outputs often turn the wrong way.
  - Full-body faces are too small for ArcFace.
- **Next:** a targeted top-up of about 190 generations: milder expressions, nose-direction wording for three-quarter views, and full body scored on an upscaled face crop or reframed to knees-up. No threshold relaxation. Details are in `docs/pilot_v1_summary.md`.
- **Training prep (not run):** `docs/pilot_v1_memory_check.md` and `templates/train_char01_pilot_v1_draft.yaml`.
  - ai-toolkit `qwen_image` with qfloat8, low_vram, cached TE and gradient checkpointing is estimated at 24-27 GB VRAM at 1024 px. Peak RAM is about 42-46 GB while quantizing on the CPU.
  - Download needed: the diffusers repo Qwen/Qwen-Image-2512, 57.7 GB. Alternatively, the ComfyUI bf16 DiT (40.86 GB) plus diffusers TE, VAE and tokenizer (about 16.9 GB). The local fp8 TE and the Edit-2511 DiT cannot substitute.
- Canon sha256 unchanged.

## 2026-10-16: char01 pilot v1 top-up (approved corrections; in progress)
- **Generation:** same Edit-2511 workflow and settings as pilot v1 (single canon reference, CFGNorm + VAE-encoded start, 40 steps, cfg 2.5, no LoRA), new seeds only. 210 seeds over the 16 unfilled slots: 16 each for three-quarter 08/11/12, 14 each for full body 20/21/22, and 12 each for 03, 04, 06, 13, 18, 19, 24, V1, V2, V4.
  - New script: `scripts/gen_pilot_topup.py`. The slot file carries its own prompt text and a per-slot `seed_base`. It checks the reference hash, uses bounded retries per seed, and is resumable.
- **Prompt corrections:**
  - Milder expressions (slight or soft smile, or relaxed).
  - Neutral soft light, with no warm, hard or rim light and no colour casts.
  - Varied outfits and settings instead of the grey T-shirt on a studio backdrop.
  - Explicit nose-direction wording on turned views ("her nose points toward the right edge of the image").
- **Scoring:** `scripts/filter_pilot_topup.py` uses the same thresholds as v1 (0.85/0.75/0.68 by yaw band, >50° to manual review with a 0.45 floor, direction ±15°, near-duplicate cos ≥ 0.97 and pHash ≤ 8). The near-duplicate check also runs against the earlier dataset's accepted, review and validation images.
  - New: every image also gets a face-crop score, computed scoring-only without editing the image. The face is cropped as a square with 50 % margin, upscaled to 512 px with Lanczos, re-detected and embedded. Full-body slots are decided on the crop score. Raw scores are always recorded.
- **Early finding (v1 full-body re-score, report only):** on the 28 v1 full-body candidates (faces about 105-140 px), the crop score differs from the raw score by at most about 0.05. Zero of 28 pass either way. Low full-body identity is therefore real drift at small face scale, not a detector-resolution artifact.
- **Merge:** `scripts/report_pilot_topup.py` builds a combined slot view (manifest plus symlinks; earlier files are never moved). It also writes per-image acceptance tables, per-category stats, contact sheets and a summary.

## 2026-10-17: char01 pilot v1 top-up results (210 gens, 3.0 h GPU)
- **Run:** 210/210 succeeded, about 51.6 s per image, no retries needed. Canon sha256 unchanged. Thresholds unchanged. Combined view built by `scripts/report_pilot_topup.py` from a manifest; earlier files were not moved.
- **Visual checks (`--visual`):** two v1 picks were demoted to review.
  - The v1 frontal 01 pick has multiple beauty marks.
  - The v1 half-body 17 pick has an extra chin mole and uncertain body geometry.
  - All top-up picks show a single image-right cheek mark where it should be visible, with no anatomy or texture defects.
- **Combined fill:** 9/24 training slots, 1/4 validation. 10 images in manual review: 6 profile, two three-quarter right at yaw about 51°, and one validation at 52°.

  | category | filled | v1 pass | top-up pass | combined pass |
  |---|---|---|---|---|
  | frontal | 4/6 | 18% | 14% | 16% |
  | three-quarter | 4/6 | 15% | 17% | 16% |
  | profile | 0/4 (all in review) | 18% | 42% | 26% |
  | half body | 0/3 | 21% | 0% | 10% |
  | full body | 0/3 | 0% | 0% | 0% |
  | camera variation | 1/2 | 6% | 0% | 4% |
  | validation | 1/4 | 10% | 6% | 8% |

- **What worked:**
  - Milder expression plus neutral soft light and new outfits/settings filled frontal 03 (cos 0.853, v1 best 0.725) and 04 (3 passes, v1 none) and validation V1 (0.939).
  - Nose-direction wording filled three-quarter image-left 08 (yaw -40, cos 0.682) and cut direction errors for 08 and 12 to 1/16 each.
  - Profile direction pass rate rose to 42%, although outputs under-rotate to about 65-76°.
- **What did not work:**
  - **Full body:** 0/42. The upscaled face-crop score was essentially identical to the raw score (mean 0.419 vs 0.419; v1 re-score 0/28 either way). Low full-body identity is real drift at small face scale (about 110 px), not a detector-resolution artifact.
  - **Half body:** 0/36. The model frames tighter than waist-up and overshoots yaw by 15-30° (18: 9/12 wrong direction).
  - **Near-front 06 (target 10°):** the turn wording overshoots to 20-42° in 9/12.
  - **Three-quarter image-right 11/12 and validation V2:** identity stays below the band thresholds (best in-band 0.66-0.73). The only direction-valid picks sit just over 50° and go to manual review.
  - **Low-angle 24:** 0/12, best 0.716.
- **Recommended corrections (no threshold change):**
  - Full body: switch to knees-up framing, or generate full body only after a first LoRA trained on the close-up/half-body set.
  - Half body: write about 15-20° when 30° is wanted, and add "camera at a distance, framed from the waist up".
  - 06: frontal wording plus "head turned only a few degrees".
  - Image-right three-quarter: target 30-35° and consider a second reference image (an accepted image-right three-quarter view). This needs approval because it changes the fixed single-reference setup.
  - 24: "camera slightly below eye level" with the original neutral outfit.

## 2026-10-17: char01 pilot v1 top-up round 2 (approved; in progress)
- **Setup:** unchanged single-reference Edit-2511 workflow (canon only, CFGNorm + VAE-encoded start, 40 steps, cfg 2.5, no LoRA). Thresholds unchanged. 160 new seeds over 11 slots: 16 each for 20/21/22, 14 each for 06, 11, 12, 18, 19, 24, V2, V4.
- **Slot corrections (prompt-side):**
  - 20-22 redefined from full body to knees-up ("framed from the knees up, camera a few meters away") with new outfits and settings. Still scored raw and on the upscaled face crop.
  - Half body 18, 19, V4: the prompt asks for about 18° while the scoring target stays 30° (new `prompt_yaw` slot field), plus "camera at a distance, framed from the waist up, hands visible".
  - 06: frontal wording plus "head turned only a few degrees" (new `direction_text` field).
  - 11, 12, V2 retargeted to 35/30/35° with nose-direction wording.
  - 24: "camera slightly below eye level" with the original neutral outfit.
- **v1 runner-ups for the demoted slots (visual check on full-res face crops, no generation):**
  - Slot 01: promoted one runner-up with a single image-right cheek mark and no near-duplicate of any selection (max cos 0.935). The other two have extra marks.
  - Slot 17: both runner-ups have extra moles and are framed chest-up rather than waist-up, so the slot stays unfilled.
- **Scripts:** `gen_pilot_topup2.py`, `filter_pilot_topup2.py` (logic unchanged) and `report_pilot_topup2.py`. The new report script merges N sources in precedence order through a manifest, with visual and decision notes applied in order.

## 2026-10-18: char01 pilot v1 top-up round 2 results (160 gens, 2.3 h GPU)
- **Run:** 160/160 succeeded, no retries. Canon sha256 unchanged. Thresholds unchanged.
  - Visual checks were applied with `report_pilot_topup2.py --notes`. The report was regenerated off the GPU host; it needs only the scored JSON files, per-image metadata and the existing thumbnails.
- **Combined fill (v1 + top-up + top-up 2, manifest only):** 14/24 training slots, 1/4 validation.
  - New this round: three-quarter right 12 (yaw 36, cos 0.684), half body 18 (yaw -45, 0.703), half body 19 (yaw 44, 0.712) and low-angle 24 (yaw -1, 0.947).
  - Frontal 01 refilled with a visually checked v1 runner-up. Slot 17's runner-ups failed the visual check (extra moles, chest-up framing).
- **Pass rates, round 2 / combined:**

  | category | round 2 | combined |
  |---|---|---|
  | frontal (06 only) | 0/14 | 13% |
  | three-quarter | 1/28 | 13% |
  | half body | 2/28 | 9% |
  | knees-up (20-22) | 0/48 | 0% |
  | camera variation | 1/14 | 5% |
  | validation | 0/28 | 5% |

- **Findings:**
  - **Knees-up:** the prompt still renders mid-thigh to waist-up framing. Faces are larger (mean 230 px vs about 115 px full body), and raw cos rose from 0.42 to 0.57, but nothing reaches threshold (best 0.707 frontal vs 0.85 required). The face-crop score again matched the raw score (0.573 vs 0.574).
  - **Halved head-turn wording (18° in the prompt for a 30° target):** outputs still land at about 40-60° (19: 12/14 too steep), so head turns in half-body shots overshoot regardless of the stated angle.
  - **06:** the frontal wording fixed direction (11/14 within ±15°), but identity stayed low (0.59-0.75) with a slight smile in a street setting.
  - **Retargeting 11/V2 to 35°:** did not lift identity (best in-band 0.63-0.67). About half flipped direction (11: 7/14).
  - **24:** "camera slightly below eye level" plus the original outfit filled the slot (0.947), but the camera angle reads only slightly low.
- **Remaining shortages and suggested corrections (thresholds unchanged):**
  - 06: neutral mouth-closed expression and a plain indoor setting, as in the 03/04 picks that filled.
  - 11, V2: the single-reference setup appears to plateau for image-right three-quarter views. Either reconsider a second reference, or defer to a first LoRA trained on the filled slots.
  - 17: regenerate with a neutral outfit and an explicit "waist-up, arms relaxed at her sides, camera at a distance".
  - 20-22: defer to after a first LoRA, or accept a waist-up redefinition.
  - V4: the same overshoot fix as half body; ask for frontal-ish "head turned slightly" and keep the ±15° gate.
  - Profiles 13-16 and V3 still await manual review.

## 2026-10-18…19: Character B stills stack (Qwen-Image 2.1 + LoRA)

- **Goal:** lock a reusable stills recipe for a fictional character LoRA (`character B`) on Qwen-Image 2.1 int8, including Turbo drafts and pose control.
- **Stack:** Qwen-Image 2.1 int8 + matching VAE + qwen3vl_8b TE; character LoRA rank 32 / ~3000 steps; Fun ControlNet Union for pose; optional Turbo int8 (official Comfy-Org, sha starts `8144a018`).
- **Timing (832×1216, models warm, 32 GB class GPU):** base 25 steps cfg 3 ≈ 9 s; Turbo 8 steps cfg 1 ≈ 2 s; base + pose CN ≈ 11–12 s; face pass adds ≈ 8 s.
- **Bust / proportion findings:** LoRA ~0.9 with stacked size wording matches a larger look but often yields round/augmented busts. Soft cue (`natural DDD proportions`) + proportion negatives + LoRA **0.7–0.8** yields softer, more natural shape but smaller volume. Stacked size adjectives alone do not recover DDD volume once pose CN + negatives hold the pose. **Decision (2026-10-19):** keep DDD as a soft cue only; LoRA **0.8 clothed / 0.7 when size cue is active**; do not stack size words.
- **Pose CN:** strength ~0.8, end 0.5, map from own renders — best match for selfie framing. Edit-mode multi-ref images hurt (posed look, re-inflated bust, slower).
- **Face pass:** crop → img2img denoise 0.3 works; Impact FaceDetailer crashes on this conditioning path.
- **Turbo:** identity holds at same seed vs base; use for drafts; finals on base (smoother skin / weaker fine detail on Turbo). Unofficial Turbo files can fail model-type detection.
- **Multi-person complex poses:** ControlNet often drops secondary contacts or rotates the subject; needs better skeletons or masked inpaint — not solved in this pass.
- **Verdict:** recipe documented in `docs/studio_workflow.md`.
- **Next:** optional depth ControlNet weights (download needs explicit OK); continue pose-map hygiene (synthetic-only maps).

## 2026-10-19: Voice workflow for history_project (Qwen3-TTS + Seed-VC)
- **Goal:** one consistent, fully synthetic narrator voice, both from text (TTS) and from a human performance (voice-to-voice). Full workflow: [history_project/voice/README.md](../history_project/voice/README.md).
- **Hardware:** Windows GPU box, 12 GB (RTX 4070 Super).
- **TTS: Qwen3-TTS (Apache-2.0).** qwen-tts 0.1.1, torch 2.8.0+cu128, Python 3.12, transformers 4.57.3, sdpa attention (no flash-attn). Models: 12Hz-1.7B-Base (clone), 1.7B-VoiceDesign, Tokenizer-12Hz.
  - Clone from a ~15–20 s reference clip of a synthetic voice, trimmed at a quiet point. x-vector-only mode when there is no transcript; ICL mode with `ref_text` matches more closely.
  - Script runner: one generation per blank-line paragraph, fixed seed per paragraph, voice prompt built once and reused, 0.4 s gaps, 24 kHz WAV, words/sec check 1.6–4.0 with one reseeded retry.
  - **Measured:** load 2.6 s; ~1.36–1.46× realtime (85 s for 62.6 s of audio); peak VRAM ~4.2–4.3 GB.
  - **Limit:** the Base clone path has no instruct, speed or emotion controls (checked in the qwen-tts source). Only VoiceDesign and CustomVoice take `instruct`, and CustomVoice only has preset speakers. Levers for a cloned voice: reference clip style, punctuation, temperature / subtalker_temperature plus picking the best take, and a 0.9–1.15× time-stretch afterwards. Whether `instruct` works on a fine-tuned `custom_voice` speaker (finetuning/sft_12hz.py) is unverified.
- **V2V: Seed-VC (GPL-3.0 code and weights; repo archived 2025-04-20; pinned commit 51383efd).** Own venv, Python 3.12.10, torch 2.8.0+cu128; requirements otherwise as pinned. Windows needed webrtcvad-wheels and `resemblyzer --no-deps`. About 8 GB to download.
  - **v1** (`inference.py`, 30 diffusion steps, cfg rate 0.7, no f0, fp16): RTF ~0.14 (6.2 s for 43 s), ~2.1 GB extra VRAM.
  - **v2:** 25–34 s for the same clip, ~3.6–3.8 GB.
  - The project owner preferred v1 in a listening test. A quiet 16 kHz source limits quality: record loud, close-mic, at 44.1 kHz or higher, without clipping.
- **Verdict:** works. Qwen3-TTS Base clone for narration, Seed-VC v1 for acted lines.
- **Next (planned, untested):** Applio (MIT, RVC fork, v3.7.0) trained on 30–60 min of TTS narrator audio for production V2V.
