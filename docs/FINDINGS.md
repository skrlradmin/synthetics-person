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
