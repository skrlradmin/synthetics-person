# Round 4 summary: char01 reference-edit pipeline (Qwen-Image-Edit-2511 int8) + LoRA pilot plan

Planning document. No LoRA was trained and no dataset was written. Details for each experiment are in `FINDINGS.md` (Round 4 Exp 1–4).

## 1. Recommendations
| Topic | Recommendation | Evidence |
|---|---|---|
| Reference method | **Config A**: canon as image1 only, via FluxKontextImageScale, using the **official 2511 template** (CFGNorm 1.0, VAE-encoded image1 start latent → 880x1184, 40 steps, euler/simple, index_timestep_zero) | Best frontal ArcFace of any arm (mean 0.928, min 0.904); turned 4/4 on the 3/4 prompt. Adding a face crop as image2 (B) gave no turned-face identity gain, took ~85% more time and multiplied the beauty mark. |
| cfg | **2.5** with the template | cfg 4 → 3/4-view cosine 0.716; cfg 2.5 sweep → 0.832, but only 2/4 turned. The template at 2.5 gets 0.747 with 4/4 turned. Lower cfg alone inflates cosine by not turning. |
| Angle approach | **Prompt-only, frame-relative wording**, with seed overgeneration and filtering on yaw and ArcFace. The fal Multiple-Angles LoRA is **not adopted**. | The LoRA hits the right yaw magnitude, but the seed decides left vs right, and 45° identity drops to 0.57–0.67 (the lookalike band). Prompt-only directions: L45 3/4, R45 1/4, L90 4/4, R90 3/4. |

## 2. Measured GPU performance (RTX 5090 32 GB, ~59 GB RAM, int8 convrot)
| Arm | s/img | Peak VRAM | Min RAM available |
|---|---|---|---|
| A, 25 steps (Exp 2 sweep) | 43.6 | ~30.3 GB | ~50.5 GB |
| A, template, 40 steps | 51.5 | 30.3 GB | 50.5 GB |
| A, template + angle LoRA | 51.5 (no measurable overhead) | 30.3 GB | 50.5 GB |
| B (image1 + face crop), 25 steps | 74.2 | ~30.3 GB | ~50.5 GB |
| B, template, 40 steps | 95.8 | 30.3 GB | 50.5 GB |

## 3. Rejected-image reasons observed (rounds 2–4)
- **Identity drift on turned views:** 3/4 cosine 0.57–0.67 sits inside the lookalike band (worst with B at cfg 4 and with the angle LoRA).
- **Beauty mark multiplied** into 3–4 moles on the image-right cheek (config B, all cfgs; worst at cfg 4).
- **Beauty mark missing or inconsistent** (config A at low cfg).
- **No turn / wrong direction:** prompt-only R45 mostly stayed frontal or turned image-left. The angle LoRA ignores left vs right.
- **Synthetic textures:** shirt-texture artifacts and over-contrast at cfg 4. Waxy, smoothed skin with the softer config A at low cfg without the template.
- **Framing errors:** the template follows the Kontext-resized image1, so some crops come out tight. One B frontal cut off the hairline.
- **ArcFace not usable past |yaw| ≈ 45°:** profiles need manual review.
- No major anatomy defects were seen in round 4 outputs.

## 4. Rejection rules for dataset curation
ArcFace cosine is measured against canon (buffalo_l). Calibration from the controls:
- canon self = 1.0
- same-prompt lookalikes s1101/1102/1105/1108 = 0.623–0.659
- a different character = 0.20–0.37
- good frontal edits = 0.90–0.95

| Yaw band (insightface) | Min cosine | Justification |
|---|---|---|
| \|yaw\| < 15° | **0.85** | All accepted template frontals scored ≥ 0.904; 0.85 is about 0.19 above the best lookalike. |
| 15–35° | **0.75** | Accepted prompt-only 3/4s at 21–32° scored 0.76–0.80; this is about 0.09 above the lookalike max. |
| 35–50° | **0.68** | Just above the lookalike max (0.659). Most LoRA quarter views (0.57–0.67) fail, which matches the visual drift. |
| > 50° (profile) | **0.45 floor + mandatory manual review** | ArcFace is unreliable here. Real profiles scored 0.37–0.60, overlapping the different-character range (0.20–0.37), so the score is only a sanity floor. |

Other hard rejects:
- **Wrong direction:** the sign of yaw must match the slot (negative = face points image-left), and |yaw| must be within ±15° of the target.
- **Near-duplicates:** a candidate is a duplicate if, against an already accepted image, ArcFace cosine ≥ 0.97 **and** pHash Hamming distance ≤ 8 (64-bit). Also reject a second image from the same seed + slot. Keep the higher-scoring one.
- **Beauty-mark side error:**
  - In frontals it must sit on the image-right cheek (the subject's left cheekbone), exactly one mark.
  - When the face points image-left, the subject's left cheek is toward camera, so the mark must be visible.
  - When the face points image-right, it may be hidden. It must never appear on the subject's right cheek.
  - Multiplied marks are a reject.
- **Anatomy defects:** hands/fingers, ears, teeth, eye asymmetry or deformation, neck or shoulder geometry.
- **Synthetic textures:** waxy or plastic skin, patterned or noisy shirt texture, over-sharpening, halos.
- **Other:** collage or multiple views, text or labels, cropped head or hairline in head framings.

## 5. Exact 24-image pilot set (Qwen-Image 2.1 t2i character LoRA)
Default source = **E**: an Exp 2 template edit from canon (config A, cfg 2.5, 40 steps, CFGNorm, encoded latent). The prompt is the identity block plus the slot wording (frame-relative direction). Each slot is overgenerated with N seeds, filtered by the section 4 rules, and the best survivor is chosen visually.

Unless the slot says otherwise, the outfit is the canon plain gray T-shirt and the background plain light gray, so the LoRA learns the face and not the clothes. The varied slots break those correlations on purpose.

| # | View / yaw target | Framing | Expression | Lighting | Setting / outfit | Source, N seeds |
|---|---|---|---|---|---|---|
| 1 | front 0° | head & shoulders | neutral | soft daylight | canon studio, gray tee | E, 4 |
| 2 | front 0° | head & shoulders | soft smile | soft daylight | studio, gray tee | E, 4 |
| 3 | front 0° | head & shoulders | broad smile (teeth) | soft daylight | studio, gray tee | E, 6 |
| 4 | front 0° | close-up face | neutral | window side-light (warm) | indoor, white tee | E, 6 |
| 5 | front 0° | head & shoulders | slight frown / serious | overcast outdoor | park, denim jacket | E, 6 |
| 6 | front ~10° | head & shoulders | laughing | golden hour | street, black knit top | E, 8 |
| 7 | 3/4 image-left −30° | head & shoulders | neutral | soft daylight | studio, gray tee | E, 4 |
| 8 | 3/4 image-left −45° | head & shoulders | soft smile | window light | living room, cream sweater | E, 6 |
| 9 | 3/4 image-left −40° | close-up | neutral | evening tungsten | café, gray tee | E, 6 |
| 10 | 3/4 image-right +30° | head & shoulders | neutral | soft daylight | studio, gray tee | E, **10** (R45 hit rate 1/4) |
| 11 | 3/4 image-right +45° | head & shoulders | soft smile | overcast | balcony, navy blouse | E, **10** |
| 12 | 3/4 image-right +40° | close-up | thoughtful | rim light, dark bg | studio dark, black tee | E, **10** |
| 13 | profile image-left −80/−90° | head & shoulders | neutral | soft daylight | studio, gray tee | E, 4 (L90 4/4) + manual review |
| 14 | profile image-left −80/−90° | head & shoulders | slight smile | outdoor shade | park, denim jacket | E, 6 + manual |
| 15 | profile image-right +80/+90° | head & shoulders | neutral | soft daylight | studio, gray tee | E, 6 (R90 3/4) + manual |
| 16 | profile image-right +80/+90° | head & shoulders | neutral | window light | kitchen, white tee | E, 6 + manual |
| 17 | front 0° | half body (waist up) | neutral | soft daylight | studio, gray tee | E, 6 |
| 18 | 3/4 image-left −30° | half body | smile | daylight | office, white shirt | E, 8 |
| 19 | 3/4 image-right +30° | half body, seated | relaxed | warm lamp | sofa, cream sweater | E, 10 |
| 20 | front 0° | full body, standing | neutral | soft daylight | studio, gray tee + jeans | E, 8 (full-body framing not validated in round 4) |
| 21 | 3/4 image-left | full body, walking | smile | overcast | sidewalk, jacket + jeans | E, 10 |
| 22 | 3/4 image-right | full body, standing | neutral | golden hour | beach path, sundress (non-revealing) | E, 10 |
| 23 | front, slight down-tilt | head & shoulders | neutral | top light / phone-selfie style | hallway indoor, hoodie | E, 8 |
| 24 | front, low camera (−15° pitch) | head & shoulders | neutral | daylight | outdoor sky bg, gray tee | E, 8 |

Composition: 9 front or near-front, 6 three-quarter head shots (3 per side), 4 profiles (2 per side), 3 half-body, 3 full-body. Every two slots share at most one non-identity attribute, to limit outfit and background leakage.

**Held-out validation (4, never trained on; same rules, different seeds and prompts):**
- V1: front 0°, head & shoulders, neutral, new lighting (cool overcast), green sweater.
- V2: 3/4 image-right +40°, smile, café.
- V3: profile image-left, studio.
- V4: half-body 3/4 image-left, outdoor, rain jacket.

These are used to score the trained LoRA's t2i outputs: ArcFace against canon per yaw band, plus a blind review.

Captions: a unique trigger token, plus framing, view, expression, lighting and setting. Do not caption identity traits (eye colour, freckles, beauty mark), so they bind to the token.

Expected compute: about 190 generations at ~52 s each ≈ 2.8 h, then scoring on CPU.

## 6. Trainer compatibility (primary sources, checked 2026-10-16)
- **ai-toolkit (ostris):** supports Qwen-Image-2.1 as model arch **`qwen_image_2`**, a separate implementation from `qwen_image`.
  - Relevant commits: 086b663 (2026-09-20, 2.1 support, alpha channel), 07abdbe (2026-09-22, `match_target_res`), f370e74 (2026-10-10, Turbo via training adapter). main HEAD is dcebc74 (2026-10-16).
  - Pin a commit that includes 07abdbe or later.
  - A third-party guide reports a fix needed for transformers 5.5.3 in `qwen_image_2/src/text_encoder.py`. Use transformers ≥ 5.17, as the Qwen model card requires.
  - The repo has no dedicated 2.1 example YAML. Start from `train_lora_qwen_image_24gb.yaml` with `arch: qwen_image_2`, `name_or_path` pointing to the Comfy-Org 2.1 repack, and the network on the DiT only (the text encoder is frozen).
- **musubi-tuner (kohya-ss), HEAD f8a1b03 (2026-09-27):** lists Qwen-Image (v1/2512/Edit) only. **No Qwen-Image-2.1 support.**
- **CFG-distillation caveat:**
  - Train on the **base** Qwen-Image-2.1, which is not distilled: 40-step sampling with CFG per the model card.
  - **Qwen-Image-2.1-Turbo** is distilled to 8 steps at CFG 1. Training on it needs ai-toolkit's training adapter (`assistant_lora_path`, live while training, off while sampling). Training directly on a distilled checkpoint degrades the distillation.
  - Do not judge a base-trained LoRA with Turbo sampling, or the reverse.

## 7. License summary (commercial use)
| Weight | License | Commercial use |
|---|---|---|
| Qwen-Image-2.1 (+ its VAE and text-encoder packaging in the 2.1 repo / Comfy-Org repack) | Qwen Research License ("Non-Commercial = research or evaluation only") | **NO** without a separate license (contact per LICENSE §2b) |
| Qwen-Image-2.1-Turbo | same research license (`license: other`) | **NO** |
| Qwen-Image-Edit-2511 (+ Comfy-Org int8 repack) | Apache-2.0 | yes |
| Qwen-Image VAE (Edit pipeline) | Apache-2.0 | yes |
| Qwen2.5-VL-7B-Instruct (Edit text encoder) | Apache-2.0 | yes |
| Qwen3-VL-8B-Instruct (upstream of the 2.1 text encoder) | Apache-2.0 upstream; the 2.1-packaged copy ships under the 2.1 research license | treat as **NO** inside the 2.1 stack |
| fal Multiple-Angles LoRA (sha256 42426ded…6765) | Apache-2.0 | yes (not adopted) |
| InsightFace code | MIT | yes |
| InsightFace buffalo_l / antelopev2 models | non-commercial research only | **NO**, analysis only, never shipped |

⚠️ **Consequence:** a char01 LoRA trained on Qwen-Image-2.1 is a derivative of research-licensed Materials and **inherits the non-commercial restriction**. A distributed model trained using Qwen outputs must also carry "Built with Qwen".

Dataset images made with Edit-2511 (Apache-2.0) are not themselves restricted, but the LoRA is. For commercial UGC, either get a Qwen commercial license or train on an Apache-2.0 base (Qwen-Image / Qwen-Image-2512, or Edit-2511 itself).

Primary sources:
- https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen-Image-2.1
- https://github.com/ostris/ai-toolkit (extensions_built_in/diffusion_models/qwen_image_2)
- https://github.com/kohya-ss/musubi-tuner
- https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
- https://github.com/deepinsight/insightface#license
- HF model cards for Qwen-Image-Edit-2511, Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct (all apache-2.0)
