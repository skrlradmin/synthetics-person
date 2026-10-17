# char01 pilot v1 dataset: summary (2026-10-16)

**Generation.** Approved Edit-2511 workflow: qwen_image_edit_2511_int8_convrot, a single reference image1 = canon s1103 (sha256 e0e7f62e… verified unchanged before and after), CFGNorm template, VAE-encoded start latent, 40 steps, cfg 2.5, 1088x1408, no LoRA, prompt-only angle wording.
- 200 generations (170 training-slot + 30 validation seeds), ~51.7 s each, peak VRAM ~30 GB. It ran detached from 16:55 to 19:48 PT.
- Full-res PNGs and per-image metadata (meta + workflow_api JSON) are untouched on the GPU host, in `char01/dataset_pilot_v1/candidates/`. This box holds 640-px JPEG thumbnails only.

**Filters** (thresholds unchanged): ArcFace vs canon per yaw band (0.85 / 0.75 / 0.68; >50° goes to manual review), |yaw − target| ≤ 15, near-duplicate (cos ≥ 0.97 and pHash ≤ 8), one best candidate per slot.

## Slot fill and acceptance by category
| Category | Slots | Filled (provisional) | Manual review | Unfilled | Candidates passing auto gates |
|---|---|---|---|---|---|
| frontal (01-06) | 6 | 3 (01, 02, 05) | 0 | 3 (03, 04, 06) | 6/34 (18%) |
| three-quarter image-left (07-09) | 3 | 2 (07, 09) | 0 | 1 (08) | 5/16 (31%) |
| three-quarter image-right (10-12) | 3 | 1 (10) | 0 | 2 (11, 12) | 2/30 (7%) |
| profile (13-16) | 4 | 0 | 3 (14, 15, 16) | 1 (13) | 7/22 (32%) direction-OK, identity manual |
| half body (17-19) | 3 | 1 (17) | 0 | 2 (18, 19) | 3/24 (13%) |
| full body (20-22) | 3 | 0 | 0 | 3 | 0/28 (0%) |
| camera variation (23-24) | 2 | 1 (23) | 0 | 1 (24) | 1/16 (6%) |
| **training total** | **24** | **8** | **3** | **13** | 24/170 (14%) |
| validation (V1-V4) | 4 | 1 (V3, profile, identity manual) | – | 3 (V1, V2, V4) | 3/30 |

Rejection reasons, counted per reason (one image can have more than one): identity below threshold 141, wrong direction 67, near-duplicate 2.

## Review list (uncertain: not accepted)
- 14_s62403 (profile image-left, yaw −73.9, cos 0.508)
- 15_s62500 (profile image-right, yaw 71.7, cos 0.462)
- 16_s62601 (profile image-right, yaw 79.5, cos 0.527)
- V3_s69305 (validation profile, yaw −74.3, cos 0.559)

All are above the 0.45 floor. They look plausible at thumbnail size, but a human check at full resolution is needed.

## Body-geometry flags
- 17_s62700 (half body, cos 0.947): **face OK, body geometry uncertain**. The waist-up framing and arms look plausible in the thumbnail but haven't been verified against a body spec.
- No full-body candidate passed the face gate. The best was 20_s63007 at cos 0.797 against a 0.85 threshold. Faces at full-body scale are ~60-90 px, where ArcFace is unreliable.

## Visual checks (thumbnail level; full-res confirmation pending)
- The beauty mark is visible as a single mark on the image-right cheek in 01 and 17, and was not seen multiplied in any accepted image. In 02, 05 and 23 it is too small to confirm at 640 px, so those are marked `visual_pending`.
- No anatomy, texture or crop defects were seen in the 8 accepted images.
- **Diversity concern:** 6 of the 8 accepted images are grey T-shirt / plain studio. The varied-clothing, warm-light, smiling and outdoor slots are the ones failing identity.

## Shortages and recommended corrections (no thresholds relaxed)
1. **Expression and lighting drift kills identity.** Smiles and laughs (03, 06), golden-hour and rim light (06, 12) and warm interiors drop cosine to 0.59-0.80. Fixes:
   - Re-run these slots with 12-16 seeds each.
   - Move strong smiles to milder "slight smile".
   - Keep lighting variation but avoid hard colour casts.
2. **Three-quarter image-right direction failures (11, 12, 08):** many outputs turn the wrong way. Fixes:
   - Add explicit frame-relative wording plus a nose cue: "her nose points toward the right edge of the image".
   - Overgenerate 16 seeds for each of 11, 12 and 08.
3. **Full body (0/3):** the face is too small for ArcFace, so the gate is not meaningful at this scale. Fixes:
   - (a) Score on a 2x-upscaled face crop, with the same thresholds.
   - (b) Generate 3/4-length (knees-up) instead of head-to-shoes.
   - (c) Or generate full body, then run a face-detail pass from canon at the same seed.
   - Until then, leave full-body slots empty rather than relax the gate.
4. **Half body 18/19, camera 24, validation V1/V2/V4:** these are near misses (cos 0.70-0.81). Re-run with 12 seeds each, using neutral expressions and cooler, even light.
5. **Estimated top-up cost:** ~13 unfilled slots + 3 validation slots × ~12 seeds ≈ 190 generations ≈ 2.8 h on the GPU host (at the measured ~52 s per image).

## Files
- `acceptance.csv` / `acceptance.json`: per image: cosine, yaw, pitch, threshold, status, rejection reason, flags, body_geometry, prompt, timing and VRAM.
- `category_stats.json`
- `approved_contact_sheet.jpg`, `review_contact_sheet.jpg`, `validation_contact_sheet.jpg`
- `unfilled_best_rejects_sheet.jpg`: the best reject per unfilled slot, for diagnosis.
- `memory_check.md`, `training_config_draft.yaml`
