# Acceptance criteria (quality first)
Effective 2026-10-16. Every experiment and every image for a character dataset is judged in this order:

1. **Identity across angles.** It must be recognisably the same fictional person (bone structure, eye shape and colour, nose, lips, jaw, defining marks) in frontal, ¾ and profile views. Use ArcFace cosine vs the canon only as supporting evidence: check it against same-prompt lookalike controls, and remember it is unreliable beyond about 45° yaw.
2. **Photographic realism.** Natural skin texture (pores, freckles, no waxy smoothing), believable eyes and catchlights, real hair strands, physically plausible light, and smartphone-camera traits where the scene calls for them.
3. **Anatomy.** Correct proportions, hands, teeth, ears and eyes, and no fused or duplicated features.
4. **Prompt adherence.** Correct pose or angle, scene, clothing and expression.
5. **Resolution and artifacts.** Sharpness, and no texture patterns, halos, banding or compression.

**Hard rejects** (regardless of other scores):
- identity failure (reads as a different person, or as a lookalike rather than her);
- major anatomy defects;
- obvious synthetic artifacts (plastic or waxy skin, crinkled or patterned fabric texture, multiplied moles or marks, haloing, duplicated features).

**Selection policy:** judge by visual results first. Don't default to faster, distilled or heavily quantized setups when a higher-quality option fits the hardware (32 GB VRAM / about 59 GB RAM). Time, VRAM and RAM are recorded only as basic logs.
