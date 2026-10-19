# Studio stills workflow (Qwen-Image 2.1)

Generic method notes for fictional adult-character UGC stills on a single 24–32 GB GPU. No persona data in this file.

## Main stills — character LoRA + Qwen-Image 2.1 int8 BASE
- UNET: `qwen_image_2.1_int8_convrot`
- Text encoder: `qwen3vl_8b_int8_convrot` (CLIPLoader type `qwen_image`)
- VAE: `qwen_image_2.1_vae_bf16`
- TextEncodeQwenImage21 (resolution 1024) + LoraLoaderModelOnly
- 25 steps, KSampler euler / simple, cfg 3, 832×1216 (864×1152 for 3:4)
- Character LoRA strength: about **0.8** for clothed / **0.7** when a larger bust cue is in play; trigger is the character token + `woman`
- Prompt: one soft size cue (e.g. `natural DDD proportions` or `full natural breasts`) plus `Proportionate natural figure, natural body proportions, realistic everyday build.`
- Negatives that help avoid round/augmented busts: exaggerated proportions, oversized chest, augmented breasts, round fake breasts, breast implants, perky lifted breasts, thick thighs, wide hips, inflated body
- Do **not** stack size words (`very large` + cup size + `full heavy`); stacked wording tends to re-inflate into a round look
- Selfies: `Front camera, 26mm lens at arm's length`; short realism line; amateur smartphone look only (no studio lighting / bokeh / retouching)
- Edit-mode refs (TextEncodeQwenImage21 `image_1` / `image_2`): tested on this stack; they hurt stills (posed/pro look, re-inflated bust, ~4× slower). Prefer not using them for stills
- Face pass (optional): crop face (~448 px) → scale to 1024 → img2img denoise 0.25–0.35, 20 steps, cfg 3, same LoRA → scale back → ImageCompositeMasked with feathered SolidMask. Impact FaceDetailer fails on Qwen 2.1 conditioning (`Tensor` has no attribute `copy`)

## Drafts — Qwen-Image 2.1 Turbo
- Official Comfy-Org: `qwen_image_2.1_turbo_int8_convrot.safetensors` (sha256 starts `8144a018`)
- Same TE, VAE, and character LoRA
- 8 steps: SamplerCustomAdvanced + ManualSigmas `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0`, KSamplerSelect euler, CFGGuider cfg 1
- About **2 s vs ~9 s** for base at 832×1216; same seed ≈ same composition; smoother skin on Turbo
- **Drafts / seed hunting only; finals on base**
- Unofficial Turbo conversions can fail with `Could not detect model type` — verify the Comfy-Org SHA

## Pose control — Fun ControlNet Union
- ModelPatchLoader `qwen_image_2.1_fun_controlnet_union_int8_convrot` → QwenImageDiffsynthControlnet, strength 0.8–1.0, start 0, end ~0.5
- Pose map at output size (DWPose/rtmlib: body+hands, drop face points)
- **Hard rule:** depth / canny / pose maps used in generation come from **synthetic renders only**. Real photos: OpenPose skeleton extraction only; never feed the photo or depth/edge maps into generation; delete temps
- Multi-person photos tangle skeletons — keep the primary subject skeleton and place others via prompt

## Ops
- Prefer image jobs without a large LLM resident (VRAM)
- Log cold vs warm times; ComfyUI caches identical prompts — nudge a value (e.g. LoRA `0.80001`) for warm timing reruns
- Paths: use placeholders like `<COMFY_ROOT>/models/...` in configs; never commit hostnames, IPs, usernames, or home paths

## QA checklist (every still)
- Finger / toe count, extra heads or people, face identity vs character B
- Bust shape vs brief (natural weight vs round/augmented)
- Amateur look; no text / watermarks
- Reject hard anatomy failures

## Standing rules
- Fully fictional adult characters only; never depict real people
- Amateur smartphone realism only
- Check each model license before commercial use of outputs
