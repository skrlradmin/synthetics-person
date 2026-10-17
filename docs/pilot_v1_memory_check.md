# Qwen-Image-2512 LoRA training: memory and download check (ai-toolkit code review, no download/install/run)

Source: the ai-toolkit files in `src/` (`aitk_qwen_image.py`, `v2_qi.py`, `mixin.py`, `base_model.py`, `qwen25_vl.py`, `BaseSDTrainProcess.py`, `SDTrainer.py`). Read only, nothing executed.

## What the code does with arch qwen_image + quantize qfloat8 + low_vram + cached TE + gradient checkpointing
- `component_load_kwargs` (base_model.py:1707): transformer qtype = `model.qtype` (qfloat8) when `quantize: true`; TE qtype = `qtype_te` when `quantize_te: true`. With `low_vram`, the transformer and TE are loaded **and quantized on the CPU**. Then `QwenImageTransformer2DModel.load()` quantizes after load (mixin.py `quantize_`, block by block).
- `aitk_qwen_image.load_model`: the vision tower is dropped *before* quantization (t2i). With low_vram, the TE stays on the CPU and is moved to the GPU on demand. The VAE gets tiled decode for samples.
- `cache_text_embeddings: true` → `unload_text_encoder` (BaseSDTrainProcess.py:232, SDTrainer.py:371-398). The TE is used once to cache prompt embeddings and then unloaded, so it takes **no VRAM during training**.
- `gradient_checkpointing: true` → `enable_gradient_checkpointing` on the transformer and the LoRA network (BaseSDTrainProcess.py:1841, 2019).
- `layer_offloading` (base_model.py:1719) is a separate opt-in, **not needed** at 32 GB. Keep it off: per BaseSDTrainProcess.py:647 it also blocks merging.

## Expected memory on Harpy (RTX 5090 32 GB, ~59 GB RAM / ~49 GB free), 1024 px buckets, bs 1, rank 16
Estimates from parameter counts, not measured.
| Phase | VRAM | System RAM |
|---|---|---|
| Load + quantize the bf16 DiT on the CPU (20.4B params) | ~0 | **peak ~42-46 GB** (bf16 40.9 GB streamed, then 8-bit ~20.4 GB). This is the tight spot: stop ComfyUI first (it holds ~20+ GB RAM/VRAM). |
| TE caching (Qwen2.5-VL 7B, qfloat8 without the vision tower) | ~8-9 GB transient | ~16 GB while loading, then freed |
| Training steps (qfloat8 DiT ~20.5 GB + LoRA/optimizer <0.5 GB + checkpointed activations) | **~24-27 GB peak** (1024), ~29-31 GB at 1328 | ~22-26 GB steady |
| Sampling (tiled VAE decode) | ~23-26 GB | – |
Verdict: fits at 1024 with ~5 GB headroom. Keep 1328 off for the pilot. ComfyUI must be stopped during training. The pilot generation peaked at ~30 GB VRAM in ComfyUI, so the two cannot share the GPU.

## Download needed (exact)
- Transformer registry: `v2_qi.py` registers ComfyUI single-file weights **only for `Qwen/Qwen-Image`** (v1), not for 2512. So `name_or_path: Qwen/Qwen-Image-2512` pulls the full diffusers repo: **~57.7 GB** (transformer ~40.9 GB + text_encoder ~16.6 GB + VAE ~0.25 GB + tokenizer and scheduler).
- **Can local files substitute?**
  - **Transformer: yes.** `name_or_path` may be a local `.safetensors`. `mixin.load_model` loads single files, and `v2_qi._load_single_file` uses `from_single_file` with config `Qwen/Qwen-Image`, which has an identical architecture config (only small config JSONs are fetched). So the ComfyUI file `qwen_image_2512_bf16.safetensors` (40.86 GB) is a drop-in. The fp8_e4m3fn version (20.43 GB) also loads, but it would be re-quantized from an fp8 cast, so it is lower quality and not recommended for training.
  - **Text encoder / VAE / tokenizer: no.** These come from `extras_name_or_path` in **diffusers layout** (`text_encoder/`, `vae/`, `tokenizer/`). If that is a `.safetensors` file, the code falls back to the `Qwen/Qwen-Image` repo. The local ComfyUI `qwen_2.5_vl_7b_fp8_scaled` and `qwen_image_vae` single files have no loader path here. The 2512 TE and VAE are byte-identical to Qwen-Image v1 (same sha256), so `extras_name_or_path: Qwen/Qwen-Image` is valid. It still downloads `text_encoder/` + `vae/` + `tokenizer/`, about **16.9 GB**.
  - **Net:** local substitution only saves space if a bf16 2512 DiT is already on disk. It isn't. Harpy has the Edit-2511 int8 DiT and the fp8 TE, and neither is usable. Options:
    - (a) the full `Qwen/Qwen-Image-2512` diffusers repo, **57.7 GB**, simplest. **Recommended.**
    - (b) ComfyUI `qwen_image_2512_bf16.safetensors` 40.86 GB + diffusers `text_encoder/ vae/ tokenizer/` ~16.9 GB = ~57.8 GB. Same size, but the bf16 single file is reused for ComfyUI inference.
  - Both need approval (>20 GB), and so does installing the trainer env. **Option (b) is preferred** if 2512 inference in ComfyUI is also planned: it avoids storing a second 40.9 GB copy of the DiT.
