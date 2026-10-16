# Qwen-Image 2.1 (production int8 checkpoint): architecture and training compatibility
This is a report only; nothing was trained. Verified 2026-10-16 from the checkpoint's tensor keys and from primary sources.

## From the checkpoint file (`qwen_image_2.1_int8_convrot.safetensors`, sha256 cb74113c…eaa57d)
- 649 tensors. Top-level groups: `transformer_blocks` (32 blocks), `txt_in` (in_layer/out_layer/text_norm 4096), `img_in` [4096,64], `proj_out` [64,4096], `time_text_embed.timestep_embedder`, a global `modulation.1` [16384,4096], and `norm_out`.
- Each block has one `attn.{to_q,to_k,to_v,to_out.0}`, `attn.norm_q/k` and one `img_mlp.{gate_up,out}`. There are **no separate text-stream weights** (no add_q_proj or txt_mlp), so it is a single-stream DiT with a shared global modulation, **not** the dual-stream 60-block MMDiT of Qwen-Image v1 / Edit-2511.
- Quantization: ComfyUI "convrot" int8. Per linear layer: I8 weight, F32 `weight_scale` [out,1], U8 `comfy_quant` descriptor. Norms, embeddings and modulation are BF16. Metadata is only `{format: pt}`, with no source or provenance fields.
- Text encoder: Qwen3-VL-8B (int8 repack; LM hidden size 4096, which matches `txt_in` 4096; includes the vision tower).
- VAE: `qwen_image_2.1_vae_bf16` (238 tensors; has `modelspec.architecture` metadata). It is **different** from the Qwen-Image v1 VAE.
- Upstream config (HF `Qwen/Qwen-Image-2.1/transformer/config.json`): `QwenImage21Transformer2DModel`, num_layers 32, 32 heads × 128 = 4096, context_in_dim 4096, in/out 64 channels, patch_size 1, mlp_ratio 3, rope axes [16,56,56], causal_condition true. **This matches the file.**
- For comparison, Qwen-Image-Edit-2511 int8 has 60 blocks, hidden 3072, `txt_in` [3072,3584] (Qwen2.5-VL-7B). Its architecture is incompatible with 2.1, so LoRAs don't transfer between them.

## Trainer support (primary sources, checked 2026-10-16)
| Trainer | Qwen-Image 2.1 LoRA | Notes / source |
|---|---|---|
| ostris/ai-toolkit | **Yes** (T2I, plus edit when control images are supplied) | README lists Qwen-Image-2.1 and 2.1-Turbo; arch `qwen_image_2`; supports convrot8 quantization. The existing character LoRA was made with ai-toolkit 0.13.25 (`ss_base_model_version: qwen_image_2`). |
| tdrussell/diffusion-pipe | **Yes, T2I only** | Changelog 2026-09-28 "Support Qwen-Image-2.1 (T2I training only)"; supported_models.md type `qwen_image21`, ComfyUI-format files, LoRA on quantized models. **Warns that 2.1 is CFG-distilled: training without a de-distillation adapter breaks distillation, so you then need CFG at inference.** |
| kohya-ss/musubi-tuner | **Yes (listed)** | README model list includes Qwen/Qwen-Image-2.1 and 2.1-Turbo. A third-party guide from 2026-09-21 said it wasn't supported yet, so support is recent; check the docs/qwen_image.md options before use. |
| kohya-ss/sd-scripts | **No** | README only mentions the Qwen-Image VAE (PR #2382); there is no 2.1 training support. |

License reminder: Qwen-Image 2.1 is under the Qwen RESEARCH LICENSE (non-commercial). LoRAs trained on it inherit that restriction.
