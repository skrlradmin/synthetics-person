# Model licenses (checked 2026-10-16 from the Hugging Face model cards / LICENSE files)

| Model | Used as | License | Commercial use? |
|---|---|---|---|
| Qwen/Qwen-Image-2.1 (int8 "convrot" ComfyUI repack) | base t2i model | **Qwen RESEARCH LICENSE (2026-09-20)** | **NO, research/evaluation only.** A separate commercial license is required from Qwen. ⚠️ |
| Qwen/Qwen-Image-2.1-Turbo | (not used) | other (Qwen license) | check before use |
| Qwen/Qwen3-VL-8B-Instruct (int8 repack) | text encoder for 2.1 | Apache-2.0 | yes |
| Qwen/Qwen-Image-Edit-2511 (Comfy-Org int8 repack) | reference editing | Apache-2.0 | yes |
| Qwen/Qwen2.5-VL-7B-Instruct (fp8 repack) | text encoder for Edit | Apache-2.0 | yes |
| Qwen-Image VAE | VAE for Edit | Apache-2.0 (Qwen-Image) | yes |
| LoRAs trained here on Qwen-Image-2.1 | character identity | derivative of a research-licensed model | inherits the non-commercial restriction |

⚠️ **Qwen-Image 2.1 appears to be research-only.** Its LICENSE grants rights "FOR NON-COMMERCIAL PURPOSES ONLY" and defines Non-Commercial as "for research or evaluation purposes only". Commercial UGC needs either a commercial license or a switch to an Apache-2.0 base (e.g. Qwen-Image / Qwen-Image-Edit-2511).
Primary source: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
