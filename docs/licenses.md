# Model licenses (checked 2026-10-16 from the Hugging Face model cards / LICENSE files)

| Model | Used as | License | Commercial use? |
|---|---|---|---|
| Qwen/Qwen-Image-2.1 (int8 "convrot" ComfyUI repack) | base t2i model | **Qwen RESEARCH LICENSE (2026-09-20)** | **NO, research/evaluation only.** A separate commercial license is required from Qwen. ⚠️ |
| Qwen/Qwen-Image-2.1-Turbo | (not used) | other (Qwen license) | check before use |
| Qwen/Qwen3-VL-8B-Instruct (int8 repack) | text encoder for 2.1 | Apache-2.0 | yes |
| Qwen/Qwen-Image-Edit-2511 (Comfy-Org int8 repack) | reference editing | Apache-2.0 | yes |
| Qwen/Qwen2.5-VL-7B-Instruct (fp8 repack) | text encoder for Edit | Apache-2.0 | yes |
| Qwen-Image VAE | VAE for Edit | Apache-2.0 (Qwen-Image) | yes |
| InsightFace (code, python package) | identity scoring only (analysis) | MIT | yes (code only) |
| InsightFace pretrained models: buffalo_l, antelopev2 (auto-downloaded by the python package) | ArcFace embeddings for identity scoring | **non-commercial research only** (trained on research-licensed data) | **NO**. Analysis only; never ship them or use them in a product. |
| fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA (`qwen-image-edit-2511-multiple-angles-lora.safetensors`, 295,140,688 bytes, sha256 `42426ded4e25fd22879d9e198b857556445ef4ca56e8da3246d0345155bb6765`, HF revision e306622) | camera-angle LoRA for Edit-2511 (rank 16, 840 bf16 targets, diffusers key format) | Apache-2.0 (model card metadata) | yes; training data = fal's synthetic Gaussian-splatting renders (provenance not further documented) |
| LoRAs trained here on Qwen-Image-2.1 | character identity | derivative of a research-licensed model | inherits the non-commercial restriction |

⚠️ **Qwen-Image 2.1 appears to be research-only.** Its LICENSE grants rights "FOR NON-COMMERCIAL PURPOSES ONLY" and defines Non-Commercial as "for research or evaluation purposes only". Commercial UGC needs either a commercial license or a switch to an Apache-2.0 base (e.g. Qwen-Image / Qwen-Image-Edit-2511).
Primary source: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE

InsightFace source: https://github.com/deepinsight/insightface#license. The README says the code is MIT, and that the training data and the models trained on it (including those auto-downloaded by the python library) are "available for non-commercial research purposes only".

Angle LoRA source: https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA (license: apache-2.0 in the model card header; base model Qwen/Qwen-Image-Edit-2511).
