# synthetics-person

Tooling and notes for generating **fully fictional, synthetic adult characters** for UGC-style (smartphone-look) imagery with ComfyUI, using the Qwen-Image model family and per-character LoRAs, on a single workstation GPU.

**Hardware:** one NVIDIA GPU with 24–32 GB VRAM (developed on a 32 GB card), 64 GB system RAM recommended, ~150 GB disk for models.

## Quickstart
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and download the models listed in [docs/licenses.md](docs/licenses.md) into its `models/` folders. Check every license first.
2. `cp config.example.yaml config.local.yaml` and fill in your paths (the local file is gitignored).
3. `pip install pyyaml`, then start ComfyUI.
4. Canonical reference candidates for a new character (no LoRA):
   `python scripts/gen_canon.py --prompt-file templates/canon_prompt.txt --neg-file templates/canon_negative.txt --out-dir out/char01 --seeds 1101 1102 1103`
5. Identity-consistency test with a character LoRA: `OUT_SUB=idexp/lora_only python scripts/idexp.py`

## Docs
- [docs/FINDINGS.md](docs/FINDINGS.md): experiment log
- [docs/licenses.md](docs/licenses.md): model licenses
- [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
- `templates/`: character spec and prompt templates

No weights, images, datasets or secrets are stored in this repo.
