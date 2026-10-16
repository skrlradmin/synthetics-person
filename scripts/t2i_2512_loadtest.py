"""Load test: Qwen-Image-2512 text-to-image in ComfyUI (no LoRA).
Model-card settings: 50 steps, true CFG 4.0, 1104x1472, official negative prompt.
Uses the shared Qwen-Image text encoder / VAE. Records time, peak VRAM, min available RAM.
Usage: python t2i_2512_loadtest.py --prompt-file templates/canon_prompt.txt --out-dir DIR [--seeds 5101 5102]
Machine/paths come from config.local.yaml (comfy_url, comfy_dir, models_2512.* optional)."""
import argparse, json, os, sys, time, shutil, subprocess, threading, urllib.request
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg = config.load(); m = cfg.get("models_2512", {}); U = cfg["comfy_url"]; C = cfg["comfy_dir"]
a = argparse.ArgumentParser(); a.add_argument("--prompt-file", required=True); a.add_argument("--out-dir", required=True)
a.add_argument("--seeds", type=int, nargs="+", default=[5101, 5102]); a.add_argument("--steps", type=int, default=50)
a.add_argument("--cfg", type=float, default=4.0); a.add_argument("--width", type=int, default=1104); a.add_argument("--height", type=int, default=1472)
a.add_argument("--shift", type=float, default=3.1); o = a.parse_args()
NEG = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"  # model card
P = open(o.prompt_file).read().strip()
post = lambda d: json.load(urllib.request.urlopen(urllib.request.Request(U + "/prompt", data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})))
peak = [0]; avail = [10**12]; stop = [False]
def mon():
    while not stop[0]:
        try: peak[0] = max(peak[0], int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0]))
        except Exception: pass
        try: avail[0] = min(avail[0], int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024)
        except Exception: pass
        time.sleep(0.3)
threading.Thread(target=mon, daemon=True).start(); os.makedirs(o.out_dir, exist_ok=True); rows = []
for seed in o.seeds:
    w = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": m.get("unet", "qwen_image_2512_fp8_e4m3fn.safetensors"), "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m.get("clip", "qwen_2.5_vl_7b_fp8_scaled.safetensors"), "type": "qwen_image", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": m.get("vae", "qwen_image_vae.safetensors")}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": P}},
         "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEG}},
         "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": o.shift}},
         "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": o.width, "height": o.height, "batch_size": 1}},
         "8": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0], "seed": seed,
               "steps": o.steps, "cfg": o.cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
         "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
         "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"qi2512_loadtest/s{seed}"}}}
    peak[0] = 0; avail[0] = 10**12; t = time.time(); pid = post({"prompt": w})["prompt_id"]
    while True:
        time.sleep(0.5); h = json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
        if pid in h: break
    dt = round(time.time() - t, 1); st = h[pid]["status"].get("status_str"); f = None
    if st == "success":
        x = h[pid]["outputs"]["10"]["images"][0]; f = os.path.join(o.out_dir, f"t2i_s{seed}.png"); shutil.copy(os.path.join(C, "output", x["subfolder"], x["filename"]), f)
    r = dict(seed=seed, status=st, sec=dt, peak_vram_mib=peak[0], min_ram_available_mib=avail[0], steps=o.steps, cfg=o.cfg, width=o.width, height=o.height, shift=o.shift, file=f, prompt_id=pid)
    json.dump({"meta": r, "workflow_api": w}, open(os.path.join(o.out_dir, f"t2i_s{seed}.workflow.json"), "w"), indent=1, ensure_ascii=False)
    rows.append(r); print(r, flush=True)
    if st != "success": print(json.dumps(h[pid]["status"])[-1500:]); break
stop[0] = True; json.dump(rows, open(os.path.join(o.out_dir, "loadtest_results.json"), "w"), indent=1)
