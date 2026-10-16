"""Exp3 camera-angle test for Qwen-Image-Edit-2511 (derived from edit_cfg_exp.py; config A + official template:
CFGNorm(1.0), VAEEncode(FluxKontextImageScale(image1)) start latent, default 40 steps, cfg 2.5).
--mode prompt: frame-relative wording appended to the identity prompt.
--mode lora: LoraLoaderModelOnly(--lora, --strength) on the UNET and the LoRA's own trained phrasing only
            ("<sks> {azimuth} eye-level shot {distance}"), as in the LoRA's published workflow.
Views: front, L45, R45, L90, R90 (prompt mode = intended image direction; lora mode = the LoRA azimuth phrase,
achieved direction is measured afterwards). Paths/machine come from config.local.yaml."""
import argparse, json, os, sys, time, shutil, math, subprocess, threading, urllib.request, datetime
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg = config.load(); me = cfg.get("models_edit", {}); U = cfg["comfy_url"]; C = cfg["comfy_dir"]; CSV = cfg.get("times_csv")
a = argparse.ArgumentParser(); a.add_argument("--ref", required=True); a.add_argument("--mode", choices=["prompt","lora"], required=True)
a.add_argument("--out-dir", required=True); a.add_argument("--tag", default="char"); a.add_argument("--lora", default="qwen-image-edit-2511-multiple-angles-lora.safetensors"); a.add_argument("--strength", type=float, default=1.0); a.add_argument("--distance", default="medium shot")
a.add_argument("--seeds", type=int, nargs="+", default=[4101, 4102, 4103, 4104]); a.add_argument("--steps", type=int, default=40)
a.add_argument("--cfgs", type=float, nargs="+", default=[2.5]); a.add_argument("--views", nargs="+", default=["front","L45","R45","L90","R90"]); a.add_argument("--width", type=int, default=1088); a.add_argument("--height", type=int, default=1408)
o = a.parse_args()
BASE = ("She is 28. Preserve her exact facial identity, including bone structure, eye spacing, eye shape, hazel-green eye color, nose, lips, "
        "jawline, skin tone, freckles, natural asymmetries and the small beauty mark below her left cheekbone. Do not redesign, beautify, "
        "rejuvenate or reinterpret her. Keep the same dark chestnut hair, natural skin texture, plain gray T-shirt, plain light-gray background "
        "and soft even daylight. Head-and-shoulders portrait, face large in frame, eye-level camera. Single photo, no collage, labels, retouching or accessories.")
INTRO = {1: "Image 1 shows a fictional woman. Create a new photo of this same woman. ",
         2: "Image 1 and image 2 are two views of the same fictional woman (image 2 is a close-up of her face). Create a new photo of this same woman. "}
T45 = "Her head is turned about 45 degrees so her face points toward the {} edge of the image, neutral relaxed expression, mouth closed."
T90 = "Her head is turned 90 degrees into a full side profile so her face points toward the {} edge of the image, neutral relaxed expression, mouth closed."
PVIEWS = {"front": "She faces the camera directly, head straight, neutral relaxed expression, mouth closed.",
          "L45": T45.format("left"), "R45": T45.format("right"), "L90": T90.format("left"), "R90": T90.format("right")}
LAZ = {"front": "front view", "L45": "front-left quarter view", "R45": "front-right quarter view", "L90": "left side view", "R90": "right side view"}
VIEWS = {v: (PVIEWS[v] if o.mode == "prompt" else f"<sks> {LAZ[v]} eye-level shot {o.distance}") for v in o.views}
NEG = "collage, multiple views, text, labels, border, heavy makeup, beauty filter, retouched, plastic skin, cartoon, CGI, deformed eyes"
def stage(p, name): dst = os.path.join(C, "input", name); shutil.copy(p, dst) if not os.path.exists(dst) else None; return name
REF = stage(o.ref, f"r4_{o.tag}_ref.png")
def enc_dims(w, h):
    s = math.sqrt(384 * 384 / (w * h)); vl = (round(w * s), round(h * s))
    s = math.sqrt(1024 * 1024 / (w * h)); lat = (round(w * s / 8) * 8, round(h * s / 8) * 8); return vl, lat
post = lambda p, d: json.load(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})))
peak = [0]; min_avail = [10**12]; stop = [False]
def mon():
    while not stop[0]:
        try: peak[0] = max(peak[0], int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0]))
        except Exception: pass
        try: min_avail[0] = min(min_avail[0], int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]))
        except Exception: pass
        time.sleep(0.3)
threading.Thread(target=mon, daemon=True).start()
rows = []; first = True; os.makedirs(o.out_dir, exist_ok=True)
o.quality = True
for cf, CFGV in [("A", v) for v in o.cfgs]:
    dual = False; alt = False; arm = (f"P_q{o.steps}s_cfg{CFGV:g}" if o.mode == "prompt" else f"L{o.strength:g}_q{o.steps}s_cfg{CFGV:g}"); d = os.path.join(o.out_dir, arm); os.makedirs(d, exist_ok=True)
    for view, vtxt in VIEWS.items():
        for seed in o.seeds:
            P = (INTRO[1] + BASE + " " + vtxt) if o.mode == "prompt" else vtxt
            w = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": me.get("unet", "qwen_image_edit_2511_int8_convrot.safetensors"), "weight_dtype": "default"}},
                 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": me.get("clip", "qwen_2.5_vl_7b_fp8_scaled.safetensors"), "type": "qwen_image", "device": "default"}},
                 "3": {"class_type": "VAELoader", "inputs": {"vae_name": me.get("vae", "qwen_image_vae.safetensors")}},
                 "4": {"class_type": "LoadImage", "inputs": {"image": REF}},
                 "5": ({"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}} if alt
                       else {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}}),
                 "20": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
                 "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": P, "vae": ["3", 0], "image1": ["5", 0]}},
                 "13": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": NEG, "vae": ["3", 0], "image1": ["5", 0]}},
                 "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": float(me.get("shift", 3.1))}},
                 "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": o.width, "height": o.height, "batch_size": 1}},
                 "9": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["6", 0], "negative": ["13", 0], "latent_image": ["8", 0], "seed": seed,
                       "steps": o.steps, "cfg": CFGV, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
                 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"r4exp3_{o.tag}/{arm}_{view}_s{seed}"}}}
            if o.quality:
                w["15"] = {"class_type": "CFGNorm", "inputs": {"model": ["7", 0], "strength": 1.0}}; w["9"]["inputs"]["model"] = ["15", 0]
                w["8"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}}
            if o.mode == "lora":
                w["16"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": o.lora, "strength_model": o.strength}}; w["7"]["inputs"]["model"] = ["16", 0]
            if dual:
                w["14"] = {"class_type": "LoadImage", "inputs": {"image": CROP}}
                w["21"] = {"class_type": "PreviewImage", "inputs": {"images": ["14", 0]}}
                w["6"]["inputs"]["image2"] = ["14", 0]; w["13"]["inputs"]["image2"] = ["14", 0]
            peak[0] = 0; min_avail[0] = 10**12; t = time.time(); pid = post("/prompt", {"prompt": w})["prompt_id"]
            while True:
                time.sleep(0.5); h = json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
                if pid in h: break
            dt = round(time.time() - t, 1); st = h[pid]["status"].get("status_str"); out = h[pid]["outputs"]
            def dims(node):
                im = out.get(node, {}).get("images", [])
                if not im: return None
                x = im[0]; return list(Image.open(os.path.join(C, x.get("type", "temp"), x["subfolder"], x["filename"])).size)
            i1 = dims("20"); i2 = dims("21") if dual else None
            if i1 is None:  # cached node: no preview emitted; reuse dims from an earlier row of the same config
                i1 = next((r["image1_dims"] for r in rows if r["config"] == cf), None)
            if dual and i2 is None: i2 = next((r["image2_dims"] for r in rows if r["config"] == cf), None)
            e1 = enc_dims(*i1) if i1 else (None, None); e2 = enc_dims(*i2) if i2 else (None, None)
            f = None
            if st == "success":
                x = out["11"]["images"][0]; f = os.path.join(d, f"{view}_s{seed}.png"); shutil.copy(os.path.join(C, "output", x["subfolder"], x["filename"]), f)
            r = dict(mode=o.mode, strength=(o.strength if o.mode == "lora" else None), prompt=P, arm=arm, config=cf, quality=o.quality, view=view, seed=seed, status=st, sec=dt, peak_vram_mib=peak[0], min_ram_available_mib=min_avail[0] // 1024, file=f, image1_dims=i1, image1_vl_dims=e1[0], image1_ref_latent_px=e1[1],
                     image2_dims=i2, image2_vl_dims=e2[0], image2_ref_latent_px=e2[1], steps=o.steps, cfg=CFGV, width=(i1 or [o.width])[0] if o.quality else o.width, height=(i1 or [0, o.height])[1] if o.quality else o.height, prompt_id=pid)
            json.dump({"meta": r, "workflow_api": w}, open(os.path.join(d, f"{view}_s{seed}.workflow.json"), "w"), indent=1)
            rows.append(r); print(r, flush=True)
            if CSV and st == "success":
                with open(CSV, "a") as fh:
                    fh.write(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')},qwen_image_edit_2511_int8_{'anglelora' if o.mode == 'lora' else 'nolora'},r4exp3_{o.tag}_{arm}_{view},{o.width}x{o.height},{o.steps},{dt},{'first' if first else 'warm'},round4/exp3/{arm}/{view}_s{seed}\n")
            first = False
            if st != "success": print(json.dumps(h[pid]["status"])[-1500:]); stop[0] = True; sys.exit(1)
stop[0] = True; json.dump(rows, open(os.path.join(o.out_dir, f"gen_results_{arm}.json"), "w"), indent=1)
