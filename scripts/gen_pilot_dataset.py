"""Generate pilot-dataset candidates with the approved Edit-2511 workflow (Round 4 Exp 2 winner):
config A = single reference (canon) via FluxKontextImageScale; official template = CFGNorm(1.0) + VAE-encoded scaled
image1 as start latent (output = Kontext-scaled canon size, 880x1184 for a 1088x1408 canon); 40 steps; cfg 2.5; euler/simple;
prompt-only, frame-relative angle wording. Slots from templates/pilot_v1_slots.json. Resumable (skips existing files).
Usage: python gen_pilot_dataset.py --ref canon.png --root DATASET_DIR [--only 01 V1 ...]
Writes DATASET_DIR/candidates/{slot}_s{seed}.png + .json and gen_log.jsonl. Paths/machine from config.local.yaml."""
import argparse, json, os, sys, time, shutil, subprocess, threading, urllib.request, datetime
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg = config.load(); me = cfg.get("models_edit", {}); U = cfg["comfy_url"]; C = cfg["comfy_dir"]; CSV = cfg.get("times_csv")
a = argparse.ArgumentParser(); a.add_argument("--ref", required=True); a.add_argument("--root", required=True); a.add_argument("--tag", default="char01")
a.add_argument("--slots", default=os.path.join(os.path.dirname(__file__), "..", "templates", "pilot_v1_slots.json")); a.add_argument("--only", nargs="*")
a.add_argument("--steps", type=int, default=40); a.add_argument("--cfg", type=float, default=2.5); o = a.parse_args()
S = json.load(open(o.slots)); ALL = [(s, False) for s in S["slots"]] + [(s, True) for s in S["validation"]]
INTRO = "Image 1 shows a fictional woman. Create a new photo of this same woman. "
ID = ("She is 28. Preserve her exact facial identity, including bone structure, eye spacing, eye shape, hazel-green eye color, nose, lips, "
      "jawline, skin tone, freckles, natural asymmetries and the small beauty mark below her left cheekbone. Do not redesign, beautify, "
      "rejuvenate or reinterpret her. Keep the same dark chestnut shoulder-length wavy hair and natural skin texture with visible pores. "
      "Single realistic photo, no collage, labels, retouching, heavy makeup or accessories.")
NEG = "collage, multiple views, text, labels, border, heavy makeup, beauty filter, retouched, plastic skin, cartoon, CGI, deformed eyes, deformed hands, extra fingers"
def direction(y):
    if abs(y) < 5: return "She faces the camera directly, head straight."
    side = "left" if y < 0 else "right"
    if abs(y) <= 15: return f"Her head is turned very slightly, about {abs(y)} degrees, toward the {side} edge of the image."
    if abs(y) <= 50: return f"Her head and shoulders are turned about {abs(y)} degrees so her face points toward the {side} edge of the image."
    return f"Her head is turned 90 degrees into a full side profile so her face points toward the {side} edge of the image; only one side of her face is visible."
def prompt(s): return f"{INTRO}{ID} {s['framing']}. {direction(s['yaw'])} {s['expr'].capitalize()}. Lighting: {s['light']}. Setting and outfit: {s['setting']}."
dst = os.path.join(C, "input", f"pilot_{o.tag}_ref.png"); shutil.copy(o.ref, dst) if not os.path.exists(dst) else None; REF = os.path.basename(dst)
post = lambda d: json.load(urllib.request.urlopen(urllib.request.Request(U + "/prompt", data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})))
peak = [0]; avail = [10**12]; stop = [False]
def mon():
    while not stop[0]:
        try: peak[0] = max(peak[0], int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0]))
        except Exception: pass
        try: avail[0] = min(avail[0], int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024)
        except Exception: pass
        time.sleep(0.3)
threading.Thread(target=mon, daemon=True).start()
cd = os.path.join(o.root, "candidates"); os.makedirs(cd, exist_ok=True); LOG = open(os.path.join(o.root, "gen_log.jsonl"), "a")
for s, val in ALL:
    if o.only and s["id"] not in o.only: continue
    base = (69000 if val else 61000) + (int(s["id"][1:]) if val else int(s["id"])) * 100
    for i in range(s["n"]):
        seed = base + i; f = os.path.join(cd, f"{s['id']}_s{seed}.png")
        if os.path.exists(f): continue
        P = prompt(s)
        w = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": me.get("unet", "qwen_image_edit_2511_int8_convrot.safetensors"), "weight_dtype": "default"}},
             "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": me.get("clip", "qwen_2.5_vl_7b_fp8_scaled.safetensors"), "type": "qwen_image", "device": "default"}},
             "3": {"class_type": "VAELoader", "inputs": {"vae_name": me.get("vae", "qwen_image_vae.safetensors")}},
             "4": {"class_type": "LoadImage", "inputs": {"image": REF}},
             "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}},
             "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": P, "vae": ["3", 0], "image1": ["5", 0]}},
             "13": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": NEG, "vae": ["3", 0], "image1": ["5", 0]}},
             "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": float(me.get("shift", 3.1))}},
             "15": {"class_type": "CFGNorm", "inputs": {"model": ["7", 0], "strength": 1.0}},
             "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
             "9": {"class_type": "KSampler", "inputs": {"model": ["15", 0], "positive": ["6", 0], "negative": ["13", 0], "latent_image": ["8", 0], "seed": seed,
                   "steps": o.steps, "cfg": o.cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
             "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
             "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"pilot_v1_{o.tag}/{s['id']}_s{seed}"}}}
        peak[0] = 0; avail[0] = 10**12; t = time.time(); pid = post({"prompt": w})["prompt_id"]
        while True:
            time.sleep(0.5); h = json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
            if pid in h: break
        dt = round(time.time() - t, 1); st = h[pid]["status"].get("status_str")
        r = dict(slot=s["id"], validation=val, seed=seed, target_yaw=s["yaw"], status=st, sec=dt, peak_vram_mib=peak[0], min_ram_available_mib=avail[0],
                 steps=o.steps, cfg=o.cfg, prompt=P, negative=NEG, prompt_id=pid, time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        if st == "success":
            x = h[pid]["outputs"]["11"]["images"][0]; shutil.copy(os.path.join(C, "output", x["subfolder"], x["filename"]), f)
        json.dump({"meta": r, "workflow_api": w}, open(f[:-4] + ".json", "w"), indent=1); LOG.write(json.dumps(r) + "\n"); LOG.flush(); print(r["slot"], seed, st, dt, flush=True)
        if CSV and st == "success":
            with open(CSV, "a") as fh: fh.write(f"{r['time']},qwen_image_edit_2511_int8_nolora,pilot_v1_{s['id']},template,{o.steps},{dt},warm,dataset_pilot_v1/candidates/{s['id']}_s{seed}\n")
        if st != "success": print(json.dumps(h[pid]["status"])[-1500:]); stop[0] = True; sys.exit(1)
stop[0] = True
