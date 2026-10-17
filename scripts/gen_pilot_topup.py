"""Top-up generation for a pilot dataset with the SAME approved Edit-2511 workflow as gen_pilot_dataset.py
(single reference via FluxKontextImageScale; CFGNorm(1.0) + VAE-encoded scaled ref as start latent; euler/simple; default 40 steps, cfg 2.5).
Differences vs gen_pilot_dataset.py: slot file carries its own prompt text and explicit seed_base per slot (so seeds never collide with
earlier runs), optional explicit nose-direction wording ("nose": true), reference hash check, bounded retries per seed (failures are
logged and skipped instead of aborting), configurable output prefix / CSV label. Resumable (skips existing files).
Usage: python gen_pilot_topup.py --ref canon.png --ref-sha256 HEX --root OUT_DIR --slots topup_slots.json [--only 08 11 ...]
Writes OUT_DIR/candidates/{slot}_s{seed}.png + .json and OUT_DIR/gen_log.jsonl. Machine paths from config.local.yaml."""
import argparse, json, os, sys, time, shutil, subprocess, threading, urllib.request, datetime, hashlib
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg = config.load(); me = cfg.get("models_edit", {}); U = cfg["comfy_url"]; C = cfg["comfy_dir"]; CSV = cfg.get("times_csv")
a = argparse.ArgumentParser(); a.add_argument("--ref", required=True); a.add_argument("--ref-sha256", required=True); a.add_argument("--root", required=True)
a.add_argument("--slots", required=True); a.add_argument("--only", nargs="*"); a.add_argument("--tag", default="char01"); a.add_argument("--label", default="pilot_topup")
a.add_argument("--steps", type=int, default=40); a.add_argument("--cfg", type=float, default=2.5); a.add_argument("--retries", type=int, default=2)
a.add_argument("--timeout", type=int, default=900); a.add_argument("--max-consecutive-fail", type=int, default=4); o = a.parse_args()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if sha(o.ref) != o.ref_sha256: sys.exit(f"reference hash mismatch for {o.ref}")
S = json.load(open(o.slots)); ALL = [(s, False) for s in S["slots"]] + [(s, True) for s in S.get("validation", [])]
INTRO, ID, NEG = S["intro"], S["identity"], S["negative"]
def direction(y, nose):
    if abs(y) < 5: return "She faces the camera directly, head straight."
    side = "left" if y < 0 else "right"; n = f" Her nose points toward the {side} edge of the image." if nose else ""
    if abs(y) <= 15: return f"Her head is turned very slightly, about {abs(y)} degrees, toward the {side} edge of the image.{n}"
    if abs(y) <= 50: return f"Her head and shoulders are turned about {abs(y)} degrees so her face points toward the {side} edge of the image.{n}"
    return f"Her head is turned 90 degrees into a full side profile so her face points toward the {side} edge of the image; only one side of her face is visible.{n}"
def prompt(s): return f"{INTRO}{ID} {s['framing']}. {direction(s['yaw'], s.get('nose'))} {s['expr'][0].upper() + s['expr'][1:]}. Lighting: {s['light']}. Setting and outfit: {s['setting']}."
dst = os.path.join(C, "input", f"{o.label}_{o.tag}_ref.png")
if not os.path.exists(dst): shutil.copy(o.ref, dst)
if sha(dst) != o.ref_sha256: sys.exit(f"ComfyUI input copy {dst} does not match reference hash")
REF = os.path.basename(dst)
def post(d): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/prompt", data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=60))
peak = [0]; avail = [10**12]; stop = [False]
def mon():
    while not stop[0]:
        try: peak[0] = max(peak[0], int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0]))
        except Exception: pass
        try: avail[0] = min(avail[0], int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024)
        except Exception: pass
        time.sleep(0.3)
threading.Thread(target=mon, daemon=True).start()
def workflow(P, seed, sid):
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": me.get("unet", "qwen_image_edit_2511_int8_convrot.safetensors"), "weight_dtype": "default"}},
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
     "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"{o.label}_{o.tag}/{sid}_s{seed}"}}}
def run(w):
    pid = post({"prompt": w})["prompt_id"]; t0 = time.time()
    while time.time() - t0 < o.timeout:
        time.sleep(0.5); h = json.load(urllib.request.urlopen(f"{U}/history/{pid}", timeout=30))
        if pid in h: return pid, h[pid]
    return pid, {"status": {"status_str": "timeout"}}
cd = os.path.join(o.root, "candidates"); os.makedirs(cd, exist_ok=True); LOG = open(os.path.join(o.root, "gen_log.jsonl"), "a"); consec = 0
for s, val in ALL:
    if o.only and s["id"] not in o.only: continue
    for i in range(s["n"]):
        seed = s["seed_base"] + i; f = os.path.join(cd, f"{s['id']}_s{seed}.png")
        if os.path.exists(f): continue
        P = prompt(s); w = workflow(P, seed, s["id"])
        for attempt in range(o.retries + 1):
            peak[0] = 0; avail[0] = 10**12; t = time.time()
            try: pid, h = run(w); st = h["status"].get("status_str")
            except Exception as e: pid, h, st = None, {"status": {"error": repr(e)}}, "exception"
            dt = round(time.time() - t, 1)
            r = dict(slot=s["id"], validation=val, seed=seed, target_yaw=s["yaw"], status=st, attempt=attempt, sec=dt, peak_vram_mib=peak[0],
                     min_ram_available_mib=avail[0], steps=o.steps, cfg=o.cfg, prompt=P, negative=NEG, prompt_id=pid, ref_sha256=o.ref_sha256,
                     time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
            if st == "success":
                x = h["outputs"]["11"]["images"][0]; shutil.copy(os.path.join(C, "output", x["subfolder"], x["filename"]), f)
                json.dump({"meta": r, "workflow_api": w}, open(f[:-4] + ".json", "w"), indent=1)
            LOG.write(json.dumps(r) + "\n"); LOG.flush(); print(r["slot"], seed, st, dt, f"attempt {attempt}", flush=True)
            if st == "success":
                consec = 0
                if CSV:
                    with open(CSV, "a") as fh: fh.write(f"{r['time']},qwen_image_edit_2511_int8_nolora,{o.label}_{s['id']},template,{o.steps},{dt},warm,{os.path.basename(os.path.normpath(o.root))}/candidates/{s['id']}_s{seed}\n")
                break
            print(json.dumps(h.get("status"))[-1500:], flush=True); time.sleep(30)
        else:
            consec += 1
            if consec >= o.max_consecutive_fail: print("ABORT: too many consecutive failures", flush=True); stop[0] = True; sys.exit(2)
stop[0] = True; print("GEN_DONE", flush=True)
