"""Generate canonical reference candidates for a new fictional character (base model, NO LoRA).
Usage: python gen_canon.py --prompt-file canon_prompt.txt --neg-file canon_negative.txt --out-dir DIR [--seeds 1 2 3] [--csv times.csv]
Machine/paths come from config.local.yaml (see config.example.yaml). Images are copied from ComfyUI output into --out-dir."""
import json, time, urllib.request, argparse, os, shutil, datetime, sys
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg = config.load()
a=argparse.ArgumentParser()
a.add_argument("--prompt-file",required=True); a.add_argument("--neg-file",required=True)
a.add_argument("--out-dir",required=True); a.add_argument("--comfy-dir",default=cfg.get("comfy_dir"))
a.add_argument("--seeds",type=int,nargs="+",default=list(range(1101,1109)))
a.add_argument("--width",type=int,default=1088); a.add_argument("--height",type=int,default=1408)
a.add_argument("--steps",type=int,default=25); a.add_argument("--cfg",type=float,default=1.0)
a.add_argument("--unet",default=cfg["models"].get("unet","qwen_image_2.1_int8_convrot.safetensors"))
a.add_argument("--clip",default=cfg["models"].get("clip","qwen3vl_8b_int8_convrot.safetensors"))
a.add_argument("--vae",default=cfg["models"].get("vae","qwen_image_2.1_vae_bf16.safetensors"))
a.add_argument("--prefix",default="canon"); a.add_argument("--csv",default=cfg.get("times_csv")); a.add_argument("--tag",default="char")
o=a.parse_args(); U=cfg["comfy_url"]
P=open(o.prompt_file).read().strip(); N=open(o.neg_file).read().strip(); os.makedirs(o.out_dir,exist_ok=True)
post=lambda p,d: json.load(urllib.request.urlopen(urllib.request.Request(U+p,data=json.dumps(d).encode(),headers={"Content-Type":"application/json"})))
res=[]
for i,seed in enumerate(o.seeds):
  w={"1":{"class_type":"UNETLoader","inputs":{"unet_name":o.unet,"weight_dtype":"default"}},
   "2":{"class_type":"CLIPLoader","inputs":{"clip_name":o.clip,"type":"qwen_image","device":"default"}},
   "3":{"class_type":"VAELoader","inputs":{"vae_name":o.vae}},
   "6":{"class_type":"TextEncodeQwenImage21","inputs":{"clip":["2",0],"prompt":P,"negative_prompt":N,"resolution":1024,"vae":["3",0]}},
   "7":{"class_type":"QwenImage21Cache","inputs":{"model":["1",0],"device":"auto","dtype":"default"}},
   "8":{"class_type":"EmptyLatentImage","inputs":{"width":o.width,"height":o.height,"batch_size":1}},
   "9":{"class_type":"KSampler","inputs":{"model":["7",0],"positive":["6",0],"negative":["6",1],"latent_image":["8",0],"seed":seed,"steps":o.steps,"cfg":o.cfg,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
   "10":{"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["3",0]}},
   "11":{"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":f"{o.prefix}/s{seed}"}}}
  t=time.time(); pid=post("/prompt",{"prompt":w})["prompt_id"]
  while True:
    time.sleep(0.5); h=json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
    if pid in h: break
  dt=time.time()-t; st=h[pid]["status"].get("status_str")
  ims=[im for out in h[pid]["outputs"].values() for im in out.get("images",[])]
  dst=None
  if st=="success" and ims and o.comfy_dir:
    dst=os.path.join(o.out_dir,f"s{seed}.png"); shutil.copy(os.path.join(o.comfy_dir,"output",ims[0]["subfolder"],ims[0]["filename"]),dst)
  res.append(dict(seed=seed,status=st,sec=round(dt,1),file=dst)); print(res[-1],flush=True)
  if o.csv:
    with open(o.csv,"a") as f: f.write(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')},{o.unet.rsplit('.',1)[0]}_nolora,t2i_canon_{o.tag},{o.width}x{o.height},{o.steps},{dt:.1f},{'first' if i==0 else 'warm'},{o.prefix}/s{seed}\n")
  if st!="success": print(json.dumps(h[pid]["status"])[-800:]); break
json.dump(res,open(os.path.join(o.out_dir,"results.json"),"w"),indent=1)
