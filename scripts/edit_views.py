"""Identity-preservation view tests with Qwen-Image-Edit from one canonical reference (no LoRA).
Usage: python edit_views.py --ref canon.png --tests templates/view_tests.yaml --out-dir DIR [--only profile]
Machine/paths/models come from config.local.yaml (comfy_url, comfy_dir, times_csv, models_edit.*).
Writes DIR/<test>/s<seed>.png, results.json and contact_sheet.jpg (seed + time labels)."""
import argparse, json, os, sys, time, shutil, urllib.request, datetime, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg=config.load(); me=cfg.get("models_edit",{}); U=cfg["comfy_url"]; C=cfg["comfy_dir"]; CSV=cfg.get("times_csv")
a=argparse.ArgumentParser(); a.add_argument("--ref",required=True); a.add_argument("--tests",required=True)
a.add_argument("--out-dir",required=True); a.add_argument("--only",nargs="*"); a.add_argument("--tag",default="char")
a.add_argument("--width",type=int,default=1088); a.add_argument("--height",type=int,default=1408); o=a.parse_args()
T=yaml.safe_load(open(o.tests)); UNET=me.get("unet","qwen_image_edit_2511_int8_convrot.safetensors")
refname=f"editref_{o.tag}_"+os.path.basename(o.ref); shutil.copy(o.ref,os.path.join(C,"input",refname))
post=lambda p,d: json.load(urllib.request.urlopen(urllib.request.Request(U+p,data=json.dumps(d).encode(),headers={"Content-Type":"application/json"})))
def sheet(res,d):
  W,H=408,int(408*o.height/o.width); S=Image.new("RGB",(W*len(res),H+40),"white"); dr=ImageDraw.Draw(S)
  try: F=ImageFont.truetype("DejaVuSans.ttf",24)
  except Exception: F=ImageFont.load_default()
  for i,r in enumerate(res):
    if r["file"]: S.paste(Image.open(r["file"]).convert("RGB").resize((W,H)),(i*W,0))
    dr.text((i*W+8,H+6),f"seed {r['seed']}  {r['sec']}s",fill="black",font=F)
  S.save(os.path.join(d,"contact_sheet.jpg"),quality=90)
first=True
for test,view in T["tests"].items():
  if o.only and test not in o.only: continue
  d=os.path.join(o.out_dir,test); os.makedirs(d,exist_ok=True); res=[]
  for seed in T["seeds"]:
    w={"1":{"class_type":"UNETLoader","inputs":{"unet_name":UNET,"weight_dtype":"default"}},
     "2":{"class_type":"CLIPLoader","inputs":{"clip_name":me.get("clip","qwen_2.5_vl_7b_fp8_scaled.safetensors"),"type":"qwen_image","device":"default"}},
     "3":{"class_type":"VAELoader","inputs":{"vae_name":me.get("vae","qwen_image_vae.safetensors")}},
     "4":{"class_type":"LoadImage","inputs":{"image":refname}},
     "5":{"class_type":"FluxKontextImageScale","inputs":{"image":["4",0]}},
     "6":{"class_type":"TextEncodeQwenImageEditPlus","inputs":{"clip":["2",0],"prompt":T["base_prompt"]+" "+view,"vae":["3",0],"image1":["5",0]}},
     "13":{"class_type":"TextEncodeQwenImageEditPlus","inputs":{"clip":["2",0],"prompt":T.get("negative",""),"vae":["3",0],"image1":["5",0]}},
     "7":{"class_type":"ModelSamplingAuraFlow","inputs":{"model":["1",0],"shift":float(me.get("shift",3.1))}},
     "8":{"class_type":"EmptySD3LatentImage","inputs":{"width":o.width,"height":o.height,"batch_size":1}},
     "9":{"class_type":"KSampler","inputs":{"model":["7",0],"positive":["6",0],"negative":["13",0],"latent_image":["8",0],"seed":seed,"steps":int(me.get("steps",25)),"cfg":float(me.get("cfg",4.0)),"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
     "10":{"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["3",0]}},
     "11":{"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":f"editviews_{o.tag}/{test}_s{seed}"}}}
    t=time.time(); pid=post("/prompt",{"prompt":w})["prompt_id"]
    while True:
      time.sleep(0.5); h=json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
      if pid in h: break
    dt=round(time.time()-t,1); st=h[pid]["status"].get("status_str"); ims=[i for x in h[pid]["outputs"].values() for i in x.get("images",[])]
    f=None
    if st=="success" and ims: f=os.path.join(d,f"s{seed}.png"); shutil.copy(os.path.join(C,"output",ims[0]["subfolder"],ims[0]["filename"]),f)
    res.append(dict(test=test,seed=seed,status=st,sec=dt,file=f)); print(res[-1],flush=True)
    if CSV and st=="success":
      with open(CSV,"a") as fh: fh.write(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')},{UNET.rsplit('.',1)[0]}_nolora,edit_view_{o.tag}_{test},{o.width}x{o.height},{me.get('steps',25)},{dt},{'first' if first else 'warm'},{o.tag}/tests/{test}/s{seed}\n")
    first=False
    if st!="success": print(json.dumps(h[pid]["status"])[-1500:]); sys.exit(1)
  json.dump(res,open(os.path.join(d,"results.json"),"w"),indent=1); sheet(res,d)
