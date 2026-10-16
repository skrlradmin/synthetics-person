"""Identity-consistency experiment: N varied SFW smartphone UGC scenes with a character LoRA,
fixed seeds, then optional VLM judge (OpenAI-compatible, multimodal) scoring same-person vs a reference.
Machine/paths/character come from config.local.yaml; OUT_SUB env sets the output subfolder."""
import json, time, urllib.request, os, sys, base64, datetime, re
sys.path.insert(0, os.path.dirname(__file__)); import config
cfg=config.load(); ch=cfg["character"]; m=cfg["models"]
U=cfg["comfy_url"]; C=cfg["comfy_dir"]; J=cfg.get("judge_url"); CSV=cfg.get("times_csv")
OUT=os.environ.get("OUT_SUB","idexp/lora_only"); REF=ch.get("reference_image")
UNET=m.get("unet","qwen_image_2.1_int8_convrot.safetensors"); CLIP=m.get("clip","qwen3vl_8b_int8_convrot.safetensors"); VAE=m.get("vae","qwen_image_2.1_vae_bf16.safetensors")
LORA=ch.get("lora",""); LSTR=float(ch.get("lora_strength",0.8)); TRIG=ch.get("trigger",""); LOOK=ch.get("appearance","")
REAL="raw unedited photo, visible skin pores, fine vellus hair, subtle skin imperfections, natural skin sheen, freckles faint, film grain, Kodak Portra 400, natural proportions"
SHOT="shot on iPhone, candid, slight motion blur, natural ambient light, real everyday place"
NEG="cartoon, illustration, deformed, extra limbs, blurry, low quality, plastic skin, airbrushed, smooth skin, waxy, doll, cgi, 3d render, overly retouched, glossy, professional photo, studio lighting, cinematic, busty, curvy, large breasts, heavy, thick, plump, wide hips, chubby"
SCENES=[
("gym","close-up selfie in a gym after a workout, flushed cheeks, wearing a grey sports top, fluorescent lighting"),
("beach","standing on a beach at sunset, wind in her hair, wearing a white sundress, warm backlight, medium shot"),
("office","sitting at an office desk with a laptop, wearing a navy blouse, overcast window light, three-quarter view"),
("grocery","in a grocery store aisle holding a basket, wearing a denim jacket and black tee, harsh overhead light, full body"),
("park","walking a dog in a city park in autumn, wearing a camel coat and scarf, soft overcast light, wide shot"),
("bathroom","bathroom mirror selfie in the morning, hair in a messy bun, wearing an oversized hoodie, soft window light"),
("night","night street selfie under neon signs, wearing a leather jacket, mixed colorful light"),
("couch","lying on a couch reading a book, wearing a cozy knit sweater, lamp light, top-down angle"),
("hike","on a mountain trail with a backpack, wearing a windbreaker and cap, bright midday sun, squinting"),
("restaurant","at a dinner table in a dim restaurant, candlelight, wearing a black slip dress and earrings, smiling"),
("train","sitting by a train window looking outside, wearing a beige trench coat, profile view, daylight"),
("rain","holding an umbrella on a rainy sidewalk, wearing a yellow raincoat, cool diffuse light"),
("bookstore","browsing shelves in a bookstore, wearing glasses and a cardigan, warm interior light, side view"),
("balcony","drinking coffee on an apartment balcony in the morning, wearing pajamas, golden sunlight"),
("concert","at an outdoor concert with friends blurred behind, wearing a band tee, stage lights, laughing"),
("airport","at an airport gate with a carry-on suitcase, wearing a white turtleneck, flat fluorescent light"),
("garden","kneeling in a garden planting flowers, wearing overalls, dappled sunlight, low angle"),
("snow","in a snowy street wearing a puffer jacket and beanie, rosy cheeks, bright overcast"),
("desk_video","front-facing webcam-style shot talking to camera in a bedroom, wearing a green sweater, ring light"),
("pool","sitting at the edge of a pool with feet in the water, wearing a sun hat and a cover-up shirt, bright sun")]
def post(url,obj): return json.load(urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"}),timeout=600))
def judge(img):
  b=lambda p:"data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode()
  q="Image 1 is a reference photo of a person; image 2 is another photo. Ignoring clothing, hair styling, lighting and background, rate how likely it is the SAME person based on facial features (1=different, 10=identical). Answer only with JSON {\"score\":N}."
  r=post(J+"/v1/chat/completions",{"messages":[{"role":"user","content":[{"type":"text","text":q},{"type":"image_url","image_url":{"url":b(REF)}},{"type":"image_url","image_url":{"url":b(img)}}]}],"max_tokens":2000,"temperature":0})
  t=r["choices"][0]["message"]["content"]; m=re.findall(r'"score"\s*:\s*(\d+)',t); return int(m[-1]) if m else None
res=[]
for n,(tag,scene) in enumerate(SCENES):
  seed=9001+n; P=", ".join(x for x in (TRIG,LOOK) if x)+f", {scene}, {SHOT}, {REAL}"
  w={"1":{"class_type":"UNETLoader","inputs":{"unet_name":UNET,"weight_dtype":"default"}},
   "12":{"class_type":"LoraLoaderModelOnly","inputs":{"model":["1",0],"lora_name":LORA,"strength_model":LSTR}},
   "2":{"class_type":"CLIPLoader","inputs":{"clip_name":CLIP,"type":"qwen_image","device":"default"}},
   "3":{"class_type":"VAELoader","inputs":{"vae_name":VAE}},
   "6":{"class_type":"TextEncodeQwenImage21","inputs":{"clip":["2",0],"prompt":P,"negative_prompt":NEG,"resolution":1024,"vae":["3",0]}},
   "7":{"class_type":"QwenImage21Cache","inputs":{"model":["12",0] if LORA else ["1",0],"device":"auto","dtype":"default"}},
   "8":{"class_type":"EmptyLatentImage","inputs":{"width":1088,"height":1408,"batch_size":1}},
   "9":{"class_type":"KSampler","inputs":{"model":["7",0],"positive":["6",0],"negative":["6",1],"latent_image":["8",0],"seed":seed,"steps":25,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
   "10":{"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["3",0]}},
   "11":{"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":f"{OUT}/{tag}_s{seed}"}}}
  if not LORA: del w["12"]
  t=time.time(); pid=post(U+"/prompt",{"prompt":w})["prompt_id"]
  while True:
    time.sleep(0.5); h=json.load(urllib.request.urlopen(f"{U}/history/{pid}"))
    if pid in h: break
  dt=time.time()-t; st=h[pid]["status"].get("status_str")
  outs=[im["filename"] for o in h[pid]["outputs"].values() for im in o.get("images",[])]
  if CSV:
    with open(CSV,"a") as f: f.write(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')},{UNET.split('.')[0]}+{LORA.split('.')[0] or 'nolora'},t2i_idexp_lora{LSTR},1088x1408,25,{dt:.1f},{'first' if n==0 else 'warm'},{OUT}/{tag}_s{seed}\n")
  res.append(dict(tag=tag,seed=seed,status=st,sec=round(dt,1),file=outs[0] if outs else None)); print(res[-1],flush=True)
if J and REF:
  for r in res:
    if r["file"]:
      try: r["judge"]=judge(f"{C}/output/{OUT}/{r['file']}")
      except Exception as e: r["judge"]=f"err {e}"
      print(r["tag"],r["judge"],flush=True)
json.dump(res,open(f"{C}/output/{OUT}/results.json","w"),indent=1); print("DONE")
