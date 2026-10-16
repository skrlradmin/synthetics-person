"""Per-image evaluation report for reference-edit outputs (CSV + JSON).
Reads the ComfyUI API workflow embedded in each PNG ("prompt" text chunk), verifies that the reference image feeds the
edit-conditioning path (LoadImage -> [scale] -> TextEncodeQwenImageEdit(Plus).image1 -> KSampler positive/negative),
and merges optional human review notes. Identity similarity is a pluggable column: left blank unless --embedder is given.
Usage: python eval_report.py --images a.png b.png ... --ref-sha256 <sha> --notes notes.json --out report  [--embedder NAME --ref ref.png]
notes.json: {"<basename>": {"detection_notes":..., "crop_quality":..., "notes":..., "provisional":"accept|reject|review"}}"""
import argparse, json, csv, os
from PIL import Image
EMBEDDERS = {}   # name -> callable(ref_path, img_path) -> float ; register e.g. an ArcFace/insightface cosine here once approved
def trace(p):
    """Return proof dict: which node ids carry the reference into conditioning."""
    loads = {k: v for k, v in p.items() if v["class_type"] == "LoadImage"}
    enc = {k: v for k, v in p.items() if v["class_type"].startswith("TextEncodeQwenImageEdit")}
    ks = [v for v in p.values() if v["class_type"] == "KSampler"][0]
    def src(link):  # follow image links back to a LoadImage
        while link and link[0] not in loads:
            n = p[link[0]]; link = n["inputs"].get("image")
        return link[0] if link else None
    used = {k: {i: src(v["inputs"].get(i)) for i in ("image1", "image2", "image3") if v["inputs"].get(i)} for k, v in enc.items()}
    return dict(load_nodes={k: v["inputs"]["image"] for k, v in loads.items()}, encoders=used,
                encoder_has_vae={k: bool(v["inputs"].get("vae")) for k, v in enc.items()},
                ksampler_pos=ks["inputs"]["positive"][0], ksampler_neg=ks["inputs"]["negative"][0],
                ref_in_positive=bool(used.get(ks["inputs"]["positive"][0])), ref_in_negative=bool(used.get(ks["inputs"]["negative"][0])))
a = argparse.ArgumentParser(); a.add_argument("--images", nargs="+", required=True); a.add_argument("--notes")
a.add_argument("--out", required=True); a.add_argument("--ref-sha256"); a.add_argument("--embedder"); a.add_argument("--ref")
o = a.parse_args(); notes = json.load(open(o.notes)) if o.notes else {}; rows = []
for f in o.images:
    im = Image.open(f); p = json.loads(im.info["prompt"]); k = [v for v in p.values() if v["class_type"] == "KSampler"][0]["inputs"]
    t = trace(p); lsha = [v.get("is_changed", [None])[0] for v in p.values() if v["class_type"] == "LoadImage"]
    lora = [v["inputs"].get("lora_name") for v in p.values() if "Lora" in v["class_type"]]
    sim = EMBEDDERS[o.embedder](o.ref, f) if o.embedder else None
    n = notes.get(os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f), notes.get(os.path.basename(f), {}))
    rows.append(dict(file=f, width=im.size[0], height=im.size[1], unet=[v["inputs"]["unet_name"] for v in p.values() if v["class_type"] == "UNETLoader"][0],
        lora=";".join(lora) or "none", seed=k["seed"], steps=k["steps"], cfg=k["cfg"], sampler=k["sampler_name"], scheduler=k["scheduler"],
        reference=";".join(t["load_nodes"].values()), reference_sha256_ok=(o.ref_sha256 in lsha) if o.ref_sha256 else "",
        ref_in_conditioning=t["ref_in_positive"], ref_in_negative=t["ref_in_negative"], conditioning_path=json.dumps(t),
        similarity="" if sim is None else round(sim, 4), similarity_method=o.embedder or "pending",
        detection_notes=n.get("detection_notes", ""), crop_quality=n.get("crop_quality", ""), notes=n.get("notes", ""), provisional=n.get("provisional", "review")))
json.dump(rows, open(o.out + ".json", "w"), indent=1)
with open(o.out + ".csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {o.out}.csv/.json")
