"""Merge an earlier pilot dataset with its top-up into a combined slot view (manifest + symlinks; earlier files are never moved or
modified), and write per-image acceptance tables, per-category stats, contact sheets, 640px thumbnails and summary.md into ROOT/deliv/.
Optional --notes / --visual JSON: {"files": {"<file>": {"status": "...", "flags": [...], "beauty_mark": "...", "anatomy": "...", "note": "..."}}}
apply carried-over or visual-inspection decisions to either dataset's files.
Usage: python report_pilot_topup.py --root TOPUP_DIR --prev-root PREV_DIR --prev-slots prev_slots.json --slots topup_slots.json [--notes f] [--visual f]"""
import argparse, json, os, glob, csv, shutil, datetime
import numpy as np, cv2
a = argparse.ArgumentParser(); a.add_argument("--root", required=True); a.add_argument("--prev-root", required=True); a.add_argument("--prev-slots", required=True)
a.add_argument("--slots", required=True); a.add_argument("--notes"); a.add_argument("--visual"); a.add_argument("--thumb-h", type=int, default=640); o = a.parse_args()
PS = json.load(open(o.prev_slots)); TS = json.load(open(o.slots)); TID = {s["id"]: s for s in TS["slots"] + TS.get("validation", [])}
SLOTS = PS["slots"] + PS["validation"]; SID = {s["id"]: s for s in SLOTS}
def cat(sid):
    if sid.startswith("V"): return "validation"
    s = SID[sid]; f = s["framing"].lower()
    if "full-body" in f: return "full body"
    if "half-body" in f: return "half body"
    if int(sid) in (23, 24): return "camera variation"
    if abs(s["yaw"]) >= 70: return "profile"
    if abs(s["yaw"]) > 15: return "three-quarter"
    return "frontal"
CATS = ["frontal", "three-quarter", "profile", "half body", "full body", "camera variation", "validation"]
prev = json.load(open(os.path.join(o.prev_root, "candidates_scored.json"))); top = json.load(open(os.path.join(o.root, "candidates_scored.json")))
cmpf = os.path.join(o.root, "compare_prev_fullbody_crop_rescore.json"); cmp = {r["file"]: r for r in json.load(open(cmpf))} if os.path.exists(cmpf) else {}
notes = {}
for f in (o.notes, o.visual):
    if f and os.path.exists(f):
        for k, v in json.load(open(f))["files"].items(): notes.setdefault(k, {}).update({kk: (notes.get(k, {}).get(kk, []) + vv if kk == "flags" else vv) for kk, vv in v.items()})
rows = []
def meta(root, fn):
    p = os.path.join(root, "candidates", fn[:-4] + ".json")
    return json.load(open(p))["meta"] if os.path.exists(p) else {}
for src, root, rs in (("v1", o.prev_root, prev), ("topup", o.root, top)):
    for r in rs:
        m = meta(root, r["file"]); c = cmp.get(r["file"], {}) if src == "v1" else {}
        x = dict(source=src, file=r["file"], slot=r["slot"], seed=r["seed"], category=cat(r["slot"]), validation=r["validation"], target_yaw=r["target_yaw"],
                 yaw=r.get("yaw"), pitch=r.get("pitch"), cosine=r.get("cosine"), crop_yaw=r.get("crop_yaw", c.get("crop_yaw")), crop_cosine=r.get("crop_cosine", c.get("crop_cosine")),
                 decision_basis=r.get("decision_basis", "raw"), decision_yaw=r.get("decision_yaw", r.get("yaw")), decision_cosine=r.get("decision_cosine", r.get("cosine")),
                 threshold=r.get("threshold"), n_faces=r.get("n_faces"), face_px=r.get("face_px"), scored_status=r["status"], auto_reasons=list(r.get("auto_reasons", [])),
                 manual_flags=list(r.get("manual_flags", [])), raw_only_reasons=r.get("raw_only_reasons"), crop_rescore_status=c.get("status"),
                 path=os.path.join(os.path.basename(os.path.normpath(root)), "candidates", r["file"]),
                 prompt=m.get("prompt"), negative=m.get("negative"), steps=m.get("steps"), cfg=m.get("cfg"), gen_sec=m.get("sec"), gen_time=m.get("time"),
                 peak_vram_mib=m.get("peak_vram_mib"), ref_sha256=m.get("ref_sha256"), _root=root)
        prof = abs(x["target_yaw"]) >= 70 or (x["decision_yaw"] is not None and abs(x["decision_yaw"]) > 50)
        x["auto_pass"] = not [q for q in x["auto_reasons"] if not q.startswith("near_duplicate")] and not (prof and (x["decision_cosine"] or 0) < 0.45)
        st = {"accepted_provisional": "accepted_provisional", "validation": "validation", "review": "review"}.get(r["status"], r["status"])
        n = notes.get(r["file"], {})
        if n.get("status"): st = n["status"]
        x["flags"] = n.get("flags", []); x["beauty_mark"] = n.get("beauty_mark", "visual_pending" if st in ("accepted_provisional", "validation", "review") else "")
        x["anatomy"] = n.get("anatomy", "visual_pending" if st in ("accepted_provisional", "validation", "review") else ""); x["note"] = n.get("note", "")
        x["combined_status"] = st; rows.append(x)
# combined slot view: earlier selections keep precedence; top-up fills empty slots
slotview = {}
for s in SLOTS:
    sel = [x for x in rows if x["slot"] == s["id"] and x["combined_status"] in ("accepted_provisional", "validation", "review")]
    sel.sort(key=lambda x: (x["combined_status"] == "review", x["source"] != "v1"))
    acc = [x for x in sel if x["combined_status"] in ("accepted_provisional", "validation")]
    if len(acc) > 1:
        for x in acc[1:]: x["combined_status"] = "pass_not_selected"; x["flags"].append("slot_already_filled")
    slotview[s["id"]] = dict(slot=s["id"], category=cat(s["id"]), target_yaw=s["yaw"], filled=bool(acc), filled_by=(acc[0]["path"] if acc else None),
                             review=[x["path"] for x in sel if x["combined_status"] == "review"])
D = os.path.join(o.root, "deliv"); TH = os.path.join(D, "thumbs"); os.makedirs(TH, exist_ok=True)
CB = os.path.join(o.root, "combined")
for d in ("accepted_provisional", "review", "validation"):
    p = os.path.join(CB, d); os.makedirs(p, exist_ok=True)
    for f in os.listdir(p): os.remove(os.path.join(p, f))  # symlinks only
for x in rows:
    st = x["combined_status"]; d = {"accepted_provisional": "accepted_provisional", "validation": "validation", "review": "review"}.get(st)
    if d: os.symlink(os.path.join(x["_root"], "candidates", x["file"]), os.path.join(CB, d, f"{x['source']}__{x['file']}"))
def thumb(x):
    t = os.path.join(TH, f"{x['source']}__{x['file'][:-4]}.jpg")
    if not os.path.exists(t):
        im = cv2.imread(os.path.join(x["_root"], "candidates", x["file"])); h, w = im.shape[:2]; s = o.thumb_h / h
        cv2.imwrite(t, cv2.resize(im, (int(round(w * s)), o.thumb_h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return t
for x in rows:
    if x["source"] == "topup" or x["combined_status"] in ("accepted_provisional", "validation", "review"): x["thumb"] = os.path.relpath(thumb(x), D)
fc = os.path.join(o.root, "face_crops")
if os.path.isdir(fc):
    os.makedirs(os.path.join(D, "face_crops"), exist_ok=True)
    for x in rows:
        if x["source"] == "topup" and TID.get(x["slot"], {}).get("fullbody"):
            p = os.path.join(fc, x["file"][:-4] + "_face.jpg")
            if os.path.exists(p): shutil.copy(p, os.path.join(D, "face_crops"))
def f3(v): return "-" if v is None else f"{v:.3f}"
def f0(v): return "-" if v is None else f"{v:.0f}"
def sheet(items, path, title, cols=6, tw=300):
    if not items:
        im = np.full((120, 900, 3), 255, np.uint8); cv2.putText(im, title + ": (none)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2); cv2.imwrite(path, im); return
    th = int(round(tw * 1184 / 880)); lh = 78; rws = (len(items) + cols - 1) // cols; im = np.full((50 + rws * (th + lh), cols * tw, 3), 255, np.uint8)
    cv2.putText(im, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, x in enumerate(items):
        r, c = divmod(i, cols); X, Y = c * tw, 50 + r * (th + lh)
        t = cv2.imread(os.path.join(D, x["thumb"])); t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA); im[Y:Y + th, X:X + tw] = t
        cp = os.path.join(fc, x["file"][:-4] + "_face.jpg")
        if x["source"] == "topup" and TID.get(x["slot"], {}).get("fullbody") and os.path.exists(cp):
            q = cv2.resize(cv2.imread(cp), (110, 110), interpolation=cv2.INTER_AREA); im[Y + 4:Y + 114, X + 4:X + 114] = q; cv2.rectangle(im, (X + 4, Y + 4), (X + 114, Y + 114), (255, 255, 255), 2)
        fl = [f for f in x["flags"] + x["manual_flags"] if "visual_pending" not in f]
        lines = [f"{x['slot']} s{x['seed']} [{x['source']}] {x['combined_status'][:10]}", f"yaw {f0(x['yaw'])} (t {x['target_yaw']}) cos {f3(x['cosine'])} crop {f3(x['crop_cosine'])}",
                 ("; ".join(fl))[:46], ("; ".join(fl))[46:92]]
        for j, L in enumerate(lines): cv2.putText(im, L, (X + 4, Y + th + 17 + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160) if j >= 2 else (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
order = {s["id"]: i for i, s in enumerate(SLOTS)}; key = lambda x: (order[x["slot"]], x["source"], x["seed"])
appr = sorted([x for x in rows if x["combined_status"] == "accepted_provisional"], key=key)
body = lambda x: any("body geometry" in f or "body_geometry" in f for f in x["flags"] + x["manual_flags"])
rev = sorted([x for x in rows if x["combined_status"] == "review" or (x["combined_status"] in ("accepted_provisional", "validation") and body(x) and x["anatomy"] != "ok")], key=key)
val = sorted([x for x in rows if x["validation"] and x["combined_status"] in ("validation", "review")], key=key)
sheet(appr, os.path.join(D, "approved_contact_sheet.jpg"), f"char01 pilot v1 + top-up: approved (accepted_provisional) {len(appr)}")
sheet(rev, os.path.join(D, "review_contact_sheet.jpg"), f"char01 pilot v1 + top-up: review / body-geometry uncertain {len(rev)} (inset = scored face crop)")
sheet(val, os.path.join(D, "validation_contact_sheet.jpg"), f"char01 pilot v1 + top-up: validation {len(val)}")
COLS = ["source", "file", "slot", "seed", "category", "validation", "target_yaw", "yaw", "pitch", "cosine", "crop_yaw", "crop_cosine", "decision_basis", "decision_yaw",
        "decision_cosine", "threshold", "n_faces", "face_px", "auto_pass", "scored_status", "combined_status", "auto_reasons", "manual_flags", "flags", "raw_only_reasons",
        "crop_rescore_status", "beauty_mark", "anatomy", "note", "path", "thumb", "steps", "cfg", "gen_sec", "gen_time", "peak_vram_mib", "ref_sha256", "prompt", "negative"]
out = [{k: x.get(k) for k in COLS} for x in sorted(rows, key=key)]
json.dump(out, open(os.path.join(D, "acceptance.json"), "w"), indent=1)
with open(os.path.join(D, "acceptance.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(COLS)
    for x in out: w.writerow(["; ".join(x[c]) if isinstance(x[c], list) else x[c] for c in COLS])
json.dump(list(slotview.values()), open(os.path.join(D, "combined_manifest.json"), "w"), indent=1)
stats = {}
for cname in CATS:
    cs = [s["id"] for s in SLOTS if cat(s["id"]) == cname]; e = {}
    for src in ("v1", "topup", "combined"):
        rr = [x for x in rows if x["category"] == cname and (src == "combined" or x["source"] == src)]; n = len(rr); p = sum(x["auto_pass"] for x in rr)
        e[src] = dict(gens=n, auto_pass=p, pass_rate=round(p / n, 3) if n else None)
    e["slots"] = cs; e["filled"] = sum(slotview[s]["filled"] for s in cs); e["total_slots"] = len(cs); e["in_review"] = [s for s in cs if slotview[s]["review"] and not slotview[s]["filled"]]
    e["empty"] = [s for s in cs if not slotview[s]["filled"] and not slotview[s]["review"]]; stats[cname] = e
tr = [x for x in rows if x["source"] == "topup"]
stats["_totals"] = dict(train_slots_filled=sum(v["filled"] for k, v in slotview.items() if not k.startswith("V")), train_slots=24,
                        validation_filled=sum(v["filled"] for k, v in slotview.items() if k.startswith("V")), validation_slots=4,
                        topup_gens=len(tr), topup_gen_seconds=round(sum(x["gen_sec"] or 0 for x in tr), 1), v1_gens=len(rows) - len(tr))
fbt = [x for x in tr if TID.get(x["slot"], {}).get("fullbody")]
stats["_fullbody_crop_effect"] = dict(
    topup=dict(n=len(fbt), raw_only_pass=sum(1 for x in fbt if x["raw_only_reasons"] == []), crop_pass=sum(1 for x in fbt if x["auto_pass"]),
               mean_raw_cos=round(float(np.mean([x["cosine"] for x in fbt if x["cosine"] is not None])), 4) if fbt else None,
               mean_crop_cos=round(float(np.mean([x["crop_cosine"] for x in fbt if x["crop_cosine"] is not None])), 4) if any(x["crop_cosine"] is not None for x in fbt) else None),
    v1_rescore=dict(n=len(cmp), raw_pass=sum(1 for r in cmp.values() if not r.get("raw_only_reasons")), crop_pass=sum(1 for r in cmp.values() if r["status"] == "pass_by_crop"),
                    crop_pass_files=[f for f, r in cmp.items() if r["status"] == "pass_by_crop"]))
json.dump(stats, open(os.path.join(D, "category_stats.json"), "w"), indent=1)
for f in ("candidates_scored.csv", "candidates_scored.json", "compare_prev_fullbody_crop_rescore.csv", "gen_log.jsonl", "topup_slots.json"):
    if os.path.exists(os.path.join(o.root, f)): shutil.copy(os.path.join(o.root, f), os.path.join(D, f))
L = [f"# char01 pilot v1 + top-up: combined summary", "", f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='minutes')}. Earlier files are referenced, not moved (combined/ holds symlinks; combined_manifest.json maps slots).", "",
     f"Training slots filled: {stats['_totals']['train_slots_filled']}/24; validation filled: {stats['_totals']['validation_filled']}/4. Top-up gens: {len(tr)} ({stats['_totals']['topup_gen_seconds']/3600:.2f} h GPU time).", "",
     "| category | slots filled | in review | empty | v1 pass | top-up pass | combined pass |", "|---|---|---|---|---|---|---|"]
for cname in CATS:
    e = stats[cname]; fm = lambda d: f"{d['auto_pass']}/{d['gens']}" + (f" ({100*d['pass_rate']:.0f}%)" if d["gens"] else "")
    L.append(f"| {cname} | {e['filled']}/{e['total_slots']} | {', '.join(e['in_review']) or '-'} | {', '.join(e['empty']) or '-'} | {fm(e['v1'])} | {fm(e['topup'])} | {fm(e['combined'])} |")
L += ["", "Pass = identity (yaw-banded ArcFace threshold; >50 deg: 0.45 floor + manual review) and direction (within 15 deg) both pass, before near-duplicate selection.", "",
      "## Slots", "", "| slot | category | target yaw | status | selected / review (yaw, cos, crop cos) | flags |", "|---|---|---|---|---|---|"]
for s in SLOTS:
    v = slotview[s["id"]]; its = [x for x in rows if x["slot"] == s["id"] and x["combined_status"] in ("accepted_provisional", "validation", "review")]
    L.append(f"| {s['id']} | {v['category']} | {s['yaw']} | {'filled' if v['filled'] else 'review' if v['review'] else 'EMPTY'} | " +
             "<br>".join(f"{x['source']}:{x['file']} ({f0(x['yaw'])}, {f3(x['cosine'])}, {f3(x['crop_cosine'])}) [{x['combined_status']}]" for x in its) + " | " +
             "<br>".join("; ".join(x["flags"] + [m for m in x["manual_flags"]]) for x in its) + " |")
e = stats["_fullbody_crop_effect"]
L += ["", "## Full body: upscaled face-crop scoring", "", f"Top-up full-body gens: {e['topup']['n']}; pass on raw score alone: {e['topup']['raw_only_pass']}; pass on face-crop score: {e['topup']['crop_pass']}; mean raw cos {e['topup']['mean_raw_cos']}, mean crop cos {e['topup']['mean_crop_cos']}.",
      f"v1 full-body candidates re-scored with the crop method (report only, not merged): {e['v1_rescore']['crop_pass']}/{e['v1_rescore']['n']} would pass ({', '.join(e['v1_rescore']['crop_pass_files']) or 'none'}).", "",
      "## Review list", "", "| file | source | slot | yaw | cos | crop cos | flags |", "|---|---|---|---|---|---|---|"]
for x in rev: L.append(f"| {x['file']} | {x['source']} | {x['slot']} | {f0(x['yaw'])} | {f3(x['cosine'])} | {f3(x['crop_cosine'])} | {'; '.join(x['flags'] + x['manual_flags'])} |")
open(os.path.join(D, "summary.md"), "w").write("\n".join(L) + "\n"); print("REPORT_DONE", stats["_totals"])
