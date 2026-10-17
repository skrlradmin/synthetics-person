"""Merge several pilot-dataset runs (e.g. v1, top-up, top-up 2) into one combined slot view (manifest + symlinks; earlier files are
never moved or modified) and write per-image acceptance tables, per-category stats, contact sheets, 640px thumbnails and summary.md
into ROOT/deliv/. Generalises report_pilot_topup.py to N sources. Sources are given in precedence order: an earlier source's
selection keeps its slot; later sources fill empty slots. The last source is the current run (ROOT).
--notes JSON files (applied in order; later wins, flags accumulate): {"files": {"<source>:<file>" or "<file>": {"status","flags","beauty_mark","anatomy","note"}}}
Usage: python report_pilot_topup2.py --root ROOT --sources v1=DIR topup=DIR topup2=ROOT --base-slots v1_slots.json
       --slot-files topup=topup_slots.json topup2=topup2_slots.json [--notes a.json b.json ...]"""
import argparse, json, os, csv, shutil, datetime
import numpy as np, cv2
a = argparse.ArgumentParser(); a.add_argument("--root", required=True); a.add_argument("--sources", nargs="+", required=True); a.add_argument("--base-slots", required=True)
a.add_argument("--slot-files", nargs="*", default=[]); a.add_argument("--notes", nargs="*", default=[]); a.add_argument("--thumb-h", type=int, default=640); o = a.parse_args()
SRC = [tuple(x.split("=", 1)) for x in o.sources]; CUR = SRC[-1][0]; SR = dict(SRC)
PS = json.load(open(o.base_slots)); SLOTS = PS["slots"] + PS["validation"]; SID = {s["id"]: s for s in SLOTS}
SF = {k: json.load(open(v)) for k, v in (x.split("=", 1) for x in o.slot_files)}
SDEF = {k: {s["id"]: s for s in v["slots"] + v.get("validation", [])} for k, v in SF.items()}
REDEF = {}
for k, d in SDEF.items():
    for sid, s in d.items():
        if s.get("redefined"): REDEF.setdefault(sid, []).append(f"{k}: {s['redefined']}")
def cat(sid):
    if sid.startswith("V"): return "validation"
    s = SID[sid]; f = s["framing"].lower()
    if "full-body" in f: return "full body / knees-up"
    if "half-body" in f: return "half body"
    if int(sid) in (23, 24): return "camera variation"
    if abs(s["yaw"]) >= 70: return "profile"
    if abs(s["yaw"]) > 15: return "three-quarter"
    return "frontal"
CATS = ["frontal", "three-quarter", "profile", "half body", "full body / knees-up", "camera variation", "validation"]
notes = {}
for f in o.notes:
    if f and os.path.exists(f):
        for k, v in json.load(open(f))["files"].items():
            e = notes.setdefault(k, {})
            for kk, vv in v.items(): e[kk] = (e.get(kk, []) + [q for q in vv if q not in e.get(kk, [])]) if kk == "flags" else vv
cmp = {}
for name, root in SRC:
    p = os.path.join(root, "compare_prev_fullbody_crop_rescore.json")
    if os.path.exists(p): cmp.update({r["file"]: r for r in json.load(open(p))})
def meta(root, fn):
    p = os.path.join(root, "candidates", fn[:-4] + ".json"); return json.load(open(p))["meta"] if os.path.exists(p) else {}
def fullbody(src, sid): return bool(SDEF.get(src, {}).get(sid, {}).get("fullbody"))
rows = []
for src, root in SRC:
    for r in json.load(open(os.path.join(root, "candidates_scored.json"))):
        m = meta(root, r["file"]); c = cmp.get(r["file"], {}) if src == SRC[0][0] else {}
        x = dict(source=src, file=r["file"], slot=r["slot"], seed=r["seed"], category=cat(r["slot"]), validation=r["validation"], target_yaw=r["target_yaw"],
                 prompt_yaw=m.get("prompt_yaw", r["target_yaw"]), slot_redefined=SDEF.get(src, {}).get(r["slot"], {}).get("redefined"),
                 yaw=r.get("yaw"), pitch=r.get("pitch"), cosine=r.get("cosine"), crop_yaw=r.get("crop_yaw", c.get("crop_yaw")), crop_cosine=r.get("crop_cosine", c.get("crop_cosine")),
                 decision_basis=r.get("decision_basis", "raw"), decision_yaw=r.get("decision_yaw", r.get("yaw")), decision_cosine=r.get("decision_cosine", r.get("cosine")),
                 threshold=r.get("threshold"), n_faces=r.get("n_faces"), face_px=r.get("face_px"), scored_status=r["status"], auto_reasons=list(r.get("auto_reasons", [])),
                 manual_flags=list(r.get("manual_flags", [])), raw_only_reasons=r.get("raw_only_reasons"), crop_rescore_status=c.get("status"),
                 path=os.path.join(os.path.basename(os.path.normpath(root)), "candidates", r["file"]),
                 prompt=m.get("prompt"), negative=m.get("negative"), steps=m.get("steps"), cfg=m.get("cfg"), gen_sec=m.get("sec"), gen_time=m.get("time"),
                 peak_vram_mib=m.get("peak_vram_mib"), ref_sha256=m.get("ref_sha256"), _root=root)
        prof = abs(x["target_yaw"]) >= 70 or (x["decision_yaw"] is not None and abs(x["decision_yaw"]) > 50)
        x["auto_pass"] = not [q for q in x["auto_reasons"] if not q.startswith("near_duplicate")] and not (prof and (x["decision_cosine"] or 0) < 0.45)
        st = r["status"]; n = dict(notes.get(r["file"], {})); n2 = notes.get(f"{src}:{r['file']}", {})
        for kk, vv in n2.items(): n[kk] = (n.get(kk, []) + vv) if kk == "flags" else vv
        if n.get("status"): st = n["status"]
        sel = st in ("accepted_provisional", "validation", "review")
        x["flags"] = n.get("flags", []); x["beauty_mark"] = n.get("beauty_mark", "visual_pending" if sel else ""); x["anatomy"] = n.get("anatomy", "visual_pending" if sel else "")
        x["note"] = n.get("note", ""); x["combined_status"] = st; rows.append(x)
rank = {s: i for i, (s, _) in enumerate(SRC)}; slotview = {}
for s in SLOTS:
    sel = [x for x in rows if x["slot"] == s["id"] and x["combined_status"] in ("accepted_provisional", "validation", "review")]
    sel.sort(key=lambda x: (x["combined_status"] == "review", rank[x["source"]]))
    acc = [x for x in sel if x["combined_status"] in ("accepted_provisional", "validation")]
    for x in acc[1:]: x["combined_status"] = "pass_not_selected"; x["flags"].append("slot_already_filled")
    slotview[s["id"]] = dict(slot=s["id"], category=cat(s["id"]), target_yaw=s["yaw"], redefined=REDEF.get(s["id"], []), filled=bool(acc),
                             filled_by=(f"{acc[0]['source']}:{acc[0]['path']}" if acc else None),
                             review=[f"{x['source']}:{x['path']}" for x in sel if x["combined_status"] == "review"])
D = os.path.join(o.root, "deliv"); TH = os.path.join(D, "thumbs"); os.makedirs(TH, exist_ok=True); CB = os.path.join(o.root, "combined")
for d in ("accepted_provisional", "review", "validation"):
    p = os.path.join(CB, d); os.makedirs(p, exist_ok=True)
    for f in os.listdir(p): os.remove(os.path.join(p, f))  # symlinks only
for x in rows:
    d = {"accepted_provisional": "accepted_provisional", "validation": "validation", "review": "review"}.get(x["combined_status"])
    if d: os.symlink(os.path.join(x["_root"], "candidates", x["file"]), os.path.join(CB, d, f"{x['source']}__{x['file']}"))
def thumb(x):
    t = os.path.join(TH, f"{x['source']}__{x['file'][:-4]}.jpg")
    if not os.path.exists(t):
        im = cv2.imread(os.path.join(x["_root"], "candidates", x["file"])); h, w = im.shape[:2]; s = o.thumb_h / h
        cv2.imwrite(t, cv2.resize(im, (int(round(w * s)), o.thumb_h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return t
for x in rows:
    if x["source"] == CUR or x["combined_status"] in ("accepted_provisional", "validation", "review"): x["thumb"] = os.path.relpath(thumb(x), D)
def cropfile(x): return os.path.join(x["_root"], "face_crops", x["file"][:-4] + "_face.jpg")
os.makedirs(os.path.join(D, "face_crops"), exist_ok=True)
for x in rows:
    if x["source"] == CUR and fullbody(CUR, x["slot"]) and os.path.exists(cropfile(x)): shutil.copy(cropfile(x), os.path.join(D, "face_crops"))
def f3(v): return "-" if v is None else f"{v:.3f}"
def f0(v): return "-" if v is None else f"{v:.0f}"
def sheet(items, path, title, cols=6, tw=300):
    if not items:
        im = np.full((120, 900, 3), 255, np.uint8); cv2.putText(im, title + ": (none)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2); cv2.imwrite(path, im); return
    th = int(round(tw * 1184 / 880)); lh = 78; rws = (len(items) + cols - 1) // cols; im = np.full((50 + rws * (th + lh), cols * tw, 3), 255, np.uint8)
    cv2.putText(im, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, x in enumerate(items):
        r, c = divmod(i, cols); X, Y = c * tw, 50 + r * (th + lh)
        t = cv2.resize(cv2.imread(os.path.join(D, x["thumb"])), (tw, th), interpolation=cv2.INTER_AREA); im[Y:Y + th, X:X + tw] = t
        if fullbody(x["source"], x["slot"]) and os.path.exists(cropfile(x)):
            q = cv2.resize(cv2.imread(cropfile(x)), (110, 110), interpolation=cv2.INTER_AREA); im[Y + 4:Y + 114, X + 4:X + 114] = q; cv2.rectangle(im, (X + 4, Y + 4), (X + 114, Y + 114), (255, 255, 255), 2)
        fl = [f for f in x["flags"] + x["manual_flags"] if "visual_pending" not in f]
        lines = [f"{x['slot']} s{x['seed']} [{x['source']}] {x['combined_status'][:10]}", f"yaw {f0(x['yaw'])} (t {x['target_yaw']}) cos {f3(x['cosine'])} crop {f3(x['crop_cosine'])}",
                 ("; ".join(fl))[:46], ("; ".join(fl))[46:92]]
        for j, L in enumerate(lines): cv2.putText(im, L, (X + 4, Y + th + 17 + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160) if j >= 2 else (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
order = {s["id"]: i for i, s in enumerate(SLOTS)}; key = lambda x: (order[x["slot"]], rank[x["source"]], x["seed"])
appr = sorted([x for x in rows if x["combined_status"] == "accepted_provisional"], key=key)
body = lambda x: any("body geometry" in f or "body_geometry" in f for f in x["flags"] + x["manual_flags"])
rev = sorted([x for x in rows if x["combined_status"] == "review" or (x["combined_status"] in ("accepted_provisional", "validation") and body(x) and x["anatomy"] != "ok")], key=key)
val = sorted([x for x in rows if x["validation"] and x["combined_status"] in ("validation", "review")], key=key)
tag = " + ".join(s for s, _ in SRC)
sheet(appr, os.path.join(D, "approved_contact_sheet.jpg"), f"char01 pilot {tag}: approved (accepted_provisional) {len(appr)}")
sheet(rev, os.path.join(D, "review_contact_sheet.jpg"), f"char01 pilot {tag}: review / body-geometry uncertain {len(rev)} (inset = scored face crop)")
sheet(val, os.path.join(D, "validation_contact_sheet.jpg"), f"char01 pilot {tag}: validation {len(val)}")
COLS = ["source", "file", "slot", "seed", "category", "validation", "target_yaw", "prompt_yaw", "slot_redefined", "yaw", "pitch", "cosine", "crop_yaw", "crop_cosine", "decision_basis",
        "decision_yaw", "decision_cosine", "threshold", "n_faces", "face_px", "auto_pass", "scored_status", "combined_status", "auto_reasons", "manual_flags", "flags", "raw_only_reasons",
        "crop_rescore_status", "beauty_mark", "anatomy", "note", "path", "thumb", "steps", "cfg", "gen_sec", "gen_time", "peak_vram_mib", "ref_sha256", "prompt", "negative"]
out = [{k: x.get(k) for k in COLS} for x in sorted(rows, key=key)]
json.dump(out, open(os.path.join(D, "acceptance.json"), "w"), indent=1)
with open(os.path.join(D, "acceptance.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(COLS)
    for x in out: w.writerow(["; ".join(x[c]) if isinstance(x[c], list) else x[c] for c in COLS])
json.dump(list(slotview.values()), open(os.path.join(D, "combined_manifest.json"), "w"), indent=1)
NAMES = [s for s, _ in SRC] + ["combined"]; stats = {}
for cname in CATS:
    cs = [s["id"] for s in SLOTS if cat(s["id"]) == cname]; e = {}
    for src in NAMES:
        rr = [x for x in rows if x["category"] == cname and (src == "combined" or x["source"] == src)]; n = len(rr); p = sum(x["auto_pass"] for x in rr)
        e[src] = dict(gens=n, auto_pass=p, pass_rate=round(p / n, 3) if n else None)
    e["slots"] = cs; e["filled"] = sum(slotview[s]["filled"] for s in cs); e["total_slots"] = len(cs); e["in_review"] = [s for s in cs if slotview[s]["review"] and not slotview[s]["filled"]]
    e["empty"] = [s for s in cs if not slotview[s]["filled"] and not slotview[s]["review"]]; stats[cname] = e
tr = [x for x in rows if x["source"] == CUR]
stats["_totals"] = dict(train_slots_filled=sum(v["filled"] for k, v in slotview.items() if not k.startswith("V")), train_slots=24,
                        validation_filled=sum(v["filled"] for k, v in slotview.items() if k.startswith("V")), validation_slots=4,
                        gens={s: sum(1 for x in rows if x["source"] == s) for s, _ in SRC}, current_gen_seconds=round(sum(x["gen_sec"] or 0 for x in tr), 1))
fbt = [x for x in tr if fullbody(CUR, x["slot"])]
mean = lambda v: round(float(np.mean(v)), 4) if v else None
stats["_crop_effect_current"] = dict(n=len(fbt), raw_only_pass=sum(1 for x in fbt if x["raw_only_reasons"] == []), crop_pass=sum(1 for x in fbt if x["auto_pass"]),
    mean_raw_cos=mean([x["cosine"] for x in fbt if x["cosine"] is not None]), mean_crop_cos=mean([x["crop_cosine"] for x in fbt if x["crop_cosine"] is not None]),
    mean_face_px=mean([x["face_px"] for x in fbt if x["face_px"]]))
stats["_redefined_slots"] = REDEF
json.dump(stats, open(os.path.join(D, "category_stats.json"), "w"), indent=1)
for f in ("candidates_scored.csv", "candidates_scored.json", "gen_log.jsonl"):
    if os.path.exists(os.path.join(o.root, f)): shutil.copy(os.path.join(o.root, f), os.path.join(D, f))
for k, v in (x.split("=", 1) for x in o.slot_files): shutil.copy(v, os.path.join(D, f"slots_{k}.json"))
for f in o.notes:
    if os.path.exists(f): shutil.copy(f, os.path.join(D, "notes_" + os.path.basename(os.path.dirname(os.path.abspath(f))) + "_" + os.path.basename(f)))
L = [f"# char01 pilot {tag}: combined summary", "", f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='minutes')}. Earlier files are referenced, not moved (combined/ holds symlinks; combined_manifest.json maps slots).", "",
     f"Training slots filled: {stats['_totals']['train_slots_filled']}/24; validation filled: {stats['_totals']['validation_filled']}/4. Gens per source: {stats['_totals']['gens']}; {CUR} GPU time {stats['_totals']['current_gen_seconds']/3600:.2f} h.", ""]
if REDEF: L += ["Redefined slots: " + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(REDEF.items())), ""]
L += ["| category | slots filled | in review | empty | " + " | ".join(f"{s} pass" for s in NAMES) + " |", "|---|---|---|---|" + "---|" * len(NAMES)]
fm = lambda d: f"{d['auto_pass']}/{d['gens']}" + (f" ({100*d['pass_rate']:.0f}%)" if d["gens"] else "")
for cname in CATS:
    e = stats[cname]; L.append(f"| {cname} | {e['filled']}/{e['total_slots']} | {', '.join(e['in_review']) or '-'} | {', '.join(e['empty']) or '-'} | " + " | ".join(fm(e[s]) for s in NAMES) + " |")
L += ["", "Pass = identity (yaw-banded ArcFace threshold; >50 deg: 0.45 floor + manual review) and direction (within 15 deg) both pass, before near-duplicate selection.", "",
      "## Slots", "", "| slot | category | target yaw | status | selected / review (yaw, cos, crop cos) | flags |", "|---|---|---|---|---|---|"]
for s in SLOTS:
    v = slotview[s["id"]]; its = sorted([x for x in rows if x["slot"] == s["id"] and x["combined_status"] in ("accepted_provisional", "validation", "review")], key=key)
    L.append(f"| {s['id']} | {v['category']} | {s['yaw']}{' (' + '; '.join(v['redefined']) + ')' if v['redefined'] else ''} | {'filled' if v['filled'] else 'review' if v['review'] else 'EMPTY'} | " +
             "<br>".join(f"{x['source']}:{x['file']} ({f0(x['yaw'])}, {f3(x['cosine'])}, {f3(x['crop_cosine'])}) [{x['combined_status']}]" for x in its) + " | " +
             "<br>".join("; ".join(x["flags"] + x["manual_flags"]) for x in its) + " |")
e = stats["_crop_effect_current"]
L += ["", f"## {CUR}: face-crop scoring on fullbody/knees-up slots", "", f"Gens: {e['n']}; pass on raw score alone: {e['raw_only_pass']}; pass on face-crop score: {e['crop_pass']}; mean raw cos {e['mean_raw_cos']}, mean crop cos {e['mean_crop_cos']}, mean face size {e['mean_face_px']} px.", "",
      "## Review list", "", "| file | source | slot | yaw | cos | crop cos | flags |", "|---|---|---|---|---|---|---|"]
for x in rev: L.append(f"| {x['file']} | {x['source']} | {x['slot']} | {f0(x['yaw'])} | {f3(x['cosine'])} | {f3(x['crop_cosine'])} | {'; '.join(x['flags'] + x['manual_flags'])} |")
open(os.path.join(D, "summary.md"), "w").write("\n".join(L) + "\n"); print("REPORT_DONE", stats["_totals"])
