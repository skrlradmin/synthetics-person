"""Score + filter top-up candidates (run in the isolated InsightFace venv; analysis only, images are never modified).
Same rules as filter_pilot_dataset.py:
  identity: ArcFace (buffalo_l) cosine vs reference >= 0.85 (|yaw|<15), 0.75 (15-35), 0.68 (35-50); |yaw|>50 -> manual review (0.45 floor)
  direction: |yaw - target| <= 15; near-duplicate: cosine >= 0.97 AND 64-bit pHash Hamming <= 8 vs any already selected image,
  including images of earlier datasets passed via --dup-pool.
Additions:
  * every image also gets a face-crop score: largest detected face, square crop of side (1+2*margin)*max(w,h) (zero-padded at borders),
    resized with Lanczos to --crop-size px, re-detected and embedded. For slots with "fullbody": true the crop score is the identity
    decision (same thresholds; yaw band and direction from the crop pose when the crop re-detects, else from the raw pose);
    raw scores are always recorded. Crops are saved to ROOT/face_crops/ for inspection only.
  * half/full-body selections get the manual flag body_geometry_visual_pending.
  * --compare-root/--compare-slots re-score an earlier dataset's candidates with the crop method (report only, nothing selected or moved).
Usage: python filter_pilot_topup.py --root DIR --ref canon.png --slots topup_slots.json --insightface-root DIR [--dup-pool DIR ...]"""
import argparse, json, os, glob, shutil, csv
import numpy as np, cv2
from insightface.app import FaceAnalysis
a = argparse.ArgumentParser(); a.add_argument("--root", required=True); a.add_argument("--ref", required=True); a.add_argument("--slots", required=True)
a.add_argument("--insightface-root", required=True); a.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
a.add_argument("--dup-pool", nargs="*", default=[]); a.add_argument("--crop-size", type=int, default=512); a.add_argument("--crop-margin", type=float, default=0.5)
a.add_argument("--compare-root"); a.add_argument("--compare-slots", nargs="*", default=[]); o = a.parse_args()
app = FaceAnalysis(name="buffalo_l", root=o.insightface_root, providers=o.providers); app.prepare(ctx_id=-1, det_size=(640, 640))
def largest(fs): return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
def face_img(im):
    fs = app.get(im); return (largest(fs), len(fs)) if fs else (None, 0)
def crop_of(im, f):
    x1, y1, x2, y2 = [float(v) for v in f.bbox]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; side = max(x2 - x1, y2 - y1) * (1 + 2 * o.crop_margin)
    X1, Y1 = int(round(cx - side / 2)), int(round(cy - side / 2)); X2, Y2 = X1 + int(round(side)), Y1 + int(round(side)); H, W = im.shape[:2]
    pad = max(0, -X1, -Y1, X2 - W, Y2 - H); P = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else im
    c = P[Y1 + pad:Y2 + pad, X1 + pad:X2 + pad]; return cv2.resize(c, (o.crop_size, o.crop_size), interpolation=cv2.INTER_LANCZOS4), int(round(side))
D = None
def phash(im):
    global D
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    if D is None:
        n = 32; D = np.array([[np.sqrt((1 if k == 0 else 2) / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n)) for i in range(n)] for k in range(n)])
    c = (D @ g @ D.T)[:8, :8].flatten(); return (c > np.median(c[1:])).astype(np.uint8)
rim = cv2.imread(o.ref); rf, _ = face_img(rim); R = rf.normed_embedding
def thr(ay): return 0.85 if ay < 15 else 0.75 if ay < 35 else 0.68 if ay <= 50 else None
def measure(p, crop_dir=None):
    im = cv2.imread(p); f, nf = face_img(im); m = dict(n_faces=nf, _phash=phash(im), image_wh=[im.shape[1], im.shape[0]])
    if f is None: m.update(yaw=None, pitch=None, cosine=None, det_score=None, face_px=None, crop_yaw=None, crop_pitch=None, crop_cosine=None, crop_det_score=None); return m
    pitch, yaw, roll = [float(x) for x in f.pose]; m.update(yaw=round(yaw, 1), pitch=round(pitch, 1), cosine=round(float(np.dot(R, f.normed_embedding)), 4),
        det_score=round(float(f.det_score), 3), face_px=int(round(max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]))), _emb=f.normed_embedding)
    c, side = crop_of(im, f); cf, _ = face_img(c); m["crop_src_px"] = side
    if crop_dir: cv2.imwrite(os.path.join(crop_dir, os.path.basename(p)[:-4] + "_face.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if cf is None: m.update(crop_yaw=None, crop_pitch=None, crop_cosine=None, crop_det_score=None)
    else:
        cp, cy, _ = [float(x) for x in cf.pose]; m.update(crop_yaw=round(cy, 1), crop_pitch=round(cp, 1), crop_cosine=round(float(np.dot(R, cf.normed_embedding)), 4), crop_det_score=round(float(cf.det_score), 3), _cemb=cf.normed_embedding)
    return m
def decide(r, t, fullbody):
    reasons, manual = [], []
    use_crop = fullbody and r.get("crop_cosine") is not None
    yaw = r["crop_yaw"] if use_crop else r["yaw"]; cos = r["crop_cosine"] if use_crop else r["cosine"]
    r["decision_basis"] = "face_crop" if use_crop else "raw"
    if fullbody and r.get("yaw") is not None and not use_crop: manual.append("crop_redetect_failed_raw_used")
    if r["yaw"] is None: reasons.append("no_face_detected"); return reasons, manual
    r["decision_yaw"], r["decision_cosine"] = yaw, cos
    if r["n_faces"] > 1: manual.append("multiple_faces")
    if abs(t) >= 70:
        if not (np.sign(yaw) == np.sign(t) and abs(yaw - t) <= 15): reasons.append(f"wrong_direction(yaw {yaw:.0f} vs target {t})")
        else:
            manual.append("profile_identity_manual(arcface unreliable >50deg)")
            if cos < 0.45: reasons.append(f"identity_below_profile_floor({cos:.3f}<0.45)")
    else:
        if abs(yaw - t) > 15: reasons.append(f"wrong_direction(yaw {yaw:.0f} vs target {t})")
        th = thr(abs(yaw)); r["threshold"] = th
        if th is None:
            manual.append("over_50deg_manual")
            if cos < 0.45: reasons.append(f"identity_below_profile_floor({cos:.3f}<0.45)")
        elif cos < th: reasons.append(f"identity_below_threshold({cos:.3f}<{th})")
    return reasons, manual
S = json.load(open(o.slots)); slots = {s["id"]: dict(s, validation=False) for s in S["slots"]}; slots.update({s["id"]: dict(s, validation=True) for s in S.get("validation", [])})
CD = os.path.join(o.root, "face_crops"); os.makedirs(CD, exist_ok=True); rows = []
for p in sorted(glob.glob(os.path.join(o.root, "candidates", "*.png"))):
    sid, seed = os.path.basename(p)[:-4].split("_s"); s = slots[sid]; fb = bool(s.get("fullbody"))
    r = dict(file=os.path.basename(p), slot=sid, seed=int(seed), validation=s["validation"], target_yaw=s["yaw"], fullbody=fb); r.update(measure(p, CD))
    r["auto_reasons"], r["manual_flags"] = decide(r, s["yaw"], fb)
    if fb and r.get("cosine") is not None:  # what the raw (uncropped) score alone would have decided
        q = dict(r); q["crop_cosine"] = None; rr, _ = decide(q, s["yaw"], False); r["raw_only_reasons"] = rr
    rows.append(r)
pool = []
for d in o.dup_pool:
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        m = measure(p)
        if m.get("_emb") is not None: pool.append(dict(file=os.path.basename(p), _emb=m["_emb"], _phash=m["_phash"], prev=True))
print("dup pool", len(pool))
sel = list(pool)
for sid in slots:
    cands = sorted([r for r in rows if r["slot"] == sid and not r["auto_reasons"]], key=lambda r: -(r.get("decision_cosine") or 0))
    chosen = None
    for r in cands:
        dup = next((q for q in sel if float(np.dot(q["_emb"], r["_emb"])) >= 0.97 and int((q["_phash"] != r["_phash"]).sum()) <= 8), None)
        if dup: r["auto_reasons"].append(f"near_duplicate_of_{dup['file']}" + ("(v1)" if dup.get("prev") else "")); continue
        chosen = r; break
    for r in [x for x in rows if x["slot"] == sid]:
        body = "half-body" in slots[sid]["framing"].lower() or slots[sid].get("fullbody")
        if r is chosen:
            if body: r["manual_flags"].append("body_geometry_visual_pending")
            sel.append(r); hard = [m for m in r["manual_flags"] if m != "body_geometry_visual_pending"]
            r["status"] = ("validation" if r["validation"] and not hard else "review" if hard else "accepted_provisional")
        elif r["auto_reasons"]: r["status"] = "rejected"
        else: r["status"] = "pass_not_selected"
        r["checks"] = dict(identity=("manual" if any("manual" in m for m in r["manual_flags"]) else ("fail" if any("identity" in x for x in r["auto_reasons"]) else "pass")),
                           direction=("fail" if any("direction" in x for x in r["auto_reasons"]) else "pass"),
                           near_duplicate=("fail" if any("duplicate" in x for x in r["auto_reasons"]) else "pass"),
                           beauty_mark="visual_pending", anatomy="visual_pending", texture="visual_pending")
for d in ["accepted_provisional", "review", "rejected", "validation"]:
    os.makedirs(os.path.join(o.root, d), exist_ok=True)
    for f in glob.glob(os.path.join(o.root, d, "*.png")): os.remove(f)  # derived views of ROOT/candidates (never touches other datasets)
for r in rows:
    if r["status"] in ("accepted_provisional", "review", "rejected", "validation"):
        shutil.copy(os.path.join(o.root, "candidates", r["file"]), os.path.join(o.root, r["status"], r["file"]))
COLS = ["file", "slot", "seed", "validation", "target_yaw", "fullbody", "yaw", "pitch", "cosine", "face_px", "crop_yaw", "crop_pitch", "crop_cosine", "decision_basis",
        "decision_yaw", "decision_cosine", "threshold", "n_faces", "status", "auto_reasons", "manual_flags", "raw_only_reasons", "checks"]
def dump(rs, base):
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]; json.dump(out, open(base + ".json", "w"), indent=1)
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLS)
        for r in out: w.writerow([("; ".join(r[c]) if isinstance(r.get(c), list) and c != "image_wh" else json.dumps(r[c]) if isinstance(r.get(c), dict) else r.get(c)) for c in COLS])
    return out
out = dump(rows, os.path.join(o.root, "candidates_scored"))
print("scored", len(out)); print({s: sum(1 for r in out if r["status"] == s) for s in ["accepted_provisional", "review", "validation", "rejected", "pass_not_selected"]})
if o.compare_root and o.compare_slots:
    cmp = []; prev = {r["file"]: r for r in json.load(open(os.path.join(o.compare_root, "candidates_scored.json")))}
    for p in sorted(glob.glob(os.path.join(o.compare_root, "candidates", "*.png"))):
        sid = os.path.basename(p).split("_s")[0]
        if sid not in o.compare_slots: continue
        pr = prev.get(os.path.basename(p), {}); t = pr.get("target_yaw"); r = dict(file=os.path.basename(p), slot=sid, seed=pr.get("seed"), validation=False, target_yaw=t, fullbody=True)
        r.update(measure(p)); r["auto_reasons"], r["manual_flags"] = decide(r, t, True); r["status"] = "pass_by_crop" if not r["auto_reasons"] else "fail_by_crop"
        r["raw_only_reasons"] = pr.get("auto_reasons"); cmp.append(r)
    dump(cmp, os.path.join(o.root, "compare_prev_fullbody_crop_rescore")); print("compare rescored", len(cmp), "pass_by_crop", sum(1 for r in cmp if r["status"] == "pass_by_crop"))
