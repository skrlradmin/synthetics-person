"""Score + filter pilot-dataset candidates (run in the isolated InsightFace venv; analysis only).
Rules (docs/round4_summary.md section 4):
  identity: ArcFace cosine vs canon >= 0.85 (|yaw|<15), 0.75 (15-35), 0.68 (35-50); |yaw|>50 -> manual review (never auto-accepted)
  direction: |yaw - target| <= 15 for every slot (profiles: identity then goes to manual review)
  near-duplicate: ArcFace cosine >= 0.97 AND 64-bit pHash Hamming <= 8 vs an already selected image -> reject
  one face exactly; beauty mark / anatomy / texture are visual checks (flagged 'visual_pending').
Per slot the best passing candidate (highest cosine) is selected -> accepted_provisional; profiles -> review.
Usage: python filter_pilot_dataset.py --root DATASET_DIR --ref canon.png --slots templates/pilot_v1_slots.json --insightface-root DIR"""
import argparse, json, os, glob, shutil, csv
import numpy as np, cv2
from insightface.app import FaceAnalysis
a = argparse.ArgumentParser(); a.add_argument("--root", required=True); a.add_argument("--ref", required=True); a.add_argument("--slots", required=True)
a.add_argument("--insightface-root", required=True); a.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"]); o = a.parse_args()
app = FaceAnalysis(name="buffalo_l", root=o.insightface_root, providers=o.providers); app.prepare(ctx_id=-1, det_size=(640, 640))
def face(p):
    im = cv2.imread(p); fs = app.get(im)
    if not fs: return None, 0
    f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])); return f, len(fs)
D = None
def phash(p):
    global D
    g = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    if D is None:
        n = 32; D = np.array([[np.sqrt((1 if k == 0 else 2) / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n)) for i in range(n)] for k in range(n)])
    c = (D @ g @ D.T)[:8, :8].flatten(); return (c > np.median(c[1:])).astype(np.uint8)
rf, _ = face(o.ref); R = rf.normed_embedding
S = json.load(open(o.slots)); slots = {s["id"]: dict(s, validation=False) for s in S["slots"]}; slots.update({s["id"]: dict(s, validation=True) for s in S["validation"]})
def thr(ay): return 0.85 if ay < 15 else 0.75 if ay < 35 else 0.68 if ay <= 50 else None
rows = []
for p in sorted(glob.glob(os.path.join(o.root, "candidates", "*.png"))):
    sid, seed = os.path.basename(p)[:-4].split("_s"); s = slots[sid]; t = s["yaw"]; f, nf = face(p); r = dict(file=os.path.basename(p), slot=sid, seed=int(seed), validation=s["validation"], target_yaw=t, n_faces=nf)
    reasons = []; manual = []
    if f is None: reasons.append("no_face_detected"); r.update(yaw=None, pitch=None, cosine=None, det_score=None)
    else:
        pitch, yaw, roll = [float(x) for x in f.pose]; cos = float(np.dot(R, f.normed_embedding)); r.update(yaw=round(yaw, 1), pitch=round(pitch, 1), cosine=round(cos, 4), det_score=round(float(f.det_score), 3))
        r["_emb"] = f.normed_embedding
        if nf > 1: manual.append("multiple_faces")
        if abs(t) >= 70:
            if not (np.sign(yaw) == np.sign(t) and abs(yaw - t) <= 15): reasons.append(f"wrong_direction(yaw {yaw:.0f} vs target {t})")
            else: manual.append("profile_identity_manual(arcface unreliable >50deg)")
        else:
            if abs(yaw - t) > 15: reasons.append(f"wrong_direction(yaw {yaw:.0f} vs target {t})")
            th = thr(abs(yaw))
            if th is None: manual.append("over_50deg_manual")
            elif cos < th: reasons.append(f"identity_below_threshold({cos:.3f}<{th})")
            r["threshold"] = th
    r["_phash"] = phash(p); r["auto_reasons"] = reasons; r["manual_flags"] = manual; rows.append(r)
sel = []
for sid in slots:
    cands = sorted([r for r in rows if r["slot"] == sid and not r["auto_reasons"]], key=lambda r: -(r["cosine"] or 0))
    chosen = None
    for r in cands:
        dup = next((q for q in sel if float(np.dot(q["_emb"], r["_emb"])) >= 0.97 and int((q["_phash"] != r["_phash"]).sum()) <= 8), None)
        if dup: r["auto_reasons"].append(f"near_duplicate_of_{dup['file']}"); continue
        chosen = r; break
    for r in [x for x in rows if x["slot"] == sid]:
        if r is chosen:
            sel.append(r); r["status"] = ("validation" if r["validation"] else ("review" if r["manual_flags"] else "accepted_provisional"))
        elif r["auto_reasons"]: r["status"] = "rejected"
        else: r["status"] = "pass_not_selected"
        r["checks"] = dict(identity=("manual" if any("manual" in m for m in r["manual_flags"]) else ("fail" if any("identity" in x for x in r["auto_reasons"]) else "pass")),
                           direction=("fail" if any("direction" in x for x in r["auto_reasons"]) else "pass"),
                           near_duplicate=("fail" if any("duplicate" in x for x in r["auto_reasons"]) else "pass"),
                           beauty_mark="visual_pending", anatomy="visual_pending", texture="visual_pending")
for d in ["accepted_provisional", "review", "rejected", "validation"]:
    for f in glob.glob(os.path.join(o.root, d, "*.png")): os.remove(f)  # folders are derived views of candidates/
for r in rows:
    if r["status"] in ("accepted_provisional", "review", "rejected", "validation"):
        shutil.copy(os.path.join(o.root, "candidates", r["file"]), os.path.join(o.root, r["status"], r["file"]))
out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
json.dump(out, open(os.path.join(o.root, "candidates_scored.json"), "w"), indent=1)
with open(os.path.join(o.root, "candidates_scored.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "slot", "seed", "validation", "target_yaw", "yaw", "pitch", "cosine", "threshold", "n_faces", "status", "auto_reasons", "manual_flags", "checks"])
    for r in out: w.writerow([r["file"], r["slot"], r["seed"], r["validation"], r["target_yaw"], r.get("yaw"), r.get("pitch"), r.get("cosine"), r.get("threshold"), r["n_faces"], r["status"], "; ".join(r["auto_reasons"]), "; ".join(r["manual_flags"]), json.dumps(r["checks"])])
print("scored", len(out)); print({s: sum(1 for r in out if r["status"] == s) for s in ["accepted_provisional", "review", "validation", "rejected", "pass_not_selected"]})
