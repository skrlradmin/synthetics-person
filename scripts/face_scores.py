"""ArcFace identity scoring vs a reference (InsightFace buffalo_l; analysis only).
NOTE: InsightFace pretrained models are licensed for non-commercial research use only.
Usage: python face_scores.py --ref ref.png --images a.png ... [--controls name=path ...] --out face_scores [--root MODEL_ROOT] [--yaw-limit 45]
Per image: detection count/confidence, face box size (fraction of image height), pose (yaw/pitch from 3D landmarks),
cosine similarity of normed embeddings. Faces with |yaw| > yaw-limit are flagged unreliable (profiles)."""
import argparse, json, csv, os, numpy as np, cv2
from insightface.app import FaceAnalysis
a = argparse.ArgumentParser(); a.add_argument("--ref", required=True); a.add_argument("--images", nargs="*", default=[])
a.add_argument("--controls", nargs="*", default=[]); a.add_argument("--out", required=True)
a.add_argument("--root", default=os.path.expanduser("~/.insightface")); a.add_argument("--yaw-limit", type=float, default=45)
a.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider"); o = a.parse_args()
app = FaceAnalysis(name="buffalo_l", root=o.root, providers=o.providers.split(",")); app.prepare(ctx_id=0, det_size=(640, 640))
prov = app.models["recognition"].session.get_providers()[0]
def faces(p):
    im = cv2.imread(p); fs = app.get(im); return im, sorted(fs, key=lambda f: -(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
rim, rf = faces(o.ref); assert rf, "no face in reference"; R = rf[0].normed_embedding
rows = []
items = [("image", p, p) for p in o.images] + [("control", c.split("=", 1)[0], c.split("=", 1)[1]) for c in o.controls]
for kind, name, p in items:
    im, fs = faces(p); h, w = im.shape[:2]
    r = dict(kind=kind, name=name if kind == "control" else os.path.relpath(p), n_faces=len(fs), det_score="", box_h_frac="", box_px="",
             yaw="", pitch="", cosine="", reliable="", note="")
    if not fs: r.update(note="DETECTION FAILURE", reliable=False)
    else:
        f = fs[0]; x1, y1, x2, y2 = f.bbox; pose = getattr(f, "pose", None)
        yaw = float(pose[1]) if pose is not None else None; pitch = float(pose[0]) if pose is not None else None
        bh = (y2 - y1) / h
        r.update(det_score=round(float(f.det_score), 3), box_h_frac=round(float(bh), 3), box_px=f"{int(x2-x1)}x{int(y2-y1)}",
                 yaw="" if yaw is None else round(yaw, 1), pitch="" if pitch is None else round(pitch, 1),
                 cosine=round(float(np.dot(R, f.normed_embedding)), 4))
        bad = []
        if yaw is not None and abs(yaw) > o.yaw_limit: bad.append(f"|yaw|>{o.yaw_limit:g} (profile) - embedding unreliable")
        if (y2 - y1) < 112: bad.append("face <112px tall - low crop quality")
        if f.det_score < 0.6: bad.append("low detection confidence")
        r.update(reliable=not bad, note="; ".join(bad))
    rows.append(r); print(r, flush=True)
meta = dict(model="insightface buffalo_l (det_10g + w600k_r50 ArcFace)", provider=prov, reference=os.path.basename(o.ref),
            yaw_limit=o.yaw_limit, license="InsightFace pretrained models: non-commercial research only")
json.dump(dict(meta=meta, rows=rows), open(o.out + ".json", "w"), indent=1)
with open(o.out + ".csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(meta)
