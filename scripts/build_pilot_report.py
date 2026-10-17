"""char01 pilot v1 report (new file; v1 left untouched). Reads candidates_scored.json + per-candidate metadata,
writes acceptance.csv/json + contact sheets. Thumbnails only (full-res originals stay on the GPU host)."""
import json,os,sys,csv,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from pilot_sheet import sheet
B=sys.argv[1] if len(sys.argv)>1 else 'dataset_pilot_v1'
rows=json.load(open(f'{B}/candidates_scored.json'))
CAT={**{k:'frontal' for k in '01 02 03 04 05 06'.split()},**{k:'three_quarter_left' for k in '07 08 09'.split()},**{k:'three_quarter_right' for k in '10 11 12'.split()},
     **{k:'profile' for k in '13 14 15 16'.split()},**{k:'half_body' for k in '17 18 19'.split()},**{k:'full_body' for k in '20 21 22'.split()},'23':'camera_variation','24':'camera_variation'}
vis=json.load(open(f'{B}/visual_review.json')) if os.path.exists(f'{B}/visual_review.json') else {}
for r in rows:
    m=json.load(open(f"{B}/candidates/{r['file'][:-4]}.json")); r['meta']=m['meta']
    r['category']='validation' if r['slot'].startswith('V') else CAT[r['slot']]
    r['body_geometry']=('face_ok_body_geometry_uncertain' if r['category'] in('half_body','full_body') and not any('identity' in x for x in r['auto_reasons']) and r.get('cosine') else 'n/a')
    v=vis.get(r['file'],{}); r['visual']=v
    r['final_status']=v.get('final',r['status']) if r['status'] in('accepted_provisional','review','validation') else r['status']
    r['rejection_reason']='; '.join(r['auto_reasons']+([v['note']] if v.get('final')=='rejected' else []))
json.dump(rows,open(f'{B}/acceptance.json','w'),indent=1)
with open(f'{B}/acceptance.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['file','slot','category','seed','target_yaw','yaw','pitch','cosine','threshold','n_faces','status','final_status','rejection_reason','manual_flags','body_geometry','visual_note','sec','peak_vram_mib','cfg','steps','prompt'])
    for r in rows:
        m=r['meta']; w.writerow([r['file'],r['slot'],r['category'],r['seed'],r['target_yaw'],r.get('yaw'),r.get('pitch'),r.get('cosine'),r.get('threshold'),r['n_faces'],r['status'],r['final_status'],r['rejection_reason'],'; '.join(r['manual_flags']),r['body_geometry'],r['visual'].get('note',''),m.get('sec'),m.get('peak_vram_mib'),m.get('cfg'),m.get('steps'),m.get('prompt')])
C=collections.defaultdict(lambda: collections.Counter())
for r in rows:
    c=C[r['category']]; c['gens']+=1; c['auto_pass']+= not r['auto_reasons']
for sid in sorted({r['slot'] for r in rows}):
    L=[r for r in rows if r['slot']==sid]; c=C[L[0]['category']]; c['slots']+=1
    f=[r for r in L if r['final_status'] in('accepted_provisional','validation')]; rv=[r for r in L if r['final_status']=='review']
    c['filled']+=bool(f); c['review']+=bool(rv and not f)
print(json.dumps({k:dict(v) for k,v in C.items()},indent=0))
json.dump({k:dict(v) for k,v in C.items()},open(f'{B}/category_stats.json','w'),indent=1)
img=lambda r:f"{B}/candidates/{r['file'][:-4]}.jpg"
lab=lambda r:(img(r),f"{r['slot']} s{r['seed']} {r['category'][:12]}",f"cos {r.get('cosine')} yaw {r.get('yaw')}")
for st,name,cols in [('accepted_provisional','approved',4),('review','review',4),('validation','validation',4)]:
    L=[r for r in rows if r['final_status']==st]
    if L: sheet([lab(r) for r in L],cols,f'{B}/{name}_contact_sheet.jpg')
# close-call sheet: best rejected per unfilled slot (for the correction analysis)
best=[]
for sid in sorted({r['slot'] for r in rows}):
    L=[r for r in rows if r['slot']==sid]
    if not any(r['final_status'] in('accepted_provisional','validation','review') for r in L):
        k=[r for r in L if r.get('cosine')]; 
        if k: best.append(max(k,key=lambda r:r['cosine']))
if best: sheet([(img(r),f"{r['slot']} s{r['seed']} REJ",f"cos {r['cosine']} yaw {r['yaw']}") for r in best],6,f'{B}/unfilled_best_rejects_sheet.jpg')
