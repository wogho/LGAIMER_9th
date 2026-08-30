from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/'candidate/COMBO-FK2-006'; OUT=ROOT/'output/submit_combo_fk2_006.zip'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 files=[('script.py',C/'script.py'),('combo_infer.py',C/'combo_infer.py'),('requirements.txt',C/'requirements.txt'),('LG_Aimers_솔루션_PPT_Phase2.pptx',ROOT/'output/LG_Aimers_솔루션_PPT_Phase2.pptx')]
 files += [(f'model/{p.relative_to(C/"model")}',p) for p in sorted((C/'model').rglob('*')) if p.is_file()]
 assert not [str(p) for _,p in files if not p.is_file()]
 with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
  for a,p in sorted(files): z.writestr(a,p.read_bytes())
 with zipfile.ZipFile(OUT) as z: assert z.testzip() is None
 r={'experiment_id':'COMBO-FK2-006','zip_path':str(OUT),'zip_sha256':sha(OUT),'members':[a for a,_ in sorted(files)],'submission_status':'HOLD'}
 (OUT.with_suffix('.manifest.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
