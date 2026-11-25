"""Export immutable original auxiliary traces without loading a GPU model."""
from common import *
import csv,io,zipfile
def main():
 environment();verify_lock()
 from binding.v19_analysis import loss_components
 cfg=read(O/'preflight.json');dest=O/'losses';dest.mkdir(exist_ok=False);exports=[];pins=[]
 for src in cfg['loss_sources']:
  p=R/src['path'];assert sha(p)==src['sha256'];data=rows(p)
  for row in data:
   exports.append(dict(job=src['job'],step=row['step'],source_id=row['source_id'],learning_rate=row['learning_rate'],seconds=row['seconds'],gradient_norm=row['gradient_norm'],**loss_components(row)))
  target=dest/(src['job']+'.steps.jsonl');target.write_bytes(p.read_bytes());pins.append(dict(path=target.name,sha256=sha(target)))
 with (dest/'auxiliary_losses.csv').open('x',newline='',encoding='utf8') as f:
  writer=csv.DictWriter(f,fieldnames=list(exports[0]));writer.writeheader();writer.writerows(exports)
 assert len(exports)==3072
 write(dest/'manifest.json',dict(status='verified',rows=len(exports),raw_sources=cfg['loss_sources'],files=pins,csv_sha256=sha(dest/'auxiliary_losses.csv'),weights=dict(ce_each_family=.5,probe_aux=.1,locality_each_family=.025),gpu_used=False))
 archive=O/'original_auxiliary_losses.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(dest.iterdir()):z.write(p,p.name)
 write(O/'loss_export_receipt.json',dict(status='complete',archive=archive.relative_to(R).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive),rows=len(exports)))
 print(json.dumps(read(O/'loss_export_receipt.json')),flush=True)
if __name__=='__main__':main()
