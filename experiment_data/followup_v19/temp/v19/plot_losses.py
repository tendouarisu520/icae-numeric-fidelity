"""Vector plots of the verified six original auxiliary traces, CPU only."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,os
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium
R=Path(__file__).resolve().parents[2];O=R/'results/v19-budget-layout-20261108';F=O/'figures';F.mkdir(exist_ok=True)
assert json.loads((O/'loss_transfer_receipt.json').read_text())['status']=='verified'
path=O/'losses/auxiliary_losses.csv';rows=list(csv.DictReader(path.open(encoding='utf8')));assert len(rows)==3072
groups=defaultdict(list)
for r in rows:
 r={k:v if k in ('job','source_id') else float(v) for k,v in r.items()};groups[r['job']].append(r)
 assert math.isclose(r['total_loss'],r['weighted_ce']+r['weighted_aux']+r['weighted_locality'],rel_tol=1e-6,abs_tol=1e-6)
for data in groups.values():assert [int(r['step']) for r in data]==list(range(1,513))
components=[('weighted_ce','Answer CE','#235789'),('weighted_aux','Auxiliary term','#BC5B30'),('weighted_locality','Locality term','#2D805A')]
methods=[('assignment','Assignment'),('relation','Assignment + locality'),('bottleneck','Weakest margin + locality')]
positive=[v[key] for data in groups.values() for v in data for key,_,_ in components if v[key]>0]
lo=math.floor(math.log10(min(positive)));hi=math.ceil(math.log10(max(positive)));assert all(v[key]>=0 for data in groups.values() for v in data for key,_,_ in components)
pdf=F/'original_auxiliary_loss_curves.pdf';W,H=900,760;c=Canvas(str(pdf),pagesize=(W,H));c.setTitle('Original auxiliary training loss components through 512 updates')
c.setFont('Helvetica-Bold',17);c.drawString(48,727,'Original auxiliary training through 512 updates')
c.setFont('Helvetica',10);c.drawString(48,707,'Actual weighted loss contributions  |  Six fixed runs  |  Raw updates and 32-update rolling means')
summaries=[]
for row,(method,label) in enumerate(methods):
 for col,seed in enumerate((73,97)):
  data=groups[f'main-compressed-{method}-seed{seed}'];x0=67+col*435;y0=506-row*207;pw=366;ph=158
  xx=lambda step:x0+pw*(step-1)/511
  yy=lambda value:y0+ph*(math.log10(value)-lo)/(hi-lo)
  c.setFillColor(HexColor('#17212B'));c.setFont('Helvetica-Bold',11);c.drawString(x0,y0+ph+13,f'{label} / seed {seed}')
  c.setFont('Helvetica',8)
  for power in range(lo,hi+1):
   y=yy(10**power);c.setStrokeColor(HexColor('#DCE2E7'));c.setLineWidth(.4);c.line(x0,y,x0+pw,y)
   c.setFillColor(HexColor('#46515C'));c.drawRightString(x0-6,y-2.7,f'1e{power}')
  for step in (1,128,256,384,512):c.drawCentredString(xx(step),y0-13,str(step))
  c.drawCentredString(x0+pw/2,y0-26,'Optimizer update')
  c.saveState();c.translate(x0-42,y0+ph/2);c.rotate(90);c.drawCentredString(0,0,'Weighted loss (log scale)');c.restoreState()
  for key,name,color in components:
   vals=[r[key] for r in data];summary=dict(method=method,seed=seed,component=key,first128_mean=sum(vals[:128])/128,preceding128_mean=sum(vals[256:384])/128,last128_mean=sum(vals[384:])/128,zero_updates=sum(v==0 for v in vals))
   summary['last_vs_preceding_pct']=100*(summary['last128_mean']/summary['preceding128_mean']-1) if summary['preceding128_mean'] else ''
   summaries.append(summary)
   if not any(vals):continue
   c.setStrokeColor(HexColor(color));c.setLineWidth(.35);c.setStrokeAlpha(.16)
   for i in range(1,len(vals)):
    if vals[i-1]>0 and vals[i]>0:c.line(xx(i),yy(vals[i-1]),xx(i+1),yy(vals[i]))
   c.setStrokeAlpha(1);c.setLineWidth(1.5)
   smooth=[sum(vals[i-31:i+1])/32 for i in range(31,len(vals))]
   for i in range(1,len(smooth)):
    if smooth[i-1]>0 and smooth[i]>0:c.line(xx(i+31),yy(smooth[i-1]),xx(i+32),yy(smooth[i]))
for i,(_,name,color) in enumerate(components):
 x=120+i*240;c.setStrokeColor(HexColor(color));c.setLineWidth(2);c.line(x,52,x+28,52);c.setFillColor(HexColor('#17212B'));c.setFont('Helvetica',10);c.drawString(x+36,49,name)
c.setFont('Helvetica',8.5);c.setFillColor(HexColor('#45505A'))
c.drawString(48,29,'Faint lines: individual updates. Solid lines: trailing 32-update means. Zero components are omitted on logarithmic axes.')
c.drawString(48,15,'Auxiliary definitions differ by method. Loss magnitudes are not gradient magnitudes; successive windows contain different source subsets.')
c.save()
doc=pdfium.PdfDocument(str(pdf));doc[0].render(scale=2).to_pil().save(F/'original_auxiliary_loss_curves.png')
with (O/'losses/loss_window_summary.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
receipt=dict(status='generated',source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=3072,runs=6,scale='log10',zero_components_omitted=True,raw_points=True,rolling_window=32,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),visual_review_pending=True)
(F/'plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(pdf=str(pdf),png=str(F/'original_auxiliary_loss_curves.png'),summary=summaries)))
