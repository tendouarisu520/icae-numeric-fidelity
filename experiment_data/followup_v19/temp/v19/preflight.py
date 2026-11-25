"""No CUDA work; freeze actual data and model identities before the queue."""
from common import *
import csv,io,subprocess,sys,zipfile
from collections import Counter
def main():
 environment();lock=verify_lock();assert not (O/'preflight.json').exists(),'Audit prior run, do not overwrite'
 original=read(R/'results/confirmation-20261104/protocol.json')
 for name,h in original['pins'].items():assert sha(R/name)==h,name
 test=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s',str(R/'src/tests'),'-p','test_v19_followup.py'],capture_output=True,text=True)
 assert test.returncode==0,test.stdout+test.stderr
 (O/'cpu-tests.txt').write_text(test.stdout+test.stderr,encoding='utf8')
 from transformers import AutoTokenizer
 from binding.v19_data import balanced_training,prospective_units
 from binding.v13_data import prepare
 train=rows(R/'data/v13-answerable-relations-20261026/train.jsonl')
 balanced=balanced_training([u for u in train if u['family']=='probe']);ordinary,crossed=prospective_units()
 dest=R/'data/v19-prospective-20261108';dest.mkdir(exist_ok=False)
 datasets={'original':ordinary,'crossed':crossed,'balanced_training':balanced}
 oldpaths=[R/'data/v13-answerable-relations-20261026'/f'{s}.jsonl' for s in ('train','development')]+[R/'results/confirmation-20261104/confirmation_units.jsonl',R/'data/v16-role-balance-20261106/development.jsonl']
 oldrows=[u for p in oldpaths for u in rows(p)];old_ids={u['source_id'] for u in oldrows};oldcontexts={c for u in oldrows for c in u['contexts']}
 assert not {u['source_id'] for u in ordinary}&old_ids
 assert not {c for u in ordinary+crossed for c in u['contexts']}&oldcontexts
 tokenizer=AutoTokenizer.from_pretrained(str(R/'models/mistral'),local_files_only=True,use_fast=False)
 info={}
 for name,data in datasets.items():
  blob=''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in data).encode();p=dest/(name+'.jsonl');p.write_bytes(blob)
  assert sha(p)==lock['generated_data_sha256'][name]
  lengths=[len(c) for u in data for c in prepare(u,tokenizer)['contexts']]
  info[name]=dict(path=p.relative_to(R).as_posix(),sha256=sha(p),units=len(data),max_tokens=max(lengths),min_tokens=min(lengths))
 sources=[]
 for job in original['jobs']:
  if job['method']=='sft':continue
  p=(R/job['checkpoint']).parent/'steps.jsonl';r=rows(p);assert len(r)==512 and [x['step'] for x in r]==list(range(1,513))
  sources.append(dict(job=job['name'],path=p.relative_to(R).as_posix(),sha256=sha(p),rows=512))
 write(O/'preflight.json',dict(status='pass',frozen_before_new_inference=True,lock_sha256=sha(R/'temp/v19/lock.json'),datasets=info,original_jobs=original['jobs'],loss_sources=sources,old_cohort_hashes={p.relative_to(R).as_posix():sha(p) for p in oldpaths},time=time.time(),cpu_tests_pass=True,reserved_test_accessed=False))
 print(json.dumps(dict(preflight='pass',datasets=info)),flush=True)
if __name__=='__main__':main()
