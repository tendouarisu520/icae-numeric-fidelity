from pathlib import Path
import hashlib,json,os,time
R=Path('E:/Research/ICAE-ConditionalMemory-20261021')
O=R/'results/v19-budget-layout-20261108'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines()]
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def environment():
 import sys
 assert R.drive.lower()=='e:';os.chdir(R);sys.path.insert(0,str(R/'src'));os.environ['PYTHONPATH']=str(R/'src')
 for k,v in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba','MPLCONFIGDIR':'cache/matplotlib'}.items():
  p=R/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
def verify_lock():
 lock=read(R/'temp/v19/lock.json')
 for name,h in lock['files'].items():assert sha(R/name)==h,name
 assert sha(R/'results/confirmation-20261104/protocol.json')==lock['original_protocol_sha256']
 return lock
def emit(path,**values):
 v=dict(time=time.time(),pid=os.getpid(),**values);write(path,v);print(json.dumps(v),flush=True)
