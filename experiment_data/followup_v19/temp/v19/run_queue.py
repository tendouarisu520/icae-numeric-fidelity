"""One bounded queue; no silent restarts or touching other users' processes."""
from common import *
import ctypes,shutil,subprocess,sys
def resources():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 free=int(text.strip().splitlines()[0]);disk=shutil.disk_usage(R).free
 class Memory(ctypes.Structure):
  _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('totalPhys','availPhys','totalPage','availPage','totalVirtual','availVirtual','availExtended')]
 m=Memory();m.dwLength=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 return dict(gpu_free_mib=free,disk_free_gib=disk/1024**3,ram_free_gib=m.availPhys/1024**3)
def main():
 environment();O.mkdir(parents=True,exist_ok=True);verify_lock();state=O/'queue_status.json'
 assert not state.exists(),'Queue already exists; inspect before resuming'
 done=[]
 def run(label,args,gpu=False,output=None):
  verify_lock()
  if gpu:
   res=resources();assert res['gpu_free_mib']>=23500 and res['disk_free_gib']>=3 and res['ram_free_gib']>=15,res
  stamp=time.strftime('%Y%m%d-%H%M%S');log=R/f'logs/v19-{label}-{stamp}'
  with Path(str(log)+'.log').open('x') as stdout,Path(str(log)+'.err.log').open('x') as stderr:
   p=subprocess.Popen([sys.executable,'-X','utf8','-B',*args],cwd=R,stdout=stdout,stderr=stderr,creationflags=0x08000000)
   emit(state,phase='running',job=label,child_pid=p.pid,output=output,stdout=str(log)+'.log',stderr=str(log)+'.err.log',completed=done)
   while p.poll() is None:time.sleep(15)
  if p.returncode:raise RuntimeError(f'{label} exited{p.returncode}; inspect {log}; no retry')
  done.append(label);emit(state,phase='job_complete',job=label,completed=done)
 run('preflight',['temp/v19/preflight.py'],output=str(O/'preflight.json'))
 run('loss-export',['temp/v19/export_losses.py'],output=str(O/'original_auxiliary_losses.zip'))
 for method in ('assignment','relation','bottleneck'):
  for seed in (73,97):
   name=f'extended-compressed-{method}-seed{seed}';out=R/'runs/v19-budget-layout-20261108'/name;assert not out.exists()
   run(name,['-m','binding.v19_training','--phase','main','--method',method,'--seed',str(seed)],True,str(out))
   v=read(out/'summary.json');assert v['status']=='success' and v['optimizer_steps']==1024 and v['step512_exact_replay']
 for seed in (73,97):
  name=f'balanced-compressed-sft-seed{seed}';out=R/'runs/v19-budget-layout-20261108'/name;assert not out.exists()
  run(name,['-m','binding.v19_training','--phase','main','--method','sft','--seed',str(seed),'--balanced'],True,str(out))
  v=read(out/'summary.json');assert v['status']=='success' and v['optimizer_steps']==1024 and v['balanced_layout']
 run('balanced-layout',['temp/v19/evaluate.py','balanced_layout'],True,str(O/'balanced_layout'))
 run('prospective',['temp/v19/evaluate.py','prospective'],True,str(O/'prospective'))
 emit(state,phase='complete',completed=done)
if __name__=='__main__':
 try:main()
 except BaseException as e:emit(O/'queue_status.json',phase='failed',error=str(e));raise
