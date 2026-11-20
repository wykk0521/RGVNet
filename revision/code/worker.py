"""Local to the rented instance: dispatch jobs on free GPUs, independent of SSH lifetime or calendar."""
from pathlib import Path
import os,sys,time,json,subprocess,fcntl,traceback
BASE=Path(__file__).resolve().parents[1]; CTRL=BASE/'control'
PY='/root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python'
SRC=Path('/root/wyk5/RGVNet')
def write(path,value):
    tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def saved(path):return json.loads(path.read_text()) if path.is_file() else {}
def main():
    guard=(CTRL/'dispatcher.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    assert len(uuids)==3
    plan=json.loads((CTRL/'training_plan.json').read_text());running={};closed={};failtypes=set()
    totals={}
    def env(g):
        return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=uuids[g],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',GDAL_NUM_THREADS='1',GDAL_CACHEMAX='8',MALLOC_ARENA_MAX='2',
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTHONPATH=':'.join(map(str,[BASE/'code',SRC/'models/hbgnet/_deps',SRC/'.python_packages',SRC])))
    print(stamp(),'started dispatcher',os.getpid(),uuids,flush=True)
    def tasks():
        ret=[]
        for r in plan:
            if r in closed:continue
            st=saved(BASE/'artifacts/logs'/r/'training_status.json')
            if st.get('status')=='completed' and st.get('epoch')==40 and (BASE/'artifacts/weights'/r/'best.pt').is_file():
                closed[r]='complete';continue
            job=BASE/'jobs'/r
            if (job/'dispatcher_failure.json').exists():closed[r]='failed'; failtypes.add(r.split('_')[0]);continue
            cp=BASE/'artifacts/weights'/r/'last.pt'
            if (BASE/'artifacts/logs'/r).exists() and not cp.is_file() and r not in [v[0] for v in running.values()]:
                closed[r]='needs_inspection';continue
            cmd=[PY,'-B','-u',str(job/'trainer.py'),'--config',str(job/'config.yaml'),'--device','cuda:0']
            if cp.is_file():cmd+=['--resume',str(cp)]
            ret.append((r,job,cmd,{'REV_JOB':r},r.split('_')[0]))
        extra=saved(CTRL/'analysis_plan.json') or []
        for a in extra:
            name=a['id'];job=BASE/'jobs'/name
            if name in closed:continue
            if (job/'complete.json').exists():closed[name]='complete';continue
            if (job/'dispatcher_failure.json').exists():closed[name]='failed';continue
            if not all(closed.get(k)=='complete' for k in a.get('needs',[])):continue
            ret.append((name,job,[PY,'-B','-u',str(BASE/'code'/a['entry']),*a.get('args',[])],a.get('env',{}),name))
        return ret
    while True:
        for g,(r,p,fd,kind,job) in list(running.items()):
            exitcode=p.poll()
            if exitcode is None:continue
            fd.close()
            if r in plan:
                st=saved(BASE/'artifacts/logs'/r/'training_status.json')
                ok=exitcode==0 and st.get('status')=='completed' and st.get('epoch')==40 and (BASE/'artifacts/weights'/r/'best.pt').is_file()
            else:ok=exitcode==0 and (job/'complete.json').is_file()
            closed[r]='complete' if ok else 'failed'
            print(stamp(),closed[r],g,r,exitcode,flush=True)
            if not ok:
                write(job/'dispatcher_failure.json',{'exit_code':exitcode,'time':stamp(),'gpu':g})
                failtypes.add(kind)
            del running[g]
        for g in range(3):
            if g in running:continue
            active={v[0] for v in running.values()}
            options=[q for q in tasks() if q[0] not in active and q[4] not in failtypes]
            if not options:continue
            r,job,cmd,en,kind=options[0];job.mkdir(exist_ok=True)
            # An exclusive lock is also held by each original training loop in its job directory.
            fd=(job/'console.log').open('ab',buffering=0);e=env(g);e.update(en)
            p=subprocess.Popen(cmd,cwd=job,env=e,stdin=subprocess.DEVNULL,stdout=fd,stderr=subprocess.STDOUT,start_new_session=True)
            running[g]=(r,p,fd,kind,job)
            write(job/'launch.json',{'gpu':g,'uuid':uuids[g],'pid':p.pid,'started':stamp(),'argv':cmd})
            print(stamp(),'launch',g,r,p.pid,flush=True)
        write(CTRL/'status.json',{'updated':stamp(),'dispatcher_pid':os.getpid(),'running':{str(g):{'job':i[0],'pid':i[1].pid} for g,i in running.items()},
              'closed':closed,'held_failed_families':sorted(failtypes),'pending_training':[r for r in plan if r not in closed and r not in {i[0] for i in running.values()}]})
        if not running and all(r in closed for r in plan) and (CTRL/'analysis_plan.json').exists():
            ee=saved(CTRL/'analysis_plan.json')
            if all(a['id'] in closed or any(closed.get(k) in ('failed','needs_inspection') for k in a.get('needs',[])) for a in ee):break
        time.sleep(4)
if __name__=='__main__':main()

