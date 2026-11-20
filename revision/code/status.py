"""Read-only snapshot; run with python3 on server 194."""
from pathlib import Path
import datetime as dt,json,re,subprocess
BASE=Path(__file__).resolve().parents[1];ctrl=BASE/'control'
def tail(p,n=27000):
    if not p.is_file():return ''
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
def read(p):return json.loads(Path(p).read_text())
def main():
    s=read(ctrl/'status.json')
    s['snapshot_china']=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
    s['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).splitlines()
    for g,r in s['running'].items():
        j=r['job'];p=BASE/'jobs'/j;r['preflight_passed']=(p/'preflight.json').exists()
        ls=tail(p/'console.log')
        mt=re.findall(r'epoch=(\d+)/(\d+) batch=(\d+)/(\d+) loss=([-+.eE\d]+)',ls)
        if mt:r['last_logged_batch']=dict(zip(['epoch','epochs','batch','batches','loss'],mt[-1]))
        t=BASE/'artifacts/logs'/j/'training_status.json'
        if t.is_file():r['last_committed_epoch']=read(t).get('epoch')
        w=BASE/'artifacts/weights'/j;r['best_available']=(w/'best.pt').is_file()
    s['oom_kill']=int(next(v.split()[1] for v in Path('/sys/fs/cgroup/memory.events').read_text().splitlines() if v.startswith('oom_kill ')))
    s['planned_training_jobs']=len(read(ctrl/'training_plan.json'))+int(any(p['id']=='segfix_73' for p in (read(ctrl/'analysis_plan.json') if (ctrl/'analysis_plan.json').exists() else [])))
    print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

