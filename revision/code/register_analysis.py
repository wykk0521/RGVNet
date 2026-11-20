"""Register dependency-bound evaluations, never by wall-clock start date."""
from pathlib import Path
import os,json,hashlib
BASE=Path(__file__).resolve().parents[1];D=BASE/'control';plans=[]
def reg(name,entry,args=(),needs=()):
    plans.append({'id':name,'entry':entry,'args':list(args),'needs':list(needs)})
def dump(p,obj):
    b=(json.dumps(obj,indent=2)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=b:raise FileExistsError(p)
    else:
        temp=p.with_suffix('.new');temp.write_bytes(b);os.replace(temp,p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
train_order=json.loads((D/'training_plan.json').read_text())
core=[r for r in train_order if r.split('_')[0] in ('rgv','hbg','drr')]
assert len(core)==6
reg('table4_ready','assemble.py',['--core-only'],['test_'+r for r in core])
for run in train_order:
    reg('test_'+run,'eval_revision.py',['--run',run], [run])
for r in ['rgv_17','rgv_41']:reg('test_'+r,'eval_revision.py',['--run',r])
reg('test_rgv_73','eval_revision.py',['--run','rgv_73','--cache'])
reg('cache_hbg_val','eval_revision.py',['--run','hbg_73','--split','val','--cache','--cache-only'])
reg('segfix_73','segfix.py',['train'],['cache_hbg_val'])
reg('test_hbg_73','eval_revision.py',['--run','hbg_73','--cache'])
reg('segfix_test_73','segfix.py',['test'],['segfix_73','test_hbg_73','test_rgv_73'])
for r in ['hbg_73','rgv_73']:reg('seam_'+r,'eval_revision.py',['--run',r,'--seam'])
reg('seam_example','seam_example.py',[],['seam_hbg_73','seam_rgv_73'])
reg('latency_final','latency.py',[],[a['id'] for a in plans])
reg('paper_tables','assemble.py',[],[a['id'] for a in plans])
for a in plans:
    assert (BASE/'code'/a['entry']).is_file() and a['id'] not in a['needs']
assert len({a['id'] for a in plans})==len(plans)
dump(D/'analysis_sources.json',{n:digest(BASE/'code'/n) for n in sorted(set(a['entry'] for a in plans)|{'register_analysis.py'})})
dump(D/'analysis_plan.json',plans)
print('Registered',len(plans),'dependent tasks (including one SegFix training).')

