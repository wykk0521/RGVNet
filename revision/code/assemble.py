"""Produce revision numerical tables. Never edits the manuscript."""
from pathlib import Path
import json,csv,statistics,math,sys
from eval_revision import BASE,SRC,atomic,sha
SEEDS=[17,41,73,101,137];SHORT=[17,41,73];METRICS=['iou','bf1','f1','goc','guc','gtc']
ARCH=SRC/'z_2_test/AI4Boundaries_1m';J=BASE/'jobs'
def read(p):return json.loads(Path(p).read_text())
def csvout(filename,rows):
    with Path(filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(kind,seed):
    if seed in SHORT and kind in ('hbg','drr','rgv'):
        suffix={'hbg':'baseline','drr':'sr_v2','rgv':'rd_v9_context'}[kind]
        dossier={('hbg',41):'hbgnet_protocol_b_v1_final_best',('drr',41):'hbgnet_protocol_b_sr_v2_final_best'}.get((kind,seed),f'hbgnet_protocol_b_seed{seed}_{suffix}_best')
        j=read(ARCH/dossier/'metrics.json');assert j['complete_split'] and j['samples']==3014 and j['config']['training']['seed']==seed
        d=dict(j['primary_metrics']);d['bf1']=d['f_boundary']
    else:
        j=read(J/f'test_{kind}_{seed}'/'complete.json');assert j['samples']==3014 and j['seed']==seed;d=j['routes']['final']
    return {k:100*d[k] for k in METRICS}
def sign(pairs):
    nz=[v for v in pairs if v!=0];n=len(nz);k=sum(v>0 for v in nz)
    return 1. if n==0 else min(1.,2*sum(math.comb(n,j) for j in range(min(k,n-k)+1))/2**n)
def core_table():
    methods=['hbg','drr','rgv'];vals={(m,s):metric(m,s) for m in methods for s in SEEDS}
    csvout('table4_seed_values_pct.csv',[dict(config=m,seed=s,**vals[m,s]) for m in methods for s in SEEDS])
    csvout('table4_mean_variance.csv',[dict(config=m,metric=k,mean_pct=statistics.mean(vv),sample_variance_pp2=statistics.variance(vv)) for m in methods for k in METRICS for vv in [[vals[m,s][k] for s in SEEDS]]])
    paired=[]
    for m,n in [('rgv','hbg'),('drr','hbg'),('rgv','drr')]:
        for k in ['bf1','iou']:
            d=[vals[m,s][k]-vals[n,s][k] for s in SEEDS];paired.append(dict(comparison=m+' minus '+n,metric=k,**{str(s):x for s,x in zip(SEEDS,d)},mean_pp=statistics.mean(d),sd_pp=statistics.stdev(d),primary_exact_p=sign(d) if (m,n,k)==('rgv','hbg','bf1') else ''))
    csvout('table4_paired_pp.csv',paired)
    return vals,paired
def main():
    vals,paired=core_table()
    if '--core-only' in sys.argv:
        atomic(Path('complete.json'),{'seeds':SEEDS,'datasets':['AI4Boundaries'],'n_test_tiles':3014,'primary_metric':'BF1','primary_comparison':'RGVNet minus HBGNet','exact_two_sided_sign_p':paired[0]['primary_exact_p'],'files':['table4_seed_values_pct.csv','table4_mean_variance.csv','table4_paired_pp.csv']})
        return
    assert sha(SRC/'tools/local_separation_refinement.py')==sha(SRC/'tools/local_separation_refinement_v2.py')
    old=read(ARCH/'hbgnet_protocol_b_sr_v1_retest_best_01/metrics.json');assert old['complete_split'] and old['config']['training']['seed']==41
    rows=[]
    for s in SHORT:
        for arch,sup,z in [('HBGNet','original',vals['hbg',s]),('HBGNet','matched',metric('matched',s)),('HBGNet+DRR','revised',vals['drr',s])]:
            rows.append(dict(seed=s,architecture=arch,supervision=sup,iou_pct=z['iou'],bf1_pct=z['bf1']))
    rows.append(dict(seed=41,architecture='HBGNet+DRR',supervision='original',iou_pct=100*old['primary_metrics']['iou'],bf1_pct=100*old['primary_metrics']['f_boundary']))
    csvout('S2_training_supervision.csv',rows)
    wins=[];delta=[];candidates=[];oldchecks=[]
    for s in SEEDS:
        dw={}
        for fam in ['rgv','win7']:
            d=read(J/f'test_{fam}_{s}'/'complete.json');assert d['samples']==3014
            dw[fam]=d['routes']['final']
            wins.append(dict(seed=s,window='3,7,15' if fam=='rgv' else '7',**{k:100*d['routes']['final'][k] for k in METRICS}))
            for direction,x in d['candidates'].items():
                assert x['accepted']<=x['proposed']
                frac=lambda a,b:100*a/b if b else None
                candidates.append(dict(seed=s,architecture=fam,branch=direction,**x,proposed_beneficial_pct=frac(x['beneficial_proposed'],x['proposed']),cev_beneficial_pct=frac(x['beneficial_cev'],x['accepted']),margin_beneficial_pct=frac(x['beneficial_margin'],x['accepted'])))
            if fam=='rgv' and s in SHORT:oldchecks.append({'seed':s,**{k:100*d['routes']['final'][k]-vals['rgv',s][k] for k in METRICS}})
        delta.append({'seed':s,**{k:100*(dw['rgv'][k]-dw['win7'][k]) for k in METRICS}})
    csvout('window_seed_values_pct.csv',wins);csvout('window_multiscale_minus_single_pp.csv',delta)
    csvout('candidate_equal_retention.csv',candidates);csvout('original_checkpoint_replay_minus_archived_pp.csv',oldchecks)
    routes=[]
    for s in SEEDS:
        r=read(J/f'test_rgv_{s}'/'complete.json')
        for path,d in r['routes'].items():routes.append(dict(seed=s,route=path,**{k:100*d[k] for k in METRICS}))
    csvout('fixed_checkpoint_routes_pct.csv',routes)
    results={j:read(J/j/'complete.json') for j in ['segfix_test_73','seam_hbg_73','seam_rgv_73','seam_example','latency_final']}
    atomic(Path('complete.json'),{'seeds':SEEDS,'primary_sign_test':paired[0]['primary_exact_p'],'S2_original_DRR_seeds':[41],'supporting_results':results})
if __name__=='__main__':main()

