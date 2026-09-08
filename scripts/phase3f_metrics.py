"""Balanced five-basis / three-initialization reference measurements."""
import numpy as np
from phase3d_metrics import single

NAMES=[f'R{r}-I{i}' for r in range(5) for i in (1,2,3)]
CATEGORIES=['rotation-robust','occasionally vulnerable','usually vulnerable','rotation-persistent']


def factorial(loss):
    x=np.asarray(loss,dtype=float)
    if x.shape!=(5,3) or not np.isfinite(x).all():raise ValueError('finite balanced5x3 required')
    centered=x-x[0,0];residual=x-x[:,:1]
    means=centered[:,0]+residual.mean(axis=1);initmeans=centered.mean(axis=0);mu=float(centered.mean())
    w=float(np.var(residual,axis=1).mean());b=float(np.var(means-means[0]));t=float(np.var(centered))
    init=float(np.var(initmeans));interaction=float(np.mean((centered-means[:,None]-initmeans[None,:]+mu)**2))
    assert abs(w+b-t)<1e-12 and abs(b+init+interaction-t)<1e-12
    h=(x>0).sum(axis=1);count=int((h>=2).sum())
    return {'overall_mean_loss':float(x.mean()),'rotation_mean_loss':(means+x[0,0]).tolist(),
            'rotation_harmful_frequency':(h/3).tolist(),'harmful_rotation_count':count,'harmful_rotation_frequency':count/5,
            'harmful_model_count':int(h.sum()),'category':CATEGORIES[0 if count==0 else 1 if count<=2 else 2 if count<=4 else 3],
            'within_rotation_initialization_variance':w,'between_rotation_variance':b,'total_model_variance':t,
            'within_init_sample_variance':float(residual.var(axis=1,ddof=1).mean()),
            'rotation_means_sample_variance':float(means.var(ddof=1)),
            'signed_rotation_variance_contrast':float(means.var(ddof=1)-residual.var(axis=1,ddof=1).mean()/3),
            'two_way_init_main_variance':init,'two_way_rotation_main_variance':b,'two_way_interaction_variance':interaction}


def measure_rotations(ids,exact,scores,truth,k=10,seed=1,samples=512,epsilon=1e-12):
    ids=np.asarray(ids,dtype=np.int64);d=np.asarray(exact,dtype=float);p=np.asarray(scores,dtype=float)
    if p.shape!=(15,len(ids)) or len(np.unique(ids))!=len(ids):raise ValueError('15 aligned scorers and unique IDs required')
    models=[];replacements=[];geometry=None
    for m in range(15):
        metric,g,repl,random=single(ids,d,p[m],truth,k,seed,samples,epsilon)
        if geometry is not None:assert geometry==g
        geometry=g;metric.update(model=NAMES[m],rotation=m//3,initialization=m%3+1,harmful=metric['ranking_recall_loss']>0)
        models.append(metric)
        for pair in repl:
            pair.pop('quantizer');pair['model']=NAMES[m];replacements.append(pair)
    order=np.argsort(d,kind='stable');inside,outside=order[:k],order[k:]
    margins=d[outside][None,:]-d[inside][:,None]
    inv=np.array([(margins>0)&(p[m,inside,None]>p[m,outside][None,:]) for m in range(15)])
    assert inv.sum(axis=(1,2)).tolist()==[m['cross_boundary_inversion_count'] for m in models]
    pairs=[]
    for a,b in zip(*np.nonzero(inv.sum(axis=0))):
        i,j=inside[a],outside[b];mask=inv[:,a,b];rc=mask.reshape(5,3).sum(axis=1);count=int(mask.sum());rotations=int((rc>0).sum())
        pairs.append({'inside_id':int(ids[i]),'outside_id':int(ids[j]),'inside_exact_rank':int(a+1),'outside_exact_rank':int(b+k+1),
            'exact_margin':float(margins[a,b]),'violation_by_model':(p[:,i]-p[:,j]).tolist(),'model_mask':mask.tolist(),
            'model_count':count,'model_frequency':count/15,'rotation_init_counts':rc.tolist(),
            'rotation_frequency':rotations,'rotation_frequency_fraction':rotations/5})
    loss=np.array([m['ranking_recall_loss'] for m in models]).reshape(5,3)
    return {**geometry,**factorial(loss),'models':models},replacements,pairs,random
