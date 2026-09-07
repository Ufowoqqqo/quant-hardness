"""Exhaustive reference score comparisons on one shared candidate-ID array."""
import numpy as np


def transition_flags(loss32_hits, loss64_hits):
    a,b=loss32_hits,loss64_hits
    return {"stable_both":a==0 and b==0, "rescued_improved":a>b,
            "persistent_harmful":a>0 and b>0, "regressed":b>a,
            "fully_rescued":a>0 and b==0,
            "partially_improved_persistent":a>b and b>0,
            "unchanged_harmful":a==b and a>0}


def measure(ids, exact, scores32, scores64, truth, k, seed, samples, epsilon,
            small_margin=.05, large_margin=.10, near_inside=5, near_outside=20):
    ids=np.asarray(ids,dtype=np.int64)
    d=np.asarray(exact,dtype=np.float64)
    scores={32:np.asarray(scores32,dtype=np.float64),64:np.asarray(scores64,dtype=np.float64)}
    if (len(ids)<=k or len(set(ids))!=len(ids) or len(d)!=len(ids)
            or any(len(v)!=len(ids) for v in scores.values())
            or not all(np.isfinite(v).all() for v in (d,*scores.values()))):
        raise ValueError("invalid common candidate pool")
    order=np.argsort(d,kind="stable")
    top,outside=order[:k],order[k:]
    exact_ids=ids[top].tolist(); A=set(exact_ids); gt=set(truth)
    hits=len(A & gt)
    sd=d[order]; dk=float(sd[k-1]); denom=max(dk,epsilon)
    rank=np.empty(len(ids),dtype=np.int32);rank[order]=np.arange(1,len(ids)+1)
    gaps={f"boundary_gap_r{r}":float(sd[k+r-1]-sd[k-r]) if len(ids)>=k+r and k>=r else None for r in (1,2,5)}
    margin=float(sd[k]-dk)
    row={"candidate_count":len(ids),"oracle_recall":hits/k,
         "exact_topk_ids":exact_ids,"exact_topk_set":sorted(A),
         "exact_boundary_tie":margin==0,"d1":float(sd[0]),"dk":dk,
         "dk_plus_1":float(sd[k]),"d20":float(sd[19]) if len(ids)>=20 else None,
         "boundary_margin":margin,"relative_boundary_margin":margin/denom,
         "local_distance_concentration":float((sd[19]-sd[0])/max(sd[0],epsilon)) if len(ids)>=20 else None,
         **gaps}
    rng=np.random.Generator(np.random.PCG64(seed))
    left=rng.integers(0,len(ids),samples);right=rng.integers(0,len(ids)-1,samples)
    right+=right>=left
    random_pairs=np.column_stack([left,right]).astype("<i4")
    pair_margin=d[outside][None,:]-d[top][:,None]
    relative=pair_margin/denom
    row["cross_boundary_pair_count"]=int(pair_margin.size)
    row["cross_boundary_exact_tie_count"]=int((pair_margin==0).sum())
    for limit,name in ((.01,"le_001"),(small_margin,"le_small"),(large_margin,"le_large")):
        row["opportunity_margin_"+name+"_count"]=int((relative<=limit).sum())
    window=order[max(0,k-6):min(len(ids),k+5)]
    inv,errors,violations,selected_out={}, {}, {}, {}
    replacements=[]
    for Q,p in scores.items():
        po=np.argsort(p,kind="stable")[:k]
        chosen=ids[po].tolist(); S=set(chosen)
        selected_out[Q]=np.array([j for j in po if ids[j] not in A],dtype=np.int64)
        error=p-d;errors[Q]=error
        absolute=np.abs(error); valid=d>epsilon
        inversions=(pair_margin>0) & (p[top][:,None]>p[outside][None,:])
        inv[Q]=inversions
        violations[Q]=error[top][:,None]-error[outside][None,:]-pair_margin
        loss_hits=hits-len(S & gt)
        chosen_inversions=(d[selected_out[Q]][None,:]>d[top][:,None]) & (
                           p[top][:,None]>p[selected_out[Q]][None,:])
        random_inv=((d[left]<d[right]) & (p[left]>p[right])) | ((d[left]>d[right]) & (p[left]<p[right]))
        displaced=sorted(A-S);intruders=sorted(S-A)
        metrics={"topk_agreement":len(A & S)/k,"fixed_candidate_recall":len(S & gt)/k,
                 "ranking_recall_loss":loss_hits/k,"loss_hits":loss_hits,
                 "displaced_count":len(displaced),"intruder_count":len(intruders),
                 "cross_boundary_inversion_count":int(inversions.sum()),
                 "cross_boundary_inversion_fraction":float(inversions.mean()),
                 "topk_items_overtaken_by_any_outside":int(np.any(inversions,axis=1).sum()),
                 "max_outside_overtaking_one_topk":int(inversions.sum(axis=1).max()),
                 "max_selected_intruders_overtaking_one_topk":int(chosen_inversions.sum(axis=1).max()),
                 "maximum_violation":float(violations[Q].max()),
                 "global_mae":float(absolute.mean()),
                 "relative_mae":float(np.mean(absolute[valid]/d[valid])) if valid.any() else None,
                 "relative_mae_omitted_count":int((~valid).sum()),
                 "random_pair_inversion_rate":float(random_inv.mean()),
                 "random_pair_inversion_count":int(random_inv.sum()),
                 "random_pq_tie_count":int((p[left]==p[right]).sum()),
                 "boundary_window_mae":float(absolute[window].mean()),
                 "topk_ids":chosen,"topk_set":sorted(S),
                 "displaced_ids":displaced,"intruder_ids":intruders,
                 "pq_boundary_tie":bool(np.sort(p)[k-1]==np.sort(p)[k])}
        row.update({f"{name}_{Q}":value for name,value in metrics.items()})
        lookup={int(identifier):i for i,identifier in enumerate(ids)}
        for i_id in displaced:
            for j_id in intruders:
                i,j=lookup[i_id],lookup[j_id]
                m=float(d[j]-d[i]);a=float(error[i]-error[j])
                replacements.append({"quantizer":Q,"displaced_id":i_id,"intruder_id":j_id,
                    "inside_exact_rank":int(rank[i]),"outside_exact_rank":int(rank[j]),
                    "displaced_is_gt":i_id in gt,"intruder_is_gt":j_id in gt,
                    "exact_margin":m,"error_difference":a,"violation":a-m,
                    "strict_inversion":bool(m>0 and p[i]>p[j])})
    row.update(transition_flags(row["loss_hits_32"],row["loss_hits_64"]))
    row["delta_loss_precision"]=(row["loss_hits_32"]-row["loss_hits_64"])/k
    for metric in ("cross_boundary_inversion_count","cross_boundary_inversion_fraction",
                   "global_mae","relative_mae","random_pair_inversion_rate",
                   "boundary_window_mae","maximum_violation","displaced_count"):
        a,b=row[metric+"_32"],row[metric+"_64"]
        row["delta_"+metric]=a-b if a is not None and b is not None else None
    row["delta_critical_inversions"]=row["delta_cross_boundary_inversion_count"]
    union=inv[32]|inv[64]
    paired=[]
    for ii,jj in zip(*np.nonzero(union)):
        i,j=top[ii],outside[jj]
        state="persistent" if inv[32][ii,jj] and inv[64][ii,jj] else "corrected" if inv[32][ii,jj] else "new"
        paired.append({"state":state,"inside_id":int(ids[i]),"outside_id":int(ids[j]),
            "inside_exact_rank":int(rank[i]),"outside_exact_rank":int(rank[j]),
            "exact_margin":float(pair_margin[ii,jj]),"relative_exact_margin":float(relative[ii,jj]),
            "error_difference_32":float(errors[32][i]-errors[32][j]),
            "error_difference_64":float(errors[64][i]-errors[64][j]),
            "violation_32":float(violations[32][ii,jj]),"violation_64":float(violations[64][ii,jj]),
            "outside_selected_32":bool(j in selected_out[32]),
            "outside_selected_64":bool(j in selected_out[64])})
    for state in ("corrected","persistent","new"):
        members=[p for p in paired if p["state"]==state]
        row[state+"_inversion_count"]=len(members)
        row[state+"_small_margin_count"]=sum(p["relative_exact_margin"]<=small_margin for p in members)
        row[state+"_large_margin_count"]=sum(p["relative_exact_margin"]>large_margin for p in members)
        row[state+"_near_rank_boundary_count"]=sum(p["inside_exact_rank"]>=near_inside and p["outside_exact_rank"]<=near_outside for p in members)
    assert row["corrected_inversion_count"]+row["persistent_inversion_count"]==row["cross_boundary_inversion_count_32"]
    assert row["new_inversion_count"]+row["persistent_inversion_count"]==row["cross_boundary_inversion_count_64"]
    return row,replacements,paired,random_pairs
