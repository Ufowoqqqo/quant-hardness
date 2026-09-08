"""Descriptive paired systems metrics; no fitted policies."""
import numpy as np

def latency_summary(samples):
    a=np.asarray(samples,dtype=float)
    if not len(a) or np.any(~np.isfinite(a)) or np.any(a<=0):raise ValueError('invalid latency')
    return {'mean_us':float(a.mean()),'median_us':float(np.median(a)),
            **{f'p{p}_us':float(np.quantile(a,p/100)) for p in [90,95,99]},'service_qps':float(1e6/a.mean())}

def gate(native,full,bounded,full_all,bounded_all,*,overhead_target=.50,saving_target=.10):
    full_overhead=full-native;bounded_overhead=bounded-native
    reduction=1-bounded_overhead/full_overhead if full_overhead>1e-9 else None
    saving=1-bounded_all/full_all
    return {'native_mean_us':native,'full_mean_us':full,'bounded_mean_us':bounded,
        'full_all16_mean_us':full_all,'bounded_all16_mean_us':bounded_all,
        'full_overhead_us':full_overhead,'bounded_overhead_us':bounded_overhead,
        'overhead_reduction':reduction,'all16_latency_saving':saving,'all16_speedup':full_all/bounded_all,
        'overhead_gate_pass':bool(full_overhead>1e-9 and bounded_overhead<=overhead_target*full_overhead),
        'all16_gate_pass':bool(saving>=saving_target),
        'meaningful_optimization':bool((full_overhead>1e-9 and bounded_overhead<=overhead_target*full_overhead) or saving>=saving_target)}

def selective_gate(gap_time,all_time,native_recall,gap_recall,all_recall,*,saving_target=.10,gain_target=.60):
    gain=all_recall-native_recall
    retained=(gap_recall-native_recall)/gain if gain>1e-12 else None
    saving=1-gap_time/all_time
    return {'gap_latency_saving':saving,'all16_recall_gain_retained':retained,
        'selectivity_gate_pass':bool(saving>=saving_target and retained is not None and retained>=gain_target)}
