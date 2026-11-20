import numpy as np, sys
exec(open('eps_signal.py').read().split('rng = np.random')[0])
rng=np.random.default_rng(2)
gs=load(42)+load(43)
def auc_norm(groups):
    v=[]
    for g in groups:
        s=(g['eps']**2).sum(1); win,lose=s[g['r']>0],s[g['r']==0]
        v.append((win[:,None]>lose[None,:]).mean())
    return np.mean(v)
for name,sel in (("group success <= 2/8",lambda g:g['r'].mean()<=0.25),("3-5/8",lambda g:0.25<g['r'].mean()<0.75),(">= 6/8",lambda g:g['r'].mean()>=0.75)):
    sub=[g for g in gs if sel(g)]
    a=auc_norm(sub)
    nl=np.array([auc_norm([{**g,'r':rng.permutation(g['r'])} for g in sub]) for _ in range(300)])
    print(f"{name:20s} groups {len(sub):5d}  AUC(bigger noise -> success) {a:.3f}  null 2.5-97.5% {np.quantile(nl,.025):.3f}-{np.quantile(nl,.975):.3f}")
