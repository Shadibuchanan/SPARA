import collections, numpy as np, sys
sys.argv=['x']; exec(open('eps_signal.py').read().split('rng = np.random')[0])
rng=np.random.default_rng(1)
data={42:load(42),43:load(43)}
by={s:collections.defaultdict(list) for s in data}
for s,gs in data.items():
    for g in gs: by[s][g['row']].append(g)
for tr,te in ((42,43),(43,42)):
    common=[m for m in by[tr] if m in by[te]]
    a=[auc(by[te][m],direction(by[tr][m])) for m in common]
    nulls=[]
    for _ in range(200):
        nulls.append(np.mean([auc(by[te][m],direction([{**g,'r':rng.permutation(g['r'])} for g in by[tr][m]])) for m in common]))
    nulls=np.array(nulls)
    print(f"per-map train {tr} -> test {te}: maps {len(common)}, test groups {sum(len(by[te][m]) for m in common)}, AUC {np.mean(a):.3f}, null 97.5% {np.quantile(nulls,.975):.3f}, p~{(nulls>=np.mean(a)).mean():.3f}")
