"""How big is the latent gap? LIVR's greedy failures on the 200 pilot maps, against 64 text
samples (job 7148708) and the diagnostics' latent random search at layer 24 (job 7148362's Q6,
run in 7148374). CPU only."""
import collections
import json

R = "/storage/ice1/5/6/psomu3/sft-run/results"
lat = [json.loads(l) for l in open(f"{R}/latent-diag-gradients-7148374/latent-search.jsonl")]
ver = {r["row_index"]: r["solved"] for r in map(json.loads, open(f"{R}/latent-diag-gradients-7148374/verdict-search.jsonl"))}
txt = {}
for l in open(f"{R}/latent-diag-sweep-Lnone-7148708/maps.jsonl"):
    r = json.loads(l)
    s = r["text"]["successes"]
    txt[r["row_index"]] = sum(s) if isinstance(s, list) else s
L24 = {r["row_index"]: r for r in lat if str(r["layer"]) == "24"}
fail = sorted(L24)
solved = {m for m, r in L24.items() if any(r["successes"])}
print("greedy failures", len(fail), "; solved by latent search", len(solved))
t0 = [m for m in fail if txt.get(m, 0) == 0]
print("never solved by 64 text samples:", len(t0), "; latent solves", len(set(t0) & solved),
      "; final-verdict search solves", sum(ver.get(m, False) for m in t0))
print("latent-only (not final-verdict):", len([m for m in solved if not ver.get(m)]),
      "; final-verdict-only:", len([m for m in fail if ver.get(m) and m not in solved]))
rate = collections.defaultdict(lambda: [0, 0])
for m in t0:
    r = L24[m]
    k = r["draws_per_scale"]
    for i, sc in enumerate(r["scales"]):
        seg = r["successes"][i * k:(i + 1) * k]
        rate[sc][0] += sum(seg)
        rate[sc][1] += len(seg)
print("success per latent try on those maps, by scale:", {s: f"{a}/{b}" for s, (a, b) in rate.items()})
