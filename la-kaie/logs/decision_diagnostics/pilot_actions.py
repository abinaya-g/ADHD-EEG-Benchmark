import numpy as np, json, glob
from collections import defaultdict
A=["A1","A2","A3","A4","A5","A6","A7","A8"]
for bench in ["CEC2022","FIR"]:
    for meth in ["LA-KAIE-Full","LA-KAIE-FixedPolicy"]:
        succ=defaultdict(list); rew=defaultdict(list); s8=[]; ex=[]
        for j in glob.glob(f"results/raw/pilot/{bench}/{meth}/inst*/run*.json"):
            z=np.load(j.replace(".json",".npz"))
            a=z["lk_ep_action"]; s=z["lk_ep_success"]; r=z["lk_ep_reward"]
            for ai,si,ri in zip(a,s,r):
                if np.isfinite(si): succ[A[int(ai)]].append(si)
                if np.isfinite(ri): rew[A[int(ai)]].append(ri)
            S=z["lk_it_S8"]; n=len(S); s8.append([S[:n//10].max(), S.max(), S[-1]]); ex.append(z["lk_it_exchanges"].mean())
        s8=np.array(s8)
        print(f"== {bench} {meth}: exchanges/iter {np.mean(ex):.1f}; S8 max(first10%) {s8[:,0].mean():.3f}, max(run) {s8[:,1].mean():.3f}, final {s8[:,2].mean():.3f}")
        print("   "+"  ".join(f"{k}:succ={np.mean(v):.2f},R={np.mean(rew[k]):+.2f},n={len(v)}" for k,v in sorted(succ.items())))
