import numpy as np, sys
sys.path.insert(0,'.')
from lakaie.benchmarks.cec2022 import CEC2022Problem, DATA_DIR
from lakaie.components import estimate_interaction, interaction_strength
rng=np.random.default_rng(0); D=20; w=np.full(D,200.)
for fid in [1,2,4,5,10]:
    p=CEC2022Problem(fid,20); o=np.array((DATA_DIR/f"shift_data_{fid}.txt").read_text().split("\n")[0].split(),float)[:20]
    for scale in [0.1,1,10]:
        for M in [350,1000]:
            X=o+rng.normal(0,scale,(M,D)); F=p(X)
            res=[]
            for t in [3.0,0.0]:
                G=estimate_interaction(X,F,o,w,sample_factor=M/231,t_crit=t)
                res.append(f"t{t:.0f}:S8={interaction_strength(G):.2f},pairs>=.3={np.mean(G[np.triu_indices(D,1)]>=0.3):.2f}")
            print(f"F{fid} scale={scale:5} M={M}: "+"  ".join(res))
