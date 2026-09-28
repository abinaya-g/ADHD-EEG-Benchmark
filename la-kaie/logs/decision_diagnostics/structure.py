import numpy as np, sys, json, glob
sys.path.insert(0,'.')
from lakaie.benchmarks.cec2022 import DATA_DIR
from lakaie.benchmarks.fir import FIRProblem
from lakaie.components import fir_prior, group_variables
print("== CEC2022 D=20 rotation matrices: density of M (|M_ij|>0.05 off-diagonal)")
for f in range(1,13):
    M=np.loadtxt(DATA_DIR/f"M_{f}_D20.txt")[:20,:20]
    off=~np.eye(20,dtype=bool)
    print(f"F{f}: dense frac={np.mean(np.abs(M[off])>0.05):.2f}  orthogonal_err={np.abs(M@M.T-np.eye(20)).max():.1e}")
print("== FIR stopband quadratic-form coupling (exact, gray-box)")
for c in range(1,9):
    G=fir_prior(FIRProblem(c).stopband_quadratic_form()); iu=np.triu_indices(31,1)
    g=group_variables(G,0.3,5)
    print(f"FIR{c}: pairs>=0.3 {np.mean(G[iu]>=0.3):.2f}, mean|G| {G[iu].mean():.2f}, groups(tau=.3,gmax=5) {len(g)}, lag1 {G[0,1]:.2f} lag5 {G[0,5]:.2f}")
