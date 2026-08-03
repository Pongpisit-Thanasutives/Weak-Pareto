"""Compare Gaussian, Fourier, and compact-bump weak test families."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from fpde_datasets import load_paper_fade, add_multiplicative_uniform_noise
from fractional_weak_form import (SeparableWeakLibrary2D, FractionalOperatorSpec,
    gaussian_test_matrix, compact_bump_test_matrix,
    periodic_spectral_directional_adjoint_on_tests, caputo_l1_adjoint_tests, fit_least_squares)
data=load_paper_fade("data/tsfade_fft.dat"); t,x,U0=data.t,data.x,data.U
dt,dx,Lx=data.dt,data.dx,data.Lx; ALPHA,BETAS,TRUTH=0.8,(1.0,1.7),np.array([-1.0,0.5])
Xg=gaussian_test_matrix(x,centers=24,width=Lx/36.0,periodic=True)
fam={"gaussian":gaussian_test_matrix(t,centers=18,width=0.5,periodic=False),
     "compact ":compact_bump_test_matrix(t,centers=18,width=0.5,periodic=False)}
def run(Tt,target):
    weak=SeparableWeakLibrary2D(t,x,Tt,Xg)
    def cols(U): return np.column_stack([weak.weak_inner(U,space_factor=periodic_spectral_directional_adjoint_on_tests(Xg,x,b)) for b in BETAS])
    def tgt(U,a):
        if target=="GL": return weak.target(U,FractionalOperatorSpec(kind="caputo",order=a,axis="t",side="left"))
        return weak.weak_inner(U,time_factor=caputo_l1_adjoint_tests(Tt,t,a))
    return cols,tgt
agrid=np.round(np.arange(0.60,1.001,0.02),2); seeds=[0,1,2,3]
for target in ["GL","L1"]:
    print(f"\n=== temporal target = {target} : coeff-error / alpha-recovery, gaussian vs compact time-tests ===")
    print(f"{'noise%':>6} | {'coef_gauss':>10} {'coef_cmpct':>10} | {'a_gauss':>8} {'a_cmpct':>8}")
    for ns in [0.0,5.0,10.0,20.0]:
        row={}
        for fname,Tt in fam.items():
            cols,tgt=run(Tt,target); ce=[]; ae=[]
            for sd in seeds:
                Un=add_multiplicative_uniform_noise(U0,ns,seed=sd) if ns>0 else U0
                Sc=cols(Un); b=tgt(Un,ALPHA); coef,_=fit_least_squares(Sc,b,ridge=1e-10,normalize=True)
                ce.append(np.linalg.norm(coef-TRUTH)/np.linalg.norm(TRUTH))
                best=min(agrid,key=lambda a:(lambda bb:np.linalg.norm(bb-cols(Un)@fit_least_squares(cols(Un),bb,ridge=1e-10,normalize=True)[0])/ (np.linalg.norm(bb)+1e-14))(tgt(Un,a)))
                ae.append(best)
                if ns==0: break
            row[fname]=(np.mean(ce),np.mean(ae))
        print(f"{ns:>6.0f} | {row['gaussian'][0]:>10.4f} {row['compact '][0]:>10.4f} | {row['gaussian'][1]:>8.3f} {row['compact '][1]:>8.3f}")
