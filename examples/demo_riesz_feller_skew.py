"""Demonstrate recovery of the skew parameter in a Riesz--Feller operator."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from fractional_weak_form import (SeparableWeakLibrary2D, FractionalOperatorSpec,
    gaussian_test_matrix, periodic_riesz_feller_adjoint_on_tests, periodic_riesz_on_tests,
    fit_least_squares)
# Generate skewed space-fractional transport: u_t = D * RieszFeller^beta_theta u (periodic, exact spectral)
L=2*np.pi; nx=192; x=np.linspace(0,L,nx,endpoint=False); dx=x[1]-x[0]
nt=160; T=1.0; t=np.linspace(0,T,nt); dt=t[1]-t[0]
beta_true, theta_true, D_true = 1.6, 0.4, 0.20
k=2*np.pi*np.fft.fftfreq(nx,d=dx)
m=-(np.abs(k)**beta_true)*np.exp(1j*np.sign(k)*theta_true*np.pi/2); m[np.isclose(k,0)]=0
u0=np.cos(x)+0.4*np.cos(2*x)+0.3*np.sin(3*x)+0.2*np.sin(5*x)
U0h=np.fft.fft(u0); U=np.vstack([np.fft.ifft(U0h*np.exp(D_true*m*tn)).real for tn in t])  # (nt,nx)
print(f"skewed data: u_t = {D_true} * D^{beta_true}_(theta={theta_true}) u   shape={U.shape}")
Tt=gaussian_test_matrix(t,centers=16,width=0.06,periodic=False)
Xx=gaussian_test_matrix(x,centers=28,width=L/40,periodic=True)
weak=SeparableWeakLibrary2D(t,x,Tt,Xx)
b=weak.target(U,FractionalOperatorSpec(kind="integer",order=1,axis="t"))  # weak u_t
def fit_theta(beta,theta):
    col=weak.weak_inner(U,space_factor=periodic_riesz_feller_adjoint_on_tests(Xx,x,beta,theta=theta))
    coef,_=fit_least_squares(col[:,None],b,ridge=1e-12,normalize=True)
    return coef[0], np.linalg.norm(b-col*coef[0])/(np.linalg.norm(b)+1e-14)
print("\nSymmetric Riesz (theta=0, the only option in the original package):")
csym=weak.weak_inner(U,space_factor=periodic_riesz_on_tests(Xx,x,beta_true))
coef_s,_=fit_least_squares(csym[:,None],b,ridge=1e-12,normalize=True)
print(f"   beta=1.6,theta=0 : D_hat={coef_s[0]:+.4f}  residual={np.linalg.norm(b-csym*coef_s[0])/np.linalg.norm(b):.4f}  (truth D=0.20)")
print("\nRiesz-Feller scan over theta at true beta=1.6:")
for th in [0.0,0.2,0.3,0.4]:
    D,res=fit_theta(beta_true,th); print(f"   theta={th:>4}: D_hat={D:+.4f}  residual={res:.4f}")
# joint (beta,theta) min-residual
grid=[(b_,th_) for b_ in np.round(np.arange(1.3,1.91,0.05),2)
      for th_ in np.round(np.arange(0.0,0.61,0.05),2) if abs(th_)<=min(b_,2-b_)+1e-9]
best=min(grid,key=lambda bt:fit_theta(*bt)[1]); Dh,rr=fit_theta(*best)
print(f"\njoint argmin: beta*={best[0]:.2f} theta*={best[1]:.2f} D*={Dh:+.4f}  (truth beta=1.6,theta=0.4,D=0.20)")
