"""Compare alternative weak temporal-target constructions on one benchmark."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from fpde_datasets import load_paper_fade, add_multiplicative_uniform_noise
from fpde_derivatives import caputo_l1_time, spectral_space_derivative
from fractional_weak_form import (SeparableWeakLibrary2D, FractionalOperatorSpec,
    gaussian_test_matrix, periodic_spectral_directional_adjoint_on_tests,
    caputo_l1_adjoint_tests, fractional_integral_adjoint_tests, caputo_corrected_field,
    adjoint_tests_1d, fit_least_squares)

data = load_paper_fade("data/tsfade_fft.dat")
t, x, U0 = data.t, data.x, data.U
dt, dx, Lx = data.dt, data.dx, data.Lx
ALPHA, BETAS, TRUTH = 0.8, (1.0, 1.7), np.array([-1.0, 0.5])
print(f"FADE  shape={U0.shape}  truth: D_t^{ALPHA} u = {TRUTH[0]}*D_x^{BETAS[0]} u + {TRUTH[1]}*D_x^{BETAS[1]} u")

# identical smooth test functions for all weak variants
T = gaussian_test_matrix(t, centers=18, width=0.5, periodic=False)
X = gaussian_test_matrix(x, centers=24, width=Lx/36.0, periodic=True)
weak = SeparableWeakLibrary2D(t, x, T, X)

def space_cols(U, betas):
    cols = []
    for b in betas:
        sf = periodic_spectral_directional_adjoint_on_tests(X, x, b)
        cols.append(weak.weak_inner(U, space_factor=sf))
    return np.column_stack(cols)

def space_cols_volterra(U, betas, alpha):
    Tint = fractional_integral_adjoint_tests(T, t, alpha, side="left")
    cols = []
    for b in betas:
        sf = periodic_spectral_directional_adjoint_on_tests(X, x, b)
        cols.append(weak.weak_inner(U, time_factor=Tint, space_factor=sf))
    return np.column_stack(cols)

def fit_err(Theta, b):
    coef, _ = fit_least_squares(Theta, b, ridge=1e-10, normalize=True)
    return coef, float(np.linalg.norm(coef - TRUTH) / np.linalg.norm(TRUTH))

def target_GL(U, alpha):   # my approach == their default Caputo mode (GL-transpose + IC subtract)
    return weak.target(U, FractionalOperatorSpec(kind="caputo", order=alpha, axis="t", side="left"))
def target_L1(U, alpha):   # exact L1-transpose (matches generator discretization)
    return weak.weak_inner(U, time_factor=caputo_l1_adjoint_tests(T, t, alpha))
def target_VOLT(U, alpha): # Volterra: LHS has NO time derivative
    return weak.weak_inner(caputo_corrected_field(U, axis=0, order=alpha, h=dt, initial="first"))

def strong_err(U, alpha, betas):  # strong-form baseline (differentiate noisy data)
    ut = caputo_l1_time(U, alpha, dt)
    valid = np.isfinite(ut).all(axis=1)
    cols = [spectral_space_derivative(U, b, Lx, riesz=False) for b in betas]
    Xs = np.column_stack([c[valid].reshape(-1) for c in cols]); ys = ut[valid].reshape(-1)
    coef, _ = fit_least_squares(Xs, ys, ridge=1e-10, normalize=True)
    return coef, float(np.linalg.norm(coef - TRUTH) / np.linalg.norm(TRUTH))

noises = [0.0, 1.0, 5.0, 10.0, 20.0]; seeds = [0,1,2,3]
print("\n=== coefficient relative error at TRUE support & orders (mean over seeds) ===")
print(f"{'noise%':>7} | {'GL-transp(MINE)':>16} | {'L1-transp(CGPT)':>16} | {'Volterra(CGPT)':>15} | {'strong':>10}")
for ns in noises:
    eg=[]; el=[]; ev=[]; es=[]
    for sd in seeds:
        Un = add_multiplicative_uniform_noise(U0, ns, seed=sd) if ns>0 else U0
        Sc = space_cols(Un, BETAS)
        eg.append(fit_err(Sc, target_GL(Un, ALPHA))[1])
        el.append(fit_err(Sc, target_L1(Un, ALPHA))[1])
        ev.append(fit_err(space_cols_volterra(Un, BETAS, ALPHA), target_VOLT(Un, ALPHA))[1])
        es.append(strong_err(Un, ALPHA, BETAS)[1])
        if ns==0: break
    print(f"{ns:>7.0f} | {np.mean(eg):>16.4f} | {np.mean(el):>16.4f} | {np.mean(ev):>15.4f} | {np.mean(es):>10.4f}")

# ---- alpha recovery at true beta-support (scan alpha), per temporal target ----
print("\n=== recovered alpha (truth 0.8) by min-residual scan at true beta-support ===")
agrid = np.round(np.arange(0.60, 1.001, 0.02), 2)
def scan_alpha(Un, target_fn, volterra=False):
    best=None; bestr=np.inf
    for a in agrid:
        if volterra:
            Sc=space_cols_volterra(Un,BETAS,a); b=target_VOLT(Un,a)
        else:
            Sc=space_cols(Un,BETAS); b=target_fn(Un,a)
        coef,_=fit_least_squares(Sc,b,ridge=1e-10,normalize=True)
        r=np.linalg.norm(b-Sc@coef)/(np.linalg.norm(b)+1e-14)
        if r<bestr: bestr=r; best=a
    return best
print(f"{'noise%':>7} | {'GL(MINE)':>9} | {'L1(CGPT)':>9} | {'Volt(CGPT)':>11}")
for ns in [0.0,5.0,10.0,20.0]:
    ag=[];al=[];av=[]
    for sd in seeds:
        Un = add_multiplicative_uniform_noise(U0, ns, seed=sd) if ns>0 else U0
        ag.append(scan_alpha(Un,target_GL)); al.append(scan_alpha(Un,target_L1)); av.append(scan_alpha(Un,None,volterra=True))
        if ns==0: break
    print(f"{ns:>7.0f} | {np.mean(ag):>9.3f} | {np.mean(al):>9.3f} | {np.mean(av):>11.3f}")
