"""Local order polish on the smooth weak residual (optional post-DE step)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from fpde_datasets import load_paper_fade
from fractional_weak_form import (SeparableWeakLibrary2D, gaussian_test_matrix,
    periodic_spectral_directional_adjoint_on_tests, caputo_l1_adjoint_tests,
    fit_least_squares, refine_orders_local)

d = load_paper_fade("data/tsfade_fft.dat"); t, x, U = d.t, d.x, d.U; Lx = d.Lx
Tt = gaussian_test_matrix(t, 18, 0.5, periodic=False)
Xx = gaussian_test_matrix(x, 24, Lx / 36, periodic=True)
weak = SeparableWeakLibrary2D(t, x, Tt, Xx)

def residual(z):
    a, b2 = float(z[0]), float(z[1])
    cols = np.column_stack([
        weak.weak_inner(U, space_factor=periodic_spectral_directional_adjoint_on_tests(Xx, x, b))
        for b in (1.0, b2)])
    b = weak.weak_inner(U, time_factor=caputo_l1_adjoint_tests(Tt, t, a))
    coef, _ = fit_least_squares(cols, b, ridge=1e-10, normalize=True)
    return float(np.linalg.norm(b - cols @ coef) / (np.linalg.norm(b) + 1e-14))

z0 = np.array([0.90, 1.50])
print("start (alpha,beta2)=", tuple(z0), " residual=%.4f" % residual(z0))
z, r = refine_orders_local(residual, z0, [(0.6, 1.05), (1.2, 2.0)])
print("polished (alpha,beta2)=(%.3f,%.3f)  residual=%.4e  (truth 0.800,1.700)" % (z[0], z[1], r))
