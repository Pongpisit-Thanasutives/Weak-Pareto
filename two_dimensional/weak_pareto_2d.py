"""
Weak-Pareto extended to two spatial dimensions -- example.

Reuses the conventions of the 1-D package verbatim:
  * Caputo target via the transpose of the L1 matrix  (fractional_weak_form.caputo_l1_adjoint_tests)
  * spatial adjoints via the conjugate Fourier multiplier
  * separable weak rows, uniform quadrature  <f,g>_h = ht hx hy sum f g
  * column-normalised ridge, disjoint train/validation rows
  * variance-normalised held-out score  E_val,  objective  log10(E_val+eps) + duplicate penalty
  * differential evolution over the continuous orders, per (power, direction) pattern
  * signed-chord elbow over support size

New in 2-D: each candidate carries a DIRECTION label d in {x, y} in addition to its
power p and continuous order beta, so the model class is

    C_0 D_t^alpha u = sum_j xi_j u^{p_j} X^{(d_j)}_{beta_j} u .

The separable projection is three mode-wise contractions; the Kronecker product of the
two spatial test bases is never formed.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# Allow the example to be run either from the supplementary-code root or
# directly from this subdirectory, without machine-specific paths.
_SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_SUPPLEMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUPPLEMENT_ROOT))
from fractional_weak_form import caputo_l1_adjoint_tests, gaussian_test_matrix  # noqa: E402

EPS = 1e-14


# ------------------------------------------------------------------ test bases
def paper_test_counts(nt, nx):
    """_default_test_counts(..., budget='paper') from the 1-D package."""
    return max(20, min(44, nt // 3)), max(32, min(80, nx // 2))


def paper_widths(t, x, n_space_tests):
    """_base_weak_widths(..., 'paper') from the 1-D package, per axis."""
    dt, dx = t[1] - t[0], x[1] - x[0]
    Lx = x[-1] - x[0] + dx
    time_width = max(3.0 * dt, 0.070 * (t[-1] - t[0] + dt))
    space_width = max(2.0 * dx, Lx / max(18.0, 0.75 * n_space_tests))
    return float(time_width), float(space_width)


def build_tests(t, x, y, Kt=6, Kx=7, Ky=7, wt=0.14, wx=0.16, wy=None):
    """Build separable tests from absolute widths on each coordinate grid."""
    wy = wx if wy is None else wy
    Th = gaussian_test_matrix(t, centers=Kt, width=float(wt), periodic=False)
    Px = gaussian_test_matrix(x, centers=Kx, width=float(wx), periodic=True)
    Py = gaussian_test_matrix(y, centers=Ky, width=float(wy), periodic=True)
    return np.asarray(Th), np.asarray(Px), np.asarray(Py)


def conj_multiplier_adjoint(P, k, beta):
    """(X_beta)^* applied to each row of the spatial test matrix P."""
    m = np.where(k != 0, (np.abs(k) ** beta) * np.exp(1j * (np.pi / 2) * beta * np.sign(k)), 0.0)
    return np.fft.ifft(np.fft.fft(P, axis=1) * np.conj(m)[None, :], axis=1).real


def contract(U, A, B, C, scale):
    """scale * sum_{ijl} U[i,j,l] A[a,i] B[b,j] C[c,l]  ->  (a,b,c) flattened."""
    tmp = np.tensordot(A, U, axes=(1, 0))       # (a, nx, ny)
    tmp = np.tensordot(tmp, B, axes=(1, 1))     # (a, ny, b)
    out = np.tensordot(tmp, C, axes=(1, 1))     # (a, b, c)
    return (scale * out).reshape(-1)


# ------------------------------------------------------- precomputed order grids
class Library2D:
    """Precompute tensor-product weak targets and directional spatial columns."""
    def __init__(self, U, t, x, y, alpha_grid, beta_grid, Kt=6, Kx=7, Ky=7,
                 spatial_width=0.16, temporal_width=None):
        self.U, self.t, self.x, self.y = U, t, x, y
        ht, hx, hy = t[1] - t[0], x[1] - x[0], y[1] - y[0]
        self.scale = ht * hx * hy
        wt_abs, _ = paper_widths(t, x, Kx)
        wt_abs = wt_abs if temporal_width is None else float(temporal_width)
        Lx = float((x[1] - x[0]) * x.size)
        Ly = float((y[1] - y[0]) * y.size)
        wx_abs = float(spatial_width) * Lx
        wy_abs = float(spatial_width) * Ly
        self.temporal_width = float(wt_abs)
        self.spatial_width_x = wx_abs
        self.spatial_width_y = wy_abs
        self.Th, self.Px, self.Py = build_tests(
            t, x, y, Kt, Kx, Ky, wt_abs, wx_abs, wy_abs
        )
        self.kx = 2 * np.pi * np.fft.fftfreq(x.size, d=hx)
        self.ky = 2 * np.pi * np.fft.fftfreq(y.size, d=hy)
        self.alpha_grid, self.beta_grid = np.asarray(alpha_grid), np.asarray(beta_grid)

        # Hoist the factors that do not vary with the searched order.  TU is shared by
        # every spatial column; USxy is shared by every temporal target.
        self._TU = np.tensordot(self.Th, U, axes=(1, 0))          # (Kt, nx, ny)
        self._TUY = np.tensordot(self._TU, self.Py, axes=(2, 1))         # (Kt, nx, Ky)
        self._TUX = np.tensordot(self._TU, self.Px, axes=(1, 1))         # (Kt, ny, Kx)
        _t = np.tensordot(U, self.Px, axes=(1, 1))                       # (nt, ny, Kx)
        self._USxy = np.tensordot(_t, self.Py, axes=(1, 1))              # (nt, Kx, Ky)
        self.B = np.stack([self.exact_target(a) for a in self.alpha_grid])
        self.F = {}
        for d in ("x", "y"):
            cols = []
            for b in self.beta_grid:
                cols.append(self.exact_column(d, b))
            self.F[d] = np.stack(cols)
        self.K = self.B.shape[1]

    def set_split(self, tr, va):
        """Precompute exact Gram tables on the train and validation row subsets.

        Because a column at order beta is the linear interpolant of its two neighbouring
        grid columns, every inner product needed by the objective is the bilinear
        interpolant of a precomputed table.  The tables are therefore EXACT, not an
        approximation: the search sees the same objective as before, evaluated in O(c^2)
        instead of O(n_rows c^2)."""
        self._split = {}
        for name, idx in (("tr", tr), ("va", va)):
            Fx, Fy = self.F["x"][:, idx], self.F["y"][:, idx]
            B = self.B[:, idx]
            self._split[name] = dict(
                n=len(idx),
                ff={("x", "x"): Fx @ Fx.T, ("x", "y"): Fx @ Fy.T,
                    ("y", "x"): Fy @ Fx.T, ("y", "y"): Fy @ Fy.T},
                fb={"x": Fx @ B.T, "y": Fy @ B.T},
                bb=B @ B.T,
                bs=B.sum(axis=1),
            )
        return self

    @staticmethod
    def _wts(grid, v):
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, grid.size - 2))
        w = (v - grid[i]) / (grid[i + 1] - grid[i])
        return i, 1.0 - w, w

    def _bilin(self, tab, g1, v1, g2, v2):
        i, a0, a1 = self._wts(g1, v1)
        j, b0, b1 = self._wts(g2, v2)
        return (a0 * (b0 * tab[i, j] + b1 * tab[i, j + 1])
                + a1 * (b0 * tab[i + 1, j] + b1 * tab[i + 1, j + 1]))

    def gram(self, which, alpha, terms):
        """Return (A, b, tt, ts, n) for the normal equations on one row subset."""
        S, g, a = self._split[which], self.beta_grid, self.alpha_grid
        c = len(terms)
        A = np.empty((c, c))
        for k, (dk, _pk, bk) in enumerate(terms):
            for l, (dl, _pl, bl) in enumerate(terms):
                A[k, l] = self._bilin(S["ff"][(dk, dl)], g, bk, g, bl)
        bvec = np.array([self._bilin(S["fb"][d], g, bb, a, alpha)
                         for d, _p, bb in terms])
        tt = self._bilin(S["bb"], a, alpha, a, alpha)
        i, w0, w1 = self._wts(a, alpha)
        ts = w0 * S["bs"][i] + w1 * S["bs"][i + 1]
        return A, bvec, tt, ts, S["n"]

    @staticmethod
    def _lerp(grid, table, v):
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, grid.size - 2))
        w = (v - grid[i]) / (grid[i + 1] - grid[i])
        return (1.0 - w) * table[i] + w * table[i + 1]

    def target(self, alpha):
        return self._lerp(self.alpha_grid, self.B, float(alpha))

    def exact_target(self, alpha):
        Wt = caputo_l1_adjoint_tests(self.Th, self.t, float(alpha))
        return (self.scale * np.tensordot(Wt, self._USxy, axes=(1, 0))).reshape(-1)

    def exact_column(self, d, beta):
        if d == "x":
            A = conj_multiplier_adjoint(self.Px, self.kx, float(beta))
            out = np.tensordot(self._TUY, A, axes=(1, 1)).transpose(0, 2, 1)   # (Kt,Kx,Ky)
        else:
            A = conj_multiplier_adjoint(self.Py, self.ky, float(beta))
            out = np.tensordot(self._TUX, A, axes=(1, 1))                      # (Kt,Kx,Ky)
        return (self.scale * out).reshape(-1)

    def column(self, d, beta):
        return self._lerp(self.beta_grid, self.F[d], float(beta))


# ------------------------------------------------------------------ ridge + score
def ridge_fit(Th_tr, b_tr, lam=1e-3):
    """Fit ridge coefficients on an explicit two-dimensional weak design."""
    s = np.linalg.norm(Th_tr, axis=0)
    s = np.where(s > 0, s, 1.0)
    Tn = Th_tr / s
    xi = np.linalg.solve(Tn.T @ Tn + lam * np.eye(Tn.shape[1]), Tn.T @ b_tr)
    return xi / s


def solve_normal(A, b, lam):
    """Ridge on the column-normalised design, from the Gram matrix alone."""
    s = np.sqrt(np.maximum(np.diag(A), EPS))
    G = A / np.outer(s, s)
    z = np.linalg.solve(G + lam * np.eye(len(s)), b / s)
    return z / s


def objective(lib, alpha, terms, tr, va, lam=1e-3, exact=False):
    """Evaluate the variance-normalised validation objective for one model."""
    if exact:
        cols = np.stack([lib.exact_column(d, b) for d, p, b in terms], axis=1)
        tgt = lib.exact_target(alpha)
        Atr = cols[tr].T @ cols[tr]
        xi = solve_normal(Atr, cols[tr].T @ tgt[tr], lam)
        r = tgt[va] - cols[va] @ xi
        Ev = np.mean(r ** 2) / (np.var(tgt[va]) + EPS)
    else:
        Atr, btr, _, _, _ = lib.gram("tr", alpha, terms)
        xi = solve_normal(Atr, btr, lam)
        Ava, bva, tt, ts, nva = lib.gram("va", alpha, terms)
        mse = max(0.0, float((tt - 2.0 * xi @ bva + xi @ Ava @ xi) / nva))
        var = max(0.0, float(tt / nva - (ts / nva) ** 2))
        Ev = mse / (var + EPS)
    pen, lam_dup, gap_min = 0.0, 0.02, 0.04
    for (d1, p1, b1), (d2, p2, b2) in itertools.combinations(terms, 2):
        if d1 == d2 and p1 == p2:
            pen += lam_dup * max(0.0, (gap_min - abs(b1 - b2)) / gap_min)
    return np.log10(Ev + EPS) + pen, xi, Ev


# --------------------------------------------------------------------- the search
def best_model_of_size(lib, c, tr, va, alpha_bounds, beta_bounds, seed=0, maxiter=24, popsize=7):
    """Search all direction patterns and orders for one support size."""
    best = (np.inf, None)
    # Candidate terms are unordered, so one sorted direction multiset represents
    # every distinct assignment pattern at a fixed support size.
    for dirs in itertools.combinations_with_replacement(("x", "y"), c):
        def f(v, _dirs=dirs):
            a, bs = v[0], v[1:]
            terms = [(_dirs[j], 0, bs[j]) for j in range(c)]
            return objective(lib, a, terms, tr, va)[0]

        res = differential_evolution(
            f, [alpha_bounds] + [beta_bounds] * c, seed=seed,
            maxiter=maxiter, popsize=popsize, polish=True, tol=1e-8, init="sobol"
        )
        if res.fun < best[0]:
            terms = [(dirs[j], 0, float(res.x[1 + j])) for j in range(c)]
            _, xi, Ev = objective(lib, float(res.x[0]), terms, tr, va)
            best = (
                float(res.fun),
                dict(alpha=float(res.x[0]), terms=terms, xi=xi, Eval=float(Ev), J=float(res.fun)),
            )
    return best[1]


def polish_and_refit(lib, model, tr, va, alpha_bounds, beta_bounds, lam=1e-3, maxiter=80):
    """Exact-order polish followed by the full-row coefficient refit.

    Mirrors the manuscript: direct operator evaluation at the polished orders removes
    interpolation error from the conditional refit, and the final coefficients are fitted
    on all weak rows rather than on the training split alone.
    """
    from scipy.optimize import minimize

    dirs = [d for d, _p, _b in model["terms"]]
    x0 = np.array([model["alpha"]] + [b for _d, _p, b in model["terms"]], dtype=float)
    lo = np.array([alpha_bounds[0]] + [beta_bounds[0]] * len(dirs))
    hi = np.array([alpha_bounds[1]] + [beta_bounds[1]] * len(dirs))

    def f(v):
        v = np.clip(v, lo, hi)
        terms = [(dirs[j], 0, v[1 + j]) for j in range(len(dirs))]
        return objective(lib, v[0], terms, tr, va, lam, exact=True)[0]

    res = minimize(f, x0, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-10))
    v = np.clip(res.x, lo, hi)
    terms = [(dirs[j], 0, float(v[1 + j])) for j in range(len(dirs))]

    cols = np.stack([lib.exact_column(d, b) for d, _p, b in terms], axis=1)
    tgt = lib.exact_target(v[0])
    xi = ridge_fit(cols, tgt, lam)
    r = tgt - cols @ xi
    out = dict(model)
    out.update(alpha=float(v[0]), terms=terms, xi=xi,
               E_fit=float(np.mean(r ** 2) / (np.var(tgt) + EPS)),
               polished=True)
    return out


def signed_chord_elbow(errs):
    """Select the signed-chord elbow from support-size winners."""
    e = np.log10(np.asarray(errs) + EPS)
    c = np.arange(1, len(e) + 1, dtype=float)
    if len(e) < 3:
        return len(e) if (len(e) == 2 and e[0] - e[1] >= 0.15) else 1
    cn = (c - c[0]) / (c[-1] - c[0])
    en = (e - e.min()) / (e.max() - e.min() + EPS)
    chord = en[0] + cn * (en[-1] - en[0])
    exc = chord - en
    return int(np.argmax(exc[1:-1]) + 2) if exc[1:-1].max() > 0 else 1


def discover(U, t, x, y, cmax=4, seed=0, alpha_bounds=(0.55, 1.25),
             beta_bounds=(0.5, 2.5), spatial_width=None, early_stop=True,
             Kt=None, Kx=None, Ky=None, maxiter=24, popsize=7,
             n_alpha_nodes=47, n_beta_nodes=59, polish=True, temporal_width=None):
    """Defaults reproduce the manuscript's declared configuration:
    DE 7x24, ridge 1e-3, 47/59 order nodes, validation fraction 0.25, the package's
    paper test-count and absolute-width rules applied per axis, exact-order polishing
    and a full-row coefficient refit. ``spatial_width`` is a fraction of each periodic
    domain length; ``temporal_width`` is an optional absolute width.

    Boundary trimming is intentionally absent. It is a control for pointwise feature
    construction in the 1-D strong-form framework; applying it to the weak Caputo target
    would either move the lower terminal away from t=0 or delete legitimate endpoint mass."""

    U = np.asarray(U, dtype=float)
    t, x, y = (np.asarray(v, dtype=float) for v in (t, x, y))
    if U.shape != (t.size, x.size, y.size):
        raise ValueError(f"U must have shape (nt, nx, ny); got {U.shape}")
    if min(t.size, x.size, y.size) < 4:
        raise ValueError("each grid axis must contain at least four samples")
    if not (np.all(np.diff(t) > 0) and np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("t, x, and y must be strictly increasing")
    if cmax < 1:
        raise ValueError("cmax must be positive")
    kt_def, kx_def = paper_test_counts(t.size, x.size)
    _, ky_def = paper_test_counts(t.size, y.size)
    Kt = kt_def if Kt is None else Kt
    Kx = kx_def if Kx is None else Kx
    Ky = ky_def if Ky is None else Ky
    if spatial_width is None:
        _, spatial_width_abs = paper_widths(t, x, Kx)
        spatial_width = spatial_width_abs / (x[-1] - x[0] + (x[1] - x[0]))
    lib = Library2D(U, t, x, y, np.linspace(*alpha_bounds, n_alpha_nodes),
                    np.linspace(*beta_bounds, n_beta_nodes), Kt, Kx, Ky, spatial_width,
                    temporal_width)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(lib.K)
    nva = int(0.25 * lib.K)
    va, tr = idx[:nva], idx[nva:]
    lib.set_split(tr, va)
    models, errs = [], []
    for c in range(1, cmax + 1):
        m = best_model_of_size(lib, c, tr, va, alpha_bounds, beta_bounds, seed=seed,
                               maxiter=maxiter, popsize=popsize)
        models.append(m)
        errs.append(m["Eval"])
        if early_stop and c >= 2 and errs[-1] > 0.7 * errs[-2]:
            break
    best = models[signed_chord_elbow(errs) - 1]
    if polish:
        best = polish_and_refit(lib, best, tr, va, alpha_bounds, beta_bounds)
    return best, models, errs
