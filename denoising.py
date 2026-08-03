"""Optional denoising utilities for weak-form fractional PDE discovery.

The weak formulation already suppresses noise by integrating against smooth test
functions.  These routines add an optional *pre-library* denoising step for the
harder case of recovering fractional orders, especially the time order alpha.
BM3D is used when the third-party ``bm3d`` package is installed.  The public API
falls back to scikit-image non-local means so the project remains runnable in a
minimal scientific Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import NDArray

DenoiseMethod = Literal["none", "gaussian", "bm3d", "nlm", "wavelet", "bm3d_or_nlm", "bm3d_or_wavelet"]
VarianceStabilization = Literal["none", "standardize", "log_positive"]

EPS = 1e-14


@dataclass(frozen=True)
class DenoiseConfig:
    """Configuration for denoising a spatiotemporal field before weak features.

    Parameters
    ----------
    method:
        ``"bm3d"`` uses the optional TUNI/PyPI BM3D wrapper when available.
        ``"bm3d_or_nlm"`` tries BM3D and falls back to non-local means;
        ``"bm3d_or_wavelet"`` falls back to a faster wavelet/Gaussian path.
    sigma:
        Absolute noise standard deviation in the original data units.  If ``None``
        it is estimated from ``noise_percent`` and the field scale, or by a robust
        high-pass MAD estimate when ``noise_percent`` is unavailable.
    sigma_factor:
        Multiplicative tuning factor for the estimated sigma.  Values in
        ``[0.5, 1.25]`` are useful in stability selection.
    transform:
        ``"standardize"`` applies denoising to a robustly standardized field.
        ``"log_positive"`` denoises log-data when all entries are positive and
        is appropriate for strictly positive multiplicative-noise observations.
    """

    method: DenoiseMethod = "none"
    sigma: float | None = None
    sigma_factor: float = 1.0
    transform: VarianceStabilization = "standardize"
    clip_percentiles: tuple[float, float] = (0.1, 99.9)
    preserve_mean: bool = True
    warn_on_fallback: bool = False


def robust_scale(U: NDArray[np.float64]) -> float:
    """Robust scale estimate based on median absolute deviation.

    Used to normalize data before optional denoising and to estimate fallback
    noise levels.  The return value is always positive, even for nearly
    constant fields.
    """
    vals = np.asarray(U, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.std(vals))
    return max(scale, EPS)


def estimate_noise_sigma(U: NDArray[np.float64], noise_percent: float | None = None) -> float:
    """Estimate additive noise sigma in data units.

    The packaged benchmarks inject multiplicative uniform noise.  For small to
    moderate noise, the induced additive standard deviation is approximately
    ``noise_percent/100 * std(U) / sqrt(3)``.  Without a declared noise level,
    use a high-pass MAD estimate.
    """
    U = np.asarray(U, dtype=float)
    if noise_percent is not None and float(noise_percent) > 0:
        return max(float(noise_percent) / 100.0 * float(np.nanstd(U)) / np.sqrt(3.0), EPS)
    try:
        from scipy.ndimage import gaussian_filter

        hp = U - gaussian_filter(U, sigma=(1.0, 1.0), mode="nearest")
        return robust_scale(hp)
    except Exception:
        return 0.05 * robust_scale(U)


def _normalize_for_denoiser(
    U: NDArray[np.float64],
    sigma_abs: float,
    config: DenoiseConfig,
) -> tuple[NDArray[np.float64], float, Any]:
    U = np.asarray(U, dtype=float)
    transform = config.transform
    if transform == "log_positive" and np.nanmin(U) > 0:
        Z = np.log(np.maximum(U, EPS))
        # Propagate absolute sigma through log locally: std(log(U)) ≈ std(U)/|U|.
        sigma_z = float(sigma_abs) / max(float(np.nanmedian(np.abs(U))), EPS)
        state = ("log_positive",)
    else:
        Z = U.copy()
        sigma_z = float(sigma_abs)
        state = ("identity",)

    if transform in {"standardize", "log_positive"}:
        med = float(np.nanmedian(Z))
        scale = robust_scale(Z)
        Zs = (Z - med) / scale
        return Zs.astype(float), max(float(sigma_z) / scale, EPS), (*state, med, scale)
    return Z.astype(float), max(float(sigma_z), EPS), (*state, 0.0, 1.0)


def _denormalize_after_denoiser(Z: NDArray[np.float64], state: Any) -> NDArray[np.float64]:
    mode, med, scale = state[0], float(state[1]), float(state[2])
    U = np.asarray(Z, dtype=float) * scale + med
    if mode == "log_positive":
        U = np.exp(U)
    return U.astype(float)


def _denoise_gaussian(Z: NDArray[np.float64], sigma_scaled: float) -> NDArray[np.float64]:
    from scipy.ndimage import gaussian_filter

    # Smooth only lightly; time derivatives are sensitive to over-smoothing.
    s = float(np.clip(0.75 * sigma_scaled, 0.25, 1.25))
    return gaussian_filter(Z, sigma=(s, s), mode="nearest")


def _denoise_nlm(Z: NDArray[np.float64], sigma_scaled: float) -> NDArray[np.float64]:
    from skimage.restoration import denoise_nl_means

    # Fast mode is sufficient for discovery; the weak projection supplies the
    # subsequent low-pass integration.
    h = max(0.6 * float(sigma_scaled), 0.03)
    return denoise_nl_means(
        Z,
        h=h,
        sigma=max(float(sigma_scaled), EPS),
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        preserve_range=True,
        channel_axis=None,
    ).astype(float)


def _denoise_wavelet(Z: NDArray[np.float64], sigma_scaled: float) -> NDArray[np.float64]:
    try:
        from skimage.restoration import denoise_wavelet
    except Exception:
        return _denoise_gaussian(Z, sigma_scaled)

    kwargs = dict(
        sigma=max(float(sigma_scaled), EPS),
        mode="soft",
        rescale_sigma=True,
    )
    try:
        return denoise_wavelet(Z, preserve_range=True, channel_axis=None, **kwargs).astype(float)
    except TypeError:
        try:
            return denoise_wavelet(Z, multichannel=False, **kwargs).astype(float)
        except TypeError:
            try:
                return denoise_wavelet(Z, **kwargs).astype(float)
            except ImportError:
                return _denoise_gaussian(Z, sigma_scaled)
    except ImportError:
        return _denoise_gaussian(Z, sigma_scaled)


def _denoise_bm3d(Z: NDArray[np.float64], sigma_scaled: float) -> NDArray[np.float64]:
    import bm3d  # type: ignore

    # The common bm3d package exposes bm3d.bm3d(image, sigma_psd=...).  Some
    # versions accept sigma as the second positional argument.  Keep both paths.
    try:
        return np.asarray(bm3d.bm3d(Z, sigma_psd=max(float(sigma_scaled), EPS)), dtype=float)
    except TypeError:
        return np.asarray(bm3d.bm3d(Z, max(float(sigma_scaled), EPS)), dtype=float)


def denoise_spacetime_field(
    U: NDArray[np.float64],
    *,
    noise_percent: float | None = None,
    config: DenoiseConfig | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return a denoised field and metadata describing the denoising path.

    This function never raises merely because BM3D is not installed when the
    method is ``"bm3d_or_nlm"``; it records the fallback in metadata instead.
    """
    cfg = config or DenoiseConfig()
    method = cfg.method
    U0 = np.asarray(U, dtype=float)
    if method == "none":
        return U0.copy(), {"method": "none", "effective_method": "none", "sigma": 0.0, "sigma_scaled": 0.0}

    sigma_abs = float(cfg.sigma if cfg.sigma is not None else estimate_noise_sigma(U0, noise_percent=noise_percent))
    sigma_abs *= float(cfg.sigma_factor)
    Z, sigma_scaled, state = _normalize_for_denoiser(U0, sigma_abs, cfg)
    # BM3D/NLM are calibrated for moderate normalized intensities.  Robustly clip
    # extremes only inside the denoiser to prevent rare outliers from determining
    # block matches; restore the original mean level after inversion if requested.
    lo, hi = np.nanpercentile(Z, cfg.clip_percentiles)
    Zc = np.clip(Z, lo, hi)
    effective = method
    fallback_reason = ""
    try:
        if method == "gaussian":
            Zd = _denoise_gaussian(Zc, sigma_scaled)
        elif method == "nlm":
            Zd = _denoise_nlm(Zc, sigma_scaled)
        elif method == "wavelet":
            Zd = _denoise_wavelet(Zc, sigma_scaled)
        elif method in {"bm3d", "bm3d_or_nlm", "bm3d_or_wavelet"}:
            try:
                Zd = _denoise_bm3d(Zc, sigma_scaled)
                effective = "bm3d"
            except Exception as exc:
                if method == "bm3d":
                    raise
                fallback_reason = repr(exc)
                if method == "bm3d_or_wavelet":
                    if cfg.warn_on_fallback:
                        warnings.warn(f"BM3D unavailable; falling back to wavelet/Gaussian denoising: {fallback_reason}", RuntimeWarning)
                    Zd = _denoise_wavelet(Zc, sigma_scaled)
                    effective = "wavelet_fallback_for_bm3d"
                else:
                    if cfg.warn_on_fallback:
                        warnings.warn(f"BM3D unavailable; falling back to non-local means: {fallback_reason}", RuntimeWarning)
                    Zd = _denoise_nlm(Zc, sigma_scaled)
                    effective = "nlm_fallback_for_bm3d"
        else:
            raise ValueError(f"unknown denoise method: {method}")
    except Exception:
        if method in {"bm3d_or_nlm", "bm3d_or_wavelet"}:
            Zd = _denoise_gaussian(Zc, sigma_scaled)
            effective = f"gaussian_fallback_for_{method}"
        else:
            raise

    Ud = _denormalize_after_denoiser(Zd, state)
    if cfg.preserve_mean:
        Ud = Ud + (float(np.nanmean(U0)) - float(np.nanmean(Ud)))
    meta = {
        "method": method,
        "effective_method": effective,
        "sigma": float(sigma_abs),
        "sigma_scaled": float(sigma_scaled),
        "sigma_factor": float(cfg.sigma_factor),
        "transform": cfg.transform,
        "fallback_reason": fallback_reason,
    }
    return Ud.astype(float), meta
