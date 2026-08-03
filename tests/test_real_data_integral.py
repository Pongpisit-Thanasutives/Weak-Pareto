import numpy as np
from scipy.special import gamma

from real_data.frozen_soil_creep_weak import fractional_integral_piecewise_constant


def test_fractional_integral_of_constant_matches_closed_form():
    alpha = 0.63
    t = np.linspace(0.0, 2.0, 101)
    h = t[1] - t[0]
    numerical = fractional_integral_piecewise_constant(np.ones_like(t), h, alpha)
    exact = t**alpha / gamma(alpha + 1.0)
    np.testing.assert_allclose(numerical, exact, rtol=2e-13, atol=2e-13)
