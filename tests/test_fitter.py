"""Tests of the vectorised profile-likelihood fitter (src/phase3_fit.py, lik="bbp").

Run with:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
import phase3_fit as F3          # noqa: E402
import bench_fitter as BF        # noqa: E402  (template builder and scipy references)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(11)
    tm, truth = BF.make_templates(rng, 2000, 12, n_ref=(30_000, 90_000))
    return rng, tm, truth


def test_profiled_beta_is_stationary(setup):
    """The closed-form beta_i is a stationary point of the NLL in beta_i (or sits at 0)."""
    rng, (s, b, d, m), truth = setup
    n, nu0 = BF.pseudo_data(rng, truth, 50)
    F = F3.Fitter(s, b, d, m + 0.5, n, nu0, "bbp")
    for mu, nu in ((0.0, 0.0), (0.7, 0.4), (2.0, -0.5)):
        a = mu * F.s[None, :]
        q = 1.0 + nu * F.delta[None, :]
        beta = F._beta(a, q)
        lam = a + beta * q
        n_over_lam = np.divide(F.n, lam, out=np.zeros_like(lam), where=lam > 0)   # n = 0 -> 0
        grad = (1 - n_over_lam) * q + 1 / F.wb - np.where(F.m > 0, F.m / np.maximum(beta, 1e-300), 0)
        interior = beta > 1e-9
        assert np.all(np.abs(grad[interior]) < 1e-6 * (1 + F.n[interior] + F.m[None, :].repeat(len(n), 0)[interior]))
        assert np.all(grad[~interior] >= -1e-8)        # at the bound the gradient points outwards


def test_analytic_gradient_matches_finite_differences(setup):
    rng, (s, b, d, m), truth = setup
    n, nu0 = BF.pseudo_data(rng, truth, 20, mu_true=1.0)
    F = F3.Fitter(s, b, d, m + 0.5, n, nu0, "bbp")
    mu = np.full(20, 0.8); nu = np.full(20, 0.3); h = 1e-6
    gm, gn, *_ = F.derivs(mu, nu)
    fd_m = (F.nll(mu + h, nu) - F.nll(mu - h, nu)) / (2 * h)
    fd_n = (F.nll(mu, nu + h) - F.nll(mu, nu - h)) / (2 * h)
    assert np.allclose(gm, fd_m, rtol=1e-4, atol=1e-5)
    assert np.allclose(gn, fd_n, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mu_true", [0.0, 1.0])
def test_vectorised_minimum_matches_scipy_references(setup, mu_true):
    """Free and background-only minima agree with per-toy Nelder-Mead and L-BFGS-B."""
    rng, tm, truth = setup
    n, nu0 = BF.pseudo_data(rng, truth, 15, mu_true=mu_true)
    mu, fh, f0 = BF.vectorised(tm, n, nu0)
    assert np.all(mu >= 0)
    fh_nm, f0_nm = BF.ref_nelder_mead(tm, n, nu0)
    fh_lb, f0_lb = BF.ref_lbfgsb(tm, n, nu0)
    for ref_h, ref_0 in ((fh_nm, f0_nm), (fh_lb, f0_lb)):
        assert np.max(fh - ref_h) < 1e-5
        assert np.max(f0 - ref_0) < 1e-5


def test_calibration_with_perfect_templates():
    """With exact templates (large m), q0 under mu = 0 gives Z > 2 at about the nominal 2.28%."""
    rng = np.random.default_rng(3)
    B = 8
    s = np.linspace(0.5, 12.0, B); bkg = np.linspace(300.0, 20.0, B)
    m = np.full(B, 1e6); w_b = bkg / m                     # effectively infinite simulation
    d = 0.05 * bkg * np.linspace(-1, 1, B)
    T = 4000
    n = rng.poisson(bkg, size=(T, B)).astype(float); nu0 = rng.normal(0, 1, T)
    F = F3.Fitter(s, (m + 0.5) * w_b, d, m + 0.5, n, nu0, "bbp")
    mu, _, fh = F.fit_multistart()
    _, _, f0 = F.fit(fix_mu=0.0)
    q0 = np.maximum(2 * (f0 - fh), 0) * (mu > 0)
    rate = np.mean(q0 > 4.0)
    assert abs(rate - 0.0228) < 4 * np.sqrt(0.0228 * 0.9772 / T)
