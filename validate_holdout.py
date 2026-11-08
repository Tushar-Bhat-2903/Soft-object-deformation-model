"""validate_holdout.py

Tests the fitted model on a clip it has never seen, to get an honest read on the fit (scoring
against the training data always looks better than it should). Only the plate's measured motion
is taken from the held-out clip; the object's response is predicted, then compared to the video.

Also fits a much dumber baseline (a single exponential recovery curve) so you can tell whether
the 3D model is actually earning its complexity, and builds an uncertainty band by resampling
the fitted parameters within their estimated error.

Usage:
  python validate_holdout.py --fit fit.json --clip holdout.npz --out holdout
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from fitlib import load_clip, run_clip, free_mask


def exp_baseline_tau(train_clips):
    """Fit the single time constant for the dumb baseline (exponential recovery after release)."""
    def cost(tau):
        s = 0.0
        for c in train_clips:
            if c["mode"] != "kinematic":
                continue
            t, h = c["t"], c["h_mm"]
            ir = int(np.argmin(np.abs(t - float(c["t_release"]))))
            h0 = float(np.median(h[:8]))
            hm = float(h[ir])
            m = t > t[ir]
            pred = h0 - (h0 - hm) * np.exp(-(t[m] - t[ir]) / tau)
            s += float(np.sum((pred - h[m]) ** 2))
        return s
    r = minimize_scalar(cost, bounds=(0.01, 5.0), method="bounded")
    return float(r.x)


def baseline_predict(c, tau):
    t, h = c["t"], c["h_mm"]
    ir = int(np.argmin(np.abs(t - float(c["t_release"]))))
    h0 = float(np.median(h[:8]))
    hm = float(h[ir])
    return np.where(t > t[ir], h0 - (h0 - hm) * np.exp(-(t - t[ir]) / tau), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="fit.json")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="holdout")
    ap.add_argument("--samples", type=int, default=16, help="parameter samples for the uncertainty band")
    ap.add_argument("--substeps", type=int, default=6)
    a = ap.parse_args()

    fit = json.load(open(a.fit))
    mesh = tuple(fit["mesh"])
    c = load_clip(a.clip, 1)
    E, nu, zeta = fit["E_Pa"], fit["nu"], fit["zeta"]
    r = run_clip((c, E, nu, zeta, mesh, a.substeps, False, fit.get("pset", 0.0)))
    t = c["t"]
    hobs, wobs = c["h_mm"], c["widths_mm"]
    fm = free_mask(c)
    hsim, wsim = 1e3 * r["h_top"], 1e3 * r["widths"]
    rmse_h = float(np.sqrt(np.mean((hsim[fm] - hobs[fm]) ** 2)))
    rmse_w = float(np.sqrt(np.mean((wsim - wobs) ** 2)))
    amp = float(np.median(hobs[:8]) - hobs.min())
    out = {"clip": a.clip, "rmse_height_mm_after_release": rmse_h, "rmse_widths_mm": rmse_w,
           "rmse_height_pct_of_compression": 100 * rmse_h / amp, "compression_mm": amp,
           "params": {"E_Pa": E, "nu": nu, "zeta": zeta, "tau_slow_s": fit["tau_slow_s"], "E_identified": fit["E_identified"]}}

    train = [load_clip(p, 1) for p in fit["train_clips"]]
    if c["mode"] == "kinematic":
        tau1 = exp_baseline_tau(train)
        bp = baseline_predict(c, tau1)
        m = np.isfinite(bp)
        out["baseline_1d_exp"] = {"tau1_s": tau1, "rmse_height_mm_after_release": float(np.sqrt(np.mean((bp[m] - hobs[m]) ** 2)))}
    names = fit["param_names"]
    cov = np.array(fit["cov_log"])
    p0 = np.array(fit["param_values"])
    band = np.zeros((0, len(t)))
    if a.samples and np.isfinite(cov).all():
        band = []
        rng = np.random.default_rng(0)
        for _ in range(a.samples):
            p = rng.multivariate_normal(p0, cov)
            i = 0
            Es = E
            if "logE" in names:
                Es = float(np.exp(p[i])); i += 1
            zs = float(np.exp(p[i])); i += 1
            ns = float(np.clip(p[i], 0.05, 0.42)) if "nu" in names else nu
            rr = run_clip((c, Es, ns, zs, mesh, a.substeps, False, fit.get("pset", 0.0)))
            band.append(1e3 * rr["h_top"])
        band = np.array(band)
        out["band_height_mm_mean_p5_p95_width_after_release"] = float((np.percentile(band, 95, axis=0) - np.percentile(band, 5, axis=0))[fm].mean())
    json.dump(out, open(a.out + "_metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(t, hobs, "k.", ms=3, label="video (held out)")
    if len(band):
        ax[0].fill_between(t, np.percentile(band, 5, axis=0), np.percentile(band, 95, axis=0), color="r", alpha=0.2,
                           label="model, 5-95% of fitted-parameter uncertainty")
    ax[0].plot(t, hsim, "r-", label="model prediction")
    if c["mode"] == "kinematic":
        ax[0].plot(t, bp, "b--", label="1D exp baseline")
    ax[0].axvline(float(c["t_release"]), color="0.6", ls=":")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("top height (mm)")
    ax[0].set_title("rmse after release %.2f mm" % rmse_h, fontsize=10)
    ax[0].legend(fontsize=7)
    for k, col in enumerate(["C0", "C1", "C2"]):
        ax[1].plot(t, wobs[:, k], ".", ms=3, color=col)
        ax[1].plot(t, wsim[:, k], "-", color=col)
    ax[1].set_title("silhouette widths 20/50/80%% height (rmse %.2f mm)" % rmse_w, fontsize=10)
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(a.out + "_compare.png", dpi=120)
    print("saved", a.out + "_metrics.json", a.out + "_compare.png")


if __name__ == "__main__":
    main()
