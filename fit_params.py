"""fit_params.py

Fit the FEM proxy to one or more training clips (output of extract_signals.py).

Free parameters: log E (Pa), log zeta (N s / m^4), nu.
  * If at least one clip has mode 'load' (known weight => known force), E is identified.
  * Otherwise only the ratio zeta/E is identifiable from displacements, so E is fixed to --E_fixed and
    the reported zeta is conditional on that choice (tau = zeta*H^2/E is the meaningful number).

Loss: robust (soft-L1) least squares on top height (after release / under load) and silhouette widths.
Uncertainty: Gauss-Newton covariance (J^T J)^-1 * s^2 at the optimum (parameters in log space).

Usage:
  python fit_params.py --clips clipA.npz clipB.npz load.npz --out fit.json --report fit_report.png
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares

from fitlib import load_clip, run_clip, residuals_of, estimate_tau


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default="fit.json")
    ap.add_argument("--report", default="fit_report.png")
    ap.add_argument("--mesh", default="4,4,5", help="nx,ny,nz cells")
    ap.add_argument("--stride", type=int, default=2, help="use every n-th video frame")
    ap.add_argument("--substeps", type=int, default=10, help="solver steps per used frame")
    ap.add_argument("--sigma_h", type=float, default=0.5, help="height noise (mm)")
    ap.add_argument("--sigma_w", type=float, default=0.8, help="width noise (mm)")
    ap.add_argument("--E_fixed", type=float, default=3e4, help="E (Pa) used when no load clip is given")
    ap.add_argument("--E0", type=float, default=3e4)
    ap.add_argument("--nu_fixed", type=float, default=None, help="fix nu instead of fitting it")
    ap.add_argument("--pset", action="store_true",
                    help="also fit a permanent-set coefficient (stress-free height shrinks by pset x deepest compression)")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    mesh = tuple(int(v) for v in a.mesh.split(","))
    clips = [load_clip(p, a.stride) for p in a.clips]
    has_load = any(c["mode"] == "load" for c in clips)
    H0 = float(np.median([c["H0_mm"] for c in clips])) * 1e-3
    kin = [c for c in clips if c["mode"] == "kinematic"]
    tau_obs = float(np.median([estimate_tau(c) for c in kin])) if kin else 0.3
    print("clips:", [(os.path.basename(c["path"]), c["mode"]) for c in clips])
    print("E identified: %s ; rough recovery time constant %.3f s" % (has_load, tau_obs))
    nw = a.workers or min(len(clips), os.cpu_count() or 1)
    pool = ProcessPoolExecutor(max_workers=nw)

    names = (["logE"] if has_load else []) + ["logzeta"] + ([] if a.nu_fixed is not None else ["nu"]) + (["pset"] if a.pset else [])

    def unpack(p):
        i = 0
        if has_load:
            E = float(np.exp(p[i])); i += 1
        else:
            E = a.E_fixed
        zeta = float(np.exp(p[i])); i += 1
        nu = a.nu_fixed if a.nu_fixed is not None else float(p[i])
        if a.nu_fixed is None:
            i += 1
        pset = float(p[i]) if a.pset else 0.0
        return E, nu, zeta, pset

    def resid(p):
        E, nu, zeta, pset = unpack(p)
        jobs = [(c, E, nu, zeta, mesh, a.substeps, False, pset) for c in clips]
        res = list(pool.map(run_clip, jobs))
        return np.concatenate([residuals_of(c, r, a.sigma_h, a.sigma_w) for c, r in zip(clips, res)])

    lb, ub, starts = [], [], []
    E0 = a.E0 if has_load else a.E_fixed
    for f in (0.5, 1.0, 2.0):
        p = []
        if has_load:
            p.append(np.log(E0))
        p.append(np.log(f * tau_obs * (a.E0 if has_load else a.E_fixed) / H0 ** 2))
        if a.nu_fixed is None:
            p.append(0.3)
        if a.pset:
            p.append(0.1)
        starts.append(np.array(p))
    if has_load:
        lb.append(np.log(1e3)); ub.append(np.log(2e6))
    lb.append(np.log(1e3)); ub.append(np.log(1e10))
    if a.nu_fixed is None:
        lb.append(0.05); ub.append(0.42)
    if a.pset:
        lb.append(0.0); ub.append(0.5)
    lb, ub = np.array(lb), np.array(ub)

    best = None
    for k, p0 in enumerate(starts):
        sol = least_squares(resid, np.clip(p0, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), loss="soft_l1", f_scale=2.0,
                            diff_step=2e-2, max_nfev=40, x_scale=np.array([1.0 if n not in ("nu", "pset") else 0.1 for n in names]))
        print("start %d: cost %.3f  params %s" % (k, sol.cost, np.round(sol.x, 3)))
        if best is None or sol.cost < best.cost:
            best = sol
    sol = best
    E, nu, zeta, pset = unpack(sol.x)
    r = sol.fun
    n, p = len(r), len(sol.x)
    J = sol.jac
    s2 = float(np.sum(r ** 2) / max(n - p, 1))
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-30))
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
        std = np.full(p, np.nan)
    tau = zeta * H0 ** 2 / E
    out = {"E_Pa": E, "nu": nu, "zeta": zeta, "tau_slow_s": tau, "E_identified": bool(has_load), "pset": pset,
           "E_note": "E fitted from the known-weight clip" if has_load else
           "E was FIXED; only zeta/E (tau) is identified from displacement-only clips",
           "param_names": names, "param_values": sol.x.tolist(), "cov_log": cov.tolist(),
           "std": dict(zip(names, std.tolist())), "reduced_chi2": s2, "jac_condition": cond,
           "mesh": list(mesh), "substeps": a.substeps, "stride": a.stride, "sigma_h": a.sigma_h, "sigma_w": a.sigma_w,
           "H0_m": H0, "train_clips": a.clips, "has_load_clip": bool(has_load), "E_fixed": a.E_fixed}
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nFIT RESULT")
    print("  E = %.3g Pa %s" % (E, "" if has_load else "(fixed)"))
    print("  nu = %.3f" % nu)
    if a.pset:
        print("  permanent-set coefficient pset = %.3f (stress-free height shrinks by pset x deepest compression)" % pset)
    print("  zeta = %.3g N s/m^4  ->  slow relaxation time tau = zeta*H^2/E = %.3f s" % (zeta, tau))
    print("  1-sigma (log-space for E,zeta; absolute for nu): %s" % dict(zip(names, np.round(std, 3))))
    print("  reduced chi2 = %.2f ; Jacobian condition number = %.1e" % (s2, cond))
    if cond > 1e4:
        print("  WARNING: ill-conditioned: some parameter combination is poorly constrained by these clips")

    jobs = [(c, E, nu, zeta, mesh, a.substeps, False, pset) for c in clips]
    res = list(pool.map(run_clip, jobs))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(clips), 2, figsize=(11, 3.2 * len(clips)), squeeze=False)
    for i, (c, rr) in enumerate(zip(clips, res)):
        t = c["t"][c["idx"]]
        ax[i, 0].plot(t, c["h_mm"][c["idx"]], "k.", ms=3, label="video")
        ax[i, 0].plot(t, 1e3 * rr["h_top"], "r-", label="model")
        ax[i, 0].set_title("%s (%s): top height" % (os.path.basename(c["path"]), c["mode"]), fontsize=9)
        ax[i, 0].set_ylabel("mm")
        for k, col in enumerate(["C0", "C1", "C2"]):
            ax[i, 1].plot(t, c["widths_mm"][c["idx"]][:, k], ".", ms=3, color=col)
            ax[i, 1].plot(t, 1e3 * rr["widths"][:, k], "-", color=col)
        ax[i, 1].set_title("silhouette width at 20/50/80% height (dots video, lines model)", fontsize=9)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(a.report, dpi=110)
    print("saved", a.out, a.report)
    pool.shutdown()


if __name__ == "__main__":
    main()
