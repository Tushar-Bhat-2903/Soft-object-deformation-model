"""fitlib.py - shared helpers: load a clip, build the simulation scenario, compute residuals."""
import numpy as np

from deform_sim import Block, simulate

XIS = (0.2, 0.5, 0.8)


def load_clip(path, stride=2):
    z = np.load(path, allow_pickle=False)
    c = {k: z[k] for k in z.files}
    c["mode"] = str(c["mode"])
    idx = np.arange(0, len(c["t"]), stride)
    c["idx"] = idx
    c["path"] = path
    return c


def scenario_of(c):
    idx = c["idx"]
    sc = {"t": c["t"][idx] - c["t"][idx][0], "mode": c["mode"]}
    if c["mode"] == "kinematic":
        sc["u"] = c["u_mm"][idx] * 1e-3
    else:
        sc["load_on"] = c["load_on"][idx]
        sc["mass_kg"] = float(c["mass_g"]) * 1e-3
    return sc


def block_of(c, mesh):
    return Block(float(c["W_mm"]) * 1e-3, float(c["D_mm"]) * 1e-3, float(c["H0_mm"]) * 1e-3, *mesh)


def free_mask(c):
    """Frames whose top height is a prediction, not an imposed boundary condition."""
    idx = c["idx"]
    t = c["t"][idx]
    if c["mode"] == "kinematic":
        return t > float(c["t_release"])
    return np.ones(len(idx), bool)


def run_clip(args):
    """Worker: simulate one clip. args = (clip, E, nu, zeta, mesh, substeps, keep_states)."""
    c, E, nu, zeta, mesh, substeps, keep = args[:7]
    pset = args[7] if len(args) > 7 else 0.0
    b = block_of(c, mesh)
    sc = scenario_of(c)
    r = simulate(b, E, nu, zeta, sc, xis=XIS, substeps=substeps, keep_states=keep, pset=pset)
    r["block"] = b if keep else None
    return r


def residuals_of(c, r, sigma_h, sigma_w):
    idx = c["idx"]
    hobs = c["h_mm"][idx]
    wobs = c["widths_mm"][idx]
    fm = free_mask(c)
    rh = (1e3 * r["h_top"] - hobs)[fm] / sigma_h
    rw = ((1e3 * r["widths"] - wobs) / sigma_w).ravel()
    bad = ~np.isfinite(r["h_top"]).all() or ~np.isfinite(r["widths"]).all()
    if bad:
        return np.full(len(rh) + len(rw), 1e3)
    return np.concatenate([rh, rw])


def estimate_tau(c):
    """Rough recovery time constant (s): time after release to regain 63% of the lost height."""
    t, h = c["t"], c["h_mm"]
    ir = int(np.argmin(np.abs(t - float(c["t_release"]))))
    h0 = float(np.median(h[:8]))
    hm = float(h[ir])
    target = hm + 0.63 * (h0 - hm)
    after = np.where(h[ir:] >= target)[0]
    if len(after) == 0:
        return 0.3
    return max(float(t[ir + after[0]] - t[ir]), 0.02)
