"""deform_sim.py

The physics engine: a soft block built from tetrahedra, pushed on by a flat plate.

Stiffness (E) and sideways bulge (nu) come from a standard rubber-like material law; a drag
term (zeta) makes the block settle back down instead of bouncing. Inertia is ignored since a
sponge moves far too slowly for it to matter.

Bottom face is glued to the table. Top nodes move freely until the plate reaches them, then
stick to it (no sliding). Two drive modes: 'kinematic' (plate height given directly, what a
video gives you) and 'load' (a known weight for a known time, needed to pin down stiffness
rather than just the stiffness/damping ratio).
"""
import itertools
import numpy as np


class Block:
    def __init__(self, W, D, H, nx=5, ny=5, nz=6):
        self.W, self.D, self.H = W, D, H
        self.nx, self.ny, self.nz = nx, ny, nz
        xs = np.linspace(-W / 2, W / 2, nx + 1)
        ys = np.linspace(-D / 2, D / 2, ny + 1)
        zs = np.linspace(0.0, H, nz + 1)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.X0 = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        self.N = self.X0.shape[0]

        def idx(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k

        tets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for perm in itertools.permutations(range(3)):
                        p = np.zeros(3, dtype=int)
                        chain = [p.copy()]
                        for a in perm:
                            p[a] = 1
                            chain.append(p.copy())
                        tets.append([idx(i + c[0], j + c[1], k + c[2]) for c in chain])
        t = np.array(tets, dtype=int)
        Dm = np.stack([self.X0[t[:, 1]] - self.X0[t[:, 0]],
                       self.X0[t[:, 2]] - self.X0[t[:, 0]],
                       self.X0[t[:, 3]] - self.X0[t[:, 0]]], axis=2)
        neg = np.linalg.det(Dm) < 0
        t[neg, 1], t[neg, 2] = t[neg, 2].copy(), t[neg, 1].copy()
        Dm = np.stack([self.X0[t[:, 1]] - self.X0[t[:, 0]],
                       self.X0[t[:, 2]] - self.X0[t[:, 0]],
                       self.X0[t[:, 3]] - self.X0[t[:, 0]]], axis=2)
        self.tets = t
        self.Bm = np.linalg.inv(Dm)
        self.BmT = np.transpose(self.Bm, (0, 2, 1))
        self.V0 = np.abs(np.linalg.det(Dm)) / 6.0
        self.Vn = np.zeros(self.N)
        for c in range(4):
            self.Vn += np.bincount(t[:, c], weights=self.V0 / 4.0, minlength=self.N)
        import scipy.sparse as sp
        nT = t.shape[0]
        self.S = sp.csr_matrix((np.ones(4 * nT), (t.ravel(), np.arange(4 * nT))), shape=(self.N, 4 * nT))
        self.h = (W * D * H / (nx * ny * nz)) ** (1.0 / 3.0)
        grid = np.arange(self.N).reshape(nx + 1, ny + 1, nz + 1)
        self.grid = grid
        self.top_idx = grid[:, :, nz].ravel()
        self.bot_idx = grid[:, :, 0].ravel()
        self.free = np.ones(self.N, dtype=bool)
        self.free[self.bot_idx] = False
        wx = np.ones(nx + 1)
        wx[[0, -1]] = 0.5
        wy = np.ones(ny + 1)
        wy[[0, -1]] = 0.5
        aw = np.outer(wx, wy)
        self.top_area_w = (aw / aw.sum()).ravel()
        self.band_cols = np.where(np.abs(xs) <= 0.2 * W)[0]
        if len(self.band_cols) == 0:
            self.band_cols = np.array([int(np.argmin(np.abs(xs)))])

    def internal_force(self, x, mu, lam, zinv=1.0):
        """Elastic push-back force at every node. zinv is 1 normally; after a hard press it's >1,
        which makes the material act as if its resting shape is already a bit shorter."""
        t = self.tets
        xt = x[t]
        Ds = np.stack([xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 0], xt[:, 3] - xt[:, 0]], axis=2)
        F = Ds @ self.Bm
        if zinv != 1.0:
            F = F.copy()
            F[:, :, 2] *= zinv
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        k0 = np.cross(c1, c2)
        J = np.einsum("ti,ti->t", c0, k0)
        invJ = 1.0 / np.where(np.abs(J) < 1e-12, 1e-12, J)
        FinvT = np.stack([k0, np.cross(c2, c0), np.cross(c0, c1)], axis=2) * invJ[:, None, None]
        logJ = np.log(np.maximum(J, 1e-2))
        P = mu * (F - FinvT) + lam * logJ[:, None, None] * FinvT
        if zinv != 1.0:
            P = P.copy()
            P[:, :, 2] *= zinv
            P = P / zinv
        G = -self.V0[:, None, None] * (P @ self.BmT)
        gs = np.transpose(G, (0, 2, 1))
        g0 = -gs.sum(axis=1, keepdims=True)
        Gn = np.concatenate([g0, gs], axis=1).reshape(-1, 3)
        return self.S @ Gn

    def tangent0(self, mu, lam):
        """Stiffness matrix at the rest shape, built once and reused by the solver every step."""
        import scipy.sparse as sp
        t = self.tets
        nT = t.shape[0]
        I3 = np.eye(3)
        rows, cols, vals = [], [], []
        for a in range(4):
            for c in range(3):
                dDs = np.zeros((nT, 3, 3))
                if a == 0:
                    dDs[:, c, :] = -1.0
                else:
                    dDs[:, c, a - 1] = 1.0
                dF = dDs @ self.Bm
                dP = mu * (dF + np.transpose(dF, (0, 2, 1))) + lam * np.trace(dF, axis1=1, axis2=2)[:, None, None] * I3
                dG = -self.V0[:, None, None] * (dP @ self.BmT)
                gs = np.transpose(dG, (0, 2, 1))
                g0 = -gs.sum(axis=1, keepdims=True)
                df = np.concatenate([g0, gs], axis=1)
                for b in range(4):
                    for cc in range(3):
                        rows.append(3 * t[:, b] + cc)
                        cols.append(np.full(nT, 3 * 0) + 3 * t[:, a] + c)
                        vals.append(-df[:, b, cc])
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(3 * self.N, 3 * self.N)).tocsr()
        return K

    def observe(self, x, xis):
        """Side-view observables: top height at centre band and silhouette width at heights xis."""
        g = x.reshape(self.nx + 1, self.ny + 1, self.nz + 1, 3)
        ztop = g[:, :, self.nz, 2].max(axis=1)
        h_top = float(ztop[self.band_cols].mean())
        zk = g[:, :, :, 2].mean(axis=(0, 1))
        wk = g[:, :, :, 0].max(axis=(0, 1)) - g[:, :, :, 0].min(axis=(0, 1))
        xi_k = zk / max(h_top, 1e-9)
        order = np.argsort(xi_k)
        w = np.interp(xis, xi_k[order], wk[order])
        return h_top, w


def material(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def simulate(block, E, nu, zeta, scenario, xis=(0.2, 0.5, 0.8), keep_states=False,
             scheme="implicit", step_frac=0.01, kappa=3.0, substeps=None, pset=0.0):
    """Run one scenario through the block and return what a camera would see at each frame time.

    scenario keys: t (frame times, s), mode ('kinematic'/'load'), u (plate height, m, kinematic
    mode), load_on + mass_kg (load mode).

    scheme='implicit' solves a small linear system each step so it can take bigger, stabler
    steps; 'explicit' is the plain, slower way, kept as a cross-check.
    pset: permanent-set amount (0-0.6) - the rest height shrinks a bit after the deepest press
    reached so far, and never grows back. 0 disables it.
    substeps: fixes the number of steps per frame so the fit doesn't get noisy as parameters change.

    Returns a dict with h_top, widths, and states (if keep_states).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    t = np.asarray(scenario["t"], float)
    mode = scenario["mode"]
    mu, lam = material(E, nu)
    xis = np.asarray(xis, float)
    Hbig = 20.0 * block.H
    if mode == "kinematic":
        u = np.asarray(scenario["u"], float).copy()
        u[~np.isfinite(u)] = Hbig
    else:
        load_on = np.asarray(scenario["load_on"], bool)
        fz = -float(scenario["mass_kg"]) * 9.81
        ftop = np.zeros((block.N, 3))
        ftop[block.top_idx, 2] = fz * block.top_area_w
    x = block.X0.copy()
    cn = (zeta * block.Vn)[:, None]
    free = block.free[:, None]
    top = block.top_idx
    free_dof = np.repeat(block.free, 3)
    if scheme == "implicit":
        K0full = block.tangent0(mu, lam)
        tau_slow = zeta * block.H ** 2 / E
        dt_cap = max(step_frac * tau_slow, 1e-5)
        lu_cache = {}
    else:
        dt_cap = 0.8 * zeta * block.h ** 2 / (16.0 * (lam + 2.0 * mu))

    zstate = {"cmax": 0.0}

    def advance(x, fext, dt, con=None, dxc=None):
        """One step. con: node indices with prescribed displacement dxc (nc,3) (implicit scheme)."""
        f = block.internal_force(x, mu, lam, 1.0 / (1.0 - pset * zstate["cmax"]) if pset else 1.0)
        if fext is not None:
            f = f + fext
        if scheme == "implicit":
            if con is None or len(con) == 0:
                key = (round(dt, 9), b"")
            else:
                key = (round(dt, 9), con.tobytes())
            if key not in lu_cache:
                cmask = np.zeros(block.N, bool)
                if con is not None and len(con):
                    cmask[con] = True
                fm = free_dof & ~np.repeat(cmask, 3)
                A = (sp.diags(np.repeat(zeta * block.Vn, 3)) + kappa * dt * K0full).tocsr()
                fidx = np.where(fm)[0]
                cidx = np.where(np.repeat(cmask, 3))[0]
                lu = spla.splu(A[fidx][:, fidx].tocsc())
                Afc = A[fidx][:, cidx] if len(cidx) else None
                if len(lu_cache) > 400:
                    lu_cache.clear()
                lu_cache[key] = (lu, Afc, fidx, cidx)
            lu, Afc, fidx, cidx = lu_cache[key]
            rhs = dt * f.ravel()[fidx]
            dx = np.zeros(3 * block.N)
            if Afc is not None:
                dx[cidx] = dxc.ravel()
                rhs = rhs - Afc @ dx[cidx]
            dx[fidx] = lu.solve(rhs)
            return x + dx.reshape(-1, 3)
        return x + np.where(free, dt * f / cn, 0.0)

    T = len(t)
    h_out = np.zeros(T)
    w_out = np.zeros((T, len(xis)))
    states = np.zeros((T, block.N, 3)) if keep_states else None
    h_out[0], w_out[0] = block.observe(x, xis)
    if keep_states:
        states[0] = x
    for n in range(T - 1):
        dT = t[n + 1] - t[n]
        ns = int(substeps) if substeps else max(1, int(np.ceil(dT / dt_cap - 1e-9)))
        dt = dT / ns
        for s_ in range(ns):
            if mode == "kinematic":
                a_ = (s_ + 1) / ns
                un = (1 - a_) * u[n] + a_ * u[n + 1]
                xn = advance(x, None, dt)
                hit = xn[top, 2] >= un
                if hit.any():
                    if scheme == "implicit":
                        for _it in range(4):
                            con = np.sort(top[hit])
                            dxc = np.zeros((len(con), 3))
                            dxc[:, 2] = un - x[con, 2]
                            xn = advance(x, None, dt, con, dxc)
                            hit2 = (xn[top, 2] >= un + 1e-12) | hit
                            if (hit2 == hit).all():
                                break
                            hit = hit2
                    else:
                        ti = top[hit]
                        xn[ti, 2] = un
                        xn[ti, 0] = x[ti, 0]
                        xn[ti, 1] = x[ti, 1]
                x = xn
            else:
                x = advance(x, ftop if load_on[n] else None, dt)
            if pset:
                zstate["cmax"] = max(zstate["cmax"], float(np.clip(1.0 - x[top, 2].mean() / block.H, 0.0, 0.6)))
        h_out[n + 1], w_out[n + 1] = block.observe(x, xis)
        if keep_states:
            states[n + 1] = x
    out = {"h_top": h_out, "widths": w_out}
    if keep_states:
        out["states"] = states
    return out


def make_press_scenario(fps, H, depth, t_down=0.3, t_hold=0.6, t_total=3.0, t_start=0.3):
    """Synthetic kinematic press: plate descends by `depth` metres, holds, lifts away instantly."""
    t = np.arange(0.0, t_total, 1.0 / fps)
    u = np.full_like(t, np.inf)
    for i, ti in enumerate(t):
        if t_start <= ti < t_start + t_down:
            u[i] = H - depth * (ti - t_start) / t_down
        elif t_start + t_down <= ti < t_start + t_down + t_hold:
            u[i] = H - depth
    return {"t": t, "mode": "kinematic", "u": u}


if __name__ == "__main__":
    import sys
    import time
    if "--selftest" in sys.argv:
        b = Block(0.06, 0.06, 0.05)
        print("nodes", b.N, "tets", len(b.tets), "total volume", b.V0.sum(), "expected", 0.06 * 0.06 * 0.05)
        print("min tet volume", b.V0.min())
        f0 = b.internal_force(b.X0, *material(4e4, 0.4))
        print("rest force max (should be ~0):", np.abs(f0).max())
        sc = make_press_scenario(60, 0.05, 0.02)
        t0 = time.time()
        r = simulate(b, 4e4, 0.4, 3e6, sc)
        print("implicit sim time %.2fs for %d frames" % (time.time() - t0, len(sc["t"])))
        t0 = time.time()
        r2 = simulate(b, 4e4, 0.4, 3e6, sc, scheme="explicit")
        print("explicit reference time %.2fs" % (time.time() - t0))
        print("max |h_top implicit - explicit| = %.3f mm ; max width diff = %.3f mm" % (
            1e3 * np.abs(r["h_top"] - r2["h_top"]).max(), 1e3 * np.abs(r["widths"] - r2["widths"]).max()))
        for i in range(0, len(sc["t"]), 6):
            print("t=%.2f h_top=%.2f mm  widths(mm)=%s" % (sc["t"][i], 1e3 * r["h_top"][i], np.round(1e3 * r["widths"][i], 1)))
