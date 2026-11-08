# Learning a soft object's deformation from video

A small 3D finite-element model of a soft block that is pressed and released, fitted to side-view video.
The fitted model then predicts a fresh press (held out from the fit) and is compared with the real video.

Pipeline: `extract_signals.py` (video to measurements) -> `fit_params.py` (fit E, nu, damping) ->
`validate_holdout.py` (predict a held-out clip, metrics, plots) -> `make_demo.py` (side-by-side video).

> Fill in the sections marked **[YOUR RESULT]** with your own numbers and figures after running on your video.

## 1. What is modelled

* **Geometry**: a box W x D x H (measured with a ruler), meshed with tetrahedra (default 4 x 4 x 5 hex cells,
  each split into 6 tetrahedra, 480 tets).
* **Elasticity**: compressible neo-Hookean material, energy density
  `psi = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2`, with `mu = E / 2(1+nu)`, `lambda = E nu / ((1+nu)(1-2nu))`.
  Parameters: Young's modulus `E` and Poisson ratio `nu`.
* **Damping / viscosity**: a viscous drag force `-zeta * V_node * velocity` on every node.
* **Dynamics**: inertia is neglected (first-order, quasi-static dynamics): `zeta V_n dx/dt = f_elastic + f_external`.
  Reason: the inertial time `H / sqrt(E/rho)` is of order milliseconds, the recovery of foam takes a few
  tenths of a second, so inertia does not matter. Recovery therefore relaxes with a time constant of
  roughly `tau = zeta H^2 / E`.
* **Boundaries**: bottom face glued to the table. The plate (a flat board) is a rigid one-sided contact: top nodes
  are clamped to the plate height while pressed and stick laterally; when the plate leaves they are free.
  A known weight (load clip) is applied as uniform pressure on the top face.
* **Solver**: linearly implicit step `(C + kappa dt K0) dx = dt f(x)` with `K0` the rest-state stiffness matrix
  (see `deform_sim.py`). Checked against an explicit reference solver (`python deform_sim.py --selftest`).
* **Optional permanent set** (`--pset` flag, off by default): the model's stress-free height can shrink
  after a deep press, `lp = 1 - pset * c_max` where `c_max` is the deepest compression fraction reached
  so far. Implemented as a one-sided multiplicative split of the deformation gradient in the vertical
  direction. Added for the real sponge (section 5.2), which does not fully recover; with `pset = 0` this
  reduces exactly to the original elastic model.

## 2. How the video informs the model

Everything is measured from the object's silhouette in a fixed side-view camera (`extract_signals.py`):

| measurement | from the video | used for |
|---|---|---|
| top height h(t) | topmost mask row in the central band | recovery curve after release; compression under a known load |
| silhouette widths at 20/50/80 % of the current height | mask width at those rows | barrel bulge, which constrains Poisson's ratio `nu` |
| plate motion u(t) | equals h(t) while the plate touches the top | the boundary condition driving the simulation (what the user did, not the material response) |
| scale | known object width in mm / its width in pixels | pixels to millimetres |

**What is identifiable.** A video contains displacements, not forces.

* With **displacement-only clips** (hand presses), the model dynamics `zeta V dx/dt = -dU/dx(E)` only depend on
  the ratio `zeta/E`, so only the relaxation time `tau` and `nu` are identified. `E` is fixed and `zeta` is
  reported conditionally.
* A **known-weight clip** (a weight of known mass left on the object, then removed) provides a known force,
  so it identifies `E` itself, and its creep also constrains `zeta`.
  `fit_params.py` uses `E` as a free parameter only when a load clip is supplied, and reports what was fixed.

## 3. Fitting and uncertainty

Robust (soft-L1) least squares over `log E`, `log zeta`, `nu` using the top-height residuals
(after release or under load) and all silhouette-width residuals, three optimiser starts, normalised by assumed
measurement noise (0.5 mm height, 0.8 mm width). Uncertainty is the Gauss-Newton covariance
`s^2 (J^T J)^-1` at the optimum. A large Jacobian condition number is reported as a warning.

## 4. Held-out prediction

`validate_holdout.py` simulates a press that was not used in the fit. The only inputs taken from that clip are
the plate motion (or load timing). The prediction of the top height after release and of the widths is
compared with the video (RMSE in mm), together with a 1D exponential-recovery baseline fitted on the same
training clips and a band showing the effect of the fitted-parameter uncertainty.

## 5. Results

### 5.1 Synthetic test (pipeline check, run before filming)

Four videos were rendered from the same simulator with known truth (`make_synthetic.py`; E = 15 kPa,
nu = 0.30, zeta = 1.2e6, tau = 0.20 s) on a finer mesh than the fit uses, then run through the whole pipeline.

Fitted on `pressA`, `pressB` and the 1 kg `load` clip; `holdout` (a faster, deeper press) was never used in the fit.

| quantity | truth | fitted | error |
|---|---|---|---|
| E (Pa) | 15000 | 14200 | -5 % |
| nu | 0.300 | 0.303 | +1 % |
| zeta (N s/m^4) | 1.20e6 | 1.24e6 | +4 % |
| relaxation time tau = zeta H^2 / E (s) | 0.200 | 0.219 | +9 % |

Held-out prediction (fresh press): RMSE of top height after release 0.44 mm (2.8 % of the 15.6 mm compression),
RMSE of the three silhouette widths 0.13 mm. A 1D exponential-recovery baseline (one fitted time constant, given the
measured compression) gets 0.18 mm on the height.

How to read these numbers honestly:

* The formal 1-sigma from the fit (0.5 % on E, 1.4 % on zeta) is **much smaller than the actual error** (5 to 9 %).
  It only reflects random noise. Systematic effects dominate: the truth uses a finer mesh (6x6x8) than the fit
  (4x4x5), and the synthetic video has a 0.4 mm bias in the measured top edge while the plate touches.
  Do not quote the formal sigma as the real uncertainty.
* Numerical error is small: against a much finer reference (8x8x10 mesh, more time steps) the fit mesh differs
  by 0.05 mm in height and 0.15 mm in widths, so most of the 0.44 mm held-out height error is the measurement
  bias above, not the solver.
* For the height curve alone, the one-parameter baseline is as good or better here, because recovery in this
  synthetic case is almost a single exponential and the baseline absorbs the measurement bias. The 3D model's added
  value is what the baseline cannot do: predict the bulge (widths), identify a force-calibrated stiffness from the load
  clip, and respond to a different loading history.

**Limits of this test.** It checks that the code recovers parameters when the model is right. It cannot show
that a neo-Hookean/viscous model describes real foam. It is the "inverse crime" setting (same model family
generates and fits the data); the finer truth mesh only partly mitigates that.

### 5.2 Real object

**Object.** A rectangular foam cleaning sponge, trimmed flat on top and bottom, pale yellow, single
material. Dimensions used: W = 70 mm (camera-facing width), D = 40 mm (assumed depth, not directly
measured from this side-on view), rest height H0 = 55.8 mm (measured from video, median of the first
clip's rest frames). No known-weight (load) clip was captured, so `E` is **fixed, not fitted**
(see section 2/3): only the ratio `zeta/E`, i.e. the relaxation time `tau`, and `nu` are identified from
these displacement-only clips.

**Capture.** Phone camera, side-on, level with the table, fixed on a support. Dark cloth backdrop, dark
plate for the press, both chosen so the sponge's yellow-beige colour is the only thing in that colour
range in frame. Two training clips (`press1`, `run1`) and one held-out clip (`run2`), each: ~1-2 s rest,
a manual hand-press with a flat dark board, held briefly, lifted quickly, then 2-3 s of recovery filmed.

**Fit** (robust least squares on `press1` + `run1`, `--pset` enabled):

| quantity | fitted value | 1-sigma |
|---|---|---|
| E (Pa) | 30000 (fixed, not identified - no load clip) | - |
| nu | 0.063 | +/- 0.019 |
| zeta (N s/m^4) | 6.34e6 | factor exp(+/-0.157) |
| relaxation time tau = zeta H^2 / E (s) | 0.659 | - |
| permanent-set coefficient | 0.101 | +/- 0.009 |
| reduced chi^2 | 23.7 | - |

The permanent-set coefficient is a small extension added specifically for this object (see note below):
the stress-free height shrinks by `pset x (deepest compression fraction reached)`, so the model relaxes
toward a slightly shorter rest state after a press instead of always returning to the original height.
Without it, the fit could not reproduce the video at all (see "Why a plain elastic model failed" below).

**Held-out prediction** (`run2`, never used in the fit): RMSE of top height after release 4.69 mm
(13.8% of the 33.9 mm compression in that clip), RMSE of the three silhouette widths 2.67 mm. The 1D
exponential-recovery baseline gets 4.77 mm on height, essentially the same as the 3D model.
Figure: `real/holdout_result_compare.png`. Demo video: `real/demo_final.mp4`.

**How to read these numbers honestly.**

* **The reduced chi^2 (23.7) is far from 1.** The model does not fully explain the video. The main
  reason is visible directly in the data: all three clips were pressed to 54-62% compression, well past
  the ~45% limit where this solver and material model are considered reliable (section 6, assumption 6).
  I did not reshoot to fix this, so this result should be read as "model behaviour outside its stated
  valid range," not as a clean test of the model.
* **The 3D model barely beats the 1-parameter baseline on height** (4.69 mm vs 4.77 mm). Unlike the
  synthetic test (section 5.1), here the model's advantage is not demonstrated by the height RMSE. Its
  real contribution on this object is the width/bulge prediction and the explicit, interpretable
  parameters (nu, tau, permanent set), not a lower height error.
* **nu = 0.063 is small** and not very physically informative here: the widths drift by several mm over
  the course of the clip in a way that is not symmetric around the press (see `press1_signals.png`,
  `run1`/`run2` plots) - most likely a camera focus/exposure change during filming rather than a real
  volume-preserving bulge. nu is therefore fitted mostly to noise in this dataset and should not be
  quoted as a measured material property.
* **E is an assumption, not a result.** No known-weight clip was captured, so 30 kPa is simply the
  fixed value the fit used; only `tau` (hence `zeta` given that `E`) is actually constrained by the
  video.
* **Why a plain elastic model failed, and the permanent-set fix.** The sponge's measured recovery does
  not go back to its original height: it snaps back most of the way within ~0.1-0.3 s and then stays
  flat a few mm short of the starting height for the rest of the clip (visible directly in
  `press1_signals.png`). A pure elastic model (section 1) can only produce a single relaxation curve
  back to the *original* rest height, so fitting it to this data forced an unphysically slow `tau`
  (around 1.9 s) that compromised between the fast snap-back and the permanent offset, fitting neither
  well. I added one parameter, `pset`, so the model's stress-free height can shrink after a deep press
  (implemented as a multiplicative decomposition of the deformation gradient in the loading direction,
  `deform_sim.py`, `internal_force(..., zinv=...)`). This is a real extension beyond the original model
  in section 1, added because the data needed it, not a standard part of the compressible neo-Hookean
  model; it is phenomenological (it does not model foam's actual plasticity/cell-collapse mechanism, it
  just lets the rest length move).

## 6. Assumptions

1. Side view is a good orthographic proxy; the object is symmetric front to back (depth D is a measured constant).
2. Homogeneous, isotropic material; one `E`, `nu`, `zeta` for the whole body.
3. Neo-Hookean elasticity. Real foam has a stress plateau and hysteresis that this does not have.
4. Damping is viscous drag proportional to nodal velocity, with no inertia. It is a proxy for internal viscosity: a
   single time constant is not the same as frequency-dependent damping.
5. Sticky contact at table and plate; the plate is flat, rigid, covers the whole top face and is lifted away quickly.
6. Compression below about 45 % of the height (the solver and the material model are not reliable beyond that).
7. The weight in a load clip acts as uniform pressure, not as a rigid body.
8. No gravity on the object itself (negligible for light foam).

## 7. What the model can and cannot capture

Can: the compression and the recovery transient of a soft block under a flat plate, the barrel bulge, the effect
of press depth and speed (through the prescribed plate motion), and a force-calibrated stiffness if a load clip
is available.

Cannot: hysteresis and permanent set, rate-dependent stiffening, the Mullins effect, friction slip at the
table, inertial effects (jiggle) of heavier or stiffer objects, anisotropy or inhomogeneity, and any motion outside
the image plane. Only a silhouette is observed, so internal strain is unobserved. Mesh error: coarse mesh and
first-order time stepping give errors of order 1 mm in height against a converged reference.

## 8. Reproduce

```
pip install -r requirements.txt
python make_synthetic.py --out synth                      # optional: synthetic test videos
python deform_sim.py --selftest                           # solver check
python extract_signals.py clip.mp4 --out clip.npz --obj_width_mm 60 --seed X,Y --plot clip.png
python extract_signals.py load.mp4 --out load.npz --obj_width_mm 60 --mode load --mass_g 500 --seed X,Y
python fit_params.py --clips clipA.npz clipB.npz load.npz --out fit.json --report fit_report.png
python validate_holdout.py --fit fit.json --clip holdout.npz --out holdout
python make_demo.py --fit fit.json --clip holdout.npz --out demo.mp4
```

For an object with visible permanent set (does not fully recover; see section 5.2), add `--pset` to the
`fit_params.py` call.

## 9. Credits

Written with Claude (Anthropic) as a coding assistant. Libraries: NumPy, SciPy (sparse linear algebra,
least squares), OpenCV (video I/O, segmentation), Matplotlib. FEM formulation follows standard references
(neo-Hookean tetrahedral FEM, e.g. Sifakis and Barbic, "FEM simulation of 3D deformable solids", SIGGRAPH
2012 course). No external simulator code is used.