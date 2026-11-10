# Learning a soft object's deformation from video

I filmed a foam sponge being pressed and released, and built a simple 3D physics model of it.
I then used the video to figure out the sponge's stiffness and damping (by fitting the model to
what the video shows), and tested the fitted model on a fresh press it had never seen.

**Pipeline:** `extract_signals.py` (turns a video into numbers) -> `fit_params.py` (fits the model
to those numbers) -> `validate_holdout.py` (tests the fit on a new clip) -> `make_demo.py` (makes a
side-by-side comparison video).

## 1. The model, in plain terms

The sponge is represented as a small 3D block made of tetrahedra (like a rough, chunky 3D mesh),
not a single rigid shape. Each little piece of the block can stretch and compress, following a
standard rubber-like material law (how hard it pushes back depends on how much it's squeezed, set
by two numbers: stiffness `E` and Poisson's ratio `nu`, which controls how much it bulges sideways
when squeezed). There's also a damping/drag term `zeta` that controls how fast it springs back
rather than bouncing forever.

The bottom of the block is glued to the table. A flat plate presses the top down and then lifts
away. I ignore inertia, since the sponge is too light and slow-moving for that to matter; it just
relaxes back according to its stiffness and damping, like a spring in honey.

I also added one extra piece specific to this real sponge: after a hard press it didn't fully
spring back to its original height, so I let the model's "rest" height shrink a little after a
deep press, instead of always snapping back to exactly where it started.

## 2. What the 3D mesh actually looks like

The block above isn't just a flat side-view shape. It's a real 3D grid of points, 150 of them,
connected into little tetrahedra and simulated in all three dimensions. The image below renders
that mesh from an angle at four moments of a real press: at rest, at the deepest point of the
press, right after the plate lifts off, and once it's settled.

![3D mesh at rest, deepest press, just after release, and recovered](real/mesh3d.png)

You can see the top layer drop during the press, the sides bulge out slightly (that's the
Poisson effect), and the top not quite return to its starting height afterward (the permanent
set). Made with `visualize_mesh3d.py`, which re-renders any clip this way:
```
python visualize_mesh3d.py --fit real/fit.json --clip real/press1.npz --out real/mesh3d.png
```

## 3. What I measured from the video, and why

Everything comes from the sponge's outline (silhouette) in a fixed side-on video:

- **Height over time** -> tells the model how fast the sponge recovers, which sets the damping.
- **Width at a few heights** -> how much it bulges sideways when squeezed, which sets Poisson's ratio.
- **Known object width (ruler)** -> converts pixels to millimetres.
- **When the plate touches/leaves** -> tells the model exactly when to start and stop pressing.

A video only shows displacement, not force. That means with hand-pressed clips alone, I can't
separately know the stiffness and the damping — only their ratio (which sets how fast it recovers).
To get the actual stiffness, you'd need a clip with a known weight resting on it. I didn't capture
one, so the stiffness number in my results is an assumption, not something measured — only the
damping (and the Poisson's ratio) are properly fitted from video.

## 4. How the fit works

I simulate the block under the plate motion read from the video, and compare the simulated height
and widths against what the video actually shows. An optimizer adjusts the model's parameters
until the simulation matches the video as closely as possible. I do this on two clips at once (so
the fit isn't overfitting to one clip's quirks), then test the result on a third, completely
separate clip the fit never saw.

## 5. Results

The object is a rectangular foam sponge, about 70 mm wide and 56 mm tall, single material
throughout.

Fitted from two training clips:
- Poisson's ratio: 0.06 (fairly tightly determined)
- Recovery time: about 0.66 seconds
- "Permanent set" amount: about 10% of the deepest press depth
- Stiffness: fixed at 30 kPa, an assumption, since no weight clip was filmed so this wasn't measured

Tested on a held-out clip the fit never saw, the model predicted the recovery height to within
about 4.7 mm of the real video. That's only slightly better than the simplest possible
comparison, a one-number exponential recovery curve, which got about 4.8 mm.

I'll be honest about this one: the model works and runs end to end, but it isn't a strong fit. I
pressed the sponge too deep in all three clips, past the depth where this kind of model is
expected to stay accurate, so this result mostly shows the model's behaviour outside its comfort
zone rather than a clean validation. See `real/` for the figures and the side-by-side demo video
(`real/demo_final.mp4`).

## 6. Assumptions

1. Side view stands in for the full 3D shape; depth is assumed, not measured from this camera angle.
2. One uniform material throughout the sponge (no soft/hard regions).
3. A standard rubber-like (neo-Hookean) stretch law — real foam actually behaves a bit differently
   (it has a "squish plateau" and doesn't return energy perfectly), which this model simplifies.
4. Damping is simple drag, not a detailed model of how foam cells actually dissipate energy.
5. The sponge sticks to the table and to the plate (no sliding).
6. The model isn't reliable once compression goes much past about 45% of the sponge's height — and
   my clips went deeper than that, which is the main weakness of this result.
7. No separate weight clip was filmed, so stiffness (E) is an assumption, not a measured result.

## 7. What it can and can't do

It can show a soft block compressing and springing back, bulging sideways, and roughly not fully
recovering after a hard press.

It can't capture real foam's more complex behaviour: energy loss between presses, getting
stiffer if pressed fast, sliding/friction effects, or anything not visible from one camera angle.
Only the outline is measured, so nothing inside the sponge is actually observed.

## 8. Interactive demo

`simulator.html` is a standalone, no-install interactive version: the same mesh shape and
fitted numbers (stiffness, damping, recovery time, permanent set), running live in a browser.
Click and drag the block to press it, let go to watch it recover. It's a real-time
approximation (a mass-spring model), not the actual solver below (that one is too slow to run
at 60 frames per second), but it's tuned to behave the same way the fit found.

To open it: it's a webpage, not a program, so double-clicking it may try to open it in a code
editor instead of a browser depending on your computer's settings. Right-click `simulator.html`
-> "Open with" -> choose Chrome, Edge, or any browser. It then runs entirely in that browser tab,
no install or internet connection needed.

## 9. How to run it

```
pip install -r requirements.txt
python deform_sim.py --selftest                 # quick solver sanity check, no video needed

python extract_signals.py clip.mp4 --pick                      # click the sponge to get a seed point
python extract_signals.py clip.mp4 --out clip.npz --obj_width_mm 70 --obj_depth_mm 40 --seed X,Y --plot clip.png --debug_video clip_dbg.mp4

python fit_params.py --clips clipA.npz clipB.npz --out fit.json --report fit_report.png --pset
python validate_holdout.py --fit fit.json --clip holdout.npz --out holdout
python make_demo.py --fit fit.json --clip holdout.npz --out demo.mp4
```

Add a load clip (known weight resting on the plate) with `--mode load --mass_g <grams>` on the
`extract_signals.py` step, and include it in `fit_params.py --clips` to actually measure stiffness
instead of assuming it.

The `real/` folder has my own run of this (videos, extracted data, the fit, and the demo video).
The `synth/` folder is a sanity check I ran first on computer-generated videos with known-true
answers, to make sure the whole pipeline recovers the right numbers before trusting it on real
footage.

https://drive.google.com/drive/u/0/folders/1k4xeaoR7jhPRRhNyPI6G49Lbctl8CA-T
The photos and videos are attached in this google drive link for access 

## 10. Credits

Built with Claude (Anthropic) as a coding assistant. Libraries: NumPy, SciPy, OpenCV, Matplotlib.
The physics approach (tetrahedral mesh, neo-Hookean material) follows standard finite-element
methods used for simulating soft bodies. No external simulator or pretrained model was used.
