# Capture guide (do this first, the rest of the pipeline depends on it)

## Object
Soft, recovers on its own, rectangular is easiest (sponge, foam block, memory-foam cube). Weigh it and measure
width (in the image plane), depth (into the image) and height with a ruler.

## Setup
* Phone on a tripod or propped against something, **side view**, camera level with the table, object centred.
* Plain background and table whose colours differ strongly from the object. A bright object (yellow/orange/pink/green sponge)
  on a grey or white table is ideal. The plate (a flat book or board) must NOT be the object's colour.
* Do not move the camera or the zoom between clips. Fixed exposure if the phone allows it.
* 60 fps minimum, 120 or 240 fps slow-motion is better (recovery takes tenths of a second). Good light, no flicker.
* Start every clip with the object untouched for at least half a second.

## Clips (name them as below)
1. `pressA`: press the plate straight down about 25 % of the height in about 0.3 s, hold about 0.5 s, then lift the plate
   **quickly and clear of the object** (do not let it ride up with the foam), keep filming for 2 s.
2. `pressB`: same idea, different depth (about 35 %) and speed.
3. `load`: place a known weight (a kitchen-scale-weighed book or a few water bottles, 0.5 to 1.5 kg, flat bottom, smaller
   than or equal to the top face) on the object, wait about 2 s, remove it quickly, film 2 s more. Writes down the mass.
   This clip is what lets E (stiffness in pascals) be identified.
4. `holdout`: another press (e.g. different depth and speed). **Never use it for fitting.**
Keep compression below about 45 % of the height.

## Commands
```
python extract_signals.py video.mp4 --pick                       # click on the object, note the seed x,y
python extract_signals.py pressA.mp4 --out pressA.npz --obj_width_mm W --obj_depth_mm D --seed X,Y --plot pressA.png --debug_video pressA_dbg.mp4
python extract_signals.py load.mp4 --out load.npz --obj_width_mm W --mode load --mass_g M --seed X,Y --plot load.png
python fit_params.py --clips pressA.npz pressB.npz load.npz --out fit.json --report fit_report.png
python validate_holdout.py --fit fit.json --clip holdout.npz --out holdout
python make_demo.py --fit fit.json --clip holdout.npz --out demo.mp4
```
Always check `*_dbg.mp4` (green contour on the object, red line on the top) and the printed contact/release times
before trusting a fit. Override with `--t_contact` / `--t_release` (seconds) if they are wrong.
