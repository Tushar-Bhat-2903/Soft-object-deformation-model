"""extract_signals.py

Turn one side-view video of a press-and-release into a .npz file the fitting code can read.
Only the object's outline is tracked, frame by frame:

  h_mm          height of the top surface
  widths_mm     silhouette width at 20%, 50%, 80% of its current height
  u_mm          plate height for kinematic clips (see below)
  load_on       whether the known weight is resting on the object (load clips)

The plate isn't tracked directly: while it's touching the object, its underside is the
object's top surface, so plate height = measured top height during that stretch. Contact is
assumed to start when the top first drops and end near the lowest point reached. Override
with --t_contact / --t_release if that guess is wrong.

Usage (kinematic press clip):
  python extract_signals.py clip1.mp4 --out clip1.npz --obj_width_mm 60 --obj_depth_mm 60 \
      --seed 640,400
Usage (known weight resting on the object):
  python extract_signals.py load.mp4 --out load.npz --obj_width_mm 60 --mode load --mass_g 200 \
      --seed 640,400
Find the seed pixel (x,y of a point on the object in the first frame) with:  --pick
"""
import argparse
import sys

import cv2
import numpy as np
from scipy.signal import savgol_filter

XIS = np.array([0.2, 0.5, 0.8])


def hsv_range_from_seed(frame_bgr, seed, half=7, tol=(12, 70, 70)):
    """Sample the colour around one clicked pixel and turn it into a lo/hi HSV range to track."""
    x, y = seed
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    patch = hsv[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1].reshape(-1, 3).astype(float)
    med = np.median(patch, axis=0)
    lo = np.clip(med - np.array(tol), 0, 255)
    hi = np.clip(med + np.array(tol), 0, 255)
    hi[0] = min(hi[0], 179)
    return lo.astype(np.uint8), hi.astype(np.uint8)


def segment(frame_bgr, lo, hi, min_area=400, anchor=None):
    """Find the object in one frame: colour-threshold, clean up the mask, pick one blob.
    Without anchor, takes the biggest blob (broke once when a hand entered frame). With anchor
    (a box known to always sit on the object), picks whichever blob overlaps it most instead."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m, bool)
    if anchor is not None:
        x0, x1, y0, y1 = anchor
        ov = np.bincount(lab[y0:y1, x0:x1].ravel(), minlength=n)[1:]
        j = 1 + int(np.argmax(ov)) if ov.max() > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[j, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(m, bool)
    return lab == j


def width_at_row(mask, row):
    """How wide the masked object is at one image row, in pixels."""
    r = int(round(row))
    if r < 0 or r >= mask.shape[0]:
        return np.nan
    xs = np.where(mask[r])[0]
    if len(xs) == 0:
        return np.nan
    return float(xs.max() - xs.min() + 1)


def top_row(mask, cx, half_band):
    """Row of the topmost object pixel, averaged over a narrow band so it doesn't jitter."""
    c0 = int(max(0, cx - half_band))
    c1 = int(min(mask.shape[1], cx + half_band + 1))
    sub = mask[:, c0:c1]
    has = sub.any(axis=0)
    if not has.any():
        return np.nan
    tops = np.argmax(sub[:, has], axis=0)
    return float(np.median(tops))


def fill_nan(a):
    """Patch over frames where the object wasn't found by interpolating between good frames."""
    a = np.asarray(a, float).copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return a
    idx = np.arange(len(a))
    a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def smooth(a, win=5):
    """Light Savitzky-Golay smoothing to take pixel-level jitter out of the measured curve."""
    a = fill_nan(a)
    win = min(win, len(a) - (1 - len(a) % 2))
    if win < 5:
        return a
    return savgol_filter(a, win, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--obj_width_mm", type=float, required=False, help="true width of the object in the image plane (mm)")
    ap.add_argument("--obj_depth_mm", type=float, default=None, help="true depth into the image (mm); default = width")
    ap.add_argument("--seed", default=None, help="x,y pixel on the object in the first frame")
    ap.add_argument("--hsv_lo", default=None, help="h,s,v lower bound (overrides --seed)")
    ap.add_argument("--hsv_hi", default=None, help="h,s,v upper bound (overrides --seed)")
    ap.add_argument("--pick", action="store_true", help="click the object in the first frame, print the seed, exit")
    ap.add_argument("--mode", choices=["kinematic", "load"], default="kinematic")
    ap.add_argument("--mass_g", type=float, default=0.0, help="mass of the weight for --mode load")
    ap.add_argument("--t_start", type=float, default=0.0, help="ignore video before this time (s)")
    ap.add_argument("--t_end", type=float, default=None)
    ap.add_argument("--n_rest", type=int, default=12, help="initial frames where the object is at rest")
    ap.add_argument("--t_contact", type=float, default=None, help="override contact start (s, clip time)")
    ap.add_argument("--t_release", type=float, default=None, help="override release time (s, clip time)")
    ap.add_argument("--debug_video", default=None, help="write an annotated mp4 here")
    ap.add_argument("--plot", default=None, help="write a signals plot png here")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit("cannot open video " + a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        sys.exit("no frames")
    i0 = int(round(a.t_start * fps))
    i1 = len(frames) if a.t_end is None else min(len(frames), int(round(a.t_end * fps)))
    frames = frames[i0:i1]

    if a.pick:
        pts = []

        def cb(ev, x, y, fl, pr):
            if ev == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                print("seed:", "%d,%d" % (x, y))
        cv2.imshow("click on the object, press q", frames[0])
        cv2.setMouseCallback("click on the object, press q", cb)
        while cv2.waitKey(50) != ord("q"):
            pass
        return
    if a.obj_width_mm is None:
        sys.exit("--obj_width_mm is required")
    if a.hsv_lo and a.hsv_hi:
        lo = np.array([int(v) for v in a.hsv_lo.split(",")], np.uint8)
        hi = np.array([int(v) for v in a.hsv_hi.split(",")], np.uint8)
    elif a.seed:
        seed = tuple(int(v) for v in a.seed.split(","))
        lo, hi = hsv_range_from_seed(frames[0], seed)
    else:
        sys.exit("give --seed x,y (use --pick to find it) or --hsv_lo/--hsv_hi")
    print("HSV range:", lo, hi, "fps", fps, "frames", len(frames))

    masks = [segment(f, lo, hi) for f in frames]
    n_rest = min(a.n_rest, len(frames))
    bb = []
    for m in masks[:n_rest]:
        ys, xs = np.where(m)
        if len(xs) == 0:
            sys.exit("object not found in the first frames; adjust --seed / HSV range")
        bb.append((xs.min(), xs.max(), ys.min(), ys.max()))
    bb = np.array(bb, float)
    cx = float(np.median((bb[:, 0] + bb[:, 1]) / 2))
    ybot = float(np.median(bb[:, 3]))
    ytop0 = float(np.median(bb[:, 2]))
    # Anchor box: bottom quarter of the object's rest position, used to re-segment every frame.
    ax0, ax1 = int(np.median(bb[:, 0])), int(np.median(bb[:, 1]))
    ay1 = int(ybot)
    ay0 = int(ybot - 0.25 * (ybot - ytop0))
    anchor = (ax0, ax1 + 1, ay0, ay1 + 1)
    masks = [segment(f, lo, hi, anchor=anchor) for f in frames]
    wpx = float(np.median([width_at_row(m, ybot - 0.5 * (ybot - ytop0)) for m in masks[:n_rest]]))
    s = a.obj_width_mm / wpx
    H0_mm = (ybot - ytop0) * s
    print("rest: width %.1f px -> %.4f mm/px ; height %.1f mm ; centre col %.1f ; table row %.1f" % (wpx, s, H0_mm, cx, ybot))

    T = len(frames)
    h = np.full(T, np.nan)
    w = np.full((T, len(XIS)), np.nan)
    ytops = np.full(T, np.nan)
    for i, m in enumerate(masks):
        yt = top_row(m, cx, 0.2 * wpx)
        ytops[i] = yt
        if np.isfinite(yt):
            h[i] = (ybot - yt) * s
            for k, xi in enumerate(XIS):
                w[i, k] = width_at_row(m, ybot - xi * (ybot - yt)) * s
    print("frames with no object found: %d" % int((~np.isfinite(h)).sum()))
    hs = smooth(h)
    ws = np.stack([smooth(w[:, k]) for k in range(len(XIS))], axis=1)
    t = np.arange(T) / fps

    sig = float(np.std(hs[:n_rest] - np.mean(hs[:n_rest])))
    thr = max(0.8, 4 * sig)
    hrest = float(np.mean(hs[:n_rest]))
    below = np.where(hs < hrest - thr)[0]
    if a.t_contact is not None:
        ic = int(round(a.t_contact * fps))
    elif len(below):
        ic = int(below[0])
    else:
        sys.exit("the object never compresses by more than %.2f mm; nothing to fit" % thr)
    imin = int(np.argmin(hs))
    hmin = hs[imin]
    near = np.where(hs <= hmin + 0.3)[0]
    if a.t_release is not None:
        ir = int(round(a.t_release * fps))
    else:
        ir = int(near.max())
    ir = max(ir, ic + 1)
    present = np.zeros(T, bool)
    present[ic:ir + 1] = True
    u = np.where(present, hs, np.inf)
    print("contact at t=%.3f s, release at t=%.3f s, max compression %.1f mm (%.0f%%)" % (
        t[ic], t[ir], hrest - hmin, 100 * (hrest - hmin) / hrest))
    if (hrest - hmin) / hrest > 0.45:
        print("WARNING: compression above 45%% of the height; the FEM proxy is not reliable there")
    if a.mode == "load" and a.mass_g <= 0:
        sys.exit("--mode load needs --mass_g")

    out = a.out or a.video.rsplit(".", 1)[0] + ".npz"
    np.savez(out, t=t, h_mm=hs, h_raw_mm=h, widths_mm=ws, u_mm=u, load_on=present,
             mode=np.array(a.mode), mass_g=a.mass_g, W_mm=a.obj_width_mm,
             D_mm=a.obj_depth_mm if a.obj_depth_mm else a.obj_width_mm, H0_mm=H0_mm, mm_per_px=s, cx_px=cx,
             ybot_px=ybot, xis=XIS, fps=fps, t_contact=t[ic], t_release=t[ir], noise_h_mm=sig,
             video=np.array(a.video), t_start=a.t_start)
    print("saved", out)

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(t, h, ".", ms=3, color="0.6", label="raw")
        ax[0].plot(t, hs, label="smoothed top height")
        ax[0].axvspan(t[ic], t[ir], color="C1", alpha=0.15, label="plate / load present")
        ax[0].set_ylabel("height (mm)")
        ax[0].legend()
        for k in range(len(XIS)):
            ax[1].plot(t, ws[:, k], label="width at %d%% height" % int(100 * XIS[k]))
        ax[1].set_ylabel("width (mm)")
        ax[1].set_xlabel("time (s)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)

    if a.debug_video:
        hgt, wid = frames[0].shape[:2]
        vw = cv2.VideoWriter(a.debug_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wid, hgt))
        for i, (f, m) in enumerate(zip(frames, masks)):
            g = f.copy()
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(g, cnt, -1, (0, 255, 0), 2)
            if np.isfinite(ytops[i]):
                cv2.line(g, (int(cx - 0.2 * wpx), int(ytops[i])), (int(cx + 0.2 * wpx), int(ytops[i])), (0, 0, 255), 2)
            cv2.line(g, (0, int(ybot)), (wid - 1, int(ybot)), (255, 0, 0), 1)
            cv2.putText(g, "t=%.2fs h=%.1fmm%s" % (t[i], hs[i], " [contact]" if present[i] else ""), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            vw.write(g)
        vw.release()
        print("saved", a.debug_video)


if __name__ == "__main__":
    main()
