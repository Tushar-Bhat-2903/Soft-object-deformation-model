"""make_demo.py

Builds a three-panel comparison video: real footage, simulation, overlay, with a height graph
underneath. The simulated panel is a real simulation driven by the plate's measured motion and
the fitted parameters, not a replay of what the video showed.

Usage:
  python make_demo.py --fit fit.json --clip holdout.npz --out demo.mp4
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

from fitlib import load_clip, run_clip


def outline_and_wire(states_t, block, cx, ybot, s_m):
    """Project the 3D mesh at one instant into the video's 2D pixel space: a filled silhouette
    plus a wireframe of the front face."""
    g = states_t.reshape(block.nx + 1, block.ny + 1, block.nz + 1, 3)
    zk = g[:, :, :, 2].mean(axis=(0, 1))
    xmin = g[:, :, :, 0].min(axis=(0, 1))
    xmax = g[:, :, :, 0].max(axis=(0, 1))

    def px(x, z):
        return np.stack([cx + x / s_m, ybot - z / s_m], axis=-1)
    left = px(xmin, zk)
    right = px(xmax, zk)
    poly = np.concatenate([left, right[::-1]]).astype(np.int32)
    front = g[:, 0, :, :]
    lines = []
    for k in range(block.nz + 1):
        lines.append(px(front[:, k, 0], front[:, k, 2]).astype(np.int32))
    for i in range(block.nx + 1):
        lines.append(px(front[i, :, 0], front[i, :, 2]).astype(np.int32))
    return poly, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="fit.json")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--video", default=None, help="override the video path stored in the clip")
    a = ap.parse_args()
    fit = json.load(open(a.fit))
    c = load_clip(a.clip, 1)
    mesh = tuple(fit["mesh"])
    r = run_clip((c, fit["E_Pa"], fit["nu"], fit["zeta"], mesh, a.substeps, True, fit.get("pset", 0.0)))
    states = r["states"]
    block = r["block"]
    vpath = a.video or str(c["video"])
    cap = cv2.VideoCapture(vpath)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    i0 = int(round(float(c["t_start"]) * float(c["fps"])))
    frames = frames[i0:i0 + len(c["t"])]
    s_mm = float(c["mm_per_px"])
    s_m = s_mm * 1e-3
    cx, ybot = float(c["cx_px"]), float(c["ybot_px"])
    Wp = float(c["W_mm"]) / s_mm
    Hp = float(c["H0_mm"]) / s_mm
    x0, x1 = int(cx - 1.2 * Wp), int(cx + 1.2 * Wp)
    y0, y1 = int(ybot - 1.4 * Hp), int(ybot + 0.12 * Hp)
    pw, ph = 480, int(480 * (y1 - y0) / (x1 - x0))
    T = min(len(frames), len(c["t"]))
    hobs = c["h_mm"]
    hsim = 1e3 * r["h_top"]
    fps = float(c["fps"])
    strip_h = 170
    W_total = pw * 3
    tmin, tmax = float(c["t"][0]), float(c["t"][T - 1])
    ymin = min(hobs.min(), hsim.min()) - 2
    ymax = max(hobs.max(), hsim.max()) + 2

    def cur(tv, hv):
        return (int(10 + (tv - tmin) / (tmax - tmin) * (W_total - 20)), int(strip_h - 20 - (hv - ymin) / (ymax - ymin) * (strip_h - 40)))
    tmp = a.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_total, ph + strip_h))
    for i in range(T):
        real = frames[i].copy()
        sim = np.full_like(real, 28)
        poly, lines = outline_and_wire(states[i], block, cx, ybot, s_m)
        u = c["u_mm"][i]
        for img in (sim,):
            cv2.fillPoly(img, [poly], (40, 140, 230))
            for l in lines:
                cv2.polylines(img, [l], False, (20, 70, 140), 1)
            cv2.line(img, (0, int(ybot)), (img.shape[1] - 1, int(ybot)), (90, 90, 90), 1)
            if np.isfinite(u) and str(c["mode"]) == "kinematic":
                yp = int(ybot - u / s_mm)
                cv2.rectangle(img, (int(cx - 0.65 * Wp), yp - 14), (int(cx + 0.65 * Wp), yp), (170, 170, 170), -1)
            if str(c["mode"]) == "load" and bool(c["load_on"][i]):
                yp = int(ybot - hsim[i] / s_mm)
                cv2.rectangle(img, (int(cx - 0.4 * Wp), yp - 30), (int(cx + 0.4 * Wp), yp), (170, 170, 170), -1)
        ov = real.copy()
        cv2.polylines(ov, [poly], True, (255, 255, 0), 2)
        panels = []
        for img, lab in ((real, "REAL"), (sim, "SIMULATED (fitted model)"), (ov, "OVERLAY")):
            p = cv2.resize(img[y0:y1, x0:x1], (pw, ph))
            cv2.putText(p, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append(p)
        top = np.concatenate(panels, axis=1)
        strip = np.full((strip_h, W_total, 3), 245, np.uint8)
        pts_o = np.array([cur(c["t"][j], hobs[j]) for j in range(T)], np.int32)
        pts_s = np.array([cur(c["t"][j], hsim[j]) for j in range(T)], np.int32)
        cv2.polylines(strip, [pts_o], False, (90, 90, 90), 2)
        cv2.polylines(strip, [pts_s], False, (0, 0, 220), 2)
        cx_i = cur(c["t"][i], hobs[i])[0]
        cv2.line(strip, (cx_i, 5), (cx_i, strip_h - 5), (200, 120, 0), 1)
        cv2.putText(strip, "top height: video (grey) vs model (red)   t=%.2fs" % c["t"][i], (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        vw.write(np.concatenate([top, strip], axis=0))
    vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
