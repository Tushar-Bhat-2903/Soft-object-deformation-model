"""make_synthetic.py

Create synthetic side-view videos from the simulator with KNOWN material parameters, to test the whole
pipeline (video -> signals -> fit -> held-out prediction). The truth uses a finer mesh than the fit does,
so the test is not completely circular, but it is still the same family of model: see README, 'Limits of
the synthetic test'.

Usage:  python make_synthetic.py --out synth
"""
import argparse
import json
import os

import cv2
import numpy as np

from deform_sim import Block, simulate, make_press_scenario

W_MM, D_MM, H_MM = 60.0, 60.0, 50.0
PX_PER_MM = 5.0
FPS = 60
TRUTH = {"E_Pa": 15e3, "nu": 0.30, "zeta": 1.2e6}
FRAME = (720, 1280)
CX, YBOT = 640, 600


def draw(states_t, block, plate_h_mm=None, load=False, rng=None):
    img = np.full((FRAME[0], FRAME[1], 3), 70, np.uint8)
    cv2.rectangle(img, (0, YBOT), (FRAME[1], FRAME[0]), (110, 110, 110), -1)
    g = states_t.reshape(block.nx + 1, block.ny + 1, block.nz + 1, 3)
    zk = g[:, :, :, 2].mean(axis=(0, 1))
    xmin = g[:, :, :, 0].min(axis=(0, 1))
    xmax = g[:, :, :, 0].max(axis=(0, 1))
    s = PX_PER_MM * 1e3
    left = np.stack([CX + xmin * s, YBOT - zk * s], axis=1)
    right = np.stack([CX + xmax * s, YBOT - zk * s], axis=1)
    poly = np.concatenate([left, right[::-1]]).astype(np.int32)
    cv2.fillPoly(img, [poly], (30, 140, 245))
    htop = g[:, :, block.nz, 2].max() * 1e3
    if plate_h_mm is not None:
        yp = int(YBOT - plate_h_mm * PX_PER_MM)
        cv2.rectangle(img, (CX - 260, yp - 30), (CX + 260, yp), (200, 80, 40), -1)
        cv2.rectangle(img, (CX - 20, yp - 200), (CX + 20, yp - 30), (200, 80, 40), -1)
    if load:
        yp = int(YBOT - htop * PX_PER_MM)
        cv2.rectangle(img, (CX - 140, yp - 90), (CX + 140, yp), (140, 140, 140), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng is not None:
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return img


def write_clip(path, block, sc, load=False, seed=0):
    rng = np.random.default_rng(seed)
    r = simulate(block, TRUTH["E_Pa"], TRUTH["nu"], TRUTH["zeta"], sc, keep_states=True, substeps=8)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME[1], FRAME[0]))
    for i in range(len(sc["t"])):
        ph = None
        if sc["mode"] == "kinematic" and np.isfinite(sc["u"][i]):
            ph = sc["u"][i] * 1e3
        vw.write(draw(r["states"][i], block, ph, load=(sc["mode"] == "load" and bool(sc["load_on"][i])), rng=rng))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    blk = Block(W_MM * 1e-3, D_MM * 1e-3, H_MM * 1e-3, 6, 6, 8)
    H = H_MM * 1e-3
    clips = {
        "pressA": make_press_scenario(FPS, H, 0.012, t_down=0.30, t_hold=0.50, t_total=3.0),
        "pressB": make_press_scenario(FPS, H, 0.018, t_down=0.20, t_hold=0.30, t_total=3.0),
        "holdout": make_press_scenario(FPS, H, 0.015, t_down=0.15, t_hold=0.40, t_total=3.0),
    }
    for k, sc in clips.items():
        write_clip(os.path.join(a.out, k + ".mp4"), blk, sc)
        print("wrote", k)
    t = np.arange(0.0, 4.0, 1.0 / FPS)
    on = (t >= 0.3) & (t < 2.0)
    sc = {"t": t, "mode": "load", "load_on": on, "mass_kg": 1.0}
    write_clip(os.path.join(a.out, "load.mp4"), blk, sc, load=True)
    json.dump({"truth": TRUTH, "W_mm": W_MM, "D_mm": D_MM, "H_mm": H_MM, "load_mass_g": 1000, "seed_xy": [CX, YBOT - 125],
               "tau_slow_s": TRUTH["zeta"] * H ** 2 / TRUTH["E_Pa"]}, open(os.path.join(a.out, "truth.json"), "w"), indent=1)
    print("wrote load and truth.json")


if __name__ == "__main__":
    main()
