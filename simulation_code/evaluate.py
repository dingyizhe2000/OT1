#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, re, csv, math, random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, t as t_dist

from network import ICNN, clip_parameters, compute_gradients

BASE_SEED = 20260527


def set_eval_seed(seed: int = BASE_SEED):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Your model & helpers
# ----------------------------

def load_model(input_size, hidden_size, act, model_path, device):
    m = ICNN(input_size, hidden_size, 1, act).to(device)
    state = torch.load(model_path, map_location=device)
    # accept both plain state_dict and a dict with "state_dict" key
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        m.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        m.load_state_dict(state["state_dict"])
    else:
        m.load_state_dict(state)
    m.eval()
    clip_parameters(m)
    return m

# ----------------------------
# L2 metrics (using your API, with type fixes)
# ----------------------------

def sign_square(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x ** 2)

def l2_evaluate(model, x_P_l2: torch.Tensor, label_P_l2: np.ndarray, device) -> float:
    # gradients -> numpy
    grad = compute_gradients(model, x_P_l2.to(device)).cpu().numpy()
    # ensure shapes match
    grad = np.asarray(grad, dtype=float)
    lab  = np.asarray(label_P_l2, dtype=float)
    return float(np.mean((lab - grad) ** 2) ** 0.5)

def L2_loss(model, transform_method: str, measure_P: str, input_size: int, device, seed: int = BASE_SEED) -> float:
    set_eval_seed(seed)
    df = 6
    t_distribution = torch.distributions.StudentT(df)

    # sample X
    if measure_P == "normal":
        x_l2 = torch.randn(10_000, input_size, device=device, dtype=torch.float32).requires_grad_(True)
    elif measure_P == "t":
        x_l2 = t_distribution.sample((10_000, input_size)).to(device).requires_grad_(True)
    else:
        raise ValueError(f"Unknown measure_P: {measure_P}")

    # labels (NUMPY) per transform
    if transform_method == "CDF":
        if measure_P == "normal":
            label_l2 = norm.cdf(x_l2.detach().cpu().numpy())
        else:  # t
            label_l2 = t_dist.cdf(x_l2.detach().cpu().numpy(), df)
    elif transform_method == "piecewise_linear":
        # y = z                         for |z| <= 1
        # y = sgn(z) * (0.5(|z|-1) + 1) for 1 < |z| <= 2
        # y = sgn(z) * (2(|z|-2) + 1.5) for |z| > 2
        z = x_l2.detach()
        abs_z = z.abs()
        sgn_z = torch.sign(z)

        y = torch.empty_like(z)

        mask1 = abs_z <= 1.0
        y[mask1] = z[mask1]

        mask2 = (abs_z > 1.0) & (abs_z <= 2.0)
        y[mask2] = sgn_z[mask2] * (0.5 * (abs_z[mask2] - 1.0) + 1.0)

        mask3 = abs_z > 2.0
        y[mask3] = sgn_z[mask3] * (2.0 * (abs_z[mask3] - 2.0) + 1.5)

        label_l2 = y.cpu().numpy()
    elif transform_method == "quadratic":
        label_l2 = sign_square(x_l2.detach()).cpu().numpy()
    else:
        raise ValueError(f"Unknown transform_method: {transform_method}")

    return l2_evaluate(model, x_l2, label_l2, device=device)

# ----------------------------
# Directory walking & evaluation
# ----------------------------

SCENARIO_RE = re.compile(r'^(?P<measure>normal|t)_(?P<transform>CDF|piecewise_linear|quadratic)_n_(?P<n>\d+)$')
DIN_RE      = re.compile(r'^d=(?P<d>\d+)$')

def eval_folder(scenario_dir: str, input_size: int, hidden_size: int, device: str, act: str = "relu", seed: int = BASE_SEED):
    """
    Evaluate all model_*.pth in `scenario_dir`, write one CSV inside it.
    CSV columns: model_idx, L2_loss, input_size, measure_P, transform_method, n, k
    """
    base = os.path.basename(scenario_dir)
    m = SCENARIO_RE.match(base)
    if not m:
        print(f"[skip] {scenario_dir} (name not recognized)")
        return

    measure   = m.group("measure")
    transform = m.group("transform")
    n_val     = int(m.group("n"))

    # discover models
    model_files = sorted(
        (f for f in os.listdir(scenario_dir) if f.startswith("model_") and f.endswith(".pth")),
        key=lambda s: int(re.search(r'(\d+)\.pth$', s).group(1)) if re.search(r'(\d+)\.pth$', s) else 10**9
    )
    if not model_files:
        print(f"[warn] no models in {scenario_dir}")
        return

    out_csv = os.path.join(
        scenario_dir,
        f"L2_error_measure_P={measure}_transform_method={transform}.csv"
    )
    os.makedirs(scenario_dir, exist_ok=True)

    use_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_idx", "L2_loss"])
        for fname in model_files:
            idx_m = re.search(r'model_(\d+)\.pth$', fname)
            if not idx_m: 
                continue
            idx = int(idx_m.group(1))
            path = os.path.join(scenario_dir, fname)
            try:
                model = load_model(input_size, hidden_size, act, path, use_device)
                loss  = L2_loss(model, transform, measure, input_size, use_device, seed=seed)
            except Exception as e:
                print(f"[error] {path}: {e}")
                loss = math.nan
            writer.writerow([idx, loss])

    print(f"[done] {scenario_dir} → {os.path.relpath(out_csv)}")

def eval_tree(root_dir: str, hidden_size: int = 15, device: str = "cpu", act: str = "relu", seed: int = BASE_SEED):
    """
    Walk a tree like:
      root_dir/
        ICNN_elu/
          d=1/
            normal_CDF_n_100/
              model_0.pth ...
          d=2/
            ...
    Evaluate every scenario folder.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # find input dimension from a parent "d=<int>"
        parts = dirpath.split(os.sep)
        d_in = None
        for p in parts:
            m = DIN_RE.match(p)
            if m:
                d_in = int(m.group("d"))
                break
        if d_in is None:
            continue

        base = os.path.basename(dirpath)
        if SCENARIO_RE.match(base):
            eval_folder(dirpath, d_in, hidden_size, device, act, seed=seed)

# ----------------------------
# Example CLI usage
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate L2 loss for ICNN model folders and write CSVs.")
    ap.add_argument("--root", required=True, help="Root folder (e.g., OT_Hutter/ICNN_elu)")
    ap.add_argument("--hidden", type=int, default=15, help="Hidden size used by the ICNN (default: 64)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Evaluation device (default: cpu)")
    ap.add_argument("--act", help="activation function")
    ap.add_argument("--seed", type=int, default=BASE_SEED, help="seed for L2 test sample generation.")
    args = ap.parse_args()

    eval_tree(args.root, hidden_size=args.hidden, device=args.device, act=args.act, seed=args.seed)
