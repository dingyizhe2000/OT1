#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, re, torch
import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats.mstats import ks_2samp

# Regex to parse "..._iterationXXXX.pt"
_ITER_RE = re.compile(r"_iteration(\d+)\.pt$")


def _list_ckpts_for_act(drug_name: str, act: str, save_path: str):
    """
    Return (paths, iters) sorted by iteration for checkpoints:
        save_path/drug_name/{act}_iterationXXXX.pt
    """
    drug_dir = save_path
    if not os.path.isdir(drug_dir):
        return [], []

    paths = []
    iters = []
    for fname in os.listdir(drug_dir):
        if not fname.startswith(act + "_iteration"):
            continue
        m = _ITER_RE.search(fname)
        if not m:
            continue
        it = int(m.group(1))
        paths.append(os.path.join(drug_dir, fname))
        iters.append(it)

    if not iters:
        return [], []

    idx = np.argsort(iters)
    paths = [paths[i] for i in idx]
    iters = [iters[i] for i in idx]
    return paths, iters


def _find_latest_iteration_ckpt(drug_name: str, act: str, save_path: str):
    """
    Scan save_path/drug_name for files named like
        '{act}_iterationXXXX.pt'
    and return (path, iteration) for the largest XXXX. If none, return (None, 0).
    """
    drug_dir = os.path.join(save_path, drug_name)
    if not os.path.isdir(drug_dir):
        return None, 0

    latest_iter = 0
    latest_path = None

    for fname in os.listdir(drug_dir):
        # ensure we only look at this activation
        if not fname.startswith(act + "_iteration"):
            continue
        m = _ITER_RE.search(fname)
        if not m:
            continue
        it = int(m.group(1))
        if it > latest_iter:
            latest_iter = it
            latest_path = os.path.join(drug_dir, fname)

    return latest_path, latest_iter


def load_latest_checkpoint(
    f_model,
    g_model,
    f_optim,
    g_optim,
    drug_name: str,
    save_path: str,
):
    """
    Load the *latest* iteration checkpoint for this drug and this activation
    into the given models and optimizers.

    The activation is inferred from `f_model.activation_name`, so files
    are expected to have names like:

        {activation}_iterationXXXX.pt

    Returns:
        (last_global_iteration, running_fl, nb)

    If no checkpoint is found for this activation, returns (0, 0.0, 0).
    """
    act = getattr(f_model, "activation_name", None)
    if act is None:
        raise AttributeError("f_model must have attribute 'activation_name' to use load_latest_checkpoint.")

    ckpt_path, last_iter = _find_latest_iteration_ckpt(drug_name, act, save_path)
    if ckpt_path is None:
        # nothing to load for this activation
        return 0, 0.0, 0

    device_f = next(f_model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device_f, weights_only=False)

    # load model states
    if "f_model" in ckpt:
        f_model.load_state_dict(ckpt["f_model"].state_dict())
    if "g_model" in ckpt:
        g_model.load_state_dict(ckpt["g_model"].state_dict())

    # load optimizer states
    if "f_optimizer" in ckpt:
        f_optim.load_state_dict(ckpt["f_optimizer"].state_dict())
    if "g_optimizer" in ckpt:
        g_optim.load_state_dict(ckpt["g_optimizer"].state_dict())

    iteration  = int(ckpt.get("iteration", last_iter))
    running_fl = float(ckpt.get("running_fl", 0.0))
    nb         = int(ckpt.get("nb", 0))

    return iteration, running_fl, nb


def save_checkpoint(step, f_model, g_model, f_optim, g_optim, drug_name, save_path,
                    running_fl, nb):  # <<< changed signature
    act = f_model.activation_name

    drug_dir = os.path.join(save_path, drug_name)
    os.makedirs(drug_dir, exist_ok=True)   # <<< ensure dir exists

    ckpt_path = os.path.join(drug_dir, f"{act}_iteration{step}.pt")
    torch.save(
        {
            "iteration": step,
            "f_model": f_model,
            "g_model": g_model,
            "f_optimizer": f_optim,
            "g_optimizer": g_optim,
            "drug": drug_name,
            "activation": act,
            "running_fl": running_fl,   # <<< store stats
            "nb": nb,                   # <<< store stats
        },
        ckpt_path,
    )

# ----------------------------
# Simple distance utilities
# ----------------------------


@torch.no_grad()
def compute_mmd(x: torch.Tensor,
                y: torch.Tensor,
                gammas=None) -> float:
    """
    Unbiased multi-scale MMD^2 with RBF kernels.

    For each gamma, uses the unbiased estimator

        MMD^2_u =
            1/(n(n-1)) * sum_{i != j} k(x_i, x_j)
          + 1/(m(m-1)) * sum_{i != j} k(y_i, y_j)
          - 2/(nm)    * sum_{i,j}    k(x_i, y_j)

    and then averages over all gammas.
    Returns a scalar float (np.nan if everything fails).
    """

    # ---- convert to numpy ----
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=float)

    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y, dtype=float)

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    n = x.shape[0]
    m = y.shape[0]

    def safe_unbiased_mmd(x_, y_, gamma_):
        # need at least 2 samples per group for unbiased estimator
        if n < 2 or m < 2:
            return np.nan
        try:
            Kxx = rbf_kernel(x_, x_, gamma_)
            Kyy = rbf_kernel(y_, y_, gamma_)
            Kxy = rbf_kernel(x_, y_, gamma_)

            # remove diagonals
            #sum_Kxx_off = Kxx.sum() - np.trace(Kxx)
            #sum_Kyy_off = Kyy.sum() - np.trace(Kyy)

            #term_xx = sum_Kxx_off / (n * (n - 1))
            #term_yy = sum_Kyy_off / (m * (m - 1))
            #term_xy = Kxy.mean()  # already 1/(nm) * sum_{i,j}

            mmd2_u = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
            return float(mmd2_u)
        except ValueError:
            return np.nan

    vals = [safe_unbiased_mmd(x, y, g) for g in gammas]
    return float(np.nanmean(vals))


def compute_ks_distance(x, y) -> float:
    """
    Average 1D KS statistic over features.
    x, y: [N, d] tensors or arrays.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=float)

    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y, dtype=float)

    assert x.shape[1] == y.shape[1]
    d = x.shape[1]
    vals = []
    for j in range(d):
        vals.append(ks_2samp(x[:, j], y[:, j]).statistic)
    return float(np.mean(vals))