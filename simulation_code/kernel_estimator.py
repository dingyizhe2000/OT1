#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, csv, argparse
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

BASE_SEED = 20260527

def sign_square(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x > 0, x**2, -x**2)

def sample_X(sample_size: int, measure_P: str, df: float, input_size: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(BASE_SEED)
    if measure_P == "normal":
        return rng.standard_normal((sample_size, input_size)).astype(np.float64)
    elif measure_P == "t":
        return rng.standard_t(df, size=(sample_size, input_size)).astype(np.float64)
    else:
        raise ValueError("measure_P must be 'normal' or 't'.")

def T0_map(x: np.ndarray, measure_P: str, transform_method: str, df: float) -> np.ndarray:
    if transform_method == "piecewise_linear":
        # y = z for |z| ≤ 1
        # y = sgn(z) * (0.5 * (|z| - 1) + 1)       for 1 < |z| ≤ 2
        # y = sgn(z) * (2 * (|z| - 2) + 1.5)       for |z| > 2

        x = np.asarray(x, dtype=float)
        absx = np.abs(x)
        signx = np.sign(x)
        y = np.empty_like(x, dtype=float)

        mask1 = (absx <= 1.0)
        mask2 = (absx > 1.0) & (absx <= 2.0)
        mask3 = (absx > 2.0)

        y[mask1] = x[mask1]
        y[mask2] = signx[mask2] * (0.5 * (absx[mask2] - 1.0) + 1.0)
        y[mask3] = signx[mask3] * (2.0 * (absx[mask3] - 2.0) + 1.5)

        return y
    elif transform_method == "quadratic":
        return sign_square(x)
    elif transform_method == "CDF":
        if measure_P == "normal":
            return stats.norm.cdf(x)
        elif measure_P == "t":
            return stats.t.cdf(x, df)
        else:
            raise ValueError("measure_P must be 'normal' or 't'.")
    else:
        raise ValueError("transform_method must be 'CDF', 'piecewise_linear', or 'quadratic'.")

def squared_euclidean_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    D = x2 + y2 - 2.0 * (X @ Y.T)
    np.maximum(D, 0.0, out=D)
    return D

def ot_permutation(X: np.ndarray, Y: np.ndarray, return_gamma: bool = True):
    D = squared_euclidean_cost(X, Y)
    r, c = linear_sum_assignment(D)
    n = D.shape[0]
    perm = np.empty(n, dtype=int)
    perm[r] = c
    cost = float(D[r, c].sum())
    if return_gamma:
        Gamma = np.zeros((n, n), dtype=np.float64)
        Gamma[r, c] = 1.0
        return perm, Gamma, cost
    else:
        return perm, cost

def empirical_ot_map_from_perm(perm: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y)[np.asarray(perm, dtype=int)]

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    D2 = x2 + y2 - 2.0 * (X @ Y.T)
    np.maximum(D2, 0.0, out=D2)
    return D2

def gaussian_kernel_from_sqdist(sq: np.ndarray, nu_kernel: float) -> np.ndarray:
    return np.exp(-nu_kernel * sq, dtype=np.float64)

def fit_rkhs_weights(K: np.ndarray, Y_tilde: np.ndarray, nu_ridge: float) -> np.ndarray:
    n = K.shape[0]
    K_reg = K.copy()
    K_reg[np.diag_indices(n)] += float(nu_ridge)
    L = np.linalg.cholesky(K_reg)
    Z = np.linalg.solve(L, Y_tilde)
    W = np.linalg.solve(L.T, Z)
    return W  # (n, d_out)

def predict_with_kernel(K_query: np.ndarray, W: np.ndarray) -> np.ndarray:
    return K_query @ W

def select_hyperparams_train_val(X_train, T_emp_train, X_val, T0X_val, nu_kernel_grid, nu_ridge_grid):
    # Precompute kernels needed for each nu_kernel
    sq_tt = pairwise_sq_dists(X_train, X_train)   # (n_tr, n_tr)
    sq_vt = pairwise_sq_dists(X_val,   X_train)   # (n_val, n_tr)

    best_val = np.inf
    best = (None, None)
    best_W = None

    for nu_k in nu_kernel_grid:
        K_tt = gaussian_kernel_from_sqdist(sq_tt, nu_k)
        K_vt = gaussian_kernel_from_sqdist(sq_vt, nu_k)
        for nu_r in nu_ridge_grid:
            W = fit_rkhs_weights(K_tt, T_emp_train, nu_r)
            T_val = K_vt @ W
            val = np.mean(np.sum((T_val - T0X_val) ** 2, axis=1))  # (1/n_val)*||.||_F^2
            if val < best_val:
                best_val = val
                best = (nu_k, nu_r)
                best_W = W
    return best[0], best[1], float(np.sqrt(best_val)), best_W  # rmse on validation

def run_pipeline_split_and_test(sample_size=100,
                                measure_P="normal",
                                transform_method="piecewise_linear",
                                input_size=3,
                                df=5,
                                rng=None,
                                nu_kernel_grid=None,
                                nu_ridge_grid=None,
                                test_sample_size=200):
    """
    - Split first half (train) / second half (val).
    - Train OT + RKHS on train; select (nu_kernel, nu_ridge) on val.
    - Report RMSE on an independent test set of size test_sample_size.
    """
    rng = rng or np.random.default_rng(BASE_SEED)

    # Step 1: generate base sample and truth
    n = sample_size
    n_tr = n // 2
    n_val = n - n_tr

    X_all  = sample_X(n, measure_P, df, input_size, rng)
    T0_all = T0_map(X_all, measure_P, transform_method, df).astype(np.float64)

    # Split deterministically: first half train, second half val
    X_train, X_val   = X_all[:n_tr], X_all[n_tr:]
    T0_train, T0_val = T0_all[:n_tr], T0_all[n_tr:]

    # Build Y only on the training half as a permutation of T0(X_train)
    perm_true_train = rng.permutation(n_tr)
    Y_train = T0_train[perm_true_train].copy()

    # Step 2: OT on train half → T_emp_train
    perm, Gamma, cost = ot_permutation(X_train, Y_train, return_gamma=True)
    T_emp_train = empirical_ot_map_from_perm(perm, Y_train)  # (n_tr, d)

    # Step 3: hyperparameter selection on validation half
    if nu_kernel_grid is None:
        nu_kernel_grid = 10.0 ** np.linspace(-9, -5, 9)
    if nu_ridge_grid is None:
        nu_ridge_grid  = 10.0 ** np.linspace(-5, -1, 9) 

    best_nu_k, best_nu_r, val_rmse, _ = select_hyperparams_train_val(
        X_train, T_emp_train, X_val, T0_val, nu_kernel_grid, nu_ridge_grid
    )

    # Refit W on the train half using the chosen hyperparameters
    K_tt_best = gaussian_kernel_from_sqdist(pairwise_sq_dists(X_train, X_train), best_nu_k)
    W_best    = fit_rkhs_weights(K_tt_best, T_emp_train, best_nu_r)

    # Step 4: independent test RMSE
    X_test  = sample_X(test_sample_size, measure_P, df, input_size, rng)
    T0_test = T0_map(X_test, measure_P, transform_method, df).astype(np.float64)

    K_test_train = gaussian_kernel_from_sqdist(pairwise_sq_dists(X_test, X_train), best_nu_k)
    T_test_pred  = predict_with_kernel(K_test_train, W_best)
    test_rmse    = float(np.sqrt(np.mean(np.sum((T_test_pred - T0_test) ** 2, axis=1))))

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "T0_train": T0_train, "T0_val": T0_val, "T0_test": T0_test,
        "Y_train": Y_train, "perm_train": perm, "Gamma_train": Gamma, "ot_cost_train": cost,
        "T_emp_train": T_emp_train,
        "best_nu_kernel": best_nu_k, "best_nu_ridge": best_nu_r,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--d',  type=int, help='training data dimension.')
    parser.add_argument('--n',  type=int, help='training data sample size.')
    parser.add_argument('--measure',  type=str, help='measure of P, either normal or t.')
    parser.add_argument('--transform', type=str, help='OT map, CDF, piecewise_linear or quadratic.')
    parser.add_argument('--seed-base', type=int, default=BASE_SEED, help='base seed; trial_i uses seed_base + i.')
    args, unknown = parser.parse_known_args()

    # ---- Config ----
    n = args.n 
    measure = args.measure 
    transform = args.transform 
    d = args.d 
    TRIALS      = 100

    # For t-distribution; ignored by "normal" but included in filenames for consistency
    DF_DEFAULT  = 6

    # Test-set size for Step 4
    TEST_SAMPLE_SIZE = 10000

    # Output directory
    OUT_DIR = "kernel_estimator_results"
    os.makedirs(OUT_DIR, exist_ok=True)

    def scenario_filename(measure, transform, n, d, df, testN):
        return f"rmse_{measure}_{transform}_n_{n}_d_{d}_df_{df}_testN_{testN}.csv"


    df = DF_DEFAULT  # used if measure == "t"; harmless otherwise
    fname = scenario_filename(measure, transform, n, d, df, TEST_SAMPLE_SIZE)
    fpath = os.path.join(OUT_DIR, fname)

    # Run 100 trials
    rmse_list = []
    for trial in range(TRIALS):
        rng = np.random.default_rng(args.seed_base + trial)
        res = run_pipeline_split_and_test(
            sample_size=n,
            measure_P=measure,
            transform_method=transform,
            input_size=d,
            df=df,
            rng=rng,
            test_sample_size=TEST_SAMPLE_SIZE
        )
        rmse_list.append((trial, float(res["test_rmse"])))

    # Write per-scenario CSV
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "test_rmse"])
        for trial, val in rmse_list:
            # store full precision; change to f"{val:.6f}" if you prefer fixed digits
            w.writerow([trial, f"{val:.12g}"])
