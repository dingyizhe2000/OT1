#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import fcntl

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import joblib
import numpy as np
import ot
import torch
from joblib import Parallel, delayed
from scipy.stats import norm, t as t_dist

from dataset import generate_raw_data


BASE_SEED = 20260527
DEFAULT_DF = 6
METADATA_FIELDS = [
    "model_idx",
    "seed",
    "d",
    "n",
    "measure",
    "transform",
    "estimator",
    "epsilon",
    "epsilon_rule",
    "epsilon_c",
    "alpha",
    "alpha_bar",
    "sinkhorn_method",
    "num_iter_max",
    "stop_threshold",
    "test_size",
    "evaluation_seed",
    "evaluation_formula",
    "l2_loss",
    "sinkhorn_niter",
    "sinkhorn_final_err",
    "sinkhorn_warnings",
    "code_version",
    "commit",
    "created_at",
]


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_code_version() -> tuple[str, str]:
    repo_dir = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
        return (f"{commit}-dirty" if dirty else commit), commit
    except Exception:
        return "unknown", "unknown"


def epsilon_from_rule(n: int, d: int, c: float = 1.0, alpha: float = 1.0) -> float:
    alpha_bar = min(float(alpha), 3.0)
    return float(c) * float(n) ** (-1.0 / (float(d) + alpha_bar + 1.0))


def choose_epsilon(
    n: int,
    d: int,
    epsilon: float | None,
    epsilon_rule: str,
    epsilon_c: float,
    alpha: float,
) -> tuple[float, str, float]:
    if epsilon is not None:
        return float(epsilon), "fixed", min(float(alpha), 3.0)
    if epsilon_rule != "pnw":
        raise ValueError(f"Unknown epsilon_rule: {epsilon_rule}")
    alpha_bar = min(float(alpha), 3.0)
    return epsilon_from_rule(n, d, c=epsilon_c, alpha=alpha), epsilon_rule, alpha_bar


def sign_square_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (x ** 2)


def true_map_np(x: np.ndarray, measure: str, transform: str, df: int = DEFAULT_DF) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if transform == "CDF":
        if measure == "normal":
            return norm.cdf(x)
        if measure == "t":
            return t_dist.cdf(x, df)
        raise ValueError(f"Unknown measure: {measure}")
    if transform == "piecewise_linear":
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        middle = sign_x * (0.5 * (abs_x - 1.0) + 1.0)
        outer = sign_x * (2.0 * (abs_x - 2.0) + 1.5)
        return np.where(abs_x <= 1.0, x, np.where(abs_x <= 2.0, middle, outer))
    if transform == "quadratic":
        return sign_square_np(x)
    raise ValueError(f"Unknown transform: {transform}")


def generate_train_data(n: int, d: int, measure: str, transform: str, seed: int, df: int = DEFAULT_DF):
    set_random_seed(seed)
    x, y = generate_raw_data(n, measure, transform, df, d)
    return x.detach().cpu().numpy().astype(np.float64), y.detach().cpu().numpy().astype(np.float64)


def generate_test_data(test_size: int, d: int, measure: str, transform: str, seed: int, df: int = DEFAULT_DF):
    set_random_seed(seed)
    if measure == "normal":
        x = torch.randn(test_size, d, dtype=torch.float32)
    elif measure == "t":
        x = torch.distributions.StudentT(df).sample((test_size, d))
    else:
        raise ValueError(f"Unknown measure: {measure}")
    x_np = x.detach().cpu().numpy().astype(np.float64)
    return x_np, true_map_np(x_np, measure, transform, df=df).astype(np.float64)


def squared_euclidean_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    cost = x2 + y2 - 2.0 * (x @ y.T)
    np.maximum(cost, 0.0, out=cost)
    return cost


def fit_entropic_ot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float,
    num_iter_max: int = 5000,
    stop_threshold: float = 1e-9,
    sinkhorn_method: str = "sinkhorn_log",
) -> tuple[dict, list[str]]:
    n = x_train.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    cost = squared_euclidean_cost(x_train, y_train)

    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coupling, log = ot.sinkhorn(
            a,
            b,
            cost,
            reg=float(epsilon),
            method=sinkhorn_method,
            numItermax=int(num_iter_max),
            stopThr=float(stop_threshold),
            log=True,
        )
    for item in caught:
        caught_warnings.append(str(item.message))

    target_log_scaling = log.get("log_v")
    source_log_scaling = log.get("log_u")
    target_scaling = log.get("v")
    source_scaling = log.get("u")
    evaluation_formula = (
        "out_of_sample_target_dual_log_scaling"
        if target_log_scaling is not None
        else "out_of_sample_target_dual_scaling"
        if target_scaling is not None
        else "out_of_sample_uniform_kernel_fallback"
    )
    model = {
        "x_train": x_train,
        "y_train": y_train,
        "coupling": np.asarray(coupling, dtype=np.float64),
        "source_log_scaling_u": None if source_log_scaling is None else np.asarray(source_log_scaling, dtype=np.float64),
        "target_log_scaling_v": None if target_log_scaling is None else np.asarray(target_log_scaling, dtype=np.float64),
        "source_scaling_u": None if source_scaling is None else np.asarray(source_scaling, dtype=np.float64),
        "target_scaling_v": None if target_scaling is None else np.asarray(target_scaling, dtype=np.float64),
        "epsilon": float(epsilon),
        "sinkhorn_log": {
            "niter": int(log.get("niter", -1)),
            "err": [float(x) for x in log.get("err", [])],
        },
        "evaluation_formula": evaluation_formula,
    }
    return model, caught_warnings


def predict_train_barycentric(model: dict) -> np.ndarray:
    coupling = np.asarray(model["coupling"], dtype=np.float64)
    y_train = np.asarray(model["y_train"], dtype=np.float64)
    row_mass = coupling.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, np.finfo(np.float64).tiny)
    return coupling @ y_train / row_mass


def predict_out_of_sample(model: dict, x_query: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    y_train = np.asarray(model["y_train"], dtype=np.float64)
    epsilon = float(model["epsilon"])
    target_log_scaling = model.get("target_log_scaling_v")
    target_scaling = model.get("target_scaling_v")
    if target_log_scaling is not None:
        log_v = np.asarray(target_log_scaling, dtype=np.float64)
    elif target_scaling is not None:
        log_v = np.log(np.maximum(np.asarray(target_scaling, dtype=np.float64), np.finfo(np.float64).tiny))
    else:
        # Fallback: normalized Gibbs kernel weights against target samples.
        # This does not use the learned Sinkhorn target dual and is recorded in metadata.
        log_v = np.zeros(y_train.shape[0], dtype=np.float64)

    preds = []
    for start in range(0, x_query.shape[0], chunk_size):
        x_chunk = np.asarray(x_query[start : start + chunk_size], dtype=np.float64)
        cost = squared_euclidean_cost(x_chunk, y_train)
        log_weights = log_v[None, :] - cost / epsilon
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
        preds.append(weights @ y_train)
    return np.vstack(preds)


def l2_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def scenario_dir(output_root: str | Path, d: int, measure: str, transform: str, n: int) -> Path:
    return Path(output_root) / f"d={d}" / f"{measure}_{transform}_n_{n}"


def model_path_for(scenario_path: Path, model_idx: int) -> Path:
    return scenario_path / f"model_{model_idx}.pkl"


def metadata_path_for(scenario_path: Path, model_idx: int) -> Path:
    return scenario_path / f"model_{model_idx}_metadata.json"


def save_metadata_json(path: Path, metadata: dict) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def write_scenario_metadata_csv(scenario_path: Path, metadata_rows: list[dict]) -> None:
    csv_path = scenario_path / "metadata.csv"
    lock_path = scenario_path / ".metadata.csv.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        rows = {}
        if csv_path.exists():
            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("model_idx") not in (None, ""):
                        rows[int(row["model_idx"])] = row
        for metadata in metadata_rows:
            rows[int(metadata["model_idx"])] = {field: metadata.get(field, "") for field in METADATA_FIELDS}

        tmp_path = scenario_path / f"metadata.csv.tmp.{os.getpid()}"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            for idx in sorted(rows):
                writer.writerow(rows[idx])
        os.replace(tmp_path, csv_path)
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def write_l2_csv(scenario_path: Path, measure: str, transform: str, rows: list[dict]) -> Path:
    out_csv = scenario_path / f"L2_error_measure_P={measure}_transform_method={transform}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_idx", "L2_loss", "epsilon", "evaluation_formula"])
        writer.writeheader()
        for row in sorted(rows, key=lambda x: int(x["model_idx"])):
            writer.writerow(
                {
                    "model_idx": row["model_idx"],
                    "L2_loss": row["l2_loss"],
                    "epsilon": row["epsilon"],
                    "evaluation_formula": row["evaluation_formula"],
                }
            )
    return out_csv


def run_single_model(
    d: int,
    n: int,
    measure: str,
    transform: str,
    model_idx: int,
    output_root: str,
    epsilon: float | None,
    epsilon_rule: str,
    epsilon_c: float,
    alpha: float,
    test_size: int,
    evaluation_seed: int,
    num_iter_max: int,
    stop_threshold: float,
    sinkhorn_method: str,
    overwrite: bool,
) -> dict:
    seed = BASE_SEED + int(model_idx)
    out_dir = scenario_dir(output_root, d, measure, transform, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for(out_dir, model_idx)
    metadata_path = metadata_path_for(out_dir, model_idx)
    eps, eps_rule_used, alpha_bar = choose_epsilon(n, d, epsilon, epsilon_rule, epsilon_c, alpha)

    start_time = time.time()
    if model_path.exists() and metadata_path.exists() and not overwrite:
        model = joblib.load(model_path)
        sinkhorn_warnings = []
    else:
        x_train, y_train = generate_train_data(n, d, measure, transform, seed)
        model, sinkhorn_warnings = fit_entropic_ot(
            x_train,
            y_train,
            eps,
            num_iter_max=num_iter_max,
            stop_threshold=stop_threshold,
            sinkhorn_method=sinkhorn_method,
        )
        model["config"] = {
            "model_idx": int(model_idx),
            "seed": int(seed),
            "d": int(d),
            "n": int(n),
            "measure": measure,
            "transform": transform,
            "epsilon_rule": eps_rule_used,
            "epsilon_c": float(epsilon_c),
            "alpha": float(alpha),
            "alpha_bar": float(alpha_bar),
            "test_size": int(test_size),
            "evaluation_seed": int(evaluation_seed),
        }
        joblib.dump(model, model_path, compress=3)

    x_test, y_test = generate_test_data(test_size, d, measure, transform, evaluation_seed)
    pred_test = predict_out_of_sample(model, x_test)
    loss = l2_loss(pred_test, y_test)

    err = model.get("sinkhorn_log", {}).get("err", [])
    code_version, commit = get_code_version()
    metadata = {
        "model_idx": int(model_idx),
        "seed": int(seed),
        "d": int(d),
        "n": int(n),
        "measure": measure,
        "transform": transform,
        "estimator": "entropic_ot_pooladian_niles_weed",
        "epsilon": float(model["epsilon"]),
        "epsilon_rule": eps_rule_used,
        "epsilon_c": float(epsilon_c),
        "alpha": float(alpha),
        "alpha_bar": float(alpha_bar),
        "sinkhorn_method": sinkhorn_method,
        "num_iter_max": int(num_iter_max),
        "stop_threshold": float(stop_threshold),
        "test_size": int(test_size),
        "evaluation_seed": int(evaluation_seed),
        "evaluation_formula": model.get("evaluation_formula", "unknown"),
        "l2_loss": float(loss),
        "sinkhorn_niter": int(model.get("sinkhorn_log", {}).get("niter", -1)),
        "sinkhorn_final_err": float(err[-1]) if err else math.nan,
        "sinkhorn_warnings": " | ".join(sinkhorn_warnings),
        "code_version": code_version,
        "commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - start_time,
    }
    save_metadata_json(metadata_path, metadata)
    return metadata


def run_scenario(
    d: int,
    n: int,
    measure: str,
    transform: str,
    model_indices: list[int],
    output_root: str,
    epsilon: float | None,
    epsilon_rule: str,
    epsilon_c: float,
    alpha: float,
    test_size: int,
    evaluation_seed: int,
    num_iter_max: int,
    stop_threshold: float,
    sinkhorn_method: str,
    n_jobs: int,
    overwrite: bool,
) -> tuple[Path, list[dict]]:
    out_dir = scenario_dir(output_root, d, measure, transform, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(run_single_model)(
            d=d,
            n=n,
            measure=measure,
            transform=transform,
            model_idx=model_idx,
            output_root=output_root,
            epsilon=epsilon,
            epsilon_rule=epsilon_rule,
            epsilon_c=epsilon_c,
            alpha=alpha,
            test_size=test_size,
            evaluation_seed=evaluation_seed,
            num_iter_max=num_iter_max,
            stop_threshold=stop_threshold,
            sinkhorn_method=sinkhorn_method,
            overwrite=overwrite,
        )
        for model_idx in model_indices
    )
    write_scenario_metadata_csv(out_dir, rows)
    write_l2_csv(out_dir, measure, transform, rows)
    return out_dir, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Pooladian--Niles-Weed style entropic OT baseline for simulations.")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100])
    parser.add_argument("--measures", nargs="+", default=["normal", "t"], choices=["normal", "t"])
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["CDF", "piecewise_linear", "quadratic"],
        choices=["CDF", "piecewise_linear", "quadratic"],
    )
    parser.add_argument("--model-start", type=int, default=0)
    parser.add_argument("--model-end", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--output-root", default="simulation_results/entropic_ot")
    parser.add_argument("--epsilon", type=float, default=None, help="fixed epsilon; overrides --epsilon-rule.")
    parser.add_argument("--epsilon-rule", default="pnw", choices=["pnw"])
    parser.add_argument("--epsilon-c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=BASE_SEED)
    parser.add_argument("--num-iter-max", type=int, default=5000)
    parser.add_argument("--stop-threshold", type=float, default=1e-9)
    parser.add_argument("--sinkhorn-method", default="sinkhorn_log")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_jobs > 9:
        raise ValueError("Local pilot is capped at 9 workers. Use --n-jobs 9 or lower.")

    model_indices = list(range(args.model_start, args.model_end))
    all_rows = []
    for d in args.dimensions:
        for n in args.sample_sizes:
            for measure in args.measures:
                for transform in args.transforms:
                    out_dir, rows = run_scenario(
                        d=d,
                        n=n,
                        measure=measure,
                        transform=transform,
                        model_indices=model_indices,
                        output_root=args.output_root,
                        epsilon=args.epsilon,
                        epsilon_rule=args.epsilon_rule,
                        epsilon_c=args.epsilon_c,
                        alpha=args.alpha,
                        test_size=args.test_size,
                        evaluation_seed=args.evaluation_seed,
                        num_iter_max=args.num_iter_max,
                        stop_threshold=args.stop_threshold,
                        sinkhorn_method=args.sinkhorn_method,
                        n_jobs=args.n_jobs,
                        overwrite=args.overwrite,
                    )
                    losses = np.asarray([row["l2_loss"] for row in rows], dtype=float)
                    print(
                        f"[done] {out_dir}: count={losses.size}, "
                        f"mean={losses.mean():.6g}, sd={losses.std(ddof=1) if losses.size > 1 else math.nan:.6g}"
                    )
                    all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    main()
