#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import itertools
import os
import time
from multiprocessing import Pool

from evaluate import eval_tree
from train import BASE_SEED, train_model


def train_each_model(d, n, measure, transform, activation, model_idx, seed_base):
    input_size = d
    hidden_size = 15
    num_epochs = 500
    num_epochs_cvx_conjugate = 500
    learning_rate = 0.001
    batch_size = 50
    df = 6
    device = "cpu"

    path = f"../simulation_results/{activation}/d={input_size}/{measure}_{transform}_n_{n}/"
    os.makedirs(path, exist_ok=True)

    model_path = f"{path}model_{model_idx}.pth"
    if os.path.exists(model_path):
        return model_path

    args = (
        model_idx,
        batch_size,
        input_size,
        hidden_size,
        activation,
        learning_rate,
        n,
        transform,
        measure,
        df,
        num_epochs,
        num_epochs_cvx_conjugate,
        path,
        device,
        seed_base + model_idx,
    )
    train_model(args)
    return model_path


def worker(hp):
    return train_each_model(**hp)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch simulation training entry point.")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--measures", nargs="+", default=["t", "normal"], choices=["normal", "t"])
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["CDF", "piecewise_linear", "quadratic"],
        choices=["CDF", "piecewise_linear", "quadratic"],
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["leaky_relu"],
        choices=["relu", "leaky_relu", "softplus"],
    )
    parser.add_argument("--model-start", type=int, default=0)
    parser.add_argument("--model-end", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=BASE_SEED)
    parser.add_argument("--eval", action="store_true", help="run eval_tree for each activation after training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_numbers = range(args.model_start, args.model_end)
    combos = [
        {
            "d": d,
            "n": n,
            "measure": measure,
            "transform": transform,
            "activation": activation,
            "model_idx": model_idx,
            "seed_base": args.seed_base,
        }
        for d, n, measure, transform, activation, model_idx in itertools.product(
            args.dimensions,
            args.sample_sizes,
            args.measures,
            args.transforms,
            args.activations,
            model_numbers,
        )
    ]

    with Pool(args.workers) as pool:
        async_results = [pool.apply_async(worker, (hp,)) for hp in combos]
        time.sleep(1)
        for ar in async_results:
            ar.wait()
            ar.get()

    if args.eval:
        for activation in args.activations:
            eval_tree(f"../simulation_results/{activation}/", act=activation, seed=args.seed_base)
