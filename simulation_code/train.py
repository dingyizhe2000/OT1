#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, time, argparse, csv, json, random, subprocess
from datetime import datetime, timezone
import fcntl

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from util import cvx_conjugate_slow
from dataset import CustomDataset, generate_raw_data
from network import ICNN, clip_parameters
from evaluate import eval_tree


BASE_SEED = 20260527
METADATA_FIELDS = [
    "model_idx",
    "seed",
    "d",
    "n",
    "measure",
    "transform",
    "activation",
    "hidden_size",
    "epochs",
    "inner_epochs",
    "batch_size",
    "learning_rate",
    "device",
    "code_version",
    "commit",
    "created_at",
]


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_code_version():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        code_version = f"{commit}-dirty" if dirty else commit
        return code_version, commit
    except Exception:
        return "unknown", "unknown"


def write_metadata(path, metadata):
    os.makedirs(path, exist_ok=True)

    json_path = os.path.join(path, f"model_{metadata['model_idx']}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    csv_path = os.path.join(path, "metadata.csv")
    lock_path = os.path.join(path, ".metadata.csv.lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        rows = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("model_idx") not in (None, ""):
                        rows[int(row["model_idx"])] = row

        rows[int(metadata["model_idx"])] = {
            field: metadata.get(field, "") for field in METADATA_FIELDS
        }

        tmp_path = f"{csv_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            for idx in sorted(rows):
                writer.writerow(rows[idx])
        os.replace(tmp_path, csv_path)
        fcntl.flock(lock_f, fcntl.LOCK_UN)



def train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer):
    for x_in, y_in in dataloader:
        # Forward pass
        loss = model(x_in).mean() + cvx_conjugate_slow(y_in, model, num_epochs_cvx_conjugate).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        clip_parameters(model)


def get_data_loader(sample_size, measure_P, transform_method, df, input_size, batch_size, device, seed=None):
    
    x, y = generate_raw_data(sample_size, measure_P, transform_method, df, input_size)

    dataset = CustomDataset(x, y, device)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    dataloader = DataLoader(dataset, batch_size, shuffle=True, generator=generator)

    return dataloader


def get_model_and_optim(input_size, hidden_size, act, learning_rate, device):

    output_size = 1
    model = ICNN(input_size, hidden_size, output_size, act)
    clip_parameters(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    return model, optimizer


def train_model(args):

    torch.set_num_threads(1)
    if len(args) == 14:
        model_num, batch_size, input_size, hidden_size, act, learning_rate, sample_size, transform_method, measure_P, df, num_epochs, num_epochs_cvx_conjugate, path, device = args
        seed = BASE_SEED + int(model_num)
    elif len(args) == 15:
        model_num, batch_size, input_size, hidden_size, act, learning_rate, sample_size, transform_method, measure_P, df, num_epochs, num_epochs_cvx_conjugate, path, device, seed = args
    else:
        raise ValueError(f"Expected 14 or 15 training arguments, got {len(args)}.")

    set_random_seed(seed)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print(f"start training model {model_num} with seed={seed}, d={input_size}, n={sample_size}, {measure_P}, {transform_method} at:" + localtime)

    dataloader = get_data_loader(sample_size, measure_P, transform_method, df, input_size, batch_size, device, seed=seed)
    model, optimizer = get_model_and_optim(input_size, hidden_size, act, learning_rate, device)

    for _ in range(num_epochs):
        train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer)

    model_path = f"{path}model_{model_num}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"store trained model {model_num} at: " + model_path)

    code_version, commit = get_code_version()
    metadata = {
        "model_idx": int(model_num),
        "seed": int(seed),
        "d": int(input_size),
        "n": int(sample_size),
        "measure": measure_P,
        "transform": transform_method,
        "activation": act,
        "hidden_size": int(hidden_size),
        "epochs": int(num_epochs),
        "inner_epochs": int(num_epochs_cvx_conjugate),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "device": str(device),
        "code_version": code_version,
        "commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(path, metadata)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--d',  type=int, help='training data dimension.')
    parser.add_argument('--n',  type=int, help='training data sample size.')
    parser.add_argument('--measure',  type=str, help='measure of P, either normal or t.')
    parser.add_argument('--transform', type=str, help='OT map, CDF, piecewise_linear or quadratic.')
    parser.add_argument('--act', type=str, help='activation of ICNN network, relu, leaky_relu, softplus.')
    parser.add_argument('--seed-base', type=int, default=BASE_SEED, help='base random seed; model_i uses seed_base + i.')
    args, unknown = parser.parse_known_args()

    # Hyperparameters
    input_size = args.d
    hidden_size = 15
    act = args.act

    num_epochs = 500
    num_epochs_cvx_conjugate = 500

    learning_rate = 0.001

    sample_size = args.n
    batch_size = 50

    measure_P = args.measure
    df = 6

    transform_method = args.transform

    device = "cpu"

    path = f"../simulation_results/{act}/d={input_size}/{measure_P}_{transform_method}_n_{sample_size}/"
    print(path)

    if not os.path.exists(path):
        os.makedirs(path)

    for input_index in range(100): 
        arguments = (input_index, batch_size, input_size, hidden_size, act, learning_rate, 
                    sample_size, transform_method, measure_P, df, 
                    num_epochs, num_epochs_cvx_conjugate, path, device, args.seed_base + input_index
                    )
        
        model_path_i = f"{path}model_{input_index}.pth"
        if not os.path.exists(model_path_i):
            train_model(arguments)

    eval_tree(f"../simulation_results/{act}/", act=act, seed=args.seed_base)
