#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, torch, argparse
import numpy as np
import scanpy as sc
from tqdm import tqdm

from network import load_network_optim
from data_loader import construct_pair_loaders, split_by_drug_to_numpy
from util import (
    DEFAULT_REALDATA_SEED,
    compute_ks_distance,
    compute_mmd,
    get_code_version,
    load_latest_checkpoint,
    save_checkpoint,
    set_random_seed,
    write_metadata_json,
)


TQDM_KW = dict(dynamic_ncols=True, leave=False, mininterval=0.2)

def train_model_minmax_icnn(
    f_model,
    g_model,
    x_iter,    # infinite iterator over control batches
    y_iter,    # infinite iterator over treated batches
    f_optim,
    g_optim,
    g_steps_per_batch: int = 10,
    n_iteration: int = 100000,     # how many minibatches define one "epoch"
    save_freq: int = 100,
    save_path: str = "",
    drug_name: str = "",           # <<< added
    pbar_desc: str = "",
):
    """
    Min–max training in a CellOT-like style, using two
    infinite iterators (x_iter, y_iter) from construct_pair_loaders.

    We resume from the latest checkpoint (if any), and run until
    global iteration index `n_iteration - 1`.

    Returns:
        Average f-loss over the (new) iterations, i.e. from last_iter to n_iteration-1.
    """

    def compute_loss_f(f_model, g_model, x_in, y_in):
        y_req = y_in.detach().requires_grad_(True)
        grad_g = g_model.transport(y_req)  # ∇g(y) via transport

        # primal objective: E[f(x)] - E[f(∇g(y))]
        loss_f = f_model(x_in).mean() - f_model(grad_g).mean()

        return loss_f


    def compute_loss_g(f_model, g_model, y_in):

        y_req = y_in.detach().requires_grad_(True)
        grad_g = g_model.transport(y_req)  # shape [B, d]

        # dual objective: E[ <y, ∇g(y)> - f(∇g(y)) ]
        loss_g = (f_model(grad_g) - torch.sum(y_req * grad_g, dim=1)).mean()

        # CellOT-style penalty on negative parts of W for g, if enabled
        if getattr(g_model, "fnorm_penalty", 0.0) > 0:
            loss_g = loss_g + g_model.penalize_w()

        return loss_g


    # load latest weights & get last global iteration index
    last_iter, running_fl, nb = load_latest_checkpoint(  
        f_model, g_model, f_optim, g_optim, drug_name, save_path
    )
    remaining = n_iteration - last_iter

    if remaining <= 0:
        print(f"[{drug_name}] nothing to resume: last_iter={last_iter}")
        return running_fl / max(nb, 1) if nb > 0 else 0.0  # <<< changed

    print(f"[{drug_name}] resuming from iteration {last_iter}, "
          f"remaining {remaining} steps to reach {n_iteration}")

    device_f = next(f_model.parameters()).device
    device_g = next(g_model.parameters()).device

    f_model.train()
    g_model.train()

    iter_range = range(last_iter, n_iteration)
    iterator = iter_range                      # <<< added default
    tbar = None
    if pbar_desc:
        from tqdm import tqdm
        # tqdm already shows elapsed and remaining time
        tbar = tqdm(iter_range,
                    desc=pbar_desc,
                    total=remaining,
                    initial=0,                
                    **TQDM_KW)
        iterator = tbar


    for step in iterator:
        # -------------------------
        # (1) Max step(s) on g (via minimizing loss_g)
        # -------------------------
        for _ in range(g_steps_per_batch):
            y_in = next(y_iter).to(device_g).float()

            loss_g = compute_loss_g(f_model, g_model, y_in)
            g_optim.zero_grad()
            loss_g.backward()
            g_optim.step()

        # -------------------------
        # (2) Min step on f
        # -------------------------
        x_in = next(x_iter).to(device_f).float()
        y_in = next(y_iter).to(device_g).float()

        loss_f = compute_loss_f(f_model, g_model, x_in, y_in)
        f_optim.zero_grad()
        loss_f.backward()
        f_optim.step()
        f_model.clamp_w()   # maintain convexity of f

        running_fl += loss_f.item()
        nb += 1

        # update tqdm postfix with current + average loss
        if tbar is not None:
            avg_loss = running_fl / max(nb, 1)
            tbar.set_postfix(
                loss_f=f"{loss_f.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )


        # save periodically *inside* the loop
        if save_freq > 0 and ((step + 1) % save_freq == 0 or (step + 1) == n_iteration):
            save_checkpoint(
                step + 1, f_model, g_model, f_optim, g_optim,
                drug_name, save_path, running_fl, nb
            )

    return running_fl / max(nb, 1)


def evaluate_checkpoints_on_val(
    drug_name: str,
    act: str,
    save_path: str,
    x_val_loader,
    y_val_loader,
    device: str = "cuda",
    val: bool = True
):
    """
    (1) Load all checkpoints for this (drug_name, act).
    (2) For each checkpoint, compute MMD, KS, SWD between:
            T_f(x_val)  (transported control)  and  y_val  (treated)
        where T_f is given by f_model.transport.
    (3) Store the resulting 3 curves + iterations as a .npz on disk.

    Assumes checkpoints are saved as:
        save_path/{act}_iterationXXXX.pt
    where ckpt["f_model"] is the trained forward ICNN potential f.
    """
    device = torch.device(device)

    # 1) List checkpoints
    
    from util import _list_ckpts_for_act
    ckpt_paths, iters = _list_ckpts_for_act(drug_name, act, save_path)
    if not ckpt_paths:
        print(f"[{drug_name} / {act}] No checkpoints found in {save_path}")
        return None

    # 2) Build full validation tensors (control & treated)
    x_val = torch.cat([xb for xb in x_val_loader], dim=0).to(device)
    y_val = torch.cat([yb for yb in y_val_loader], dim=0).to(device)

    # Optional: match counts (just in case)
    n = min(x_val.size(0), y_val.size(0))
    x_val = x_val[:n]
    y_val = y_val[:n]

    mmd_curve = []
    ks_curve = []

    if val: 
        print(f"[{drug_name} / {act}] Evaluating {len(ckpt_paths)} checkpoints on VAL set...")
    else: 
        print(f"[{drug_name} / {act}] Evaluating {len(ckpt_paths)} checkpoints on TEST set...")

    for ckpt_path, _ in tqdm(zip(ckpt_paths, iters), total=len(ckpt_paths), dynamic_ncols=True):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"  [WARN] Failed to load {ckpt_path}: {e}")
            mmd_curve.append(np.nan)
            ks_curve.append(np.nan)
            continue

        if "f_model" not in ckpt:
            print(f"  [WARN] 'f_model' not found in {ckpt_path}, skipping.")
            mmd_curve.append(np.nan)
            ks_curve.append(np.nan)
            continue

        # 2a) Get f_model and move to eval mode
        f_model = ckpt["f_model"].to(device)
        f_model.eval()

        # 2b) Compute transport T_f(x_val) = ∇f(x_val)
        with torch.enable_grad():
            x_req = x_val.detach().clone().requires_grad_(True)
            y_pred = f_model.transport(x_req)  # [N,d]

        # 2c) Distances
        mmd_val = compute_mmd(y_pred, y_val)
        ks_val = compute_ks_distance(y_pred, y_val)

        mmd_curve.append(mmd_val)
        ks_curve.append(ks_val)

    mmd_curve = np.asarray(mmd_curve, dtype=float)
    ks_curve  = np.asarray(ks_curve, dtype=float)
    iters_arr = np.asarray(iters, dtype=int)

    # 3) Save curves to disk
    os.makedirs(save_path, exist_ok=True)
    if val:
        out_file = os.path.join(save_path, f"{drug_name}_{act}_val_curves.npz")
    else:
        out_file = os.path.join(save_path, f"{drug_name}_{act}_test_curves.npz")

    meta = {
        "drug": drug_name,
        "activation": act,
        "save_path": save_path,
        "device": str(device),
        "n_val": int(x_val.size(0)),
    }

    np.savez_compressed(
        out_file,
        iterations=iters_arr,
        MMD=mmd_curve,
        KS=ks_curve,
        meta=str(meta),
    )
    print(f"[{drug_name} / {act}] Saved evaluation curves → {out_file}")

    return {
        "iterations": iters_arr,
        "MMD": mmd_curve,
        "KS": ks_curve,
        "meta": meta,
    }



if __name__=="__main__":
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--act', type=str, required=True,
                        help='activation of ICNN network, e.g. relu, leaky_relu, requ, softplus.')
    parser.add_argument('--seed', type=int, default=DEFAULT_REALDATA_SEED,
                        help='base random seed; per-drug seed is seed + drug index.')
    parser.add_argument('--data-path', type=str, default="../4i/8h.h5ad",
                        help='path to the 4i .h5ad file.')
    parser.add_argument('--feature-file', type=str, default="../4i/features.txt",
                        help='path to the feature-name text file.')
    parser.add_argument('--save-path', type=str, default="../4idata_results",
                        help='root directory for real-data checkpoints and metrics.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='torch device, e.g. cpu or cuda.')
    args, unknown = parser.parse_known_args()

    # Hyperparameters
    act = args.act

    drug_names = [
        "cisplatin",
        "cisplatin_olaparib",
        "crizotinib",
        "dabrafenib",
        "dacarbazine",
        "dasatinib",
        "decitabine",
        "dexamethasone",
        "erlotinib",
        "everolimus",
        "hydroxyurea",
        "imatinib",
        "ixazomib",
        "ixazomib_lenalidomide_dexamethasone",
        "lenalidomide",
        "melphalan",
        "midostaurin",
        "mln2480",
        "olaparib",
        "paclitaxel",
        "palbociclib",
        "panobinostat",
        "pomalidomide_carfilzomib_dexamethasone",
        "regorafenib",
        "sorafenib",
        "staurosporine",
        "trametinib",
        "temozolomide",
        "trametinib_dabrafenib",
        "trametinib_erlotinib",
        "trametinib_midostaurin",
        "trametinib_panobinostat",
        "ulixertinib",
        "vemurafenib_cobimetinib",
        "vindesine",
    ]

    device = args.device

    d_in  = 48
    width = 64
    depth = 4

    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.9

    batch_train_size = 256
    batch_val_test_size = 600
    val_fraction = 0.1
    test_fraction = 0.2
    swd_n_proj = 128 

    g_steps_per_batch = 10
    n_iteration = 100000
    save_freq = 100

    save_path = args.save_path


    try:
        data4i = sc.read_h5ad(args.data_path)
    except FileNotFoundError as e:
        print(e)

    
    feature_names_txt = args.feature_file
    drug_to_matrix = split_by_drug_to_numpy(
        data4i,
        feature_names_txt
    )


    for job_index, drug_name in enumerate(drug_names):
        job_seed = int(args.seed) + job_index
        set_random_seed(job_seed)

        # Build loaders
        x_train_iter, y_train_iter, x_val_loader, y_val_loader, x_test_loader, y_test_loader = construct_pair_loaders(
            drug_to_matrix,
            drug_name=drug_name,
            batch_train_size = batch_train_size, 
            batch_val_test_size = batch_val_test_size, 
            device=device,
            val_fraction=val_fraction,       # desired global ratio
            test_fraction=test_fraction,      # desired global ratio
            random_seed=job_seed,
        )

        f_model, g_model, f_optim, g_optim = load_network_optim(
            d_in = d_in, 
            width = width, 
            depth = depth, 
            act = act, 
            lr = lr, 
            beta1 = beta1, 
            beta2 = beta2
        )


        os.makedirs(save_path, exist_ok=True)
        save_path_drug = os.path.join(save_path, f"{drug_name}")
        run_id = f"{drug_name}_{act}_seed{job_seed}"
        metadata = {
            "run_id": run_id,
            "job_index": job_index,
            "base_seed": int(args.seed),
            "seed_derivation": "job_seed = base_seed + job_index",
            "drug_name": drug_name,
            "activation": act,
            "seed": job_seed,
            "split_seed": job_seed,
            "torch_seed": job_seed,
            "d_in": d_in,
            "width": width,
            "depth": depth,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "batch_train_size": batch_train_size,
            "batch_val_test_size": batch_val_test_size,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "g_steps_per_batch": g_steps_per_batch,
            "n_iteration": n_iteration,
            "save_freq": save_freq,
            "device": device,
            "data_path": args.data_path,
            "feature_file": feature_names_txt,
            "code_version": get_code_version(repo_dir=os.path.dirname(os.path.dirname(__file__))),
        }
        write_metadata_json(save_path_drug, metadata)
        

        avg_fl = train_model_minmax_icnn(
            f_model, g_model,
            x_iter=x_train_iter,
            y_iter=y_train_iter,
            f_optim=f_optim,
            g_optim=g_optim,
            g_steps_per_batch=g_steps_per_batch,
            n_iteration=n_iteration,
            save_freq=save_freq,
            save_path=save_path_drug,
            drug_name=drug_name,
            pbar_desc=f"[{drug_name}]"
        )

        evaluate_checkpoints_on_val(
            drug_name=drug_name,
            act=act,
            save_path=save_path_drug,
            x_val_loader=x_val_loader,
            y_val_loader=y_val_loader,
            device=device, 
            val=True
        )

        evaluate_checkpoints_on_val(
            drug_name=drug_name,
            act=act,
            save_path=save_path_drug,
            x_val_loader=x_test_loader,
            y_val_loader=y_test_loader,
            device=device, 
            val=False
        )

    
