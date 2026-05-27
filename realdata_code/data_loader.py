#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

from util import DEFAULT_REALDATA_SEED


def load_train_test_loaders(
    drug_name,
    device,
    batch_train_size,
    batch_val_test_size,
    val_fraction,
    test_fraction,
    random_seed=DEFAULT_REALDATA_SEED,
    data_path="../4i/8h.h5ad",
    feature_names_txt="../4i/features.txt",
):
    data4i = sc.read_h5ad(data_path)
    drug_to_matrix = split_by_drug_to_numpy(data4i, feature_names_txt)

    return construct_pair_loaders(
        drug_to_matrix,
        drug_name=drug_name,
        batch_train_size=batch_train_size,
        batch_val_test_size=batch_val_test_size,
        device=device,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )


def split_by_drug_to_numpy(adata, feature_names_txt):
    """
    Splits the dataset into numpy matrices grouped by drug conditions.

    Parameters
    ----------
    adata : AnnData
        The single-cell dataset containing `obs['drug']` and feature data.
    feature_names_txt : str
        Path to a text file containing one feature (column name) per line.

    Returns
    -------
    dict
        Mapping: { drug_name : numpy array of shape (n_cells, n_features) }
    """

    # --- Load feature names ---
    with open(feature_names_txt, "r") as f:
        feature_list = [line.strip() for line in f.readlines()]

    # Filter to features actually present in the dataset
    feature_list = [f for f in feature_list if f in adata.var_names]

    if len(feature_list) == 0:
        raise ValueError("None of the features in the feature list exist in adata.var_names.")

    # Ensure drug column is string-typed
    drug_labels = adata.obs['drug'].astype(str)

    # Identify all unique drug conditions (including control)
    drug_list = sorted(drug_labels.unique())

    # ---- Split into numpy arrays ----
    result = {}

    for drug in drug_list:
        idx = (drug_labels == drug)
        if idx.sum() == 0:
            continue
        
        # Convert extracted slice to dense numpy array
        result[drug] = adata[idx, feature_list].X.toarray()

    return result

# --------------------------
# Dataset that just takes tensors already balanced to equal length
# --------------------------
class CustomDataset(Dataset):
    def __init__(self, x, device):
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def _infinite_loader(loader):
    """Turn a DataLoader into an infinite iterator (minimal CellOT-style)."""
    while True:
        for batch in loader:
            yield batch


def construct_pair_loaders(
    drug_to_matrix,
    drug_name,
    batch_train_size=256,
    batch_val_test_size=600,
    device="cpu",
    val_fraction=0.1,       # desired global ratio
    test_fraction=0.2,      # desired global ratio
    random_seed=DEFAULT_REALDATA_SEED,
):
    """
    Creates matched (control, drug) datasets and returns infinite train loaders
    and finite test loaders.
    
    Modified so that final train/val/test = 0.7 / 0.1 / 0.2.
    """

    control = drug_to_matrix["control"]
    treated = drug_to_matrix[drug_name]

    x = np.array(control, dtype=float)
    y = np.array(treated, dtype=float)

    # -------------------------------
    # 1) First split: test set (20%)
    # -------------------------------
    x_train, x_test = train_test_split(
        x, test_size=test_fraction, shuffle=True, random_state=random_seed
    )

    y_train, y_test = train_test_split(
        y, test_size=test_fraction, shuffle=True, random_state=random_seed
    )

    # ---------------------------------------------
    # 2) Second split: validation fraction should be
    #    val_fraction of the whole dataset (e.g. 0.1)
    #
    # Since test_fraction was removed, adjust:
    # val_fraction_adjusted = 0.1 / 0.8 = 0.125
    # ---------------------------------------------
    train_pool_fraction = 1.0 - test_fraction
    val_fraction_adjusted = val_fraction / train_pool_fraction

    x_train, x_val = train_test_split(
        x_train,
        test_size=val_fraction_adjusted,   # MINIMAL CHANGE
        shuffle=True,
        random_state=random_seed
    )
    y_train, y_val = train_test_split(
        y_train,
        test_size=val_fraction_adjusted,   # MINIMAL CHANGE
        shuffle=True,
        random_state=random_seed
    )

    # --- Wrap tensors ---
    x_train = CustomDataset(x_train, device)
    x_val   = CustomDataset(x_val,   device)
    x_test  = CustomDataset(x_test,  device)

    y_train = CustomDataset(y_train, device)
    y_val   = CustomDataset(y_val,   device)
    y_test  = CustomDataset(y_test,  device)

    # --- Loaders ---
    x_generator = torch.Generator()
    y_generator = torch.Generator()
    x_generator.manual_seed(int(random_seed))
    y_generator.manual_seed(int(random_seed) + 1)

    x_train_loader = DataLoader(
        x_train, batch_size=batch_train_size, shuffle=True, generator=x_generator
    )
    y_train_loader = DataLoader(
        y_train, batch_size=batch_train_size, shuffle=True, generator=y_generator
    )

    x_val_loader   = DataLoader(x_val,   batch_size=batch_val_test_size, shuffle=False)
    y_val_loader   = DataLoader(y_val,   batch_size=batch_val_test_size, shuffle=False)

    x_test_loader  = DataLoader(x_test,  batch_size=batch_val_test_size, shuffle=False)
    y_test_loader  = DataLoader(y_test,  batch_size=batch_val_test_size, shuffle=False)

    # --- Infinite training iterators ---
    x_train_iter = _infinite_loader(x_train_loader)
    y_train_iter = _infinite_loader(y_train_loader)

    return x_train_iter, y_train_iter, x_val_loader, y_val_loader, x_test_loader, y_test_loader
