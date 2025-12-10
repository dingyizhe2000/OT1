#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import os, time, argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from util import cvx_conjugate_slow
from dataset import CustomDataset, generate_raw_data
from network import ICNN, clip_parameters
from evaluate import eval_tree



def train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer):
    for x_in, y_in in dataloader:
        # Forward pass
        loss = model(x_in).mean() + cvx_conjugate_slow(y_in, model, num_epochs_cvx_conjugate).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        clip_parameters(model)


def get_data_loader(sample_size, measure_P, transform_method, df, input_size, batch_size, device):
    
    x, y = generate_raw_data(sample_size, measure_P, transform_method, df, input_size)

    dataset = CustomDataset(x, y, device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

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
    model_num, batch_size, input_size, hidden_size, act, learning_rate, sample_size, transform_method, measure_P, df, num_epochs, num_epochs_cvx_conjugate, path, device = args
    
    localtime = time.asctime( time.localtime(time.time()) )
    print(f"start training model {model_num} with d={input_size}, n={sample_size}, {measure_P}, {transform_method} at:" + localtime)

    dataloader = get_data_loader(sample_size, measure_P, transform_method, df, input_size, batch_size, device)
    model, optimizer = get_model_and_optim(input_size, hidden_size, act, learning_rate, device)

    for _ in range(num_epochs):
        train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer)

    model_path = f"{path}model_{model_num}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"store trained model {model_num} at: " + model_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--d',  type=int, help='training data dimension.')
    parser.add_argument('--n',  type=int, help='training data sample size.')
    parser.add_argument('--measure',  type=str, help='measure of P, either normal or t.')
    parser.add_argument('--transform', type=str, help='OT map, CDF, piecewise_linear or quadratic.')
    parser.add_argument('--act', type=str, help='activation of ICNN network, relu, leaky_relu, softplus.')
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
                    num_epochs, num_epochs_cvx_conjugate, path, device
                    )
        
        model_path_i = f"{path}model_{input_index}.pth"
        if not os.path.exists(model_path_i):
            train_model(arguments)

    eval_tree(f"../simulation_results/{act}/", act=act)