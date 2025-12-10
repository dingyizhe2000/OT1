#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2017 LocusLab
# Modifications Copyright (c) 2025 Yizhe Ding

# ---------------------------------------------------------------------
# NOTICE:
# This file includes an adaptation of the Input Convex Neural Network (ICNN) originally
# implemented in TensorFlow and licensed under the Apache License 2.0 by the original authors:
# Brandon Amos, Lei Xu and Zico Kolter.
# The original TensorFlow code can be found here: https://github.com/locuslab/icnn.
#
# Modifications have been made to convert the code to PyTorch. These modifications are 
# licensed under the Apache License 2.0. A copy of this license is included in this 
# repository as `Apache_LICENSE`, or you can access it here: 
# http://www.apache.org/licenses/LICENSE-2.0.
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeakyReLUModule(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ScaledSoftplus(nn.Module):
    def __init__(self, scale: float = 1.05):
        super().__init__()
        self.softplus = nn.Softplus()
        self.c0 = float(self.softplus(torch.zeros(1)))
        self.scale = float(torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * (self.softplus(x) - self.c0)
    

def make_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("leaky_relu", "lrelu"):
        # default slope 0.2 to match CellOT's LeakyReLU(0.2)
        return LeakyReLUModule(negative_slope=0.2)
    if name in ("softplus_scaled", "softplus"):
        return ScaledSoftplus()
    raise ValueError("Unsupported activation.")


class ICNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(ICNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        self.fc2_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2_x = nn.Linear(input_size, hidden_size)
        
        self.fc3_z = nn.Linear(hidden_size, output_size, bias=False)
        self.fc3_x = nn.Linear(input_size, output_size)
    
        self.activation = make_activation_module(activation)

    def forward(self, x):
        z = self.fc1(x)
        z = self.activation(z)
        
        z = self.fc2_z(z) + self.fc2_x(x)
        z = self.activation(z)
        
        z = self.fc3_z(z) + self.fc3_x(x)
        
        return z
    
def clip_parameters(model):
    # Ensure non-negative weights for specific layers
    for name, param in model.named_parameters():
        if 'fc2_z' in name or 'fc3_z' in name or 'fc4_z' in name:
            param.data = torch.clamp(param.data, min=0.0)


def compute_gradients(model, x: torch.Tensor) -> torch.Tensor:
    x_ = x.clone().detach().requires_grad_(True)
    model.eval()
    out = model(x_)
    model.zero_grad(set_to_none=True)
    out.backward(torch.ones_like(out))
    return x_.grad.detach()


def load_model(input_size, hidden_size, model_path):
    
    output_size = 1
    
    model = ICNN(input_size, hidden_size, output_size)

    model.load_state_dict(torch.load(model_path))
    
    return model