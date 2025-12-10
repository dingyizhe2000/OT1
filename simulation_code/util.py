#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import torch
import torch.optim as optim

def inner_product(a, b):
    """Batch inner product ⟨a_i, b_i⟩ over arbitrary trailing dims."""
    return (a * b).reshape(a.size(0), -1).sum(dim=1)


# Function to compute the convex conjugate phi_star(y) using optimization
def cvx_conjugate_slow(y, phi, inner_epoch):
    x = torch.zeros_like(y, requires_grad=True)  # Initialize x as zero tensor with gradients
    optimizer = optim.SGD([x], lr=0.001) 

    # Closure to compute the objective function and its gradient
    def closure():
        optimizer.zero_grad()
        loss = (phi(x) - inner_product(x, y)).sum()
        loss.backward()
        return loss
    
    # Run the optimizer
    for _ in range(inner_epoch):
        optimizer.step(closure)
    
    # Return the optimal value of the convex conjugate
    phi_star_y = (inner_product(x, y) - phi(x))
    
    return phi_star_y