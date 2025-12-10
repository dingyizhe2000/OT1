#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Yizhe Ding

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

def load_network_optim(d_in, width, depth, act, lr, beta1, beta2):
    f_model = ICNN(d_in=d_in, width=width, depth=depth, activation=act)
    g_model = ICNN(d_in=d_in, width=width, depth=depth, activation=act)

    f_optim = optim.Adam(f_model.parameters(), lr=lr, betas=(beta1, beta2))
    g_optim = optim.Adam(g_model.parameters(), lr=lr, betas=(beta1, beta2))

    return f_model, g_model, f_optim, g_optim

# -------------------------------------------------
# Activations (your style, with slope 0.2 for LeakyReLU)
# -------------------------------------------------

class LeakyReLUModule(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ReQUModule(nn.Module):
    """ReQU(x) = (ReLU(x))^2."""
    def forward(self, x):
        return F.relu(x) ** 2

class ScaledSoftplus(nn.Module):
    def __init__(self, scale: float = 1.20):
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
        return ReQUModule()
    if name in ("softplus_scaled", "softplus"):
        return ScaledSoftplus()
    raise ValueError("Unsupported activation.")


# -------------------------------------------------
# ICNN (CellOT-style forward, your interface)
# -------------------------------------------------

class ICNN(nn.Module):
    """
    ICNN in CellOT style but with your interface:

        z0 = sigma(A[0](x));  z0 = z0^2
        for l = 0..L-2:
            z <- sigma(W[l] z + A[l+1](x))
        y = W[-1](z) + A[-1](x)

    where A[l]: x -> layer l, W[l]: z_{l} -> z_{l+1}.

    transport(x) = ∇_x φ(x), φ(x) = y (scalar per sample).
    """

    def __init__(
        self,
        d_in: int = 48,
        width: int = 64,
        depth: int = 4,
        activation: str = "leaky_relu",
        fnorm_penalty: float = 1.0,
        kernel_init_fxn=None,
    ):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        self.d_in = d_in
        self.width = width
        self.depth = depth
        self.activation_name = activation
        self.fnorm_penalty = fnorm_penalty

        # activation module (LeakyReLU has slope 0.2 via make_activation_module)
        self.sigma = make_activation_module(activation)

        # hidden units: [width, width, ..., width] (length = depth)
        hidden_units = [width] * depth
        units = hidden_units + [1]  # last is output dimension 1

        # W: z_l -> z_{l+1}, bias=False, all nonnegative enforced by clamp_w/penalize_w
        self.W = nn.ModuleList(
            [nn.Linear(idim, odim, bias=False) for idim, odim in zip(units[:-1], units[1:])]
        )

        # A: x -> z_l (and output); each layer has its A_l(x) term
        self.A = nn.ModuleList(
            [nn.Linear(d_in, odim, bias=True) for odim in units]
        )

        # ---- Optional kernel initialization (CellOT-style) ----
        def init_fxn(w):
            nn.init.uniform_(w, a=0.0, b=0.1)

        # initialize A layers
        for layer in self.A:
            init_fxn(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # initialize W layers
        for layer in self.W:
            init_fxn(layer.weight)


    # --------------------
    # Forward (CellOT-style)
    # --------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

            z = sigma(A[0](x))
            z = z^2

            for W, A in zip(W[:-1], A[1:-1]):
                z = sigma(W(z) + A(x))

            y = W[-1](z) + A[-1](x)

        Returns y with shape [N, 1].
        """
        # first hidden layer: sigma(A[0](x)), then square
        z = self.sigma(self.A[0](x))
        z = z * z

        # intermediate hidden layers
        for W_layer, A_layer in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(W_layer(z) + A_layer(x))

        # final linear layer
        y = self.W[-1](z) + self.A[-1](x)   # [N, 1]
        return y

    # --------------------
    # Brenier map T(x) = ∇φ(x)
    # --------------------
    def transport(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇_x φ(x), where φ(x) is the scalar potential (forward output).
        Mirrors CellOT's `transport` method.
        """
        assert x.requires_grad, "x must require grad for transport()"

        (output,) = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size(0), 1), device=x.device).float(),
        )
        return output  # [N, d_in]

    # --------------------
    # Convexity helpers
    # --------------------
    def clamp_w(self):
        """
        Project W weights onto [0, ∞) to enforce W_l >= 0 (convexity).
        Call this after each optimizer step if you want hard constraints.
        """
        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0.0)

    def penalize_w(self) -> torch.Tensor:
        """
        L2 penalty on negative parts of W weights, scaled by fnorm_penalty.
        Can be added to the loss if you prefer soft penalization instead of (or in addition to) clamp_w.
        """
        if self.fnorm_penalty == 0:
            # return a scalar-zero tensor on the correct device
            return torch.zeros([], device=self.W[0].weight.device)

        return self.fnorm_penalty * sum(
            (torch.relu(-w.weight)).norm() for w in self.W
        )