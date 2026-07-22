#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: physics_model.py
#
# Physics-aware network components for the HJ-Communicative RL study.
#
# Contents
#   1. PhysicsEncoder  -- small CNN that ingests the extra physics channels.
#   2. MultiHeadCommNet -- the PDF's "Step 8": replaces CommNet's mean/single-head
#                          pooling with genuine multi-head self-attention across
#                          agents. (The original repo ALREADY has single-head
#                          softmax attention via `attention=True`; this is the
#                          real multi-head upgrade the PDF describes.)
#   3. hj_residual_loss / consistency_loss -- the physics loss terms, written so
#                          they are internally consistent (both use the geodesic
#                          field, unlike the PDF which mixes geodesic + Euclidean).
#
# These modules mirror the shapes used in DQNModel.py:
#   input : (batch, agents, frame_history, D, W, H)   for the image stream
#   physics: (batch, agents, C_phys,       D, W, H)   for the physics stream
#   output: (batch, agents, number_actions)
#
# NOTE: this file is torch code and depends on the same torch version as the
# rest of the repo. It has not been executed in this environment (no GPU/torch
# here), but it is structured to be a drop-in sibling of DQNModel.CommNet.

import torch
import torch.nn as nn


# =============================================================================
# 1. Physics encoder branch  (PDF Step 2)
# =============================================================================
class PhysicsEncoder(nn.Module):
    """Encodes the (speed / potential / distance / gradient) channels into a
    feature vector concatenated with the image-CNN features before communication.
    Deliberately lightweight -- the physics maps are smooth and low-frequency."""

    def __init__(self, in_channels=3, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.PReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.PReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, 3, padding=0), nn.PReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, out_features)

    def forward(self, x):                      # x: (B*agents, C, D, W, H)
        h = self.net(x).flatten(1)
        return self.fc(h)


# =============================================================================
# 2. Multi-head attention communication  (PDF Step 8, done properly)
# =============================================================================
class MultiHeadCommNet(nn.Module):
    """CommNet with true multi-head self-attention between agents at each stage.

    Compared to the repo's CommNet:
      * mean pooling  -> nn.MultiheadAttention (query/key/value over the agent
        axis), with a residual connection and LayerNorm (a transformer block).
      * optionally fuses a PhysicsEncoder feature into the shared representation.
    """

    def __init__(self, agents, frame_history, number_actions,
                 phys_channels=0, n_heads=4, xavier=True):
        super().__init__()
        self.agents = agents
        self.frame_history = frame_history
        self.use_phys = phys_channels > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- shared image CNN (identical to DQNModel.CommNet backbone) --------
        def conv(i, o, k, p):
            return nn.Conv3d(i, o, k, padding=p)
        self.conv0 = conv(frame_history, 32, 5, 1); self.pool0 = nn.MaxPool3d(2); self.act0 = nn.PReLU()
        self.conv1 = conv(32, 32, 5, 1);           self.pool1 = nn.MaxPool3d(2); self.act1 = nn.PReLU()
        self.conv2 = conv(32, 64, 4, 1);           self.pool2 = nn.MaxPool3d(2); self.act2 = nn.PReLU()
        self.conv3 = conv(64, 64, 3, 0);           self.act3 = nn.PReLU()
        self.img_dim = 512

        # --- physics branch ---------------------------------------------------
        if self.use_phys:
            self.phys = PhysicsEncoder(phys_channels, out_features=128)
            self.fuse = nn.Linear(self.img_dim + 128, self.img_dim)

        # --- three transformer communication blocks over the agent axis -------
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(self.img_dim, n_heads, batch_first=True)
            for _ in range(3)])
        self.norm = nn.ModuleList([nn.LayerNorm(self.img_dim) for _ in range(3)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(self.img_dim, self.img_dim), nn.PReLU())
            for _ in range(3)])

        # --- per-agent heads --------------------------------------------------
        self.head = nn.ModuleList([
            nn.Sequential(nn.Linear(self.img_dim, 256), nn.PReLU(),
                          nn.Linear(256, number_actions))
            for _ in range(agents)])
        self.to(self.device)
        if xavier:
            for m in self.modules():
                if isinstance(m, (nn.Conv3d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)

    def _cnn(self, x):                          # x: (B, frame_history, D, W, H)
        x = self.act0(self.pool0(self.conv0(x)))
        x = self.act1(self.pool1(self.conv1(x)))
        x = self.act2(self.pool2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        return x.reshape(x.size(0), -1)[:, :self.img_dim]

    def forward(self, image, physics=None):
        """image: (B, agents, frame_history, D, W, H)
           physics: (B, agents, C, D, W, H) or None"""
        image = image.to(self.device) / 255.0
        B = image.size(0)
        feats = []
        for i in range(self.agents):
            f = self._cnn(image[:, i])
            if self.use_phys and physics is not None:
                pf = self.phys(physics[:, i].to(self.device))
                f = self.fuse(torch.cat([f, pf], dim=-1))
            feats.append(f)
        x = torch.stack(feats, dim=1)           # (B, agents, img_dim)

        for attn, norm, ff in zip(self.attn, self.norm, self.ff):
            a, _ = attn(x, x, x)                 # attention across agents
            x = norm(x + a)                      # residual + norm
            x = x + ff(x)

        out = torch.stack([self.head[i](x[:, i]) for i in range(self.agents)], dim=1)
        return out.cpu()


# =============================================================================
# 3. Physics loss terms  (PDF Step 7, made internally consistent)
# =============================================================================
def hj_residual_loss(V, F):
    """L_HJ = ( ||grad V|| * F - 1 )^2 on a predicted potential field V.

    V, F: (B, D, W, H) tensors. Uses finite differences for the gradient.
    Only needed if you actually *learn* V with a PINN head. If V is precomputed
    offline (recommended), this term is unnecessary -- V already satisfies the
    equation by construction.
    """
    gx = V[:, 1:, :, :] - V[:, :-1, :, :]
    gy = V[:, :, 1:, :] - V[:, :, :-1, :]
    gz = V[:, :, :, 1:] - V[:, :, :, :-1]
    # pad back to equal size
    gx = torch.nn.functional.pad(gx, (0, 0, 0, 0, 0, 1))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1, 0, 0))
    gz = torch.nn.functional.pad(gz, (0, 1, 0, 0, 0, 0))
    grad_norm = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-8)
    return ((grad_norm * F - 1.0) ** 2).mean()


def consistency_loss(V_pred, V_geodesic):
    """L_Consistency = ||V_pred - V_geodesic||^2.

    IMPORTANT FIX vs the PDF: the target is the GEODESIC field V_geodesic (the
    solution of the Eikonal equation), NOT the Euclidean distance d(x). Using
    Euclidean d, as the PDF does, directly contradicts L_HJ whenever the speed
    field F varies (which is the entire point of the physics). See EVALUATION.md
    for the ~48-voxel discrepancy this causes on real ADNI data.
    """
    return ((V_pred - V_geodesic) ** 2).mean()


def communication_loss(agent_features):
    """L_Comm = sum_i ||z_i - mean_j z_j||^2  (encourages a shared code)."""
    z_bar = agent_features.mean(dim=1, keepdim=True)
    return ((agent_features - z_bar) ** 2).mean()
