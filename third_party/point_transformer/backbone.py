"""Hengshuang Zhao Point Transformer backbone (encoder only).

Adapted from qq456cvb/Point-Transformers ``models/Hengshuang/model.py`` so that
construction takes plain kwargs instead of a Hydra config object.
"""

import torch.nn as nn

from .pointnet_util import PointNetSetAbstraction
from .transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super(TransitionDown, self).__init__()
        self.sa = PointNetSetAbstraction(
            k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True
        )

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Backbone(nn.Module):
    """Point Transformer encoder that returns per-point features at the coarsest level."""

    def __init__(self, num_point, nblocks, nneighbor, input_dim, transformer_dim):
        super(Backbone, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    num_point // 4 ** (i + 1),
                    nneighbor,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
        self.out_channels = 32 * 2 ** nblocks

    def forward(self, x):
        # x: (B, N, input_dim) with xyz in the first 3 channels when input_dim >= 3
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
        return points


__all__ = ["Backbone", "TransitionDown"]
