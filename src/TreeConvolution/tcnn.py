# < begin copyright >
# Copyright Ryan Marcus 2019
#
# This file is part of TreeConvolution.
#
# TreeConvolution is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TreeConvolution is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TreeConvolution.  If not, see <http://www.gnu.org/licenses/>.
#
# < end copyright >

import torch
import torch.nn as nn

class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data):
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        zero_vec = zero_vec.to(results.device)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)

class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]), x[1])

class TreeLayerNorm(nn.Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)

class DynamicPooling(nn.Module):
    def forward(self, x):
        return torch.max(x[0], dim=2).values

class OneHot(nn.Module):
    def forward(self, x):

        one_hots = []
        for recon in x:
            # One-hot along dim=0
            idx = recon.argmax(dim=0, keepdim=True)  # [1, 62]
            one_hot = torch.zeros_like(recon).scatter_(0, idx, 1.0)

            # Compute max per column
            max_vals = recon.max(dim=0, keepdim=True).values  # [1, 62]

            # Detect columns where all values == max (padded columns)
            all_equal_max = (recon == max_vals).all(dim=0, keepdim=True)  # [1, 62]

            # Build mask: keep only columns that are NOT all equal to max
            mask = (~all_equal_max).float()

            # Apply mask
            one_hot = one_hot * mask
            one_hots.append(one_hot)

        x = torch.stack(one_hots, dim=0)  # [B, 9, 62]

        return x
