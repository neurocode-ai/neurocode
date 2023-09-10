"""
MIT License

Copyright (c) 2023 Neurocode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-09-10
Last updated: 2023-09-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):
    def __init__(self, numFeatures, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean) * torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):
    def __init__(self, sizeHidden=512, normMode="layerNorm"):
        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":

            def normLayer(x):
                return nn.InstanceNorm1d(x, affine=True)

        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):
    def __init__(
        self, dimEncoded, dimOutput, keepHidden, nLevelsGRU, mode="GRU", reverse=False
    ):
        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )
        elif mode == "RNN":
            self.baseNet = nn.RNN(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )
        else:
            self.baseNet = nn.GRU(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):
        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class CPCModel(nn.Module):
    def __init__(self, encoder, AR):
        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label
