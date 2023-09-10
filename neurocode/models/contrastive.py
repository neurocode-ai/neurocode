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


class ProjectionHead(nn.Module):
    def __init__(self, encoder, emb_size, latent_size, dropout=0.25):
        super(ProjectionHead, self).__init__()
        self.f = encoder
        self.g = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, latent_size),
        )

    def forward(self, x):
        features = self.f(x)
        latent = self.g(features)

        return latent


class ContrastiveRPNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=0.5):
        super(ContrastiveRPNet, self).__init__()
        self.emb = emb
        self.clf = nn.Sequential(nn.Linear(emb_size, 1))

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1 - z2))


class ContrastiveTSNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=0.5):
        super(ContrastiveTSNet, self).__init__()
        self.emb = emb
        self.clf = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

    def forward(self, x):
        z1, z2, z3 = [self.emb(t) for t in x]
        return self.clf(torch.cat((torch.abs(z1 - z2), torch.abs(z2 - z3)), dim=1))
