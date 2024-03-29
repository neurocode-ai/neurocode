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

import numpy as np
import torch

from torch import nn


class Blanco2020(nn.Module):
    """Sleep staging architecture from Blanco et al 2020.

    Convolutional neural network for sleep staging described in [Blanco2020]_.
    A series of seven convolutional layers with kernel sizes running down from 7 to 3,
    in an attempt to extract more general features at the beginning, while more specific
    and complex features were extracted in the final stages.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chans : int
        Number of convolutional channels. Set to 20 in [Blanco2020]_.
    n_groups : int
        Number of groups for the convolution. Set to 2 in [Blanco2020]_ for 2 Channel EEG.
        controls the connections between inputs and outputs. n_channels and n_conv_chans must be
        divisible by n_groups.
    input_size_s : float
        Size of the input, in seconds.
    n_classes : int
        Number of classes.
    dropout : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.

    References
    ----------
    .. [Blanco2020] Fernandez-Blanco, E., Rivero, D. & Pazos, A. Convolutional
        neural networks for sleep stage scoring on a two-channel EEG signal.
        Soft Comput 24, 4067–4079 (2020). https://doi.org/10.1007/s00500-019-04174-1
    """

    def __init__(
        self,
        n_channels,
        sfreq,
        n_conv_chans=20,
        input_size_s=30,
        n_classes=5,
        n_groups=2,
        max_pool_size=2,
        dropout=0.5,
        apply_batch_norm=False,
        return_feats=False,
    ):
        super().__init__()

        input_size = np.ceil(input_size_s * sfreq).astype(int)

        self.n_channels = n_channels

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_channels, n_conv_chans, (1, 7), groups=n_groups, padding=0),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 7), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 3), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 3), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
        )

        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self._return_features = return_feats
        if not return_feats:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.len_last_layer, self.len_last_layer),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.len_last_layer, n_classes),
            )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, n_channels, 1, input_size)
            )  # batch_size, n_channels, height, width
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if x.ndim == 3:
            x = x.unsqueeze(2)

        feats = self.feature_extractor(x).flatten(start_dim=1)
        if self._return_features:
            return feats

        return self.fc(feats)
