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
import numpy as np

from .base import PretextTaskSampler


class RecordingSampler(PretextTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, n_channels, n_views=2, **kwargs):
        self.n_channels = n_channels
        self.n_views = n_views

    def _sample_pair(self):
        batch_anchors = []
        batch_samples = []
        recordings = self.rng.choice(
            self.info["n_recordings"], size=(self.batch_size), replace=False
        )
        for reco_idx1 in recordings:
            win_idx1 = self._sample_window(recording_idx=reco_idx1)
            win_idx2 = self._sample_window(recording_idx=reco_idx1)

            batch_anchors.append(self.data[reco_idx1][win_idx1][0][None])
            batch_samples.append(self.data[reco_idx1][win_idx2][0][None])

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))

        return (ANCHORS, SAMPLES)
