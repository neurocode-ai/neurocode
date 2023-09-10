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


class RelativePositioningSampler(PretextTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, tau_neg=30, tau_pos=2, **kwargs):
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos

    def _sample_pair(self):
        batch_anchors = list()
        batch_samples = list()
        batch_labels = list()
        reco_idx = self._sample_recording()
        for _ in range(self.batch_size):
            win_idx1 = self._sample_window(recording_idx=reco_idx)
            pair_type = self.rng.binomial(1, 0.5)
            win_idx2 = -1

            if pair_type == 0:
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while (
                    np.abs(win_idx1 - win_idx2) < self.tau_neg or win_idx1 == win_idx2
                ):
                    win_idx2 = self._sample_window(recording_idx=reco_idx)
            elif pair_type == 1:
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while (
                    np.abs(win_idx1 - win_idx2) > self.tau_pos or win_idx1 == win_idx2
                ):
                    win_idx2 = self._sample_window(recording_idx=reco_idx)

            batch_anchors.append(self.data[reco_idx][win_idx1][0][:2][None])
            batch_samples.append(self.data[reco_idx][win_idx2][0][:2][None])
            batch_labels.append(float(pair_type))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, SAMPLES, LABELS)
