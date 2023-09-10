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


class TSSampler(PretextTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, tau_pos=5, tau_neg=5, **kwargs):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def _sample_pair(self):
        """ """
        batch_anchors = list()
        batch_middles = list()
        batch_samples = list()
        batch_labels = list()

        for _ in range(self.batch_size):
            pair_type = self.rng.binomial(1, 0.5)
            reco_idx = self._sample_recording()
            anchor_idx = self._sample_window(recording_idx=reco_idx)
            sample_idx = self._sample_window(recording_idx=reco_idx)
            middle_idx = self._sample_window(recording_idx=reco_idx)

            # Resample the other anchor index until it is inside the positive
            # context relative to the anchor.
            while (
                (np.abs(anchor_idx - sample_idx) >= self.tau_pos)
                or (anchor_idx == sample_idx)
                or (np.abs(anchor_idx - sample_idx) < 2)
            ):
                sample_idx = self._sample_window(recording_idx=reco_idx)

            if pair_type == 0:
                # Negative sample, they are not ordered
                while (anchor_idx <= middle_idx <= sample_idx) or (
                    sample_idx <= middle_idx <= anchor_idx
                ):
                    middle_idx = self._sample_window(recording_idx=reco_idx)

            elif pair_type == 1:
                # Positive sample, they are ordered
                while (
                    not (anchor_idx <= middle_idx <= sample_idx)
                    and not (anchor_idx >= middle_idx >= sample_idx)
                ) or (middle_idx == anchor_idx or middle_idx == sample_idx):
                    middle_idx = self._sample_window(recording_idx=reco_idx)

            batch_anchors.append(self.data[reco_idx][anchor_idx][0][None])
            batch_middles.append(self.data[reco_idx][middle_idx][0][None])
            batch_samples.append(self.data[reco_idx][sample_idx][0][None])
            batch_labels.append(float(pair_type))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        MIDDLES = torch.Tensor(np.concatenate(batch_middles, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, MIDDLES, SAMPLES, LABELS)
