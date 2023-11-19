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
Last updated: 2023-11-19
"""

import torch
import numpy as np

from .base import PretextTaskSampler
from neurocode.datautil import (
    CropResizeTransform,
    PermutationTransform,
)
from .contrastive_view import ContrastiveViewGenerator


class SignalSampler(PretextTaskSampler):
    """pretext task sampler for the SimCLR pipeline,
    applying data augmentations T to signals S(t), of the SLEMEG
    data. Currently only applies to transformations,
    CropResizeTransformation and PermutationTransformation.
    Literature has showed their strength together, but could be
    insightful to explore other transformations as well.

    Attributes
    ----------
    n_channels: int
        The amount of signal channels that is included in the input data.
        Decides the resulting dimensionality of the samples tensors.
    n_views: int
        Number of transformations to apply to the original signal S(t).
    crop_partitions: int
        The number of partitions to create when performing CropResize
        transformation. TODO investigate how this effects learning.
    permutation_partitions: int
        The number of partitions to create when performing Permutation
        transformation. TODO investigate how this effects learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(
        self,
        n_channels,
        n_views=2,
        crop_partitions=5,
        permutation_partitions=10,
        **kwargs
    ):
        self.n_channels = n_channels
        self.n_views = n_views

        self._transforms = [
            CropResizeTransform(n_partitions=crop_partitions),
            PermutationTransform(n_partitions=permutation_partitions),
        ]

        self.transformer = ContrastiveViewGenerator(self._transforms, n_views)

    def _sample_pair(self):
        """sample two augmented views (t1, t2) of the same original
        signal S(t), as per the SimCLR framework. Randomly picks a
        recording and corresponding window to transform. The transformation
        is performed directly on the signal, and supports both single- and
        multi-channel data. To see the specific parameters for the transforms,
        see the neurocode.datautil.transforms module.

        Returns
        -------
        ANCHORS: torch.tensor
            The signals which has had the first transform applied to them.
            Resulting shape should be (N, 1, C, T) where N is the batch size,
            C is the amount of signal channels included, and T is the size of
            the time window.
        SAMPLES: torch.tensor
            Same as above, but second transformation was applied to the original
            signal S(t). See neurocode.datautil.transforms for documentation.

        """
        batch_anchors = list()
        batch_samples = list()
        for _ in range(self.batch_size):
            reco_idx = self._sample_recording()
            wind_idx = self._sample_window(recording_idx=reco_idx)

            x = self.data[reco_idx][wind_idx][0]
            T1, T2 = self.transformer(x)

            batch_anchors.append(T1.unsqueeze(0))
            batch_samples.append(T2.unsqueeze(0))

        ANCHORS = torch.cat(batch_anchors).unsqueeze(1)
        SAMPLES = torch.cat(batch_samples).unsqueeze(1)

        return (ANCHORS, SAMPLES)

    def _extract_features(self, model, device):
        """heuristically sample windows from each
        recording and use f() to extract features.
        Labels are pairwise sampled to the corresponding
        features, otherwise the tSNE plots are useless.

        Parameters
        ----------
        model: torch.nn.Module
            The neural network model, encoder f(), which is yet to be
            or has already been trained and is used to perform the
            feature extraction. Make sure to set the model in evaluation
            mode so that batch normalization layers and dropout layers
            are inactivated; otherwise you get inaccurate features.
            Furthermore, enable returning of features, otherwise the
            features are fed to the projection head g() and we are not
            interested in the features in the mapped space.
        device: torch.device | str
            The device on which to perform feature extraction, either
            CPU, CUDA or some GPU:0...N, should be the same as that
            of the provided model.

        Returns
        -------
        X: np.array
            The extracted features, casted to numpy arrays and forced to
            move to the CPU if they were on another device. The amount
            of features to extract is a bit arbitrary, and depends
            on the window_size_s of the pipeline and the amount of
            recordings provided to the sampler instance.
        Y: np.array
            The corresponding labels for the extracted features. Used
            such that the tSNE plots can be labeled accordingly, and
            has to be the same length as X.

        """
        X, Y = [], []
        model.eval()
        model._return_features = True
        with torch.no_grad():
            for recording in range(len(self.data)):
                for window in range(len(self.data[recording])):
                    if window % 1 == 0:
                        window = (
                            torch.Tensor(self.data[recording][window][0][None])
                            .float()
                            .to(device)
                        )
                        feature = model(window.unsqueeze(0))
                        X.append(feature[0, :][None])
                        Y.append(*self.labels[recording])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        model._return_features = False

        return (X, Y)
