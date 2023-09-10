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

File created: 2023-09-07
Last updated: 2023-09-07
"""

import numpy as np

from scipy import signal


class BaseTransform(object):
    """the base data augmentation object, defining the structure that all
    implemented transforms inherit from. The transform has to be callable,
    thus, all other transforms have to implement _parameters and transform.

    See the implementations inheriting from the BaseTransform object for
    more information about actualy transforms on signal data.

    Methods
    -------
    __call__(x):
        Apply the transform on the input x, requiring that the transform T
        inheriting from this object has implemented transform func.

    """

    def __init__(self, *args, **kwargs):
        self._parameters(**kwargs)

    def __call__(self, x):
        return self.transform(x)

    def _parameters(self, *args, **kwargs):
        raise NotImplementedError("No parameter setup implemented for BaseTransform.")

    def transform(self, x):
        raise NotImplementedError(
            "This BaseTransform is not callable, transform not implemented yet."
        )


class CropResizeTransform(BaseTransform):
    """data augmentation module T for the SimCLR pipeline, applying
    the Crop&Resize transform to a given input data x, retaining
    the dimensionalities. Supports multi-channel data, but might
    have slower execution times since the transform has to be
    applied to each channel individually (?)...

    Inherits from the BaseTransform object that defines the outlining
    functions for the transform.

    Attributes
    ----------
    n_partitions: int | None
        The number of partitions to create and sample from, i.e. given
        an input signal S(t) we create n number of windowed
        signals {s_1, ..., s_n}. The number of partitions has to be
        able to divide the number of samples of the provided signal S(t).
    pick_one: bool
        Sets the transformation mode of the class, if true; then only one
        partition is selected uniformly at random and then resampled to
        the original dimensionality of S(t), if false; then we
        uniformly at random pick one partition to throw away, and
        resample all n-1 partitions to the original dimensionality.

    Methods
    -------
    _parameters(*args, n_partitions=None, pick_one=False):
        sets up the provided attributes of the class according to the given parameters
    transform(x):
        applies the implemented data augmentation transform on the given input array x.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, n_partitions=None, pick_one=True):
        if not n_partitions:
            n_partitions = 5

        self.n_partitions = n_partitions
        self.pick_one = pick_one

    def transform(self, x):
        """func expects input np.array to have dimensionality (C, T)
        even if only one channel is used. Hopefully more channels
        in the SimCLR pipeline can yield better results, since there
        are spatio-temporal relations that are prevalent, not only
        temporal relations that would be uncovered by looking at
        one channel only.

        Parameters
        ----------
        x: np.array
            The array on which to crop&resize according to set parameters in
            constructor. See main comments in class for information on what
            the parameters do to the array.

        Returns
        -------
        resampled: np.array
            The resampled np.array, if pick_one is false then the information loss
            is substantially decreased, since that would only leave one out.
            Retains the input dimensionalities of x.

        """
        n_channels, n_samples = x.shape
        partitions = np.random.choice([1, 2, 4, 5])

        if n_samples % partitions:
            raise ValueError(
                f"Can`t partition x with {n_samples=} into {partitions}"
                " partitions without information loss."
            )

        size = n_samples // partitions
        indices = np.arange(partitions)
        if self.pick_one:
            # pick one partition uniformly at random to resample to original size
            # and return these channels. the alternative is to pick one and leave it
            # out, instead of using it. this choice remains for ALL channels in x.
            choice = np.random.choice(indices)
            start, end = [np.ceil((choice + i) * size).astype(int) for i in [0, 1]]
            resampled = np.zeros(x.shape).astype(x.dtype)

            for channel in range(n_channels):
                resampled[channel, :] = signal.resample(
                    x[channel, start:end], n_samples
                )

            return resampled

        else:
            raise NotImplementedError(
                "Leave one out has not been implemented yet, not used in literature either."
            )

        return None


class PermutationTransform(BaseTransform):
    """data augmentation module T for the SimCLR pipeline, applying the
    Permutation transformation on the provided input data array x,
    retaining its dimensionalities and NOT applying it inplace.

    Inherits from the BaseTransform object that defines the outlining
    functions for the transform.

    Attributes
    ----------
    n_partitions: int | None
        The number of partitions to create and shuffle. See above CropResizeTransform
        for more information on how to partitioning is performed.

    Methods
    -------
    _parameters(*args, n_partitions=None):
        sets up the provided attributes of the class according to the given parameters
    transform(x):
        applies the implemented data augmentation transform on the given input array x.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, n_partitions=None):
        if not n_partitions:
            n_partitions = 10

        self.n_partitions = n_partitions

    def transform(self, x):
        """func applies the Transform to the input x, preserving the dimensionalities
        but expects it to be of shape (n_channels, n_samples)

        Parameters
        ----------
        x: np.array
            The broadcastable array on which to apply transformation. Does not do this inplace,
            i.e. the class creates a new numpy array and returns it.

        Returns
        -------
        permuted: np.array
            The permuted np.array, partitioned into n_partitions and uniformly at random
            shuffled to generate 'new' data.
        """
        n_channels, n_samples = x.shape
        partitions = np.random.choice([4, 5, 10, 20])

        if n_samples % partitions:
            raise ValueError(
                f"Can`t partition x with {n_samples=} into {partitions}"
                " partitions without information loss."
            )

        size = n_samples // partitions
        indices = np.random.permutation(partitions)

        # get the partitioning indices based on the input size and permuted base indices
        samples = [
            (np.ceil(i * size).astype(int), np.ceil((i + 1) * size).astype(int))
            for i in indices
        ]
        permuted = np.zeros(x.shape).astype(x.dtype)

        # apply the permutation on all channels of x, according to above shuffling
        for channel in range(n_channels):
            for idx, (start, end) in enumerate(samples):
                nstart, nend = [np.ceil((idx + i) * size).astype(int) for i in [0, 1]]
                permuted[channel, nstart:nend] = x[channel, start:end]

        return permuted


class AmplitudeScaleTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, minscale=0.5, maxscale=2.0):
        self.minscale = minscale
        self.maxscale = maxscale

    def transform(self, x):
        scale = np.random.uniform(self.minscale, self.maxscale, 1)
        scaled = x.copy() * scale
        return scaled


class ZeroMaskingTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, samples=0.5):
        self.samples = samples

    def transform(self, x):
        n_channels, n_samples = x.shape
        percent = np.random.uniform(0.1, self.samples, 1)[0]
        ttt = int(n_samples * ((1 - percent) / 2))
        offset = np.random.randint(-ttt, high=ttt)

        lidx = np.floor(n_samples * ((1 - percent) / 2)).astype(int) + offset
        ridx = (
            np.floor(n_samples * (percent + ((1 - percent) / 2))).astype(int) + offset
        )

        zeroed = x.copy()
        zeroed[:, lidx:ridx] = 0.0
        zeroed = zeroed.reshape(x.shape)

        return zeroed
