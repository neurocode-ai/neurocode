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


class ContrastiveViewGenerator(object):
    """callable object that generated n_views
    of the given input data x.

    Attributes
    ----------
    T: tuple | list
        A collection holding the transforms, either torchvision.transform or
        BaseTransform from neurocode.datautil, number of transforms should
        be the same as n_views. No stochastic choice on transform is made
        in this module, but could be implemented.
    n_views: int
        The number dictating the amount of augmentations/transformations
        to apply to input x, and decides the length of the resulting list
        after invoking __call__ on the object.

    """

    def __init__(self, T, n_views):
        self.transforms = T
        self.n_views = n_views

    def __call__(self, x):
        return [torch.Tensor(self.transforms[t](x)) for t in range(self.n_views)]
