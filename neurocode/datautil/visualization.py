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
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def manifold_plot(
    X: np.ndarray,
    y: list,
    title: str,
    *,
    technique: str = 'tSNE',
    n_components: int = 2,
    perplexity: int = 30,
    savefig: bool = True,
    **kwargs: dict,
):
    """
    Parameters
    ----------
        X: np.ndarray
            The embeddings that you want to visualize using the specified
            manifold technique.
        y: list
            A list of tuples with corresponding labels for each embedding.
        title: str
            The title of plot.
        technique: str
            Specifies what manifold technique to use for embedding dimensionality
            reduction, either one of `tSNE` or `UMOD`.
    """

