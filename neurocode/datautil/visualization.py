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
Last updated: 2023-11-19
"""

import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.manifold import TSNE

from typing import (
    List,
    Dict,
    Tuple,
)

__all__ = (
    "manifold_plot",
    "history_plot",
)

_manifolds = {
    "tSNE": TSNE,
    "UMAP": UMAP,
}


def manifold_plot(
    X: np.ndarray,
    y: List[Tuple],
    title: str,
    *,
    technique: str = "tSNE",
    perplexity: int = 30,
    savefig: bool = True,
    **kwargs: Dict,
):
    """Apply the a non-linear dimensionality reduction technique (manifold), either t-SNE
    or UMAP, to a set of embedding vectors. Color code plots based on provided labels.

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
            reduction, either one of `tSNE` or `UMAP`.
        perplexity: int
            Set the hyperparameter for t-SNE, changes the complexity of resulting
            visualization. See sklearn documentation for more information.
        savefig: bool
            Specify whether or not to save the produced matplotlib figures.

    """
    manifold_kwargs = {perplexity: perplexity} if technique == "tSNE" else {}
    manifold = _manifolds[technique](**manifold_kwargs)

    components = manifold.fit_transform(X)

    n_samples = len(y)
    labels = {
        "sleep": np.ones((n_samples,)),
        "eyes": np.ones((n_samples,)),
        "recording": np.ones((n_samples,)),
        "gender": np.ones((n_samples,)),
        "age": np.ones((n_samples,)),
        "RTrecipCTR": np.ones((n_samples,)),
        "RTrecipPSD": np.ones((n_samples,)),
        "RTctr": np.ones((n_samples,)),
        "RTpsd": np.ones((n_samples,)),
        "RTdiff": np.ones((n_samples,)),
        "lapseCTR": np.ones((n_samples,)),
        "lapsePSD": np.ones((n_samples,)),
    }

    for idx, (
        subj_id,
        reco_id,
        gender,
        age,
        RTrecipCTR,
        RTrecipPSD,
        RTctr,
        RTpsd,
        RTdiff,
        minor_lapses_ctr,
        minor_lapses_psd,
    ) in enumerate(y):
        labels["sleep"][idx] = int(reco_id // 2)
        labels["eyes"][idx] = int(reco_id % 2)
        labels["recording"][idx] = int(reco_id)
        labels["gender"][idx] = int(gender)
        labels["age"][idx] = int(age)
        labels["RTrecipCTR"][idx] = RTrecipCTR
        labels["RTrecipPSD"][idx] = RTrecipPSD
        labels["RTctr"][idx] = RTctr
        labels["RTpsd"][idx] = RTpsd
        labels["RTdiff"][idx] = RTdiff
        labels["lapseCTR"][idx] = minor_lapses_ctr
        labels["lapsePSD"][idx] = minor_lapses_psd

    unique_labels = {
        "sleep": [0, 1],
        "eyes": [0, 1],
        "recording": [0, 1, 2, 3],
        "gender": [0, 1],
        "age": np.unique(labels["age"]),
        "RTrecipCTR": np.unique(labels["RTrecipCTR"]),
        "RTrecipPSD": np.unique(labels["RTrecipPSD"]),
        "RTctr": np.unique(labels["RTctr"]),
        "RTpsd": np.unique(labels["RTpsd"]),
        "RTdiff": np.unique(labels["RTdiff"]),
        "lapseCTR": np.unique(labels["lapseCTR"]),
        "lapsePSD": np.unique(labels["lapsePSD"]),
    }

    unique_ll = {
        "sleep": ["control", "psd"],
        "eyes": ["closed", "open"],
        "recording": [
            "control eyes-closed",
            "control eyes-open",
            "psd eyes-closed",
            "psd eyes-open",
        ],
        "gender": ["female", "male"],
        "age": unique_labels["age"],
        "RTrecipCTR": unique_labels["RTrecipCTR"],
        "RTrecipPSD": unique_labels["RTrecipPSD"],
        "RTctr": unique_labels["RTctr"],
        "RTpsd": unique_labels["RTpsd"],
        "RTdiff": unique_labels["RTdiff"],
    }

    reactiontimes_ = [
        "RTrecipCTR",
        "RTrecipPSD",
        "RTctr",
        "RTpsd",
        "RTdiff",
        "lapseCTR",
        "lapsePSD",
    ]

    for key in labels:
        fig, ax = plt.subplots()
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels[key]))
        ]

        if key in reactiontimes_:
            colors = labels[key]
            sc = ax.scatter(
                components[:, 0],
                components[:, 1],
                c=colors[:],
                cmap=plt.cm.coolwarm,
                alpha=0.8,
                s=5.0,
            )
            plt.colorbar(sc)

        else:
            for idx, (k, col) in enumerate(zip(unique_labels[key], colors)):
                mask = labels[key] == k
                xy = components[mask]
                ax.scatter(
                    xy[:, 0], xy[:, 1], alpha=0.8, s=5.0, label=unique_ll[key][idx]
                )

        handles, lbls = ax.get_legend_handles_labels()
        uniques = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, lbls))  # noqa E741
            if l not in lbls[:i]
        ]
        ax.legend(*zip(*uniques))
        fig.suptitle(f"{technique} of embeddings, subject {key}, {title} training")
        if savefig:
            plt.savefig(f"{technique}_{key}_{title}-training.png")
        plt.show()


def history_plot(history, savefig=True):
    """Takes a dictionary of training metrics and visualizes them in a combined plot.
    If you want more customizability then use your own plotting.

    Parameters
    ----------
    history: dict
        Dictionary containing training/testing metrics, valid keys
        are: `tloss`, `vloss`, `tacc`, `vacc`.
    savefig: bool
        Saves the produces plot to the curent working directory of the user.

    """
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = None

    if "tacc" in history or "vacc" in history:
        ax2 = ax1.twinx()

    ax1.plot(
        history["tloss"],
        ls="-",
        marker="1",
        ms=5,
        alpha=0.7,
        color="tab:blue",
        label="training loss",
    )

    if "vloss" in history:
        ax1.plot(
            history["vloss"],
            ls=":",
            marker="1",
            ms=5,
            alpha=0.7,
            color="tab:blue",
            label="validation loss",
        )

    if "tacc" in history:
        ax2.plot(
            history["tacc"],
            ls="-",
            marker="2",
            ms=5,
            alpha=0.7,
            color="tab:orange",
            label="training acc",
        )

    if "vacc" in history:
        ax2.plot(
            history["vacc"],
            ls=":",
            marker="2",
            ms=5,
            alpha=0.7,
            color="tab:orange",
            label="validation acc",
        )

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    lines1, labels1 = ax1.get_legend_handles_labels()

    if ax2:
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylabel("Accuracy [%]", color="tab:orange")
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
    else:
        ax1.legend(lines1, labels1)

    plt.tight_layout()

    if savefig:
        plt.savefig(f"training_history.png")
