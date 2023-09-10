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
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict


class SimCLR(object):
    """Simple framework for Contrastive Learning on visual Representations (SimCLR)
    object, implementing Info NCE Loss and training+validation procedure.
    Currently doesn't utilize the implemented Info NCE Loss, but favours the
    NTXentLoss implemented in pytorch-metric-learning. See its documentation
    for more information on implementation.

    Attributes
    ----------
    model: torch.nn.Module
        The initialized neural network to train, usually encoder f() for SimCLR.
    device: torch.device | str
        The device on which to place tensors, should be the same as the model is on.
    optimizer: torch.optim | None
        The optimization algorithm to apply gradients on. Defaults to Adam.
    criterion: torch.nn.loss | pytorch_metric_learning.loss | None
        The loss function to minimize by utilizing the provided optimizer. For
        Info NCE Loss you should use torch.nn.CrossEntropyLoss
    scheduler: torch.optim | None
        The pytorch scheduler to use, defaults to CosineAnnealingLR.
        See documentation at pytorch.org for more information.
    batch_size: int
        The batch size of the sampler, for SimCLR this is expanded multiplied by
        n_views, since one datapoint x is augmented that amount of times.
    epochs: int
        The number of epochs, full iteration of n_samples, to train for.
    temperature: float | int
        The temperature constant `tau` to use with NT-Xent Loss, and Info NCE Loss.
        See literature for more information about its characteristics.
    n_views: int | None
        The number of aguemnted views to produce given a datapoint x.

    Methods
    -------
    info_nce_loss(features):
        Calculates the similarity measures of the input features, torch.tensor,
        and returns the logits and labels associated with them. Loss is implemented
        based on the Oord et al. 2016 Contrastive Predictive Coding paper.
    fit(samplers, plot=False):
        Traing the provided encoder f() on the given sampler datasets. The attributes
        decide majority of parameters during training. Plot enables visualizing the
        positive sample data transformations.

    """

    def __init__(
        self,
        model,
        device,
        optimizer=None,
        criterion=None,
        scheduler=None,
        batch_size=256,
        epochs=100,
        temperature=0.7,
        n_views=2,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.temperature = temperature
        self.n_views = n_views

    def info_nce_loss(self, features):
        """implementation of Info NCE Loss, as per defined by Oord et al. 2016,
        takes in the features, expect them the positive pair to lie one batch
        size deeper in the tensor, i.e. the features tensor has
        dimensionality (2*N, D) where N is batch size and D is the embedding size.

        To use cosine-similarity it is required that the features are normalized,
        and non-zero. Then, discard the diagonal from both labels and similarity
        matrix. Select and combine the multiple positives, and select negatives

        placing all positive logits first results in all correct index
        labels ending up at position 0, so the zero matrix and the
        corresponding logits matrix can now simply be sent to
        the criterion to calculate loss.

        Parameters
        ----------
        features: torch.tensor
            The extracted features from f(), 2N samples where N is the batch size.
            Thus, there are 2(N-1) negative samples contra the positive samples.

        Returns
        -------
        logits: torch.tensor
            The log-similarities of the provided positive and negative samples.
        labels: torch.tensor
            The labels for where the positive samples are, i.e. 1/2(N-1) possibility
            to randomly pick the right one. So larger N, tougher contstraint on
            training and more reasonable features.

        """
        labels = torch.cat(
            [torch.arange(self.batch_size) for _ in range(self.n_views)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(self.device)

        features = F.normalize(features, dim=1)
        sim_mat = torch.matmul(features, features.T)

        assert (
            sim_mat.shape == labels.shape
        ), "Labels and similarity matrix doesn`t have matching shapes."

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1)

        positives = sim_mat[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_mat[~labels.bool()].view(sim_mat.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return (logits, labels)

    def fit(self, samplers, plot=False, save_model=False, paramspath="params.pth"):
        """fit the SimCLR model to the provided samplers, conditionally plot the positive pair
        augmentations when running for validity, or just for interest. Majority of
        parameters for training are set up a priori, i.e. in optimizer, scheduler, which can
        be found as attributes in the base class.

        Parameters
        ----------
        samplers: dict
            Dictionary containing keys 'train' and 'valid', for the respective datasets.
            The values are PretextSamplers that should be iterated over, yielding batches
            of samples according to the pretext task at hand.

        Returns
        -------
        history: dict
            Dictionary of training+validation results, keys are 'train' and 'valid' and values
            are lists of loss. Should be passed to history_plot function from datautil.visualization

        """
        history = defaultdict(list)
        for epoch in range(self.epochs):
            self.model.train()
            tloss, tacc = 0.0, 0.0
            vloss, vacc = 0.0, 0.0
            for images in samplers["train"]:
                # broadcast the images to their respective anchors and samples, concatenate them
                # such that on position N in [0, ..., 2N) the first augmented image has its
                # positive sample. Then, the labels can easily be created with arange func.
                anchors, samples = images
                images = torch.cat((anchors, samples)).float().to(self.device)
                embeddings = self.model(images)

                if plot:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(torch.swapaxes(images[0, :].cpu(), 0, 2).numpy())
                    axs[1].imshow(
                        torch.swapaxes(images[anchors.shape[0], :].cpu(), 0, 2).numpy()
                    )
                    plt.show()

                indices = torch.arange(0, anchors.size(0), device=anchors.device)
                labels = torch.cat((indices, indices)).to(self.device)

                loss = self.criterion(embeddings, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tloss += loss.item()

            with torch.no_grad():
                self.model.eval()
                for images in samplers["valid"]:
                    anchors, samples = images
                    images = torch.cat((anchors, samples)).float().to(self.device)
                    embeddings = self.model(images)

                    indices = torch.arange(0, anchors.size(0), device=anchors.device)
                    labels = torch.cat((indices, indices)).to(self.device)

                    loss = self.criterion(embeddings, labels)
                    vloss += loss.item()

            self.scheduler.step()
            tloss /= len(samplers["train"])
            vloss /= len(samplers["valid"])
            history["tloss"].append(tloss)
            history["vloss"].append(vloss)
            print(
                f"     {epoch + 1:02d}            {tloss:.4f}              {vloss:.4f}                  {tacc:.2f}%                 {vacc:.2f}%"
            )

        if save_model:
            print(f"Done training, saving model parameters to {paramspath}")
            torch.save(self.model, paramspath)

        return history
