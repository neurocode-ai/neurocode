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

File created: 2022-09-10
Last updated: 2023-09-23
"""

from __future__ import annotations

import logging
import mne
import numpy as np

from torch.utils.data import Dataset
from neurocode.datasets.simulated import SimulatedDataset

from collections import OrderedDict
from typing import (
    Any,
    Union,
)

logger = logging.getLogger(__name__)


class RecordingDataset(Dataset):
    def __init__(
        self,
        data: Union[list[mne.io.Raw], dict[str, mne.io.Raw]],
        labels: Union[list[list[mne.label.Label]], dict[str, list[mne.io.Label]]],
        **kwargs: dict,
    ):
        """ """
        super(RecordingDataset, self).__init__()

        if isinstance(data, list) and isinstance(labels, list):
            raise ValueError(
                f"Can not infer recording names when both `data` and `labels` are of "
                f"type `list`. At least one of them have to be of type `dict`."
            )

        self._info = {}
        self._format_data_and_labels(data, labels, **kwargs)

    def __len__(self) -> int:
        """ """
        return self._info["n_recordings"]

    def __getitem__(
        self,
        indices: tuple[Union[int, str], int],
    ) -> Union[int, float, np.ndarray]:
        """ """
        recording, window = indices

        if isinstance(recording, int):
            recording = self._data.keys().index(recording)

        return self._data[recording][window]

    def __iter__(self) -> tuple[mne.io.Raw, list]:
        for name in range(len(self)):
            yield (self._data[name], self._labels[name])

    def _format_data_and_labels(
        self,
        data: Union[list[mne.io.Raw], dict[str, mne.io.Raw]],
        labels: Union[list[list[mne.label.Label]], dict[str, list[mne.io.Label]]],
        **kwargs,
    ):
        """ """

        if isinstance(data, list):
            data = OrderedDict((name, raw) for name, raw in zip(labels.keys(), data))

        if isinstance(labels, list):
            labels = OrderedDict(
                (name, label) for name, label in zip(data.keys(), labels)
            )

        info = {}
        info = {**info, **kwargs}
        info["n_recordings"] = len(data)
        info["lengths"] = {name: len(raw) for name, raw in data.items()}

        self._data = data
        self._labels = labels
        self._info = info

    def data(self) -> dict[str, mne.io.Raw]:
        """ """
        return self._data

    def labels(self) -> dict[str, list[mne.label.Label]]:
        """ """
        return self._labels

    def info(self) -> dict[str, Any]:
        """ """
        return self._info

    def train_valid_split(
        self,
        *,
        ratio: float = 0.6,
        shuffle: bool = True,
    ) -> tuple[RecordingDataset, RecordingDataset]:
        """ """
        split_idx = int(len(self) * ratio)
        indices = np.arange(len(self))

        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        X_train = {}
        Y_train = {}
        for i, name in enumerate(self._data.keys()):
            if i in train_indices:
                X_train[name] = self._data[name]
                Y_train[name] = self._labels[name]

        X_valid = {}
        Y_valid = {}
        for i, name in enumerate(self._data.keys()):
            if i in valid_indices:
                X_valid[name] = self._data[name]
                Y_valid[name] = self._labels[name]

        train_dataset = RecordingDataset(
            data=X_train,
            labels=Y_train,
            **self._info,
        )

        valid_dataset = RecordingDataset(
            data=X_valid,
            labels=Y_valid,
            **self._info,
        )

        return (train_dataset, valid_dataset)

    @classmethod
    def from_simulated(
        cls,
        dataset: SimulatedDataset,
        **kwargs: dict,
    ) -> RecordingDataset:
        """ """
        cls(
            data=dataset.data(),
            labels=dataset.labels(),
            **kwargs,
        )
