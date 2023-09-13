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
Last updated: 2023-09-13
"""

import numpy as np

from torch.utils.data import Dataset


class RecordingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self._setup(*args, **kwargs)

    def __len__(self):
        return self._info["n_recordings"]

    def __getitem__(self, indices):
        recording, window = indices
        return self._data[recording][window]

    def __iter__(self):
        for idx in range(len(self)):
            yield (self._data[idx], self._labels[idx])

    def _setup(self, datasets, labels, formatted=False, **kwargs):
        if not formatted:
            datasets = {
                recording: dataset for recording, dataset in enumerate(datasets)
            }
            labels = {recording: label for recording, label in enumerate(labels)}

        lengths = {recording: len(d) for recording, d in enumerate(datasets.values())}
        info = {
            "lengths": lengths,
            "n_recordings": len(datasets),
        }
        info = {**info, **kwargs}

        self._data = datasets
        self._labels = labels
        self._info = info

    def get_data(self):
        return self._data

    def get_labels(self):
        return self._labels

    def get_info(self):
        return self._info

    def split(self, ratio=0.6, shuffle=True):
        split_idx = int(len(self) * ratio)
        indices = list(range(len(self)))

        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        X_train = {idx: self.data[k] for idx, k in enumerate(train_indices)}
        Y_train = {idx: self.labels[k] for idx, k in enumerate(train_indices)}
        X_valid = {idx: self.data[k] for idx, k in enumerate(valid_indices)}
        Y_valid = {idx: self.labels[k] for idx, k in enumerate(valid_indices)}

        train_dataset = RecordingDataset(
            X_train, Y_train, formatted=True, sfreq=self.info["sfreq"]
        )
        valid_dataset = RecordingDataset(
            X_valid, Y_valid, formatted=True, sfreq=self.info["sfreq"]
        )
        return (train_dataset, valid_dataset)
