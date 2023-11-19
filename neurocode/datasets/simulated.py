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

File created: 2023-09-23
Last updated: 2023-11-19
"""

from __future__ import annotations

import numpy as np
import mne
import logging

from .utils import mne_datasets
from mne import make_ad_hoc_cov
from mne.datasets import (
    sample,
    has_dataset,
)
from mne.simulation import (
    simulate_sparse_stc,
    simulate_raw,
    add_noise as simulated_noise,
    add_ecg as simulated_ecg,
    add_eog as simulated_eog,
)
from torch.utils.data import Dataset
from tqdm import tqdm

from typing import (
    Callable,
    Union,
    Optional,
)

logger = logging.getLogger(__name__)


class SimulatedDataset(Dataset):
    def __init__(
        self,
        name: str,
        *,
        seed: Union[int, float] = 19489184,
        **kwargs: dict,
    ):
        """ """
        super(SimulatedDataset, self).__init__()

        if name not in mne_datasets:
            raise ValueError(f"No MNE dataset called: `{name}`.")

        data_path = mne_datasets[name].data_path()
        self._name = name
        self._data_path = data_path

        self._raw_data = {}
        self._labels = {}
        self._raw_simulated = {}
        self._rng = np.random.RandomState(seed)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __iter__(self):
        pass

    def read_from_file(
        self,
        fname: str,
        *,
        set_eeg_reference: bool = True,
        eeg_projection: bool = True,
        **kwargs: dict,
    ) -> SimulatedDataset:
        """ """

        if fname in self._raw_data:
            return self._raw_data[fname]

        raw = mne.io.read_raw_fif(self._data_path / fname)

        if set_eeg_reference:
            raw.set_eeg_reference(projection=eeg_projection)

        labels = mne.read_labels_from_annot(
            self._name,
            subjects_dir=self._data_path / "subjects",
        )

        self._raw_data[fname] = raw
        self._labels[fname] = labels

        return self

    def read_from_files(
        self,
        fnames: list[str],
        **kwargs: dict,
    ) -> SimulatedDataset:
        """ """

        for fname in fnames:
            self._read_from_file(fname, **kwargs)

        return self

    def simulate(
        self,
        fwd_fname: str,
        *,
        n_dipoles: int = 4,
        epoch_duration: float = 2.0,
        harmonic_number: int = 0,
        data_fn: Optional[Callable] = None,
        add_noise: bool = True,
        add_ecg: bool = True,
        add_eog: bool = True,
        iir_filter_coefficients: Optional[list[float]] = None,
        **kwargs: dict,
    ) -> SimulatedDataset:
        """ """

        for name, raw in (pbar := tqdm(self._raw_data.items())):
            pbar.set_description(f"Simulating data for `{name}`.")
            times = raw.times[: int(raw.info["sfreq"] * epoch_duration)]

            fwd = mne.read_forward_solution(self._data_path / fwd_fname)
            src = fwd["src"]
            stc = simulate_sparse_stc(
                src,
                n_dipoles=n_dipoles,
                times=times,
                random_state=self._rng,
            )

            raw_sim = simulate_raw(raw.info, [stc] * 10, forward=fwd)

            if add_noise:
                cov = make_ad_hoc_cov(raw_sim.info)
                simulated_noise(
                    raw_sim,
                    cov,
                    iir_filter=iir_filter_coefficients,
                    random_state=self._rng,
                )

            if add_ecg:
                simulated_ecg(raw_sim, random_state=self._rng)

            if add_eog:
                simulated_eog(raw_sim, random_state=self._rng)

            self._raw_simulated[name] = raw_sim._data

        return self

    def data(self) -> dict[str, mne.io.Raw]:
        """ """
        return self._raw_simulated

    def labels(self) -> dict[str, list[mne.label.Label]]:
        """ """
        return self._labels
