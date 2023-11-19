<div align="center">
<br/>
<div align="left">
<br/>
<p align="center">
<a href="https://github.com/neurocode-ai/neurocode">
<img align="center" width=40% src="https://github.com/neurocode-ai/neurocode/blob/main/docs/images/neurocode_logo.webp"></img>
</a>
</p>
</div>
  
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Lint style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPi version](https://img.shields.io/pypi/v/neurocode.svg)](https://pypi.org/project/neurocode/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/neurocode-ai/neurocode/graph/badge.svg?token=IQD60CY83U)](https://codecov.io/gh/neurocode-ai/neurocode)
[![CI](https://github.com/neurocode-ai/neurocode/actions/workflows/ci.yml/badge.svg)](https://github.com/neurocode-ai/neurocode/actions/workflows/ci.yml)
[![Tests](https://github.com/neurocode-ai/neurocode/actions/workflows/tests.yml/badge.svg)](https://github.com/neurocode-ai/neurocode/actions/workflows/tests.yml)

</div>

## 🔎 Overview
A minimalistic Python library for EEG/MEG deep learning research, primarely focused on self-supervised learning. 

## 📦 Installation
Either clone this repository and perform a local install accordingly
```
git clone https://github.com/neurocode-ai/neurocode.git
cd neurocode
poetry install
```
or install the most recent release from the Python Package Index (PyPI).
```
pip install neurocode
```

## 🚀 Example usage
Below you can see an example adapted for a SSL training workflow using the SimCLR framework.

```python
import torch

from pytorch_metric_learning import losses
from neurocode.datasets import SimulatedDataset, RecordingDataset
from neurocode.samplers import SignalSampler
from neurocode.models import SignalNet
from neurocode.training import SimCLR
from neurocode.datautil import manifold_plot, history_plot

sample_data = SimulatedDataset("sample", seed=7815891891337)
sample_data.read_from_file("MEG/sample/sample_audvis_raw.fif")

# create random extrapolated data from the raw MEG recording,
# you need to provide a location to a forward solution (source space) to use
sample_data.simulate("MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif")

dataset = RecordingDataset(sample_data.data(), sample_data.labels(), sfreq=200)
train, valid = dataset.train_valid_split(split=0.75)

samplers = {
  'train': SignalSampler(train.data(), train.labels(), train.info(), ...),
  'valid': SignalSampler(valid.data(), valid.labels(), valid.info(), ...),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignalNet(...)
optimizer = torch.optim.Adam(model.parameters(), ...)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ...)
criterion = losses.NTXentLoss()

# train the neural network using the self-supervised learning SimCLR framework,
# save or plot the history to see training loss evolution
simclr = SimCLR(model, device, ...)
history = simclr.fit(samplers, save_model=True)

```

## 📋 License
All code is to be held under a general MIT license, please see [LICENSE](https://github.com/neurocode-ai/neurocode/blob/main/LICENSE) for specific information.
