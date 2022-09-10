""" --------------------------------------------------------------------
Copyright [2022] [Wilhelm Ågren]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


File created:   10-09-2022
Last edited:    10-09-2022

Implementation of the base PretextTaskSampler, inheriting from the 
PyTorch sampler object.
-------------------------------------------------------------------- """
import numpy as np
from torch.utils.data.sampler import Sampler

class PretextTaskSampler(Sampler):
    def __init__(self, data, labels, *args, **kwargs):
        self._data = self.data
        self._labels = self.labels
        self._rng = np.random.RandomState(seed=0)

    def __len__(self):
        raise NotImplementedError(
            f''
        )
