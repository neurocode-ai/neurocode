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

Unit tests for the neurocode.samplers module.
-------------------------------------------------------------------- """
import unittest
from neurocode.samplers import PretextTaskSampler
from torch.utils.data.sampler import Sampler

class TestPretextTaskSampler(unittest.TestCase):
    def test_constructor(self):
        sampler = PretextTaskSampler([], [])
        assert isinstance(sampler, Sampler)
