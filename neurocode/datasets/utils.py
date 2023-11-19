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

from mne.datasets import *  # noqa  F821

mne_datasets = {
    "brainstorm.bst_auditory": brainstorm.bst_auditory,
    "brainstorm.bst_resting": brainstorm.bst_resting,
    "brainstorm.bst_raw": brainstorm.bst_raw,
    "fnirs_motor": fnirs_motor,
    "hf_sef": hf_sef,
    "kiloword": kiloword,
    "limo": limo,
    "misc": misc,
    "mtrf": mtrf,
    "multimodal": multimodal,
    "opm": opm,
    "sleep_physionet.age": sleep_physionet.age,
    "sleep_physionet.temazepam": sleep_physionet.temazepam,
    "sample": sample,
    "somato": somato,
    "spm_face": spm_face,
    "visual_92_categories": visual_92_categories,
    "phantom_4dbti": phantom_4dbti,
    "refmeg_noise": refmeg_noise,
    "ssvep": ssvep,
    "erp_core": erp_core,
    "epilepsy_ecog": epilepsy_ecog,
    "eyelink": eyelink,
}
