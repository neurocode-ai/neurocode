# Neurocode
A minimalistic Python library for EEG/MEG deep learning research and analysis, primarely focused on self-supervised learning. 

## What does it aim to do?
The goal of `neurocode` is to provide a simple and easy to use API for neuroscientists and EEG/MEG practitioners that may not have the most programming experience. It allows for out-of-the-box loading, reading, and parsing of the most common file types [.fif, .edf, .bdf] based on your predefined directory structure of the local data.

Furthermore, a number of example self-supervised learning samplers and deep neural networks are available to give the practitioners an example of what can be done with the data. 

Finally, in our opinion, feature analysis is an important aspect of any deep learning application. Analysing your neural networks latent space is the only way to truly understand what is being learned, and as such, `neurocode` provides streamlined functions to extract, visualize, and analyse the features of your data.

# License
All code in this repository is to be held under an Apache-2.0 styled license, please see [LICENSE](https://github.com/neurocode-ai/neurocode/blob/main/LICENSE) for more information.