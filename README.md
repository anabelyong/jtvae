# Junction Tree Variational Autoencoder for Molecular Graph Generation (Python 3.7)

Official implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364). Inspired by [https://github.com/Bibyutatsu/FastJTNNpy3] and [https://github.com/hello-maker/JunctionTreeVAE/tree/main]

# Accelerated Version
I have implemented the JTVAE with 'fast_molvae/' and 'fast_jtnn/'. 

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2019.09.01)
* Python (version == 3.7)
* PyTorch (version >= 1.13)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Tutorial
Head to 'fast_molvae/README.md'

# Quick Start
The following directories contains the most up-to-date implementations of our model:
* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The following directories provides scripts for the experiments in our original ICML paper:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.


# Contact
Wengong Jin (wengong@csail.mit.edu)
Anabel Yong (ucabyon@ucl.ac.uk/ anabel.yong@ignotalabs.ai)
