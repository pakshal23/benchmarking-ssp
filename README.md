# A Statistical Framework to Investigate the Optimality of Signal-Reconstruction Methods
This repository contains the code related to the paper "A Statistical Framework to Investigate the Optimality of Signal-Reconstruction Methods" [[paper](https://ieeexplore.ieee.org/abstract/document/10141672)] [[arXiv version](https://arxiv.org/pdf/2203.09920.pdf)], which presents a statistical framework to benchmark the performance of reconstruction algorithms for 1D linear inverse problems, in particular, neural-network-based methods that require large quantities of training data.

Features
------------------
**Setup**: Linear inverse problems (denoising, deconvolution, Fourier sampling / MRI) where the signal is a realization of a 1D sparse stochastic process
* Availability of the goldstandard MMSE estimator (upper limit on reconstruction performance)
* Availability of (unlimited) training data for neural-network-based approaches

Requirements
------------------
* PyTorch (for reconstruction using neural networks)
* [GlobalBioIm](https://github.com/Biomedical-Imaging-Group/GlobalBioIm) library (for reconstruction using $\ell_1$-norm and log penalty regularizers)

Developers
------------------
This framework was developed at the Biomedical Imaging Group, École polytechnique fédérale de Lausanne (EPFL), Switzerland. This work was supported in part by the Swiss National Science Foundation under Grant 200020_184646 / 1 and in part by the European Research Council (ERC Project FunLearn) under Grant 101020573.

Contributors: Pakshal Bohra (pakshalbohra@gmail.com), Pol del Aguila Pla, Jean-François Giovanelli

