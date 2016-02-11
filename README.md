## Autoencoding beyond pixels using a learned similarity measure

*[Anders Boesen Lindbo Larsen](https://github.com/andersbll)*, *[Søren Kaae Sønderby](https://github.com/skaae)*, *[Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh)*, *[Ole Winther](http://cogsys.imm.dtu.dk/staff/winther)*

Implementation of the method described in our [Arxiv paper](http://arxiv.org/abs/1512.09300).


### Abstract
We present an autoencoder that leverages learned representations to better measure similarities in data space.
By combining a variational autoencoder with a generative adversarial network we can use learned feature representations in the GAN discriminator as basis for the VAE reconstruction objective.
Thereby, we replace element-wise errors with feature-wise errors to better capture the data distribution while offering invariance towards e.g. translation.
We apply our method to images of faces and show that it outperforms VAEs with element-wise similarity measures in terms of visual fidelity.
Moreover, we show that the method learns an embedding in which high-level abstract visual features (e.g. wearing glasses) can be modified using simple arithmetic.


### Getting started
We have tried automatizing everything from data fetching to generating pretty images.
This means that you can get started in two steps:

 1. Install [CUDArray](https://github.com/andersbll/cudarray) and [DeepPy](https://github.com/andersbll/deeppy).
 2. Run `python celeba_aegan.py`.

You can also try out the other scripts if you want to experiment with different models/datasets.


### Examples
Coming soon ...


### Implementation references
We wish to thank the authors of the following projects for inspiration.
Our method would never have gotten off the ground without the insights gained from inspecting their code.
 - [The Eyescream Project](https://github.com/facebook/eyescream).
 - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code).
 - Joost van Amersfoort's VAE implementations ([Theano](https://github.com/y0ast/Variational-Autoencoder) and [Torch](https://github.com/y0ast/VAE-Torch)).
 - [Ian Goodfellow's GAN implementation](https://github.com/goodfeli/adversarial).
 - [Parmesan](https://github.com/casperkaae/parmesan)
