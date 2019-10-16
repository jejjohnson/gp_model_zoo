# Kernel Functions

---
## Deep Kernel Learning

This is a Probabilistic Neural Network (PNN). It's when we try to learn features through a Neural Network and then on the last layer, we fit a Gaussian Process. It's a great idea and I think that this has a lot of potential. One of the criticisms of people in the GP community ([Bonilla et. al., 2016](https://arxiv.org/abs/1610.05392)) is that we don't typically use very expressive kernels. That's where the power of GPs come from. So if we can have kernels from Neural Networks (one of the most expressive ML methods available to date), then we can get a potentially great ML algorithm. Even in practice, [a developer](https://fehiepsi.github.io/blog/deep-gaussian-process/) have stated that we can get state-of-the-art results with some minimum tweaking of the architecture.

**Comments**: 
* I've also heard this called "Deep Feature Extraction".
* This is NOT a Deep GP. I've seen one paper that incorrectly called it that. A deep GP is where we stack GPs on top of each other. See the [deep GP](deep_gps.md) guide for more details.


**Literature**

* [Deep Kernel Learning](https://arxiv.org/abs/1511.02222) - Wilson et. al. (2015)
* [Stochastic Variational Deep Kernel learning](https://papers.nips.cc/paper/6426-stochastic-variational-deep-kernel-learning) - Wilson et. al. (2016)
* [A Representer Theorem for Deep Kernel Learning](http://jmlr.org/papers/volume20/17-621/17-621.pdf) - Bohn et. al. (2019)


**Code**

* [TensorFlow Implementation](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb) | [GP Dist Example](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VariationalGaussianProcess)
* [Pyro Implementation](https://pyro.ai/examples/dkl.html)
* [GPFlow Implementation](https://nbviewer.jupyter.org/github/GPflow/GPflow/blob/develop-2.0/doc/source/notebooks/tailor/gp_nn.ipynb)
* [GPyTorch Implementation](https://gpytorch.readthedocs.io/en/latest/examples/05_Scalable_GP_Regression_Multidimensional/KISSGP_Deep_Kernel_Regression_CUDA.html)

---
## Software

* Multiple Kernel Learning - [MKLpy](https://github.com/IvanoLauriola/MKLpy)
* Kernel Methods - [kernelmethods](https://github.com/raamana/kernelmethods)
* [pykernels](https://github.com/gmum/pykernels/tree/master)
    > A huge suite of different python kernels.
* [kernpy](https://github.com/oxmlcs/kerpy)
  > Library focused on statistical tests
* [keops](http://www.kernel-operations.io/keops/index.html)
  > Use kernel methods on the GPU with autograd and without memory overflows. Backend of numpy and pytorch.
* [pyGPs]()
  > This is a GP library but I saw quite a few graph kernels implemented with different Laplacian matrices implemented.
* [megaman]()
  > A library for large scale manifold learning. I saw quite a few different Laplacian matrices implemented.