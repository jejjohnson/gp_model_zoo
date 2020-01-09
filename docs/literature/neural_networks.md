# Neural Networks and Gaussian Processes


---
## Neural Networks & Deep Gaussian Processes

* [Building Bayesian Neural Networks with Blocks:
On Structure, Interpretability and Uncertainty](https://arxiv.org/pdf/1806.03563.pdf) - Zhou et. al. (2018)


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
## Latest

* [Deep Probabilistic Kernels for Sample-Efficient Learning](https://paperswithcode.com/paper/deep-probabilistic-kernels-for-sample) - Mallick et. al. (2019) [**arxiv**]
  > Propose a deep probabilistic kernel to address 1) traditional GP kernels aren't good at capturing similarities between high dimensional data points and 2) deep neural network kernels are not sample efficient. Has aspects such as Random Fourier Features, semi-supervised learning and utilizes the Stein Variational Gradient Descent algorithm.
* [On the expected behaviour of noise regulariseddeep neural networks as Gaussian processes](https://paperswithcode.com/paper/on-the-expected-behaviour-of-noise) - Pretorius et al (2019) [**arxiv**]
  > They study the impact of noise regularization via droput on Deep Kernel learning.

---
## SOTA