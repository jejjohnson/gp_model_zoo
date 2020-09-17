# My Ideas


## Datasets

* Complex 1D Time Series
* MultiOutput
  * Ask Jordi (crops?)
  * Ocean Data
* Heteroscedastic Noise Models
* FLUX Data (ask Alvaro)
* Earth System Data Cube (ESDC) 
  > A potentially complex Spatial Temporal dataset. We could get some nice pretty maps.
* ISP Stuff
  > Some tutorials - [Courses](https://isp.uv.es/courses.html)
* Light Curves
  > It would be nice to involve the astrophysics community. A first pass example: [Exoplanet Example](https://docs.exoplanet.codes/en/stable/tutorials/gp/)

## Tutorials

> It would be nice to recreate some staple tutorials that have either been in schools or have appeared on the documentation of websites.

* GPs from scratch - [PyTorch](https://github.com/ebonilla/gaussianprocesses/blob/master/notebooks/gp-inference.ipynb) | [JAX](https://jejjohnson.github.io/research_journal/appendix/gps/1_introduction/) | [Marc Slides](https://deisenroth.cc/teaching/2019-20/linear-regression-aims/lecture_gaussian_processes.pdf)
  > It would be nice to do a GP from scratch using different libraries. And then slowly refactoring until we become one with the library. A must-do for newcomers but then...don't ever do it again.
* Comparing Fully Bayesian (Hierarchical, Priors on Params) - [Paper](http://proceedings.mlr.press/v118/lalchand20a/lalchand20a.pdf) | [Code](https://github.com/vr308/Bayesian-GPs) | [Demo](https://github.com/vr308/Bayesian-GPs/blob/master/sparse_gp_regression.py)
  > A good and simple paper where they compare different GP implementations with hierarchical parameters (so priors on parameters). Shows MCMC and VI. I didn't see any priors on the params within the code.
* [Pyro - GP Tutorial](http://pyro.ai/examples/gp.html)
  > An excellent tutorial showing step-by-step from simple GP, Sparse GP and Sparse Variational GP. It would be great to have the same thing with GPyTorch, TensorFlow and PyMC3.
* [GPSS Labs 2020](http://gpss.cc/gpss20/labs)
  > These are great labs. But they need to stop using GPy. Outdated software. So I would like to rewrite this in GPFlow (the successor) and perhaps in **pymc3** and **GPyTorch**. [Example - GPs w. GPyTorch](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/GP/gp_solution.ipynb)
* [GPs w. GPyTorch](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/GP/gp_solution.ipynb) by DeepBayes2019
  > An excellent tutorial and an exact replica as the GPy tutorial.
* [GP Intro in Python](https://adamian.github.io/talks/Damianou_GP_tutorial.html)
  > A good tutorial from Andreas Damianou.

## Graphs

* A While Algorithm should you choose chart - [example](https://github.com/siruil/GPyTorch-Wrapper/blob/master/flowchart.pdf)
  > I might be able to ask the devs this one, but it would be nice to have a chart about which algorithm would one recommend.
* Picture of the Relations - [Slides](https://www.semanticscholar.org/paper/A-Tutorial-on-Gaussian-Processes-%28or-why-I-don%27t-Ghahramani/e837f153e86d4c0a580a22df07a5140c4259530d?p2df)
  > A nice chart to show the connection between weights, functions, kernels and bayesian.
* Picture of the Libraries and their parents - [Example for Numpy](https://www.nature.com/articles/s41586-020-2649-2/figures/2)
* GP graphical models - [Package](https://docs.daft-pgm.org/en/latest/) | 
* ReDo Francois Plot - Train vs Model

## Algorithms

* Exact GP (GP)
* Sparse GP (SGP)
* Stochastic Variational GP (SVGP)
* Deep Gaussian Processes (DGP) - [GPyTorch](https://docs.gpytorch.ai/en/latest/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html) | [GPFlow2.0](https://github.com/FelixOpolka/Deep-Gaussian-Process) | [Pyro](https://fehiepsi.github.io/blog/deep-gaussian-process/)
* Deep Kernel Learning (DKL)
* Random Fourier Features

### Inference Schemes

* Exact
* Variational Inference
* Monte Carlo
* Expectation Propagation
* Importance Weighted Sampling

## Packages

#### PyMC3

I think this is definitely worth looking it. It's a staple package in the Bayesian community that deserves some love. 

**Tutorials**
* [Exact GP](https://docs.pymc.io/notebooks/GP-Marginal.html)
* [Fully Bayesian Exact GP](https://docs.pymc.io/notebooks/GP-Latent.html) - MCMC sampling methods
* [Sparse GP](https://docs.pymc.io/notebooks/GP-SparseApprox.html) - FITC, VFE
* Fully Bayesian Sparse GP?
* Mauna Loa Example - [Part 1](https://docs.pymc.io/notebooks/GP-MaunaLoa.html) | [Part 2](https://docs.pymc.io/notebooks/GP-MaunaLoa2.html)

#### Edward2

> An attempt to do "drop-in" GP layers that are keras-like. This is just awesome. I anticipate there will be tricks to this, but I feel like this is definitely the future so I'd like to explore this.


* [Bayesian Layers](https://github.com/google/edward2/tree/master/edward2/tensorflow/layers#bayesian-layers)


## Improvements

#### Reproducibility

* Colab Notebooks!
* Binder!

#### Logging

* `tqdm` for the loops
* `wandb` for the logging
* `TensorBoard` for others

#### BoilerPlate Code

* [Demo Wrapper](https://github.com/yucho147/GP)
* `pytorch-lightning` for the PyTorch boilerplate code
* `skorch` for the PyTorch boilerplate code and GP capabilities

#### Plotting

It would be nice to demonstrate some neat plotting features in all of the tutorials. Perhaps with more increasing complexity with every tutorial.

* Bokeh - pymc3 tutorial with [Mauna Loa](https://docs.pymc.io/notebooks/GP-MaunaLoa.html)
* [GPy](https://gpy.readthedocs.io/en/deploy/GPy.plotting.html) - All the plotting functions from the GP library
* [Probflow](https://github.com/brendanhasz/probflow) - All of the Bayesian stuff from the Probflow library


## Misc Resources


### JAX + GPs

**GPs w. Tensorflow**

* Exact GP 
  * [Marginal Likelihood](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/GaussianProcessRegressionModel#optimize_model_parameters_via_maximum_marginal_likelihood)
  * [MCMC Sampling](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/GaussianProcessRegressionModel#marginalization_of_model_hyperparameters)
* Variational GP
  * [ELBO](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions/VariationalGaussianProcess#usage_examples)

**GPs + Numpyro**

* Exact GP
  * [NUTS Sampler](http://pyro.ai/numpyro/examples/gp.html)


## Library Tutorials

### TensorFlow 2.0

* [Eat TensorFlow2 in 30 Days](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)


### PyMC3

* [Bayesian Regression in PYMC3 using MCMC & Variational Inference](https://alexioannides.com/2018/11/07/bayesian-regression-in-pymc3-using-mcmc-variational-inference/)


## Algorithms


### Inference


#### Expectation Propagation

* Deep Gaussian Processes for Regression using Approximate Expectation Propagation - [Paper](http://jmlr.org/proceedings/papers/v48/bui16.pdf) | [Code](https://github.com/thangbui/deepGP_approxEP/) | [Code v2](https://github.com/thangbui/geepee)
* A Unifying Framework for Sparse Gaussian Process Approximation using Power Expectation Propagation - [Paper](https://arxiv.org/abs/1605.07066) | [Code](https://github.com/thangbui/sparseGP_powerEP/)