---
title: Intro to GPs
description: Introduction to GPs
authors:
    - J. Emmanuel Johnson
path: docs/
source: intro.md
---
# Inference


---
## Variational Inference 

This section outlines a few interesting papers I found where they are trying to improve how we do variational inference. I try to stick to methods where people have tried and succeeded at applying them to GPs. Below are a few key SOTA objective functions that you may come across in the GP literature.  The most common is definitely the Variational ELBO but there are a few unknown objective functions that came out recently and I think they might be useful in the future. We just need to get them implemented and tested. Along the way there have been other modifications.

---
### Variational Evidence Lower Bound (ELBO)

This is the standard objective function that you will find the literature.

**[Scalable Variational Gaussian Process Classification](https://arxiv.org/abs/1411.2005)** - Hensman et. al. (2015)

??? details "Details"
    $$
    \mathcal{L}_{ELBO} = \sum_{i=1}^{N} \mathbb{E}_{q(\mathbf{u})}
    \left[ \mathbb{E}_{f(f|\mathbf{u})} 
    \left[ \log p(y_i | f_i) \right] \right] - \beta 
    D_{KL}\left[ q(\mathbf{u} || p(\mathbf{u})) \right]
    $$

    where:

    * $N$ - number of data points 
    * $p(\mathbf{u})$ - prior distribution for the inducing function values
    * $q(\mathbf{u})$ - variational distribution for the inducing function values
    * $\beta$ - free parameter for the $D_{KL}$ regularization penalization


!!! info "The Variational Gaussian Process by Tran et. al. (2016)"
    -> [Paper](https://arxiv.org/abs/1511.06499)

    -> [Code]()

    -> [Video](http://videolectures.net/iclr2016_tran_variational_gaussian/)

??? info "Scalable Training of Inference Networks for Gaussian-Process Models - Shi et. al. (2019)"
    -> [Paper](https://arxiv.org/pdf/1905.10969.pdf)

    -> [Code](https://github.com/thjashin/gp-infer-net)


??? info "Sparse Orthogonal Variational Inference for Gaussian Processes - Shi et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/1910.10596)

    -> [Code](https://github.com/thjashin/solvegp)


-> [Slides](https://github.com/suhailshergill/ProPL-meetup/tree/master/talks/2016/08_18_Variational_Gaussian_Process)

> Some excellent Slides

---
## Monte Carlo


??? info "Automated Augmented Conjugate Inference for Non-conjugate Gaussian Process Models - Galy-Fajou et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/2002.11451)

    -> [Code](https://github.com/theogf/AugmentedGaussianProcesses.jl)

    -> [Tweet](https://twitter.com/theo_gf/status/1233082060430020609?s=19)


---
### Importance Weighted Variational Inference (IWVI)

**[Deep Gaussian Processes with Importance-Weighted Variational Inference](https://github.com/hughsalimbeni/DGPs_with_IWVI)** - Salimbeni et. al. (2019) -  [Paper](https://arxiv.org/abs/1905.05435) | [Code](https://github.com/hughsalimbeni/DGPs_with_IWVI) | [Video](https://slideslive.com/38917895/gaussian-processes) | [Poster](https://twitter.com/HSalimbeni/status/1137856997930483712/photo/1)  | [ICML 2019 Slides](https://icml.cc/media/Slides/icml/2019/101(12-11-00)-12-12-05-4880-deep_gaussian_p.pdf) | [Workshop Slides](http://tugaut.perso.math.cnrs.fr/pdf/workshop02/salimbeni.pdf)

  > They propose a way to do importance sampling coupled with variational inference to improve single layer and multi-layer GPs and have shown that they can get equivalent or better results than just standard variational inference.

* [Importance Weighting and Variational Inference](https://papers.nips.cc/paper/7699-importance-weighting-and-variational-inference) - Domke & Sheldon (2018)


### Hybrid Schemes

??? info "Automated Augmented Conjugate Inference for Non-conjugate Gaussian Process Models - Galy-Fajou et. al. (2020)"
    > Combines SVI and Gibbs Sampling

    -> [Paper](http://proceedings.mlr.press/v108/galy-fajou20a/galy-fajou20a.pdf)

    -> [Code](https://github.com/theogf/AACI_EXP)

    -> [Package](https://github.com/theogf/AugmentedGaussianProcesses.jl)

---
### Predictive Log Likelihood (PLL)

**[Sparse Gaussian Process Regression Beyond Variational Inference](https://arxiv.org/abs/1910.07123)** - Jankowiak et. al. (2019)

??? details "Details"
    $$
    \begin{aligned}
    \mathcal{L}_{PLL} &= \mathbb{E}_{p_{data}(\mathbf{y}, \mathbf{x})}
    \left[ \log p(\mathbf{y|x})\right] - \beta 
    D_{KL}\left[ q(\mathbf{u}|| p(\mathbf{u})\right] \\
    &\approx  \sum_{i=1}^{N} \log \mathbb{E}_{q(\mathbf{u})}
    \left[ \int p(y_i |f_i) p(f_i | \mathbf{u})df_i \right] - \beta 
    D_{KL}\left[ q(\mathbf{u}) || p(\mathbf{u}) \right]
    \end{aligned}
    $$

    where:

    * $N$ - number of data points 
    * $p(\mathbf{u})$ - prior distribution for the inducing function values
    * $q(\mathbf{u})$ - variational distribution for the inducing function values

---
### Generalized Variational Inference (GVI)


**[Generalized Variational Inference](https://paperswithcode.com/paper/generalized-variational-inference)** - Knoblauch et. al. (2019)
  > A generalized Bayesian inference framework. It goes into a different variational family related to Renyi's family of Information theoretic methods; which isn't very typical because normally we look at the Shannon perspective. They had success applying it to Bayesian Neural Networks and Deep Gaussian Processes.
  
  * [Deep GP paper](https://arxiv.org/abs/1904.02303)

---
## Gradient Descent Regimes

### Natural Gradients (NGs)

**[Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models](https://arxiv.org/abs/1803.09151)** - Salimbeni et. al. (2018) | [Code](https://github.com/GPflow/GPflow/blob/develop/gpflow/training/natgrad_optimizer.py)

  > This paper argues that training sparse GP algorithms with gradient descent can be quite slow due to the need to optimize the variational parameters $q_\phi(u)$ as well as the model parameters. So they propose to use the natural gradient for the variational parameters and then the standard gradient methods for the remaining parameters. They show that the SVGP and the DGP methods all converge much faster with this training regime. I imagine this would also be super useful for the BayesianGPLVM where we also have variational parameters for our inputs as well.

  -> [Blog](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) - Agustinus Kristiadi

  -> [Lecture 3.5 Natural Gradient Optimization (I) | Neural Networks | MLCV 2017](https://www.youtube.com/watch?v=eio6l-Po83o&feature=youtu.be)


* [Noisy Natural Gradient as Variational Inference](https://arxiv.org/abs/1712.02390) - Zhang (2018) - [Code](https://github.com/wlwkgus/NoisyNaturalGradient)
* [PyTorch Implementation](https://github.com/wiseodd/natural-gradients/tree/master/pytorch)


??? info "Parallel training of DNNs with Natural Gradient and Parameter Averaging - Povey et. al. (2014)"
    > A seamingly drop-in replacement for stochastic gradient descent with some added benefits of being shown to improve generalization tasks, stability of the training, and can help obtain high quality uncertainty estimates.

    -> [Paper](https://arxiv.org/abs/1410.7455)

    -> [Code](https://github.com/YiwenShaoStephen/NGD-SGD)

    -> [Blog](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)


??? info "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm - Lui & Wang (2016)"
    > A tractable approach for learning high dimensional prob dist using Functional Gradient Descent in RKHS. It's from a connection with the derivative of the KL divergence and the Stein's identity.

    -> Stein's Method [Webpage](https://sites.google.com/site/steinsmethod/home)

    -> [Pyro Implementation](http://docs.pyro.ai/en/stable/inference_algos.html#module-pyro.infer.svgd)

---
??? info "Robust, Accurate Stochastic Optimization for Variational Inference - Dhaka et. al. (09-2020)"
    > More robust stochastic variational inference method using Polyak-Ruppert averaging with MCMC.

    -> [Paper](https://arxiv.org/abs/2009.00666)

    -> [Tweet](https://twitter.com/avehtari/status/1301792837214797827?s=09)


---
## Uncategorized






---
## Regularization

* [Regularized Sparse Gaussian Processes](https://paperswithcode.com/paper/regularized-sparse-gaussian-processes) - Meng & Lee (2019) [**arxiv**]
  > Impose a regularization coefficient on the KL term in the Sparse GP implementation. Addresses issue where the distribution of the inducing inputs fail to capture the distribution of the training inputs.
