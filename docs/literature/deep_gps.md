---
title: Deep Gaussian Processes
description: Deep Gaussian Processes
authors:
    - J. Emmanuel Johnson
path: docs/literature
source: deep_gps.md
---
# Deep Gaussian Processes

These are GP models that stack GPs one after the other. As far as understanding, the best would be lectures as I have highlighted below.

---
## 👨🏽‍🏫 | 👩🏽‍🏫 Resources

**Neil Lawrence @ MLSS 2019**
> I would say this is the best lecture to understand the nature of GPs and why we would might want to use them.
>
> [Blog](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html) | [Lecture](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html) | [Slides](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html)

**Neil Lawrence @ GPSS 2019**
  > [Notes](http://inverseprobability.com/talks/notes/introduction-to-deep-gps.html) | [Lecture](https://youtu.be/eS_na-6ZlCI)

**Maurizio Filippone @ DeepBayes.ru 2018**
  > I would say this is the second best lecture because Maurizio gives a nice overview of the GP methods there already are (at the time).
  >
  > [Lecture](https://www.youtube.com/watch?v=zBEV5ezyYmI) | [Slides](http://www.eurecom.fr/~filippon/Talks/talk_dgps_deep_bayes_summer_school_2018.pdf) | [New Slides](http://www.eurecom.fr/~filippon/Talks/talk_deep_bayes_moscow_2019.pdf)



---

## Algorithms

The literature isn’t so big but there a number of different implementations depending on the lab:

1. Variational Inference
   > This is the most popular method and has been pursued the most. It's also the implementation that you will find standard in libraries like GPyTorch, GPFlow and Pyro.
2. Expectation Propagation
   > This group used expectation propagation to train the GP. They haven't really done so much since then and I'm not entirely sure why this line of the DGP has gone a bit dry. It would be nice if they resumed. I suspect it may be because of the software. I haven't seen too much software that focuses on clever expectation propagation schemes; they mainly focus on variational inference and MC sampling schemes.
3. MC sampling
   > One lab has tackled this where you can use some variants of MC sampling to train a Deep GP. You'll find this standard in many GP libraries because it's fairly easy to integrate in almost any scheme. MC sampling is famous for being slow but the community is working on it. I imagine a break-through is bound to happen.
4. Random Feature Expansions
   > This uses RFF to approximate a GP and then stacks these on top. I find this a big elegent and probably the simplest. But I didn't see too much research on the tiny bits of the algorithm like the training or the initialization procedures.

I don’t think there is any best one because I’m almost certain noone has done any complete comparison. I can say that the VI one is the most studied because that lab is still working on it. In the meantime, personally I would try to use implementations in standard libraries where the devs have ironed out the bugs and allowed for easy customization and configuration; so basically the doubly stochastic. 

---

### Variational Inference

**[Deep Gaussian Processes](http://adamian.github.io/publications.html#DeepGPs)** - Damianou & Lawrence (2013)

> This paper is the original method of Deep GPs. It might not be useful for production but there are still many insights to be had from the originators.
> [Code](http://htmlpreview.github.io/?https://github.com/SheffieldML/deepGP/blob/master/deepGP/html/index.html)

**[Nested Variational Compression in Deep Gaussian Processes]()** - Hensman & Lawrence (2014)

**[Doubly Stochastic Variational Inference for Deep Gaussian Processes](https://arxiv.org/abs/1705.08933)** - Salimbeni & Deisenroth (2017)

> This paper uses stochastic gradient descent for training the Deep GP. I think this achieves the state-of-the-art results thus far. It also has the most implementations in the standard literature.

* [Code](https://github.com/ICL-SML/Doubly-Stochastic-DGP) | [Pyro](https://fehiepsi.github.io/blog/deep-gaussian-process/) | [GPyTorch](https://gpytorch.readthedocs.io/en/latest/examples/13_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html)


---

### Random Fourier Features


**[Random Feature Expansions for Deep Gaussian Processes](https://arxiv.org/abs/1610.04386)** - Cutjar et. al. (2017)

> This implementation uses ideas from random fourier features in conjunction with Deep GPs.

* [Paper II](https://pdfs.semanticscholar.org/bafa/7e2d586e7bfe77d9a55ac1cff4eb2f6ff292.pdf) |  [Video](https://vimeo.com/238221933) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features)
* [Lecture I]() | [Slides]() | 
* [Lecture (Maurizio)](https://www.youtube.com/watch?v=750fRY9-uq8&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=19&t=0s) | [Slides](http://www.eurecom.fr/~filippon/Talks/talk_deep_bayes_moscow_2019.pdf) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features/blob/master/code/dgp_rff.py)


---

### MC Sampling

**[Learning deep latent Gaussian models with Markov chain Monte Carlo]()** - Hoffman (2017)

**[Inference in Deep Gaussian Processes Using Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1806.05490)** - Havasi et. al. (2018)

---

### Expectation Propagation

**[Deep Gaussian Processes for Regression using Approximate Expectation Propagation](https://arxiv.org/abs/1602.04133)** - Bui et. al. (2016)

> This paper uses an approximate expectation method for the inference in Deep GPs. 

[Paper](https://arxiv.org/abs/1602.04133) | [Code](https://github.com/thangbui/geepee)

---

### Hybrids

**[Deep Gaussian Processes with Importance-Weighted Variational Inference](https://github.com/hughsalimbeni/DGPs_with_IWVI)** - Salimbeni et. al. (2019)

This paper uses the idea that our noisy inputs are instead 'latent covariates' instead of additive noise or that our input itself is a latent covariate. They also propose a way to do importance sampling coupled with variational inference to improve single layer and multiple layer GPs and have shown that they can get equivalent or better results than just standard variational inference. The latent variables alone will improve performance for both the IWVI and the VI training procedures.
* [Paper](https://arxiv.org/abs/1905.05435) | [Code](https://github.com/hughsalimbeni/DGPs_with_IWVI) | [Video](https://slideslive.com/38917895/gaussian-processes) | [Poster](https://twitter.com/HSalimbeni/status/1137856997930483712/photo/1)  | [ICML 2019 Slides](https://icml.cc/media/Slides/icml/2019/101(12-11-00)-12-12-05-4880-deep_gaussian_p.pdf) | [Workshop Slides](http://tugaut.perso.math.cnrs.fr/pdf/workshop02/salimbeni.pdf) 



---

## Misc

??? tip "Inter-domain Deep Gaussian Processes - Rudner et. al. (2020)"
      -> [Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/5904-Paper.pdf)

      -> [Slides](https://icml.cc/media/Slides/icml/2020/virtual(no-parent)-16-19-00UTC-6718-inter-domain_de.pdf)

??? info "Interpretable Deep Gaussian Processes with Moments - Lu et. al. (2020)"
    -> [Paper](http://proceedings.mlr.press/v108/lu20b.html)

??? info "Beyond the Mean-Field: Structured Deep Gaussian Processes Improve the Predictive Uncertainties - LIngdinger et al (2020)"
      -> [Paper](https://arxiv.org/abs/2005.11110)

      -> [Code](https://github.com/boschresearch/Structured_DGP)

---

## Insights




### Problems with Deep GPs

* Deep Gaussian Process Pathologies - [Paper](http://proceedings.mlr.press/v33/duvenaud14.pdf)
  > This paper shows how some of the kernel compositions give very bad estimates of the functions between layers; similar to how residual NN do much better.

