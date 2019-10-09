# Deep Gaussian Processes

These are GP models that stack GPs one after the other. As far as understanding, the best would be lectures as I have highlighted below.

---
## Resources

* Neil Lawrence @ MLSS 2019
    > I would say this is the best lecture to understand the nature of GPs and why we would might want to use them.
  * [Blog](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html) | [Lecture](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html) | [Slides](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html)
* Neil Lawrence @ GPSS 2019
  * [Notes](http://inverseprobability.com/talks/notes/introduction-to-deep-gps.html) | [Lecture](https://youtu.be/eS_na-6ZlCI)
* Maurizio Filippone @ DeepBayes.ru 2018
  > I would say this is the second best lecture because Maurizio gives a nice overview of the GP methods there already are (at the time).
  * [Lecture](https://www.youtube.com/watch?v=zBEV5ezyYmI) | [Slides](http://www.eurecom.fr/~filippon/Talks/talk_dgps_deep_bayes_summer_school_2018.pdf) | [New Slides](http://www.eurecom.fr/~filippon/Talks/talk_deep_bayes_moscow_2019.pdf)



---
## Algorithms

The literature isn’t so big but there a number of different implementations depending on the lab:

1. Stochastic Variational Inference
2. Expectation Propagation
3. Random Feature Expansions

I don’t think there is any best one because I’m almost certain noone has done any complete comparison. But I would try to use implementations in standard libraries where the devs have ironed out the bugs and allow customization; so in our case the doubly stochastic. 

---
### Deep Gaussian Processes - Damianou & Lawrence (2013)

> This paper is the original method of Deep GPs. 

[Paper](http://adamian.github.io/publications.html#DeepGPs) | [Code](http://htmlpreview.github.io/?https://github.com/SheffieldML/deepGP/blob/master/deepGP/html/index.html)

### Doubly Stochastic Variational Inference for Deep Gaussian Processes - Salimbeni & Deisenroth (2017)

> This paper uses stochastic gradient descent for training the Deep GP. I think this achieves the state-of-the-art results thus far. It also has the most implementations in the standard literature.

* [Paper](https://arxiv.org/abs/1705.08933)
* [Code](https://github.com/ICL-SML/Doubly-Stochastic-DGP) | [Pyro](https://fehiepsi.github.io/blog/deep-gaussian-process/) | [GPyTorch](https://gpytorch.readthedocs.io/en/latest/examples/13_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html)


### Deep Gaussian Processes for Regression using Approximate Expectation Propagation - Bui et. al. (2016)

> This paper uses an approximate expectation method for the inference in Deep GPs.

[Paper](https://arxiv.org/abs/1602.04133) | [Code](https://github.com/thangbui/geepee)


### Random Feature Expansions for Deep Gaussian Processes - Cutjar et. al. (2017)

> This implementation uses ideas from random fourier features in conjunction with Deep GPs.

* [Paper I](https://arxiv.org/abs/1610.04386) | [Paper II](https://pdfs.semanticscholar.org/bafa/7e2d586e7bfe77d9a55ac1cff4eb2f6ff292.pdf) |  [Video](https://vimeo.com/238221933) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features)
* [Lecture I]() | [Slides]() | 
* [Lecture (Maurizio)](https://www.youtube.com/watch?v=750fRY9-uq8&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=19&t=0s) | [Slides](http://www.eurecom.fr/~filippon/Talks/talk_deep_bayes_moscow_2019.pdf) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features/blob/master/code/dgp_rff.py)

### Importance Weighted Sampling 

* [Paper](https://arxiv.org/abs/1905.05435) | [Code](https://github.com/hughsalimbeni/DGPs_with_IWVI) | [Video](https://slideslive.com/38917895/gaussian-processes) | [Poster](https://twitter.com/HSalimbeni/status/1137856997930483712/photo/1)  | [ICML 2019 Slides](https://icml.cc/media/Slides/icml/2019/101(12-11-00)-12-12-05-4880-deep_gaussian_p.pdf) | [Workshop Slides](http://tugaut.perso.math.cnrs.fr/pdf/workshop02/salimbeni.pdf) 


---
## Insights


### Problems with Deep GPs

* Deep Gaussian Process Pathologies - [Paper](http://proceedings.mlr.press/v33/duvenaud14.pdf)
  > This paper shows how some of the kernel compositions give very bad estimates of the functions between layers; similar to how residual NN do much better.

### Relations to Neural Networks
