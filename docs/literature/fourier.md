---
title: Fourier
description: GPs and Fourier Representation
authors:
    - J. Emmanuel Johnson
path: docs/literature/
source: fourier.md
---
# GPs + Fourier Representations

> Any content related to GPs and Fourier representations

## Key Words

* Random Fourier Features
* Sparse Spectrum


## Random Fourier Features

??? tip "Random Features for Large-Scale Kernel Machines by Rahimi & Recht"
    -> [Paper](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines)


??? info "Nyström Method vs Random Fourier Features: A Theoretical and Empirical Comparison by Yang et al (2012)"
    -> [Paper](https://papers.nips.cc/paper/4588-nystrom-method-vs-random-fourier-features-a-theoretical-and-empirical-comparison)

??? info "The Geometry of Random Features by Choromanski (2012)"
    > A paper showing how orthogonal random features might be better.

    -> [Paper](http://proceedings.mlr.press/v84/choromanski18a.html)

🌐 [Reflections on Random Kitchen Sinks](http://www.argmin.net/2017/12/05/kitchen-sinks/) by Rahimi and Recht (12-2017)
> An interesting blog from the authors about the origins of their idea.

🌐 [Random Fourier Features](http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/) (12-2019) by Gregory Gundersen
> Excellent blog post going step-by-step of how to do Fourier features.

## Fast-Food Approximations

??? info "Fastfood: Approximate Kernel Expansions in Loglinear Time by Viet Le et. al. (2014)"
    -> [Paper](https://arxiv.org/abs/1408.3060)

    -> [Video](http://videolectures.net/nipsworkshops2012_smola_kernel/)

    -> [Code](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.kernel_approximation.Fastfood.html)

??? info "A la Carte - Learning Fast Kernels by Yang et. al. (2014)"
    -> [Paper](https://arxiv.org/abs/1412.6493)

??? info "Efficient Approximate Inference with Walsh-Hadamard Variational Inference by Rossi et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/1912.00015)

---

## Sparse Spectrum Gaussian Processes

These are essentially the analogue to the random fourier features for Gaussian processes.

??? info "Sparse Spectrum Gaussian Process Regression - Lázaro-Gredilla et. al. (2010)"
    > The original algorithm for SSGP.

    -> [Paper](http://jmlr.csail.mit.edu/papers/v11/lazaro-gredilla10a.html)


### Code

📝 [Numpy](https://github.com/marcpalaci689/SSGPR)

📝 [GPFlow](https://github.com/jameshensman/VFF/blob/master/VFF/ssgp.py)

📝 [GPyTorch](https://docs.gpytorch.ai/en/v1.2.0/examples/01_Exact_GPs/Spectral_Delta_GP_Regression.html)
> They call it a mixture of Deltas.

---

## Variational

The SSGP algorithm had a tendency to overfit. So they added some additional parameters to account for the noise in the inputs making the marginal likelihood term intractable. They added variational methods to deal with the


??? info "Improving the Gaussian Process Sparse Spectrum Approximation by Representing Uncertainty in Frequency Inputs - Gal et. al. (2015)"
    > "...proposed variational inference in a sparse spectrum model that is derived from a GP model." - Hensman et. al. (2018)

    -> 📝 [Theano](https://github.com/yaringal/VSSGP)

    -> 📝 [autograd](https://github.com/marcpalaci689/VSSGPR)

    -> Yarin Gal's Stuff - [website](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Improving)

??? info "Variational Fourier Features for Gaussian Processes -  Hensman et al (2018)"  
    > "...our work aims to directly approximate the posterior of the true models using a variational representation." - Hensman et. al. (2018)

    -> [Paper](http://www.jmlr.org/papers/volume18/16-579/16-579.pdf)

    -> [GPFlow](https://gpflow.readthedocs.io/en/develop/notebooks/advanced/variational_fourier_features.html)

??? info "Know Your Boundaries: Constraining Gaussian Processes by Variational Harmonic Features by Solin & Kok (2019)"
    -> [Paper](https://arxiv.org/abs/1904.05207)

    -> [GPFlow](https://github.com/AaltoML/boundary-gp)

---

### Uncertain Inputs

I've only seen one paper that attempts to extend this method to account for uncertain inputs.

??? info "Prediction under Uncertainty in Sparse Spectrum Gaussian Processes with Applications to Filtering and Control - Pan et. al. (2017)"
    > This is the only paper I've seen that tries to extend this method

    -> [Paper](http://proceedings.mlr.press/v70/pan17a.html)

---

??? info "Deep Sigma Point Processes by Jankowiak et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/2002.09112)

    -> [Code]()