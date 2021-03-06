---
title: Sparse GPs
description: Sparse Gaussian Processes
authors:
    - J. Emmanuel Johnson
path: docs/literature
source: sparse_gps.md
---
# Sparse GPs

## 📜 Papers


### Subset Methods

> All of these methods in some way shape or form, are trying to reduce the size of the kernel matrix.

??? fire "Nystrom Approximation"
    -> [Using Nystrom to Speed Up Kernel Machines](https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf)- Williams & Seeger (2001)

    ---

    -> [scikit-learn](https://scikit-learn.org/stable/modules/kernel_approximation.html#nystroem-kernel-approx)
    
    -> [KRR Setting](https://github.com/jejjohnson/kernellib/blob/04691f8a8c058d83addb2556ed99d342dc3c8dfc/kernellib/regression/large_scale.py#L98)

    -> [Nice Blog](https://maelfabien.github.io/machinelearning/largescale/#iv-nystr%C3%B6m-approximation) (**Slow to Load**)


??? info "Random Fourier Features"


??? info "FastFood"
    -> [Fastfood: Approximate Kernel Expansions in Loglinear Time](https://arxiv.org/abs/1408.3060) by Viet Le et. al. (2014) | [Video](http://videolectures.net/nipsworkshops2012_smola_kernel/) | [Code](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.kernel_approximation.Fastfood.html)

    -> [A la Carte - Learning Fast Kernels](https://arxiv.org/abs/1412.6493)  by Yang et. al. (2014)

    -> [Efficient Approximate Inference with Walsh-Hadamard Variational Inference](https://arxiv.org/abs/1912.00015) by Rossi et. al. (2020)

### Mixture of Experts


??? info "Deep Structured Mixtures of Gaussian Processes - Trapp et. al. (2020)"
    -> [Paper](http://proceedings.mlr.press/v108/trapp20a/trapp20a.pdf)

    -> [Cde](https://github.com/trappmartin/DeepStructuredMixtures)

## Inducing Points

??? fire "Fully Independent Training Conditional (FITC)"
    -> [Sparse Gaussian Processes Using Pseudo-Inputs](http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf) - Snelson and Ghahramani (2006)
    
    -> [Flexible and Efficient GP Models for Machine Learning](http://www.gatsby.ucl.ac.uk/~snelson/thesis.pdf) - Snelson (2007)

    -> [Variational Orthogonal Features](https://paperswithcode.com/paper/variational-orthogonal-features) - by Burt et al (2020)

??? fire "Parametric Gaussian Process Regressors by Jankowiak et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/1910.07123)

    -> [GPyTorch Code](https://docs.gpytorch.ai/en/v1.2.0/examples/04_Variational_and_Approximate_GPs/Approximate_GP_Objective_Functions.html#Objective-Funtion-2)-The-Predictive-Log-Likelihood)

??? fire "Rethinking Sparse Gaussian Processes: Bayesian Approaches to Inducing-Variable Approximations by Rossi et al (2020)"
    -> [Paper](https://arxiv.org/abs/2003.03080)

## Posterior Approximation


#### Variational Free Energy


??? tip "Variational Learning of Inducing Variables in Sparse GPs - Titsias (2009)"
    > The OG for this method. You'll see this paper cited a lot!

    -> [Paper](https://pdfs.semanticscholar.org/9c13/b87b5efb4bb011acc89d90b15f637fa48593.pdf)

??? tip "Understanding Probabilistic Sparse GP Approx - Bauer et. al. (2016)"
    > A good paper which highlights some import differences between the FITC, DTC and VFE. It provides a clear notational differences and also mentions how VFE is a special case of DTC.

    -> [Paper](https://arxiv.org/pdf/1606.04820.pdf)


??? info "Other Papers"

    -> [On Sparse Variational meethods and the KL Divergence between Stochastic Processes](https://arxiv.org/pdf/1504.07027.pdf) - Matthews et. al. (2015)


#### Stochastic Variational Inference (SVI)

??? tip "Gaussian Processes for Big Data - Hensman et al. (2013)"

    -> [Paper](https://arxiv.org/pdf/1309.6835.pdf)



    
#### Expectation Propagation (EP)


??? info "A Unifying Framework for Gaussian Process Pseudo-Point Approximations using Power Expectation Propagation - Bui (2017)"
    > A good summary of all of the methods under one unified framework called the Power Expectation Propagation formula.

    -> [Paper](http://jmlr.org/papers/volume18/16-603/16-603.pdf)

    ---

    -> [Code](https://github.com/thangbui/sparseGP_powerEP/tree/master/python/sgp): Exact and Sparse Power EP

    -> [Updated](https://github.com/thangbui/geepee) | [Other](https://github.com/emakryo/gaussian_process/blob/master/gaussian_process/expectation_propagation.py)

    -> [Related Code](https://github.com/AaltoML/kalman-jax)
  
#### Variational

??? fire "**Rates of Convergence for Sparse Variational Gaussian Process Regression** - Burt et. al. (2019)"
    > All you need to do is cite this paper whenever people don't believe that Sparse GPs aren't good at approximating Exact GPs.
    
    -> [Paper](https://arxiv.org/abs/1903.03571) | 💻 [Code](https://github.com/DavidBurt2/Rates-of-Convergence-SGPR)
    -> [Convergence of Sparse Variational Inference in Gaussian Processes Regression](http://jmlr.org/papers/v21/19-1015.html) | [Code](https://github.com/markvdw/RobustGP)

---

### Latest

* [Deep Structured Mixtures of Gaussian Processes](https://arxiv.org/abs/1910.04536) - Trapp et. al. (2019)
  > Going back to the old days of improving the local-expert technique. 
* [Sparse Gaussian Process Regression Beyond Variational Inference]() - Jankowiak et. al. (2019)

---

## Other


??? info "Adversarial Robustness Guarantees for Classification with Gaussian Processes - Blass et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/1905.11876)

---
## Thesis Explain

Often times the papers that people publish in conferences in Journals don't have enough information in them. Sometimes it's really difficult to go through some of the mathematics that people put  in their articles especially with cryptic explanations like "it's easy to show that..." or "trivially it can be shown that...". For most of us it's not easy nor is it trivial. So I've included a few thesis that help to explain some of the finer details. I've arranged them in order starting from the easiest to the most difficult.


* [GPR Techniques](https://github.com/HildoBijl/GPRT) - Bijl (2016)    
  * Chapter V - Noisy Input GPR
* [Non-Stationary Surrogate Modeling with Deep Gaussian Processes](https://lib.ugent.be/fulltxt/RUG01/002/367/115/RUG01-002367115_2017_0001_AC.pdf) - Dutordoir (2016)
  * Chapter IV - Finding Uncertain Patterns in GPs
* [Nonlinear Modeling and Control using GPs](http://mlg.eng.cam.ac.uk/pub/pdf/Mch14.pdf) - McHutchon (2014)
  * Chapter II - GP w/ Input Noise (NIGP)
* [Deep GPs and Variational Propagation of Uncertainty](http://etheses.whiterose.ac.uk/9968/1/Damianou_Thesis.pdf) - Damianou (2015)
  * Chapter IV - Uncertain Inputs in Variational GPs
  * Chapter II (2.1) - Lit Review
* [Bringing Models to the Domain: Deploying Gaussian Processes in the Biological Sciences](http://etheses.whiterose.ac.uk/18492/1/MaxZwiesseleThesis.pdf) - Zwießele (2017)
  * Chapter II (2.4, 2.5) - Sparse GPs, Variational Bayesian GPLVM

??? info "Sparse Gaussian Process Approximations and Applications by Van der Wilk (2018)"
    -> [Thesis](https://markvdw.github.io/vanderwilk-thesis.pdf)

### Presentations

* [Variational Inference for Gaussian and Determinantal Point Processes](http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf) - Titsias (2014)


### Notes

* [On the paper: Variational Learning of Inducing Variables in Sparse Gaussian Processees](http://mlg.eng.cam.ac.uk/thang/docs/talks/rcc_vargp.pdf) - Bui and Turner (2014)

## Gory Details Blogs

> Some resources that break down some of the intricate mathematical details that are sometimes lost within the literature.

* Bill Engels
  * [Inducing point methods to speed up GPs](https://bwengals.github.io/inducing-point-methods-to-speed-up-gps.html) - (01-06-2017)
  * [FITC and VFE](https://bwengals.github.io/fitc-and-vfe.html) - (28-06-2018)
  * [PyMC3 FITC/VFE implementation notes](https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html) - (29-06-2018)
  * [VFE approximation for Gaussian processes, the gory details](https://bwengals.github.io/vfe-approximation-for-gaussian-processes-the-gory-details.html) - (20-08-2018)
* [Variational Free Energy for Sparse GPs](https://gonzmg88.github.io/blog/2018/04/19/VariationalFreeEnergy) - Gonzalo (04-2018)
* [Sparse and Variational Gaussian Process — What To Do When Data is Large](https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7) - Wei Yi (06-2020) | [code](https://gist.github.com/jasonweiyi)
* [A Cheatsheet for Sparse Variational Gaussian Processes](https://tiao.io/post/sparse-variational-gaussian-processes/) - Louis Tiao (09-2020)
* [Derivation of SGPR Equations](https://gpflow.readthedocs.io/en/master/notebooks/theory/SGPR_notes.html)
* [GaussianProcesses.jl](https://stor-i.github.io/GaussianProcesses.jl/latest/sparse/) - Lots of details in here.

## Code Examples

> Some examples where people have implemented the algorithms very didactically.

* [SVGP](https://github.com/pmorenoz/RecyclableGP/blob/master/models/svgp.py) - Recyclable GP