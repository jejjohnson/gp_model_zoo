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


### Sparse Spectrum Gaussian Processes

These are essentially the analogue to the random fourier features for Gaussian processes.

#### SSGP

1. Sparse Spectrum Gaussian Process Regression - Lázaro-Gredilla et. al. (2010) - [PDF](http://jmlr.csail.mit.edu/papers/v11/lazaro-gredilla10a.html)
   > The original algorithm for SSGP.
2. Prediction under Uncertainty in Sparse Spectrum Gaussian Processes
with Applications to Filtering and Control - Pan et. al. (2017) - [PDF](http://proceedings.mlr.press/v70/pan17a.html)
    > This is a moment matching extension to deal with the uncertainty in the inputs at prediction time.

* Python Implementation
  * [Numpy](https://github.com/marcpalaci689/SSGPR)
  * [GPFlow](https://github.com/jameshensman/VFF/blob/master/VFF/ssgp.py)


#### Variational SSGPs

The SSGP algorithm had a tendency to overfit. So they added some additional parameters to account for the noise in the inputs making the marginal likelihood term intractable. They added variational methods to deal with the 
1. Improving the Gaussian Process Sparse Spectrum Approximation by Representing Uncertainty in Frequency Inputs - Gal et. al. (2015) 
   > "...proposed variational inference in a sparse spectrum model that is derived from a GP model." - Hensman et. al. (2018)
2. Variational Fourier Features for Gaussian Processes -  Hensman et al (2018)  [Paper](http://www.jmlr.org/papers/volume18/16-579/16-579.pdf)
   > "...our work aims to directly approximate the posterior of the true models using a variational representation." - Hensman et. al. (2018)

* Yarin Gal's Stuff - [website](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Improving)
* Code
  * [Numpy](https://github.com/marcpalaci689/SSGPR)
  * [Theano](https://github.com/yaringal/VSSGP)



#### Uncertain Inputs

I've only seen one paper that attempts to extend this method to account for uncertain inputs.

* [Prediction under Uncertainty in Sparse Spectrum Gaussian Processes with Applications to Filtering and Control](http://proceedings.mlr.press/v70/pan17a.html) - Pan et. al. (2017) 
  > This is the only paper I've seen that tries to extend this method


---
### Latest

* [Know Your Boundaries: Constraining Gaussian Processes by Variational Harmonic Features](https://paperswithcode.com/paper/know-your-boundaries-constraining-gaussian) - Solin & Kok (2019)