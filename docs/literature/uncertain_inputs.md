# Uncertain Inputs in Gaussian Processes

Please go to my dedicated repository where I explore all things to do with uncertain Gaussian processes.

<<<<<<< HEAD
It can be found here: [jejjohnson.github.io/uncertain_gps/](https://jejjohnson.github.io/uncertain_gps/#/literature)
=======
- [Motivation](#motivation)
- [Algorithms](#algorithms)
  - [Error-In-Variables Regression](#error-in-variables-regression)
  - [Monte Carlo Sampling](#monte-carlo-sampling)
  - [Taylor Expansion](#taylor-expansion)
  - [Moment Matching](#moment-matching)
  - [Covariance Functions](#covariance-functions)
  - [Iterative](#iterative)
  - [Linearized (Unscented) Approximation](#linearized-unscented-approximation)
  - [Heteroscedastic Likelihood Models](#heteroscedastic-likelihood-models)
  - [Latent Variable Models](#latent-variable-models)
  - [Latent Covariates](#latent-covariates)
  - [Variational Strategies](#variational-strategies)
- [Next Steps?](#next-steps)
    - [1. Apply these algorithms to different problems (other than dynamical systems)](#1-apply-these-algorithms-to-different-problems-other-than-dynamical-systems)
    - [2. Improve the Kernel Expectation Calculations](#2-improve-the-kernel-expectation-calculations)
    - [3. Think about the problem differently](#3-think-about-the-problem-differently)
    - [4. Think about pragmatic solutions](#4-think-about-pragmatic-solutions)
    - [5. Figure Out how to extend it to Deep GPs](#5-figure-out-how-to-extend-it-to-deep-gps)
- [Appendix](#appendix)
  - [Kernel Expectations](#kernel-expectations)
    - [Literature](#literature)
    - [Toolboxes](#toolboxes)
  - [Connecting Concepts](#connecting-concepts)
    - [Moment Matching](#moment-matching-1)
    - [Derivatives of GPs](#derivatives-of-gps)
    - [Extended Kalman Filter](#extended-kalman-filter)
  - [Uncertain Inputs in other ML fields](#uncertain-inputs-in-other-ml-fields)
  - [Key Equations](#key-equations)
---

## Motivation


This is my complete literature review of all the ways the GPs have been modified to allow for uncertain inputs.

---
## Algorithms



---
### Error-In-Variables Regression

This isn't really GPs per say but it is probably the first few papers that actually publish about this problem in the Bayesian community (that we know of).

* [Bayesian Analysis of Error-in-Variables Regression Models]() - Dellaportas & Stephens (1995)
* [Error in Variables Regression: What is the Appropriate Model?](http://orca.cf.ac.uk/54629/1/U585018.pdf) - Gillard et. al. (2007) [**Thesis**]

---
### Monte Carlo Sampling

So almost all of the papers in the first few years mention that you can do this. But I haven't seen a paper explicitly walking through the pros and cons of doing this. However, you can see the most implementations of the PILCO method as well as the Deep GP method do implement some form of this.

### Taylor Expansion



* [Learning a Gaussian Process Model with Uncertain Inputs]() - Girard & Murray-Smith (2003) [**Technical Report**]



---
### Moment Matching




This is where we approximate the mean function and the predictive variance function to be Gaussian by taking the mean and variance (the moments needed to describe the distribution).

<details>

$$\begin{aligned}
m(\mu_{x_*}, \Sigma_{x_*}) &= \mu(\mu_{x_*})\\
v(\mu_{x_*}, \Sigma_{x_*}) &= \nu^2(\mu_{x_*}) + 
\frac{\partial \mu(\mu_{x_*})}{\partial x_*}^\top
\Sigma_{x_*}
\frac{\partial \mu(\mu_{x_*})}{\partial x_*} +
\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \nu^2(\mu_{x_*})}{\partial x_* \partial x_*^\top}  \Sigma_{x_*}\right\}
\end{aligned}$$

</details>


* [Gaussian Process Priors With Uncertain Inputs – Application to Multiple-Step Ahead Time Series Forecasting]() - Girard et. al. (2003)
* [Approximate Methods for Propagation of Uncertainty in GP Models]() - Girard (2004) [**Thesis**]
* [Prediction at an Uncertain Input for Gaussian Processes and Relevance Vector Machines Application to Multiple-Step Ahead Time-Series Forecasting]() - Quinonero-Candela et. al. (2003) [**Technical Report**]
* [Analytic moment-based Gaussian process filtering]() - Deisenroth et. al. (2009)
  * [PILCO: A Model-Based and Data-Efficient Approach to Policy Search]() - Deisenroth et. al. (2011)
    * Code - [TensorFlow](https://github.com/nrontsis/PILCO) | [GPyTorch](https://github.com/jaztsong/PILCO-gpytorch) | [MXFusion I](https://github.com/amzn/MXFusion/blob/master/examples/notebooks/pilco.ipynb) | [MXFusion II](https://github.com/amzn/MXFusion/blob/master/examples/notebooks/pilco_neurips2018_mloss_slides.ipynb)
* [Efficient Reinforcement Learning using Gaussian Processes]() - Deisenroth (2010) [**Thesis**]
  * Chapter IV - Finding Uncertain Patterns in GPs (Lit review at the end)


---
### Covariance Functions

<details>

Daillaire constructed a modification to the RBF covariance function that takes into account the input noise.

$$K_{ij} = \left| 2\Lambda^{-1}\Sigma_x + I \right|^{1/2} \sigma_f^2 \exp\left( -\frac{1}{2}(x_i - x_j)^\top (\Lambda + 2\Sigma_x)^{-1}(x_i - x_j) \right)$$

for $i\neq j$ and

$$K_{ij}=\sigma_f^2$$

for $i=j$. This was shown to have bad results if this $\Sigma_x$ is not known. You can see the full explanation in the thesis of McHutchon (section 2.2.1) which can be found in Iterative section below.


</details>


* [An approximate inference with Gaussian process to latent functions from uncertain data]() - Dallaire et. al. (2011) | [Prezi](https://s3.amazonaws.com/academia.edu.documents/31116309/presentation_iconip09.pdf?response-content-disposition=inline%3B%20filename%3DLearning_Gaussian_Process_Models_from_Un.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20191016%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20191016T123012Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=92785eb3561f2822752b538ea8f232fc127d9bc6db94a307e165ac73e62a3601) | [Code](https://github.com/maka89/noisy-gp)


---
### Iterative



* [Gaussian Process Training with Input Noise]() - McHutchon & Rasmussen (2011) | [Code](https://github.com/HildoBijl/GPRT/tree/master/NIGP)
  * [Nonlinear Modelling and Control using GPs]() - McHutchon (2014) [**Thesis**] 
    * Chapter IV - Finding Uncertain Patterns in GPs
* [System Identification through Online Sparse Gaussian Process Regression with Input Noise](https://arxiv.org/pdf/1601.08068.pdf) - Bijl et. al. (2017) | [Code](https://github.com/HildoBijl/SONIG)
  * [Gaussian Process Regression Techniques]() - Bijl (2018) [**Thesis**] | [Code](https://github.com/HildoBijl/GPRT)
    * Chapter V - Noisy Input GPR


---
### Linearized (Unscented) Approximation

This is the linearized version of the Moment-Matching approach mentioned above. Also known as unscented GP. In this approximation, we only change the predictive variance. You can find an example colab notebook [here](https://colab.research.google.com/drive/1AOtGvOVRzqPaLkAzSH5tjkG-8OKOJ43R) with an example of how to use this with the GPy library.

<details>

$$\begin{aligned}
\tilde{\mu}_f(x_*) &= \underbrace{k_*^\top K^{-1}y}_{\mu_f(x_*)} \\
\tilde{\nu}^2(x_*) &= \underbrace{k_{**} - k_*^\top K^{-1} k_*}_{\nu^2(x_*)} + \partial \mu_f \text{ } \Sigma_x \text{ } \partial \mu_f^\top
\end{aligned}$$

**Note**: The inspiration of this comes from the Extended Kalman Filter (links below) which tries to find an approximation to a non-linear transformation, $f$ of $x$ when $x$ comes from a distribution $x \sim \mathcal{N}(\mu_x, \Sigma_x)$.

</details>



* [GP-BayesFilters: Bayesian Filtering Using Gaussian Process Prediction]() - Ko and Fox (2008)
  > They originally came up with the linearized (unscented) approximation to the moment-matching method. They used it in the context of the extended Kalman filter which has a few more elaborate steps in addition to the input uncertainty propagation.
* [Expectation Propagation in Gaussian Process Dynamical Systems]() - Deisenroth & Mohamed (2012)
  > The authors use expectation propagation as a way to propagate the noise through the test points. They mention the two ways to account for the input uncertainty referencing the GP-BayesFilters paper above: explicit moment-matching and the linearized (unscented) version. They also give the interpretation that the Moment-Matching approach with the kernel expectations is analogous to doing the KL-Divergence between prior distribution with the uncertain inputs $p(x)$ and the approximate distribution $q(x)$.
* [Accounting for Input Noise in Gaussian Process Parameter Retrieval]() - Johnson et. al. (2019)
  > My paper where I use the unscented version to get better predictive uncertainty estimates. 
  >
  > **Note**: I didn't know about the unscented stuff until after the publication...unfortunately.
* [Unscented Gaussian Process Latent Variable Model: learning from uncertain inputs with intractable kernels]() - Souza et. al. (2019) [**arxiv**]
  > A very recent paper that's been on arxiv for a while. They give a formulation for approximating the linearized (unscented) version of the moment matching approach. Apparently it works better that the quadrature, monte carlo and the kernel expectations approach.



---
### Heteroscedastic Likelihood Models

* [Heteroscedastic Gaussian Process Regression]() - Le et. al. (2005)
* [Most Likely Heteroscedastic Gaussian Process Regression]() - Kersting et al (2007)
* [Variational Heteroscedastic Gaussian Process Regression]() - Lázaro-Gredilla & Titsias (2011)
* [Heteroscedastic Gaussian Processes for Uncertain and Incomplete Data]() - Almosallam (2017) [**Thesis**]
* [Large-scale Heteroscedastic Regression via Gaussian Process](https://arxiv.org/abs/1811.01179) - Lui et. al. (2019) [**arxiv**] | [Code](https://github.com/LiuHaiTao01/SVSHGP)

---
### Latent Variable Models


* [Gaussian Process Latent Variable Models for Visualisation of High Dimensional Data]() - Lawrence (2004)
* [Generic Inference in Latent Gaussian Process Models]() - Bonilla et. al. (2016)
* [A review on Gaussian Process Latent Variable Models]() - Li & Chen (2016)


---
### Latent Covariates

* [Gaussian Process Regression with Heteroscedastic or Non-Gaussian Residuals]() - Wang & Neal (2012)
* [Gaussian Process Conditional Density Estimation](https://arxiv.org/pdf/1810.12750.pdf) - Dutordoir et. al. (2018)
* [Decomposing feature-level variation with Covariate Gaussian Process Latent Variable Models]() - Martens et. al. (2019)
* [Deep Gaussian Processes with Importance-Weighted Variational Inference]() - Salimbeni et. al. (2019)

---
### Variational Strategies



* [Bayesian Gaussian Process Latent Variable Model]() - Titsias & Lawrence (2010)
* [Nonlinear Modelling and Control using GPs]() - McHutchon (2014) [**Thesis**]
* [Variational Inference for Uncertainty on the Inputs of Gaussian Process Models]() - Damianou et. al. (2014)
  * [Deep GPs and Variational Propagation of Uncertainty]() - Damianou (2015) [**Thesis**]
    * Chapter IV - Uncertain Inputs in Variational GPs
    * Chapter II (2.1) - Lit Review
  * [Processes Non-Stationary Surrogate Modeling with Deep Gaussian]() - Dutordoir (2016) [**Thesis**]
    > This is a good thesis that walks through the derivations of the moment matching approach and the Bayesian GPLVM approach. It becomes a little clearer how they are related after going through the derivations once.
* [Bringing Models to the Domain: Deploying Gaussian Processes in the Biological Sciences](http://etheses.whiterose.ac.uk/18492/1/MaxZwiesseleThesis.pdf) - Zwießele (2017) [**Thesis**]
  * Chapter II (2.4, 2.5) - Sparse GPs, Variational Bayesian GPLVM




---
## Next Steps?

So after all of this literature, what is the next step for the community? I have a few suggestions based on what I've seen:

#### 1. Apply these algorithms to different problems (other than dynamical systems)

It's clear to me that there are a LOT of different algorithms. But in almost every study above, I don't see many applications outside of dynamical systems. I would love to see other people outside (or within) community use these algorithms on different problems. Like Neil Lawrence said in a recent MLSS talk; "we need to stop jacking around with GPs and actually **apply them**" (paraphrased). There are many little goodies to be had from all of these methods; like the linearized GP predictive variance estimate for better variance estimates is something you get almost for free. So why not use it? 

#### 2. Improve the Kernel Expectation Calculations

So how we calculate kernel expectations is costly. A typical sparse GP has a cost of O(NM^2). But when we do the calculation of kernel expectations, that order goes back up to O(DNM^2). It's not bad considering but it is still now an order of magnitude larger for high dimensional datasets. This is going backwards in terms of efficiency. Also, many implementations attempt to do this in parallel for speed but then the cost of memory becomes prohibitive (especially on GPUs). There are some other good approximation schemes we might be able to use such as advanced Bayesian Quadrature techniques and the many moment transformation techniques that are present in the Kalman Filter literature. I'm sure there are tricks of the trade to be had there.

#### 3. Think about the problem differently

An interesting way to approach the method is to perhaps use the idea of covariates. Instead of the noise being additive, perhaps it's another combination where we have to model it separately. That's what Salimbeni did for his latest Deep GP and it's a very interesting way to look at it. It works well too!


#### 4. Think about pragmatic solutions

Some of these algorithms are super complicated. It makes it less desireable to actually try them because it's so easy to get lost in the mathematics of it all. I like pragmatic solutions. For example, using Drop-Out, Ensembles and Noise Constrastive Priors are easy and pragmatic ways of adding reliable uncertainty estimates in Bayesian Neural Networks. I would like some more pragmatic solutions for some of these methods that have been listed above. **Another Shameless Plug**: the method I used is very easy to get better predictive variances almost for free.

#### 5. Figure Out how to extend it to Deep GPs

So the original Deep GP is just a stack of BGPLVMs and more recent GPs have regressed back to stacking SVGPs. I would like to know if there is a way to improve the BGPLVM in such a way that we can stack them again and then constrain the solutions with our known prior distributions. 


---
## Appendix


---
### Kernel Expectations


So [Girard 2003] came up with a name of something we call kernel expectations $\{\mathbf{\xi, \Omega, \Phi}\}$-statistics. These are basically calculated by taking the expectation of a kernel or product of two kernels w.r.t. some distribution. Typically this distribution is normal but in the variational literature it is a variational distribution. 


<details>

The three kernel expectations that surface are:

$$\mathbf \xi(\mathbf{\mu, \Sigma}) = \int_X \mathbf k(\mathbf x, \mathbf x)\mathcal{N}(\mathbf x|\mathbf \mu,\mathbf  \Sigma)d\mathbf x$$

$$\mathbf \Omega(\mathbf{y, \mu, \Sigma}) = \int_X \mathbf k(\mathbf x, \mathbf y)\mathcal{N}(\mathbf x|\mathbf \mu,\mathbf  \Sigma)d\mathbf x$$

$$\mathbf \Phi(\mathbf{y, z, \mu, \Sigma}) = \int_X \mathbf k(\mathbf x, \mathbf y)k(\mathbf x, \mathbf z)\mathcal{N}(\mathbf x|\mathbf \mu,\mathbf  \Sigma)d\mathbf x$$

</details>

To my knowledge, I only know of the following kernels that have analytically calculated sufficient statistics: Linear, RBF, ARD and Spectral Mixture. And furthermore, the connection is how these kernel statistics show up in many other GP literature than just uncertain inputs of GPs; for example in Bayesian GP-LVMs and Deep GPs.

#### Literature


* Oxford M:
  * [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature]() - Gunter et. al. (2014)
  * [Batch Selection for Parallelisation of Bayesian Quadrature]() - 
    * [Code](https://github.com/OxfordML/bayesquad])
* Prüher et. al
  * [On the use of gradient information in Gaussian process quadratures]() (2016)
    > A nice introduction to moments in the context of Gaussian distributions.
  * [Gaussian Process Quadrature Moment Transform]() (2017)
  * [Student-t Process Quadratures for Filtering of Non-linear Systems with Heavy-tailed Noise]() (2017)
    * Code: [Nonlinear Sigma-Point Kalman Filters based on Bayesian Quadrature](https://github.com/jacobnzw/SSMToybox)
      > This includes an implementation of the nonlinear Sigma-Point Kalman filter. Includes implementations of the
      * [Moment Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L11)
      * [Linearized Moment Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L49)
      * [MC Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L62)
      * [Sigma Point Transform]([SigmaPointTransform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L102)),
      * [Spherical Radial Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L152)
      * [Unscented Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L207)
      * [Gaussian Hermite Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L296)
      * [Fully Symmetric Student T Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L363)
      > And a few experimental transforms:
      * Truncated Transforms:
        * [Sigma Point Transform](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L588)
        * [Spherical Radial]()
        * [Unscented]()
        * [Gaussian Hermite]()
      * [Taylor GPQ+D w. RBF Kernel](https://github.com/jacobnzw/SSMToybox/blob/master/ssmtoybox/mtran.py#L668)



#### Toolboxes

* [Emukit](https://nbviewer.jupyter.org/github/amzn/emukit/blob/master/notebooks/Emukit-tutorial-Bayesian-quadrature-introduction.ipynb)



---
### Connecting Concepts


---
#### Moment Matching

---
#### Derivatives of GPs


* [Derivative observations in Gaussian Process Models of Dynamic Systems]() - Solak et. al. (2003)
* [Differentiating GPs](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf) - McHutchon (2013)
  > A nice PDF with the step-by-step calculations for taking derivatives of the linear and RBF kernels.
* [Exploiting gradients and Hessians in Bayesian optimization and
Bayesian quadrature](https://arxiv.org/pdf/1704.00060.pdf) - Wu et. al. (2018)

---
#### Extended Kalman Filter

This is the origination of the Unscented transformation applied to GPs. It takes the Taylor approximation of your function


* [Wikipedia](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
* Blog Posts by Harveen Singh - [Kalman Filter](https://towardsdatascience.com/kalman-filter-interview-bdc39f3e6cf3) | [Unscented Kalman Filter](https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d) | [Extended Kalman Filter](https://towardsdatascience.com/extended-kalman-filter-43e52b16757d)
* [Intro to Kalman Filter and Its Applications](https://www.intechopen.com/books/introduction-and-implementations-of-the-kalman-filter/introduction-to-kalman-filter-and-its-applications) - Kim & Bang (2018)
* [Tutorial](https://www.cse.sc.edu/~terejanu/files/tutorialEKF.pdf) - Terejanu
* Videos
  * [Lecture](https://youtu.be/DE6Jn2cB4J4) by Cyrill Stachniss
  * [Lecture](https://www.youtube.com/watch?v=HFYmz6Y7Xrw) by Robotics Course | [Notes](https://drive.google.com/drive/folders/1S9FfOKmYFbj7EgHSOvu9FeS8Wn3nubOY)
  * [Lecture](https://www.youtube.com/watch?v=0M8R0IVdLOI) explained with Python Code


---
### Uncertain Inputs in other ML fields

* Statistical Rethinking 
  * [Course Page](https://github.com/rmcelreath/statrethinking_winter2019)
  * [Lecture](https://youtu.be/UgLF0aLk85s) | [Slides](https://speakerdeck.com/rmcelreath/l20-statistical-rethinking-winter-2019) | [PyMC3 Implementation](https://nbviewer.jupyter.org/github/pymc-devs/resources/blob/master/Rethinking/Chp_14.ipynb)

---
### Key Equations


**Predictive Mean for latent function, $f$**

$$\mu_f(x_*) = k_*^\top K^{-1}y$$
$$\sigma^2_f(x_*) = k_{**} - k_*^\top K^{-1} k_*$$

**Predictive Mean for outputs, $y$**

$$\mu_y(x_*) = k_*^\top K^{-1}y$$
$$\sigma^2_y(x_*) = \sigma^2 + k_{**} - k_*^\top K^{-1} k_*$$

**Kernel Expectations**





>>>>>>> c932531f57ce7d91e9b7ee84aed035dc77414ba1
