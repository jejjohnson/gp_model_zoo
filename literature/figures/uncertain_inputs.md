# Uncertain Inputs in Gaussian Processes


## Motivation




---
## Algorithms

In the context of Gaussian Processes, there are 

---
### 

---
### Monte Carlo Sampling

---
### Taylor Expansion

* [Learning a Gaussian Process Model w. Uncertain Inputs]() - Girard & Murray-Smith (2003)

---
### Moment Matching

This is where we approximate the mean function and the predictive variance function to be Gaussian by taking the mean and variance (the moments needed to describe the distribution).


#### Linearized (Unscented) Approximation

This is known as unscented GP. In this approximation, we only change the predictive variance. 

$$\begin{aligned}
\tilde{\mu}_f(x_*) &= \underbrace{k_*^\top K^{-1}y}_{\mu_f(x_*)} \\
\tilde{\nu}^2(x_*) &= \underbrace{k_{**} - k_*^\top K^{-1} k_*}_{\nu^2(x_*)} + \partial \mu_f \text{ } \Sigma_x \text{ } \partial \mu_f^\top
\end{aligned}$$

**Note**: The inspiration of this comes from the Extended Kalman Filter (links below) which tries to find an approximation to a non-linear transformation, $f$ of $x$ when $x$ comes from a distribution $x \sim \mathcal{N}(\mu_x, \Sigma_x)$.


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

---
### Latent Variable Models


---
### Latent Covariates

* [Gaussian Process Regression with Heteroscedastic or Non-Gaussian Residuals]() - Wang & Neal (2012)
* [Decomposing feature-level variation with Covariate Gaussian Process Latent Variable Models]() - Martens et. al. (2019)



---
## Appendix


---
### Connecting Concepts


#### Extended Kalman Filter

This is the origination of the Unscented transformation applied to GPs. It takes the Taylor approximation of your function


* [Wikipedia](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
* Blog Posts by Harveen Singh - [Kalman Filter](https://towardsdatascience.com/kalman-filter-interview-bdc39f3e6cf3) | [Unscented Kalman Filter](https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d) | [Extended Kalman Filter](https://towardsdatascience.com/extended-kalman-filter-43e52b16757d)
* Videos
  * [Lecture](https://youtu.be/DE6Jn2cB4J4) by Cyrill Stachniss
  * [Lecture](https://www.youtube.com/watch?v=HFYmz6Y7Xrw) by Robotics Course | [Notes](https://drive.google.com/drive/folders/1S9FfOKmYFbj7EgHSOvu9FeS8Wn3nubOY)
  * [Lecture](https://www.youtube.com/watch?v=0M8R0IVdLOI) explained with Python Code


---
### Key Equations


**Predictive Mean for latent function, $f$**

$$\mu_f(x_*) = k_*^\top K^{-1}y$$
$$\sigma^2_f(x_*) = k_{**} - k_*^\top K^{-1} k_*$$

**Predictive Mean for outputs, $y$**

$$\mu_y(x_*) = k_*^\top K^{-1}y$$
$$\sigma^2_y(x_*) = \sigma^2 + k_{**} - k_*^\top K^{-1} k_*$$

**Kernel Expectations**
