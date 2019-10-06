# Literature Review - Timeline

---
## Key Words

Generally speaking, we all don't speak the same language even though we all speak english. Therefore it's useful to know some useful rough synonyms

**Gaussian Process**
* Kriging
* Kalman Filter
* Bayesian (Kernel) Linear Regression

**Taylor Expansion**
* Moment Matching
* Error Propagation
* Dynamical System Inputs

**Uncertain Inputs**
* Latent
* Localization Uncertainty

---

## Timeline

#### Error-in-Variables Regression

* Dellaportas, Stephans (1995)

#### Taylor Approximation of GPs (Integrating)

* Girard, Murray-Smith (2003)

#### Moment Matching

* Girard et al. (2003)
* Candela et al. (2003)
* Deisenroth (2010)
* Deisenroth and Rasmussen - PILCO (2011)

#### Stochastic Measurement Points

* Dallaire et al. (2009) - uncertainty incorporating SE covariance function
* Girard and Murray-Smith (2003) - taylor approx. uncertainty incorporating SE covariance function

**Note**: None of these methods take into account the posterior data (i.e. the derivative)

* McHutchon and Rasmussen (2011) - NIGP
* Bijl et al. (2017) - SONIG

#### Variational Inference

* Titsias (2009)
* Titsias and Lawrence (2010) - Bayesian GPLVM
* MchHutchon (2014)
* Damianou et al. (2016)

#### Heteroscedastic 

* Golberg et al. (1998)
* Le et al. (2005)
* Snelson and Ghahramani (2006)
* Kersting et al. (2007)
* Lazaro-Gredilla and Titsias (2011)
* Wang and Neal (2012)







---
#### Citations

Typically these people:

* **Girard, Quinanero-Candela** 
    * *Propagate Uncertainty across the predictive sequence*
    * *Consider the input uncertainty only at test time**
    * Propagate the test input uncertainty through a non-linear GP results in a non-Gaussian predictive density.
    * Rely on momement matching to obtain the predictive mean and covariance or develope a scheme based on simulations
* **McHutchon** 
    * rely on local approximations inside the latent mapping function, rather than modelling the approximate posterior densities directly.



**Accounting for Variance in the Input Space**
* Bayesian analysis of errors-in-variables regression models - Dellaportas, Stephens - 1995, Journal
* Gaussian Process Priors with Uncertain Inputs: Application to Multiple-Step Ahead Time Series Forcasting - Girard, Rasmussen, Candela, Murray-Smith - 2003 - Book

* Bayesian Inference for the uncertainty distribution of computer model outputs - Oakley, O'Hagan - 2002 - Journal
* Propagation of Uncertainty in Bayesian Kernel Models: Application to Multiple-Step Ahead Forecasting -  Quinonero-Candela, Girard, Larsen, Rassmussen - 2003 - Conference
* Bayesian Uncertainty Analysis for Complex Computer Codes - Oakley - 1991 - PhD
* Estimating percentiles of uncertain computer code outputs - Oakley - 2004 - Journal
* Gaussian Process Training with Input Noise - McHutchon, Rasmussen - 2011 - Conference

**Heteroscedastic Gaussian Process Regression** (uncertainties are measure)
* Most Likely Heterscedastic gaussian process regression - Kersting, Plagemann, Pfaff, Burgard - 2007 - Conference
* Regression with input-dependent noise: A GP treatment - Golberg, Williams, Bishop - 1998 - Book
* Variational Heteroscedastic gaussian process regression - Lazaro-Gredilla, Titsias - 2011 - Conference


---

## Gaussian Process Regression Techniques - A Student Version of the Thesis

> When applying Gaussian process regression we have always assumed that there is noise on the output measurements fm, but not on the input points x. This assumption does ofcourse notalways hold: the inputpoints can be subject to noise as well. 

> When the trial inputpoints x∗ are subject to noise, we can integrate over all possible trial input points. This will not result in a Gaussian distribution for the output though. One solution is to switch to numerical techniques. The more conventional solution is to apply moment matching instead. When we do, we analytically calculate the mean and covari- ance ofthe resulting distribution and use those to approximate the result as a Gaussian distribution.

>When the measurementinputpoints xm are stochastic, these tricks do notwork anymore. One way to work around this is to take the noise on the input points xm into account through the output noise variance ˆΣm. The more the function we are approximating is sloped, the more we should take inputnoise into account like this. We can apply this idea to regularGaussian process regression, resultingin theNIGPalgorithm, orwecan apply it to sparse methods like FITC, resultingin the SONIG algorithm. The SONIG algorithm is capable ofincorporatingnewmeasurements ( ˆxm, ˆfm) one byone in a computationally efficientmanner. When doing so, it provides us with Gaussian ap- proximations ofthe posteriordistributions ofboth the inputandthe output, as well as the posterior covariance between these distributions. With this data it is possible to setup for instance a nonlinear system identification algorithm.


---
## Deep Gaussian Processes and Variational Propagation of Uncertainty - Damianou, 2015, Thesis

> In many real-world applications it is unrealistic to consider the inputs to a regression model as absolutely certain. For example when the inputs are measurements coming from noisy sensors, or when the inputs are coming from bootstrapping or extrapolating from a trained regressor. In the general setting, we assume that the actual inputs to the regression model are not observed; instead we only observe their noisy versions. In this case, the GP methodology cannot be trivially extended to account for the variance associated with the input space.

In his thesis he treats the unobserved inputs as latent variables. He marginalises these variables in order to obtain an approximate posterior over a full distribution over the inputs. There is a heavy relation to **latent variable modeling** with GPs.

---
## Girard, Murray-Smith - Learning a Gaussian Process Model with Uncertain Inputs (2003, Technical Report)


#### Theory

Expand the original process around the input mean using the **delta method** (i.e. Taylor approximations). They assume that the noise is random and normally distributed. This results in new covariance functions that account for the randomness of the input and they test their methods. 

They reason that normally we have the following problem:

$$E(y|x)=f(x)$$

where:
* $y=f(x) + \epsilon_y$
* $\epsilon_y \sim \mathcal{N}(0, \sigma_y)$

But if they want to account for input noise, then they have the following integral to describe their problem:

$$E(y|u, \sigma_x) = \int f(x)p(x)dx$$

where:
* $x=u+\epsilon_x$
* $\epsilon_x \sim \mathcal{N}(u, \sigma_x I)$,

and the integral cannot be solved analytically as we need some approximations for $f(x)$. They do a 2nd order Taylor expansion about the mean $u$ of $x$:

$$f(x) = f(u) + (x-u)^Tf'(u) + \frac{1}{2}(x-u)^{T}f''(u)(x-u) + \mathcal{O}(||x-u||^2)$$

They get the following expected value function:

$$E(y|u, \sigma_x) \approx \int \left[ f(u) + (x-u)^Tf'(u) + \frac{1}{2}(x-u)^{T}f''(u)(x-u) \right]p(x)dx$$

and this reduces to

$$E(y|u, \sigma_x) \approx f(u) + \frac{\sigma_x}{2} Tr[f''(u)]$$

which results in the final generative model for the data:

$$y = f(u, \sigma_x) + \sigma_y$$
$$f(u, \sigma_x) = f(u) + \frac{\sigma_x}{2} Tr[f''(u)]$$

However, this only takes care of the mean function. In the case of the covariance, they use the theory of random functions.


---

* Gaussian Process Training with Input Noise - [Paper](http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf) | [Author Website](http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon) | [Code](https://github.com/jejjohnson/fumadas/tree/master/NIGP)
    * Differentiating Gaussian Process - [Notes](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf)
* Learning a Gaussian Process Model with Uncertain Inputs - [Technical Report](http://www.dcs.gla.ac.uk/~rod/publications/GirMur03-tr-144.pdf)
* Gaussian Processes: Prediction at a Noisy Input and Application to Iterative Multiple-Step Ahead Forecasting of Time-Series - [Paper](http://www.dcs.gla.ac.uk/~rod/publications/GirMur05.pdf)
* GP Regression with Noise Inputs - [Presentation](http://dcervone.com/slides/GP_noisy_inputs.pdf)
* Approximate Methods for Propagation of Uncertainty with GP Models - [Thesis](http://www.dcs.gla.ac.uk/~rod/publications/Gir04.pdf) | [Code](https://github.com/maka89/noisy-gp)
* Learning Gaussian Process Models from Uncertain Data - [Paper](https://www.researchgate.net/publication/221140644_Learning_Gaussian_Process_Models_from_Uncertain_Data) | [Github](https://github.com/maka89/noisy-gp)


---

### Scratch

**Taylor Series Approach**: Girard and Murray-Smith (2003), local approximation to the GP function. Simplier to work with rather than working with the function implied by the GP prior. They add terms up to the quadratic term and take the expected value for computational reasons. Derivatives of GPs are also GPs so their derivatives have a mean and variance in closed form. Needs 4th and 5th order derivatives for the covariance function and gradient respectively. The use of only first order terms is impossible due to them disappearing so it's a minimum of 2nd order terms.


**Heteroscedastic Noise**:  Goldbeg et al. (1998), ; Learning the noise model. A function is applied to the noise coefficient $\sigma_y$, computationally better with an approximate version - Kersing et al (2007) but does not take into account the uncertainty in the noise GP and opts for the mean instead; variance in MAP Yuan and Wahba (2004), Le et al. (2005); importance of points in pseudo-training set as well as Copula and Wishart Process methods Wilson and Ghahramani (2006 ,2010,2011); \textbf{but} none of these methods exploit the structure in the input noise datasets.

**NIGP**:  Introduced the NIGP based on the Taylor expansion of the posterior distribution. The method processes the input noise through the Taylor expansion as being proportional to the square of the posteriors mean function's gradient. Optional, one could add an additional order which would capture the uncertainty in the gradient. This is fitted and are modeling the GP assuming that the inputs are noisy. They also are able to estimate a noise variance on the input per dimension. They showcase this in time-series data with output at time $t-1$ becomes the input of time $t$ in the case where that is clearly not noise-free.

**Variational**: Titsias and Lawrence (2010), a choise of variational distribution can lead to a tractable lower bound on the marginal likelihood of a GP model under the assumption that there is uncertainty in the inputs. They also utilize the GP latent variable model Lawrence (2004) and show how this can be used in the problem of input noise. Since we cannot compute the true marginal likelihood for GP regression with noisy inputs, a lower bound on the marginal likelihood as a functional of the variational distribution of the inputs. Computational expensive without the use of deep learning architecture.

**Covariance**:  Dallaire et al. (2008); Situation where the Gaussian distributions are known on the training points. Optimizing the Hyperparameters adds no flexibility.

**Inverse Uncertainty**: (Baumgaertel et al., 2014)
