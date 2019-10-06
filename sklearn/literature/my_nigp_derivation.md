# My NIGP Derivation

This document goes through my NIGP version. It is very similar the the original NIGP algorithm but the key difference is that the input error uncertainties are known. So we don't have to go through iterations in order to solve for the unknown quantities.

---

## Problem

The standard GP algorithm has the following assumption:

$$y=f(x, \theta)+\epsilon_y$$

where $\epsilon_y \sim \mathcal{N}(0, \sigma^2_y)$. Our model has a different specification. Let's assume that the true input is corrupted by noise and thus defined by $x = \bar{x} + \epsilon_x$ where $\epsilon_x \sim \mathcal{N}(0, \sigma_x)$. So our new model is:

$$y \approx f(\bar{x} + \epsilon_x, \theta) + \epsilon_y$$

where $f \sim \mathcal{GP}(m, K)$. 

### Taylor Approximation

We use a first order Taylor approximation over the term which contains the input noise to obtain:

$$y \approx f(x, \theta) + \epsilon_x^T  \frac{\partial f(\bar{x}, \theta)}{\partial \bar{x}} + \epsilon_y$$

### Moment Matching

So now we have $p(y|\bar{x}, \epsilon_x, \theta)$ which is given by the above equation. We can compute the moments of said equation by using the Taylor series approximation.

#### 1st Moment: Expectation

The first moment is given by the expectation of $y$, $\mathbb{E}[y]$. This is the expected value of the $y$ given the GP prior which gives the exact same mean as the standard GP with no additional terms:

$$\mathbb{E}[y] = \mathbb{E}_{f, \epsilon_y}\left[ f(\bar{x}, \theta) + \epsilon_x^T \frac{\partial f(\bar{x}, \theta)}{\partial \bar{x}} + \epsilon_y \right]$$
$$\mathbb{E}[y] = \mathbb{E}[f(\bar{x}, \theta)]$$
$$\mathbb{E}[y] = m(y)$$

#### 2nd Moment: Variance

$$\mathbb{V}[y] = \mathbb{V}_{f, \epsilon_y}\left[ f(\bar{x}, \theta) + \epsilon_x^T \frac{\partial f(\bar{x}, \theta)}{\partial \bar{x}} + \epsilon_y \right]$$

$$\mathbb{V}[y] = \mathbb{V}_f[f(\bar{x}, \theta) ] + \mathbb{V}_f[\epsilon_x^T \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}}] + \mathbb{V}_f[\sigma^2_y]$$

Term I: This is the prior variance of the latent function; which in this case is just the kernel function:

$$\mathbb{V}_f[f(\bar{x}, \theta) ] = K(\bar{x}, \bar{x}) = K$$

**NOTE:This is the derivation assuming a Gaussian dist for the covariance matrix.**
Term II: The derivative of a GP is also a GP. Our input covariance function is also a constant because it is given to us. So that leaves us left with the product of two Gaussian distributed vectors which leaves us with a non-Gaussian distribution.

$$\mathbb{V}_f[\epsilon_x^T \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}}] = \text{Tr} \left\{ \Sigma_x \mathbb{V}\left[ \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}} \right] \right\}
+ \mathbb{E}\left[ \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}} \right]^T \cdot \Sigma_x \cdot \mathbb{E}\left[  \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}} \right]$$

##### Zero Terms

**Note**: There are some zero terms here that I neglected to put. 

Term I: This covariance term is zero because the noise is independent of the GP function.

$$-2 \mathbb{C}_{f, \epsilon_y} [ f(\bar{x}, \theta), \epsilon_x^T \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}} ] = 0$$

Term II: This covariance term is also zero because the noise is independent of the GP function.

$$-2 \mathbb{C}_{f, \epsilon_y} [ f(\bar{x}, \theta), \epsilon_y ] = 0$$

Term III:

$$-2 \mathbb{C}_{f, \epsilon_y} \left[ \epsilon_x^T \frac{\partial f (\bar{x}, \theta)}{\partial \bar{x}}, \epsilon_y \right] = \mathbb{E}[\epsilon_x] \cdot \mathbb{C} \left[ \frac{\partial f(\bar{x}, \theta)}{\partial \bar{x}} \right]
+ \mathbb{E}\left[ \frac{\partial f(\bar{x}, \theta)}{\partial \bar{x}} \right] \cdot \mathbb{C}\left[ \epsilon_x, \epsilon_y \right]$$

* The first term is zero because the expected value of the input noise is zero.
* The second term is zero because there should be zero covariance between the input noise and the output noise (not sure how true is this assumption tbh).


### Training

Recall our posterior distribution:

$$p(y|\bar{x}, \theta)=\mathcal{N}\left(y; 0, K(\bar{X}, \bar{X}) + \tilde{\Sigma}(\bar{X}) + \sigma_y^2 I\right)$$

The log likelihood of this distribution is defined as:

$$-log P(y|\bar{x}, \theta) = \frac{D}{2} \log 2\pi + \frac{1}{2} \log \left| K_n \right| + \frac{1}{2} Y^T K_n^{-1}Y$$

where:

* $K_n = K + \tilde{\Sigma}(\bar{X}) + \sigma^2_y I$

