# GPs w/ Error Propagation


### Resources

* Gaussian Process Training with Input Noise - [Paper](http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf) | [Author Website](http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon) | [Code](https://github.com/jejjohnson/fumadas/tree/master/NIGP)
    * Differentiating Gaussian Process - [Notes](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf)
* Learning a Gaussian Process Model with Uncertain Inputs - [Technical Report](http://www.dcs.gla.ac.uk/~rod/publications/GirMur03-tr-144.pdf)
* Gaussian Processes: Prediction at a Noisy Input and Application to Iterative Multiple-Step Ahead Forecasting of Time-Series - [Paper](http://www.dcs.gla.ac.uk/~rod/publications/GirMur05.pdf)
* GP Regression with Noise Inputs - [Presentation](http://dcervone.com/slides/GP_noisy_inputs.pdf)
* Approximate Methods for Propagation of Uncertainty with GP Models - [Thesis](http://www.dcs.gla.ac.uk/~rod/publications/Gir04.pdf) | [Code](https://github.com/maka89/noisy-gp)
* Learning Gaussian Process Models from Uncertain Data - [Paper](https://www.researchgate.net/publication/221140644_Learning_Gaussian_Process_Models_from_Uncertain_Data) | [Github](https://github.com/maka89/noisy-gp)

---

### Gaussian Process

A gaussian process is:

* "... a stochastic process which is used in machine learning to describe a distribution directly into the function space." -  *Learning Gaussian Process Models from Uncertain Data* - [paper](https://www.researchgate.net/profile/Camille_Besse/publication/221140644_Learning_Gaussian_Process_Models_from_Uncertain_Data/links/0912f508ff851b4894000000/Learning-Gaussian-Process-Models-from-Uncertain-Data.pdf)

* GP Class by Rassmussen - [Website](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/)
    * Modeling Data - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/modelling%20data.pdf)
    * Linear in the Parameters Regression - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/linear%20in%20the%20parameters%20regression.pdf)
    * Likelihood and the Concept of Noise - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/likelihood%20and%20noise.pdf)
    * Probability Fundamentals - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/probability%20fundamentals.pdf)
    * Bayesian Inference and Prediction with Finite Regression Models - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/bayesian%20finite%20regression.pdf)
    * Marginal Likelihood - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/marginal%20likelihood.pdf)
    * Parameters and Functions - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/parameters%20and%20functions.pdf)
    * Marginal Likelihood - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/marginal%20likelihood.pdf)
    * Gaussian Process - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/gaussian%20process.pdf)
    * Posterior Gaussian Process - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/gp%20and%20data.pdf)
    * GP Marginal Likelihood and Hyperparameters - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/hyperparameters.pdf)
    * Linear Models and GPs - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/correspondence.pdf)
    * Finite and Infinite Basis - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/infinite.pdf)
    * Covariance Functions - [Slides](http://mlg.eng.cam.ac.uk/teaching/4f13/1718/covariance%20functions.pdf)
    * GPML Toolbox Introduction - [Code Website](http://www.gaussianprocess.org/gpml/code/matlab/doc/)

---
### Covariance Functions


#### RBF Function

The typical squared exponential kernel:

$$K(x, y) = exp\left( -\frac{||x-y||^2_2}{2\lambda_d^2} \right)$$

Remember the distance calculation:

$$
d_{ij} = ||x-y||^2_2 = (x-y)^{\top}(x-y) = x^{\top}x-2x^{\top}y-y^{\top}y
$$

$$D=$$

Alternatively, one could write the kernel function in standard matrix notation:

$$k(x,y)=exp\left[-\frac{1}{2}(x-y)^{\top}\Lambda^{-1}(x-y)\right]$$

where $\Lambda$ is an ($D \times D$) matrix whos diagonal entries are $\lambda^2$.

We can also get the analytical solutions to the gradient of this kernel matrix. Which is useful for later:

$$\frac{\partial K(x,y)}{\partial \Lambda}=(x-y)^{\top}\Lambda_2^{-1}(x-y)\cdot K(x,y)$$

where $\Lambda_2$ is an ($D \times D$) matrix whos diagonal entries are $\lambda^3$.


* Gradient of RBF Kernel - [stackoverflow](https://math.stackexchange.com/questions/1030534/gradients-of-marginal-likelihood-of-gaussian-process-with-squared-exponential-co/1072701#1072701)
* Euclidean Distance Matrices (Essential Theory, Algorithms and Applications) - [Arxiv](https://arxiv.org/pdf/1502.07541.pdf)

---
### Gaussian Process Training with Input Noise

With GPs, normally we have the standard two assumptions:

1. The inputs, $x$ are noise-free.
2. The outputs, $y$ are corrupted by the constant-variance Gaussian noise.

But what happens in the case where the Input and Ouput measurements are corrupted by noise? A lot of data these days are derived from models. An example would be the IASI dataset that gives temperature measurements which are derived from other models which use the radiance as inputs. The objective would be to use a GP model to try and learn a function to predict temperature based off of the radiances. Why would you want to do this? Well often times statistical models can be a lot faster than physical models. Plus there are useful techniques that one could use to assess your model using statistical techniques. Now, there is no reason not to believe that the inputs themselves are noise. I imagine any measurement that we make from satellites will have some noise associated with it. So it is safe to assume that there is some noise associated with that particular value making the standard assumptions invalid.

So now for some equations. Let's assume that we have a $D$-dimensional vector $x$ and a set of outputs for $y$. From the 2nd assumption listed above, we can say that

$$y=\tilde{y} + \epsilon_y$$

where $y$ is the ouput which we assume to be noisy, $\tilde{y}$ is the output from the model, and $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2)$.

Negating the 1st assumption, let's write the equation:

$$x = \tilde{x} + \epsilon_x$$

where $x$ is the measurement we assume to be noisy, $\tilde{x}$ is the actual input and $\epsilon_x \sim \mathcal{N}(0, \Sigma_x)$ (essentially each dimension is independently corrupted by noise). This means the $\sum_x$ is a $D$-dimensional vector.

Now let's say there exists a function $f(\cdot)$ which is a model on the latent variables. We can write an expression of an observed output as a function of the observed inputs like so:

$$y=f(x-\epsilon_x)+\epsilon_y$$

We can do a Taylor expansion at the observed state $x$:

$$f(x-\epsilon_x) = f(x) + \epsilon_x^{\top}\frac{\partial f(x)}{\partial x} + \ldots$$

**Note**: Differentiation is a linear operation so the derivative of a Gaussian process is another Gaussian process [[1](http://mlg.eng.cam.ac.uk/pub/pdf/SolMurLeietal03.pdf)]. Modeling this as a GP would be fairly expensive as one would have to consider a distribution over Taylor expansions. So essentially do the entire maximum likelihood process to find the hyperparameters of two different GPs; each with their own respective mean function and variance function. Might not be feasible or worth it. Instead there is an approximate method that we can use.

Let's take the derivative of the mean of the GP instead. Let $\partial_{\bar{f}}$ denote the derivative of one GP function wrt the D-Dimensional input and $\triangle_{\bar{f}}$ be an $N\times D$ matrix for the derivative of $N$ function values. **Note**: differentiating the mean function corresponds to ignoring the uncertainty about the derivative (unless we want to consider higher order terms but then this could become a bit recursive). So, this gives us a nice linear model:

$$y=f(x) + \epsilon_x^{\top}\partial_{\bar{f}} + \epsilon_y$$

So the probability of an observation $y$ can be written as:

$$P(y|f) = \mathcal{N}(f, \sigma_y^2 + \partial_{\bar{f}}^{\top}\Sigma_x \partial_{\bar{f}})$$

We also keep the usual GP prior for $f$:

$$P(f|X)=\mathcal{N}(0, K(X,X))$$ 

where $K(X,X)$ is an $N\times N$ training covariance matrix and $X$ is an $N \times D$ matrix of input observations.

Now if we combine the probabilities, we can get the predictive posterior mean function:

$$\mathbb{E}[f_*|X,y,x_*]=k(x_*,X) \left[ K(X,X) + \sigma_y^2I+diag \left\{ \triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top} \right\} \right]^{-1}y$$

and the variance function:

$$\mathbb{V}[f_*|X,y,x_*]=k(x_*,x_*) - k(x_*, X)\left[ K(X,X) + \sigma_y^2I+diag \left\{ \triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top} \right\} \right]^{-1}k(X,x_*)$$

where:

* $\partial_{\bar{f}}$ is a $D$-dimensional vector
* $\triangle_{\bar{f}}$ is a ($N \times D$)-dimensional matrix
* $\triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top}$ is a $(N\times D)(D \times D)(D \times D)=(N\times N)$-dimensional matrix

---

### Training

This new model introduces $D$-hyperparameters compared to the standard GP. So:

* $D$-dimensional vector $\sum_x$ (training input noise)
* $D$-dimensional vector $\sigma_{y}^2$ (noise variance)

##### Algorithm

Let $K=exp(\gamma ||x-x_*||^2_2)$

1. Evaluate a standard GP with the training data.

Maximize:

$$log(y | X, \theta) = -\frac{1}{2}y^{\top}\left[K+\sigma_y^2I  \right]y-\frac{1}{2}\log|K+\sigma_y^2I| -\frac{n}{2}\log(2\pi)$$

where $\theta=[\gamma, \sigma_y^2]$

2. Take the derivative of the posterior mean of the result.

$$\triangle_{\bar{f}}=\frac{\partial \mu(x)}{\partial x}$$

3. Add the diagonal terms to the predictive posterior mean and variance.

$$K+\sigma_y^2+diag\{ \triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top}  \}$$

4. Calculate the marginal likelihood of the GP with the corrected variance. (Calculate the derivative with respect to the initial parameters with the added noise variance)

Maximize:

$$log(y | X, \theta) = -\frac{1}{2}y^{\top}\left[K+\sigma_y^2I + diag\{ \triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top}  \} \right]y-\frac{1}{2}\log|K+\sigma_y^2I + diag\{ \triangle_{\bar{f}} \Sigma_x \triangle_{\bar{f}}^{\top}  \}| -\frac{n}{2}\log(2\pi)$$

where $\theta=[\gamma, \sigma_y^2, \Sigma_x]$