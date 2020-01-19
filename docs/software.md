# Software



Software for Gaussian processes (GPs) have really been improving for quite a while now. It is now a lot easier to not only to actually use the GP models, but also to modify them improve them.

## Library Classification

<p align="center">
  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/spectrum-of-workflows.png" alt="drawing" width="600"/>
</p>

**Photo Credit**: Francois Chollet [Tweet](https://twitter.com/fchollet/status/1052228463300493312/photo/1)

So how to classify a library's worth is impossible because it's completely subjective. But I like this chart by Francois Chollet who put the different depths a package can go to in order to create a package that caters to different users.Libraries

## Quick Overview

---

### Sklearn

The GP implementation in the [scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html) library are already sufficient to get people started with GPs in scikit-learn. Often times when I'm data wrangling and I'm exploring possible algorithms, I'll already have the sklearn library installed in my conda environment so I typically start there myself especially for datasets less than 2,000 points. 


#### Sample Code Snippet

The sklearn implementation is as basic as it gets. If you are familiar with the scikit-learn API then you will have no problems using the GPR module. It's a three step process with very little things to change.

<!-- tabs:start -->

#### ** Model **

```python
# define kernel function
kernel = \
RBF(length_scale=1.0) \
  + WhiteKernel(noise_level=0.1)
  
# initialize GP model
gpr_model = GaussianProcessRegressor(
  kernel=kernel_gpml,
  alpha=0,
  optimizer=None, 
  normalize_y=True
)
```

#### ** Training **

```python
# train GP model
gpr_model.fit(Xtrain, ytrain)
```

#### ** Predictions **

```python
# get predictions
y_pred, y_std = gpr_model.predict(Xtest, return_std=True)
```
<!-- tabs:end -->

Again, this is the simplest API you will find and for small data problems, you'll find that this works fine out-of-the-box. I highly recommend this when starting especially if you're not a GP connoisseur. What I showed above is as complicated as it gets. Any more customization outside of this is a bit difficult as the scikit-learn API for GPs isn't very modular and wasn't designed as such. But as a first pass, it's good enough.

---

### [GPy](./../gpy/README.md)

GPy is the most **comprehensive research library** I have found to date. It has the most number of different special GP "corner case" algorithms of any package available. The GPy [examples](https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html) and [tutorials](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb) are very comprehensive. The major caveat is that the [documentation](https://gpy.readthedocs.io/en/deploy/) is very difficult to navigate. I also found the code base to be a bit difficult to really understand what's going on because there is no automatic differentiation to reduce the computations so there can be a bit of redundancy. I typically wrap some typical GP algorithms with some common parameters that I use within the sklearn `.fit()`, `.predict()`, `.score()` framework and call it a day. The standard algorithms will include the Exact GP, the Sparse GP, and Bayesian GPLVM. A **warning** though: this library does not get updated very often so you will likely run into very silly bugs if you don't use strict package versions that are recommended. There are rumors of a GPy2 library that's based on [MXFusion](https://github.com/amzn/MXFusion) but I have failed to see anything concrete yet. 

**Idea**: Some of the main algorithms such as the sparse GP implementations are mature enough to be dumped into the sklearn library. For small-medium data problems, I think this would be extremely beneficial to the community. Some of the key papers like the (e.g. the [FITC-SGP](https://papers.nips.cc/paper/2857-sparse-gaussian-processes-using-pseudo-inputs), [VFE-SGP](http://proceedings.mlr.press/v5/titsias09a.html), [Heteroscedastic GP](https://dl.acm.org/doi/10.1145/1273496.1273546), [GP-LVM](https://dl.acm.org/doi/10.5555/2981345.2981387)) certainly pass some of the [strict sklearn criteria](https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms). But I suspect that it wouldn't be a joy to code because you would need to do some of the gradients from scratch. I do feel like it might make GPs a bit more popular if some of the mainstream methods were included in the scikit-learn library.

#### Sample Code Snippet

The GPy implementation is also very basic. If you are familiar with the scikit-learn API then you will have no problems using the GPR module. It's a three step process with very little things to change.


<!-- tabs:start -->

#### ** Model **

```python
# define kernel function
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
  
# initialize GP model
gpr_model = GPy.models.GPRegression(
  Xtrain, ytrain,
  kern=kernel
)
```

#### ** Training **

```python
# train GP model
m.optimize(messages=True)
```

#### ** Predictions **

```python
# get predictions
y_pred, y_std = gpr_model.predict(Xtest)
```

<!-- tabs:end -->

So as you can see, the API is very similar to the scikit-learn API with some small differences; the main one being that you have to initiate the GP model with the data. The rest is fairly similar. You should definitely take a look at the GPy docs if you are interested in some more advanced examples.

---
### [GPyTorch](./../gpytorch/README.md) (TODO)

This is my defacto library for **applying GPs** to large scale data. Anything above 10,000 points, and I will resort to this library. It has GPU acceleration and a large suite of different GP algorithms depending upon your problem. I think this is currently the dominant GP library for actually using GPs and I highly recommend it for utility. They have many options available ranging from latent variables to multi-outputs. Recently they've just revamped their entire library and documentation with some I still find it a bit difficult to really customize anything under the hood. But if you can figure out how to mix and match each of the modular parts, then it should work for you.

#### Sample Code Snippet

In GPyTorch, the library follows the pythonic way of coding that became super popular from deep learning frameworks such as Chainer and subsequently PyTorch. It consists of a 4 step process which is seen in the snippet below.

<!-- tabs:start -->

#### ** Model **

```python
class MyGP(gpytorch.models.ExactGP):
     def __init__(self, train_x, train_y, likelihood):
         super().__init__(train_x, train_y, likelihood)

         # Mean Function
         self.mean_module = gpytorch.means.ZeroMean()

         # Kernel Function
         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

     def forward(self, x):
         mean = self.mean_module(x)
         covar = self.covar_module(x)
         return gpytorch.distributions.MultivariateNormal(mean, covar)

 # train_x = ...; train_y = ...
 likelihood = gpytorch.likelihoods.GaussianLikelihood() model = MyGP(train_x, train_y, likelihood)
 model = MyGP(train_x, train_y, likelihood)
```

<!-- tabs:end -->

**[Source](https://gpytorch.readthedocs.io/en/latest/models.html#exactgp)** - GPyTorch Docs

I am only scratching the surface with this quick snippet. But I wanted to highlight how this fits into

---

### [Pyro](./../pyro/README.md)
This is my defacto library for doing **research with GPs**. In particular for GPs, I find the library to be super easy to mix and match priors and parameters for my GP models. Also pyro has a great [forum](https://forum.pyro.ai/) which is very active and the devs are always willing to help. It is backed by Uber and built off of PyTorch so it has a strong dev community. I also talked to the devs at the ICML conference in 2019 and found that they were super open and passionate about the project. 

#### 

<!-- tabs:start -->

#### ** Model **

```python
kernel2 = gp.kernels.RBF(
    input_dim=1, 
    variance=torch.tensor(0.1),
    lengthscale=torch.tensor(10.)
)
gpr_model = gp.models.GPRegression(X, y, kernel2, noise=torch.tensor(0.1))
```

#### ** Training **

```python
# define optimizer
optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)

# define loss function
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 1_000

# typical PyTorch boilerplate code
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
```
<!-- tabs:end -->

**[Source](http://pyro.ai/examples/gp.html)**: Pyro Docs

---
### [GPFlow](./../gpflow/README.md) (TODO)

What Pyro is to PyTorch, GPFlow is to TensorFlow. A few of the devs from GPy went to GPFlow so it has a very similar style as GPy. But it is a lot cleaner due to the use of autograd which eliminates all of the code used to track the gradients. Many researchers use this library as a backend for their own research code so I would say it is the second most used library in the research domain. I didn't find it particularly easy to customize in tensorflow =<1.14 because of the session tracking which wasn't clear to me from the beginning. But now with the addition of tensorflow 2.0 and GPFlow adopting that new framework, I am eager to try it out again.

---

### [TensorFlow Probability](./../tensorflow/README.md) (TODO) 
This library is built into Tensorflow already and they have a few GP modules that allow you to train GP algorithms. In edition, they have a keras-like GP layer which is very useful for using a GP as a final layer in probabilistic neural networks. The GP community is quite small for TFP so I haven't seen too many examples for this.

---

### [Edward2](./../edward2/README.md) (TODO)
This is the most exciting one in my opinion because this library will allow GPs (and Deep GPs) to be used for the most novice users and engineers. It features the GP and sparse GP as bayesian keras-like layers. So you can stack as many of them as you want and then call the keras `model.fit()`. I think this is a really great feature and will put GPs on the map because it doesn't get any easier than this.

---
## Library Classification

Below you have a few plots which show the complexity vs flexible scale of different architectures for software. The goal of keras and tensorflow is to accommodate ends of that scale. 


<p align="center">
  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/model-building-spectrum.png" alt="drawing" width="800"/>
</p>

**Figure**: Photo Credit - Francois Chollet

<p align="center">
  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/model-training-spectrum.png" alt="drawing" width="800"/>
</p>

**Figure**: Photo Credit - Francois Chollet





---

## GPU Support

| **Package**              | **Backend** | **GPU Support** |
| ------------------------ | ----------- | --------------- |
| GPy                      | Numpy       | ✓               |
| Scikit-Learn             | Numpy       | ✗               |
| PyMC3                    | Theano      | ✓               |
| TensorFlow (Probability) | TensorFlow  | ✓               |
| Edward                   | TensorFlow  | ✓               |
| GPFlow                   | TensorFlow  | ✓               |
| Pyro.contrib             | PyTorch     | ✓               |
| GPyTorch                 | PyTorch     | ✓               |
| PyMC4                    | TensorFlow  | ✓               |

---

## Algorithms Implemented

| **Package**               | **GPy** | **Scikit-Learn** | **PyMC3** | **TensorFlow (Probability)** | **GPFlow** | **Pyro** | **GPyTorch** |
| ------------------------- | ------- | ---------------- | --------- | ---------------------------- | ---------- | -------- | ------------ |
| Exact                     | ✓       | ✓                | ✓         | ✓                            | ✓          | ✓        | ✓            |
| Moment Matching GP        | ✓       | ✗                | ✓         | ✗                            | S          | S        | ✓            |
| SparseGP - FITC           | ✓       | ✗                | ✓         | ✗                            | ✓          | ✓        | ✓            |
| SparseGP - PEP            | ✓       | ✗                | ✗         | ✗                            | ✗          | ✗        | ✗            |
| SparseSP - VFE            | ✓       | ✗                | ✗         | ✗                            | ✓          | ✓        | ✓            |
| Variational GP            | ✓       | ✗                | ✗         | ✓                            | ✓          | ✓        | ✗            |
| Stochastic Variational GP | ✓       | ✗                | ✗         | S                            | ✓          | ✓        | ✓            |
| Deep GP                   | ✗       | ✗                | ✗         | S                            | S          | ✓        | D            |
| Deep Kernel Learning      | ✗       | ✗                | ✗         | S                            | S          | S        | ✓            |
| GPLVM                     | ✓       | ✗                | ✗         | ✗                            | ✗          | ✓        | ✓            |
| Bayesian GPLVM            | ✓       | ✗                | ✗         | ✗                            | ✓          | ✓        | ✓            |
| SKI/KISS                  |         | ✗                | ✗         | ✗                            | ✗          | ✗        | ✓            |
| LOVE                      | ✗       | ✗                | ✗         | ✗                            | ✗          | ✗        | ✓            |


**Key**

| Symbol | Status          |
| ------ | --------------- |
| **✓**  | **Implemented** |
| ✗      | Not Implemented |
| D      | Development     |
| S      | Supported       |
| S(?)   | Maybe Supported |

---

## 


