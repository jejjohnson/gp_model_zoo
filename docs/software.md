# Software

Software for Gaussian processes (GPs) have really been improving for quite a while now. It is now a lot easier to not only to actually use the GP models, but also to modify them improve them.

## Library Classification


<p align="center">
  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/spectrum-of-workflows.png" alt="drawing" width="600"/>
</p>

**Figure**: Photo Credit - Francois Chollet

So how to classify a library's worth is impossible because it's completely subjective. But I like this chart by Francois Chollet who put the different depths a package can go to in order to create a package that caters to different users.

---
## Python Packages

|       **Package**        | **Backend** | **GPU Support** |
| :----------------------: | :---------: | :-------------: |
|           GPy            |    Numpy    |        ✓        |
|       Scikit-Learn       |    Numpy    |        ✗        |
|          PyMC3           |   Theano    |        ✓        |
| TensorFlow (Probability) | TensorFlow  |        ✓        |
|          Edward          | TensorFlow  |        ✓        |
|          GPFlow          | TensorFlow  |        ✓        |
|       Pyro.contrib       |   PyTorch   |        ✓        |
|         GPyTorch         |   PyTorch   |        ✓        |
|          PyMC4           | TensorFlow  |        ✓        |

---
## Algorithms Implemented

|        **Package**        | **GPy** | **Scikit-Learn** | **PyMC3** | **TensorFlow (Probability)** | **GPFlow** | **Pyro** | **GPyTorch** |
| :-----------------------: | :-----: | :--------------: | :-------: | :--------------------------: | :--------: | :------: | :----------: |
|           Exact           |    ✓    |        ✓         |     ✓     |              ✓               |     ✓      |    ✓     |      ✓       |
|    Moment Matching GP     |    ✓    |        ✗         |     ✓     |              ✗               |     S      |    S     |      ✓       |
|      SparseGP - FITC      |    ✓    |        ✗         |     ✓     |              ✗               |     ✓      |    ✓     |      ✓       |
|      SparseGP - PEP       |    ✓    |        ✗         |     ✗     |              ✗               |     ✗      |    ✗     |      ✗       |
|      SparseSP - VFE       |    ✓    |        ✗         |     ✗     |              ✗               |     ✓      |    ✓     |      ✓       |
|      Variational GP       |    ✓    |        ✗         |     ✗     |              ✓               |     ✓      |    ✓     |      ✗       |
| Stochastic Variational GP |    ✓    |        ✗         |     ✗     |              S               |     ✓      |    ✓     |      ✓       |
|          Deep GP          |    ✗    |        ✗         |     ✗     |              S               |     S      |    ✓     |      D       |
|   Deep Kernel Learning    |    ✗    |        ✗         |     ✗     |              S               |     S      |    S     |      ✓       |
|           GPLVM           |    ✓    |        ✗         |     ✗     |              ✗               |     ✗      |    ✓     |      ✓       |
|      Bayesian GPLVM       |    ✓    |        ✗         |     ✗     |              ✗               |     ✓      |    ✓     |      ✗       |
|         SKI/KISS          |         |        ✗         |     ✗     |              ✗               |     ✗      |    ✗     |      ✓       |
|           LOVE            |    ✗    |        ✗         |     ✗     |              ✗               |     ✗      |    ✗     |      ✓       |


**Key**
| Symbol |     Status      |
| :----: | :-------------: |
| **✓**  | **Implemented** |
|   ✗    | Not Implemented |
|   D    |   Development   |
|   S    |    Supported    |
|  S(?)  | Maybe Supported |

---
## Libraries


### **[Sklearn](sklearn/README.md)**

Often times if I have a The [sklearn docs](https://scikit-learn.org/stable/modules/gaussian_process.html) are already sufficient to get people started with GPs in scikit-learn. However, I have a few algorithms in there that I personally use. I often use the sklearn library when I want to apply GPs for datasets less than 2000 points. 

**Note**: I do want to point out that the likelihood is absent in the GP model and you have to instead use the white kernel. This threw me off at first and I made many mistakes with this and tbh I think this should be made clear and fixed.

#### Sample Code Snippet

The sklearn implementation is as basic as it gets. If you are familiar with the scikit-learn API then you will have no problems using the GPR module. It's a three step process with very little things to change.

1. Define your GPR model with the appropriate `kernel_function`.

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

2. Train your GPR model using the familiar `model.fit()` API.

   ```python
   gpr_model.fit(Xtrain, ytrain)
   ```

3. Get predictions from your model using the familiar `model.predict()` API.

   ```python
   y_pred, y_std = gpr_model.predict(Xtest, return_std=True)
   ```

Again, this is the simplest API you will find and for small data problems, you'll find that this works fine out-of-the-box. I highly recommend this when starting. What I showed above is as complicated as it gets. Any more customization outside of this is a bit difficult as the scikit-learn API for GPs isn't very modular and wasn't designed as such. But as a first pass, it's good enough.

---

### **[GPy](./../gpy/README.md)**

GPy is the most **comprehensive research library** I have found to date. It has the most number of different special case GP algorithms of any package available. The GPy [examples](https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html) and [tutorials](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb) are good but I personally found the [docs](https://gpy.readthedocs.io/en/deploy/) very difficult to navigate. I also found the code base to be a bit difficult to really understand what's going on. I typically wrap some typical GP algorithms within the sklearn `.fit()`, `.predict()`, `.score()` framework. The standard algorithms will include some like the Sparse GP, Heteroscedastic GP and Bayesian GPLVM. I typically use this library if my dataset is between 2,000 and 10,000 points. It also doesn't get updated very often so I'm assuming the devs have moved on to other things. There are rumors of a GPy2 library that's based on MXFusion but I have failed to see anything concrete yet.

#### Sample Code Snippet

The sklearn implementation is as basic as it gets. If you are familiar with the scikit-learn API then you will have no problems using the GPR module. It's a three step process with very little things to change.

1. Define your GPR model with the appropriate `kernel_function`.

   ```python
   # define kernel function
   kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
     
   # initialize GP model
   gpr_model = GPy.models.GPRegression(
     Xtrain, ytrain,
     kern=kernel
   )
   ```

2. Train your GPR model using the `optimize` method.

   ```python
   m.optimize(messages=True)
   ```

3. Get predictions from your model using the familiar `model.predict()` API.

   ```python
   y_pred, y_std = gpr_model.predict(Xtest)
   ```

So as you can see, the API is very similar to the scikit-learn API with some small differences; the main one being that you have to initiate the GP model with the data. The rest is fairly similar. You should definitely take a look at the GPy docs if you are interested in some more advanced examples.

---

### **[GPyTorch](./../gpytorch/README.md) (TODO)**
This is my defacto library for **applying GPs** to large scale data. Anything above 10,000 points, and I will resort to this library. It has GPU acceleration and a large suite of different GP algorithms depending upon your problem. I think this is the dominant GP library for actually using GPs and I highly recommend it for utility. I still find it a bit difficult to really customize anything under the hood. But if you can figure out how to mix and match each of the modular parts, then it should work for you.

#### Sample Code Snippet

In GPyTorch, the library follows the pythonic way of coding that became super popular for deep learning from scikit-learn and deep learning frameworks such as Chainer and subsequently PyTorch. It consists of a 4 step process:

1. We need to define our Gaussian process model where we put in our `mean_function` and `covariance_function`.

```python
# Create a GPModel Class
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # create a constant mean function
        self.mean_function = gpytorch.means.ConstantMean()
        
        # create an RBF kernel function
        self.covariance_function = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
		
    # Forward pass for the model 
    def forward(self, x):
        # get the mean value
        mean_x = self.mean_function(x)
        
        # get the covariance
        covar_x = self.covariance_function(x)
        
        # pass the mean and covariance through a Gaussian
        # multivariate distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

2. Then we need to initialize our regression model which includes our `gaussian_process` and our `gaussian_likelihood`.

```python
# initialize Gaussian Likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# intialize full GP model with likelihood
gpr_model = GPModel(train_x, train_y, likelihood)
```

3. Then we need to do an explicit training loop as we modify the parameters using automatic differentiation until convergence. If you have a GPU, this step will be much faster.

```python
# Put model in 'training mode'
gpr_model.train()
likelihood.train()

# Optimizer for GPR - adam optimizer
optimizer = torch.optim.Adam([
    {'params': gpr_model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# Loss Function for GPR - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = gpr_model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    # step forward for the optimization
    optimizer.step()
   
```

If you are used to PyTorch or TensorFlow 2.0 then this training loop will be very familiar to you. It's a bit cumbersome but you can easily through this into a training loop to make the code much cleaner.

4. Finally we can make predictions using our trained regression model.

```python
# Get into evaluation (predictive posterior) mode
gpr_model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
```

I am only scratching the surface with this quick snippet. But I wanted to highlight how this fits into

---

### **[Pyro](./../pyro/README.md) (TODO)**
This is my defacto library for doing **research with GPs**. In particular for GPs, I find the library to be super easy to mix and match priors and parameters for my GP models. Also pyro has a great [forum](https://forum.pyro.ai/) which is very active and the devs are always willing to help. It is backed by Uber and built off of PyTorch so it has a strong dev community. I also talked to the devs at the ICML conference in 2019 and found that they were super open and passionate about the project. 

---

### **[GPFlow](./../gpflow/README.md) (TODO)**
What Pyro is to PyTorch, GPFlow is to TensorFlow. A few of the devs from GPy went to GPFlow so it has a very similar style as GPy. But it is a lot cleaner due to the use of autograd which eliminates all of the code used to track the gradients. Many researchers use this library as a backend for their own research code so I would say it is the second most used library in the research domain. I didn't find it particularly easy to customize in tensorflow =<1.14 because of the session tracking which wasn't clear to me from the beginning. But now with the addition of tensorflow 2.0 and GPFlow adopting that new framework, I am eager to try it out again.

---

### **[TensorFlow Probability](./../tensorflow/README.md) (TODO)** 
This library is built into Tensorflow already and they have a few GP modules that allow you to train GP algorithms. In edition, they have a keras-like GP layer which is very useful for using a GP as a final layer in probabilistic neural networks. The GP community is quite small for TFP so I haven't seen too many examples for this.

---

### **[Edward2](./../edward2/README.md) (TODO)** 
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




