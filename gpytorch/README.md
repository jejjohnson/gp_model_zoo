# Algorithms Implemented

I did a light wrapper of the following algorithms:

1. [Exact GP](exact.py)
2. [Sparse GP](sparse.py)
3. Uncertain Inputs
    * Linearized Moment-Matching (**TODO**)
      * Exact GP
      * Sparse GP
    * Moment-Matching (**TODO**)
      * Exact
      * Sparse
    * Variational Prior
      * [Sparse GP](uncertain.py)


---
## Example Script

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
