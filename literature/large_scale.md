


---
### SOTA

* [Exact GP on a Million Data Points](https://arxiv.org/abs/1903.08114) - Wang et. al. (2019) | [Code](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Simple_GP_Regression/Simple_MultiGPU_GP_Regression.ipynb)
  > The authors manage to train a GP with multiple GPUs on a million or so data points. They use the matrix-vector-multiplication strategy. Quite amazing actually...


* [Constant-Time Predictive Distributions for GPs](https://arxiv.org/abs/1803.06058) - Pleiss et. al. (2018) | [Code]()
  > Using MVM techniquees, they are able to make constant time predictive mean and variance estimates; addressing the computational bottleneck of predictive distributions for GPs.