# Components

This is a poor name but I want to put all of the papers here that seek to improve specific components within the GP algorithms, e.g. the inference scheme.


---
## Natural Gradients

**Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models** - Salimbeni et. al. (2018) | [Paper](https://arxiv.org/abs/1803.09151) | [Code](https://github.com/GPflow/GPflow/blob/develop/gpflow/training/natgrad_optimizer.py)



> This paper argues that training sparse GP algorithms with gradient descent can be quite slow due to the need to optimize the variational parameters $q_\phi(u)$ as well as the model parameters. So they propose to use the natural gradient for the variational parameters and then the standard gradient methods for the remaining parameters. They show that the SVGP and the DGP methods all converge much faster with this training regime.






**Related Papers & Code**

  * [Noisy Natural Gradient as Variational Inference](https://arxiv.org/abs/1712.02390) - Zhang (2018) - [Code](https://github.com/wlwkgus/NoisyNaturalGradient)
  * [Parallel training of DNNs with Natural Gradient and Parameter Averaging](https://arxiv.org/abs/1410.7455) - Povey et. al. (2014) | [Code](https://github.com/YiwenShaoStephen/NGD-SGD)
* [PyTorch](https://github.com/wiseodd/natural-gradients/tree/master/pytorch)


---
## Generalized Variational Inference

> In this paper, the author looks at a generalized variational inference technique that can be applied to deep GPs.

[VI](https://arxiv.org/pdf/1904.02063.pdf) | [DeepGP](https://arxiv.org/pdf/1904.02303.pdf)
