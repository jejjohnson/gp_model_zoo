# Software

Software for Gaussian processes (GPs) have really been improving for quite a while now. It is now a lot easier to not only to actually use the GP models, but also to modify them improve them.

## Library Classification


<p align="center">
  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/spectrum-of-workflows.png" alt="drawing" width="600"/>
</p>

**Figure**: Photo Credit - Francois Chollet

So how to classify a library's worth is impossible because it's completely subjective. But I've

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
|           MMGP            |    ✓    |                  |     ✓     |                              |     ✓      |    ✓     |      ✓       |
|      SparseGP - FITC      |    ✓    |                  |     ✓     |                              |     ✓      |    ✓     |      ✓       |
|      SparseSP - VFE       |    ✓    |                  |           |                              |     ✓      |    ✓     |      ✓       |
|      Variational GP       |    ✓    |                  |           |              ✓               |     ✓      |    ✓     |
| Stochastic Variational GP |    ✓    |                  |           |                              |     ✓      |    ✓     |      ✓       |
|          Deep GP          |         |                  |           |                              |            |    ✓     |      D       |
|   Deep Kernel Learning    |         |                  |           |              S               |            |    S     |      ✓       |
|           GPLVM           |    ✓    |                  |           |                              |            |    ✓     |      ✓       |
|      Bayesian GPLVM       |    ✓    |                  |           |                              |     ✓      |    ✓     |
|         SKI/KISS          |         |                  |           |                              |            |          |      ✓       |
|           LOVE            |         |                  |           |                              |            |          |      ✓       |


**Key**

| **✓** | **Implemented** |
| :---: | :-------------: |
|   ✗   | Not Implemented |
|   D   |   Development   |
|   S   |    Supported    |
| S(?)  | Maybe Supported |

---
## Libraries


### **[Sklearn](sklearn/README.md)**

Often times if I have a The [sklearn docs](https://scikit-learn.org/stable/modules/gaussian_process.html) are already sufficient to get people started with GPs in scikit-learn. However, I have a few algorithms in there that I personally use. I often use the sklearn library when I want to apply GPs for datasets less than 2000 points. 

**Note**: I do want to point out that the likelihood is absent in the GP model and you have to instead use the white kernel. This threw me off at first and I made many mistakes with this and tbh I think this should be made clear and fixed.

### **[GPy](./../gpy/README.md)**

GPy is the most **comprehensive research library** I have found to date. It has the most number of different special case GP algorithms of any package available. The GPy [examples](https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html) and [tutorials](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb) are good but I personally found the [docs](https://gpy.readthedocs.io/en/deploy/) very difficult to navigate. I also found the code base to be a bit difficult to really understand what's going on. I typically wrap some typical GP algorithms within the sklearn `.fit()`, `.predict()`, `.score()` framework. The standard algorithms will include some like the Sparse GP, Heteroscedastic GP and Bayesian GPLVM. I typically use this library if my dataset is between 2,000 and 10,000 points. It also doesn't get updated very often so I'm assuming the devs have moved on to other things. There are rumors of a GPy2 library that's based on MXFusion but I have failed to see anything concrete yet.

### **[GPyTorch](./../gpytorch/README.md) (TODO)**
This is my defacto library for **applying GPs** to large scale data. Anything above 10,000 points, and I will resort to this library. It has GPU acceleration and a large suite of different GP algorithms depending upon your problem. I think this is the dominant GP library for actually using GPs and I highly recommend it for utility. I still find it a bit difficult to really customize anything under the hood. But if you can figure out how to mix and match each of the modular parts, then it should work for you.

### **[Pyro](./../pyro/README.md) (TODO)**
This is my defacto library for doing **research with GPs**. In particular for GPs, I find the library to be super easy to mix and match priors and parameters for my GP models. Also pyro has a great [forum](https://forum.pyro.ai/) which is very active and the devs are always willing to help. It is backed by Uber and built off of PyTorch so it has a strong dev community. I also talked to the devs at the ICML conference in 2019 and found that they were super open and passionate about the project. 
  
### **[GPFlow](./../gpflow/README.md) (TODO)**
What Pyro is to PyTorch, GPFlow is to TensorFlow. A few of the devs from GPy went to GPFlow so it has a very similar style as GPy. But it is a lot cleaner due to the use of autograd which eliminates all of the code used to track the gradients. Many researchers use this library as a backend for their own research code so I would say it is the second most used library in the research domain. I didn't find it particularly easy to customize in tensorflow =<1.14 because of the session tracking which wasn't clear to me from the beginning. But now with the addition of tensorflow 2.0 and GPFlow adopting that new framework, I am eager to try it out again.
  
### **[TensorFlow Probability](./../tensorflow/README.md) (TODO)** 
This library is built into Tensorflow already and they have a few GP modules that allow you to train GP algorithms. In edition, they have a keras-like GP layer which is very useful for using a GP as a final layer in probabilistic neural networks. The GP community is quite small for TFP so I haven't seen too many examples for this.

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




