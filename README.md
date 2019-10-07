# Gaussian Process Model Zoo

I have included some of the algorithms I work with or have worked with during my PhD. My [lab](https://isp.uv.es/) works with kernel methods and we frequently use GPs for different applications, e.g. emulation, ocean applications and parameter retrievals. We typically use GPs for the following reasons with #1 being the most important (as it with research groups whether they admit it or not).

1. It's what we've been doing... We are a kernel lab, use mainly kernel methods and GPs are essentially a Bayesian treatment of kernel methods for regression and classification applications.
2. The GP treatment of uncertainty via confidence intervals is essentially when dealing with physical data.
3. We can often use sensible priors and a fairly consistent way of tuning the hyperparameters.
4. Somewhat robust to overfitting via built-in regularization.

I created this repo because I didn't want my code to go to waste in case there were people who are new to GPs and want to see a few key algorithms implemented. Also, it allows me to centralize my code for all my projects. Hopefully it will be of some use to others.

#### Special Interest

I will focus a bit more on utilizing the GP models to accommodate uncertain inputs. This is a peculiar research topic of mine so there will be a bit of emphasis on that aspect within the repo.


## Table of Contents


### **[Sklearn](sklearn/README.md)**

Often times if I have a The [sklearn docs](https://scikit-learn.org/stable/modules/gaussian_process.html) are already sufficient to get people started with GPs in scikit-learn. However, I have a few algorithms in there that I personally use. I often use the sklearn library when I want to apply GPs for datasets less than 2000 points. 

**Note**: I do want to point out that the likelihood is absent in the GP model and you have to instead use the white kernel. This threw me off at first and I made many mistakes with this and tbh I think this should be made clear and fixed.

### **[GPy](gpy/README.md)**

GPy is the most **comprehensive research library** I have found to date. It has the most number of different special case GP algorithms of any package available. The GPy [examples](https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html) and [tutorials](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb) are good but I personally found the [docs](https://gpy.readthedocs.io/en/deploy/) very difficult to navigate. I also found the code base to be a bit difficult to really understand what's going on. I typically wrap some typical GP algorithms within the sklearn `.fit()`, `.predict()`, `.score()` framework. The standard algorithms will include some like the Sparse GP, Heteroscedastic GP and Bayesian GPLVM. I typically use this library if my dataset is between 2,000 and 10,000 points. It also doesn't get updated very often so I'm assuming the devs have moved on to other things. There are rumors of a GPy2 library that's based on MXFusion but I have failed to see anything concrete yet.

### **[GPyTorch](gpytorch/README.md) (TODO)**

This is my defacto library for **applying GPs** to large scale data. Anything above 10,000 points, and I will resort to this library. It has GPU acceleration and a large suite of different GP algorithms depending upon your problem. I think this is the dominant GP library for actually using GPs and I highly recommend it for utility. I still find it a bit difficult to really customize anything under the hood. But if you can figure out how to mix and match each of the modular parts, then it should work for you.


### **[Pyro](pyro/README.md) (TODO)**

This is my defacto library for doing **research with GPs**. In particular for GPs, I find the library to be super easy to mix and match priors and parameters for my GP models. Also pyro has a great [forum](https://forum.pyro.ai/) which is very active and the devs are always willing to help. It is backed by Uber and built off of PyTorch so it has a strong dev community. I also talked to the devs at the ICML conference in 2019 and found that they were super open and passionate about the project. 

### **[GPFlow](gpflow/README.md) (TODO)**

What Pyro is to PyTorch, GPFlow is to TensorFlow. A few of the devs from GPy went to GPFlow so it has a very similar style as GPy. But it is a lot cleaner due to the use of autograd which eliminates all of the code used to track the gradients. Many researchers use this library as a backend for their own research code so I would say it is the second most used library in the research domain. I didn't find it particularly easy to customize in tensorflow =<1.14 because of the session tracking which wasn't clear to me from the beginning. But now with the addition of tensorflow 2.0 and GPFlow adopting that new framework, I am eager to try it out again.

### **[TensorFlow Probability](tensorflow/README.md) (TODO)** 

This library is built into Tensorflow already and they have a few GP modules that allow you to train GP algorithms. In edition, they have a keras-like GP layer which is very useful for using a GP as a final layer in probabilistic neural networks. The GP community is quite small for TFP so I haven't seen too many examples for this.

### **[Edward2](edward2/README.md) (TODO)** 

This is the most exciting one in my opinion because this library will allow GPs (and Deep GPs) to be used for the most novice users and engineers. It features the GP and sparse GP as bayesian keras-like layers. So you can stack as many of them as you want and then call the keras `model.fit()`. I think this is a really great feature and will put GPs on the map because it doesn't get any easier than this.

---
## Library Classification

Below you have a few plots which show the complexity vs flexible scale of different architectures for software. The goal of keras and tensorflow is to accommodate ends of that scale. 

![Spectrum of Keras workflows](https://keras-dev.s3.amazonaws.com/tutorials-img/spectrum-of-workflows.png)

**Figure**: Photo Credit - Francois Chollet

![Model definition: spectrum of workflows](https://keras-dev.s3.amazonaws.com/tutorials-img/model-building-spectrum.png)

**Figure**: Photo Credit - Francois Chollet

![Model training: spectrum of workflows](https://keras-dev.s3.amazonaws.com/tutorials-img/model-training-spectrum.png)

**Figure**: Photo Credit - Francois Chollet
