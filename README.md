# Gaussian Process Model Zoo

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

---
## Motivation

I recently ran into someone at a conference who said, "*a lot of research dies in Graduate students laptops*" (it was actually this [scientist right here](https://twitter.com/jennifermarsman)). So I decided to go through all of my stuff, organize it a little bit and make it public.

I have included some of the algorithms that I use or have worked with during my PhD. My [lab](https://isp.uv.es/) works with kernel methods and we frequently use GPs for different Earth science applications, e.g. emulation, ocean applications and parameter retrievals. We typically use GPs for the following reasons with number 1 being the most important (as it with research groups whether they admit it or not).

1. It's what we've been doing... We are a kernel lab, use mainly kernel methods and GPs are essentially a Bayesian treatment of kernel methods for regression and classification applications.
2. The GP treatment of uncertainty via confidence intervals is essentially when dealing with physical data.
3. We can often use sensible priors and a fairly consistent way of tuning the hyperparameters.
4. Somewhat robust to overfitting via built-in regularization.

I created this repo because I didn't want my code to go to waste in case there were people who are new to GPs and want to see a few key algorithms implemented. Also, it allows me to centralize my code for all my projects. Hopefully it will be of some use to others.

---
## Special Interest

I will focus a bit more on utilizing the GP models to accommodate uncertain inputs. This is a peculiar research topic of mine so there will be a bit of emphasis on that aspect within the repo.

---
## What you'll find here

* [**Literature**](literature/README.md)

There are many resources on the internet and I try to compile as much as I can.

* [**Software**](software.md)

I like to keep track of what's going on. So I've listed the libraries that I am aware of as well as some things I've noticed about them. 

* [**Model Zoo**](model_zoo.md)

I've only just started to finish cleaning everything but I have wrapped algorithms in a few libraries such as [GPy](https://sheffieldml.github.io/GPy/), [GPFlow](https://www.gpflow.org/), [Pyro](https://pyro.ai/), and [GPyTorch](https://gpytorch.ai/). Each library has its pros and cons but I like tinkering so I tend to try things out. I'll keep a running list of stuff that I have already implemented [here](model_zoo.md).



  
