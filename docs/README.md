---
title: Home
description: GP Model Zoo
authors:
    - J. Emmanuel Johnson
path: docs/
source: README.md
---
# Gaussian Process Model Zoo

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: [jejjohnson.github.io/gp_model_zoo/](https://jejjohnson.github.io/gp_model_zoo/)
* Personal Website: [jejjohnson.netlify.app](https://jejjohnson.netlify.app)

---
## Motivation

I recently ran into someone at a conference who said,

> *A lot of research dies in Graduate students laptops*.

(it was actually this [scientist right here](https://twitter.com/jennifermarsman)). So I decided to go through all of my stuff, organize it a little bit and make it public.

I have included some of the algorithms that I use or have worked with during my PhD. My [lab](https://isp.uv.es/) works with kernel methods and we frequently use Gaussian Processes (GPs) for different Earth science applications, e.g. emulation, ocean applications and parameter retrievals. We typically use GPs for the following reasons with number 1 being the most important (as it with research groups whether they admit it or not).

1. It's what we've been doing... We are a kernel lab, use mainly kernel methods and GPs are essentially a Bayesian treatment of kernel methods for regression and classification applications.
2. Properly handling uncertainty is essential when dealing with physical data.
3. With GPs, can often use sensible priors and a fairly consistent way of tuning the hyperparameters.
4. Somewhat robust to overfitting via built-in regularization.

I created this repo because I didn't want my code to go to waste in case there were people who are new to GPs and want to see a few key algorithms implemented. Also, it allows me to centralize my code for all my projects. Hopefully it will be of some use to others.

---
## What you'll find here


### [**Beginners**](intro.md)

> These are the resources I consider to be the best when it comes to learning about GPs. The field has grown a lot but we still have newcomers. Start here!

### [**Literature**](literature/README.md)

> There are many resources on the internet and I try to compile as much as I can. I do try and keep track of the SOTA and peruse arxiv from time to time.

### [**Software**](software.md)

> I like to keep track of what's going on. So I've listed the libraries that I am aware of as well as some things I've noticed about them.
