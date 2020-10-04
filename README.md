# Gaussian Process Model Zoo [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jejjohnson/gp_model_zoo)

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Docsify Website: [jejjohnson.github.io/gp_model_zoo/](https://jejjohnson.github.io/gp_model_zoo/)
* Personal Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

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

* [**Getting Started**](https://jejjohnson.github.io/gp_model_zoo/intro/)

Getting started with GPs can seem a bit daunting and there are many resources on the internet. I tried to compile the ones I could find and highlight my personal favourites.

* [**Literature**](https://jejjohnson.github.io/gp_model_zoo/literature)

So. Much. Literature. Really. There are a lot. I try to keep a running record of the literature and also try to organize it somewhat by topic.

* [**Software**](https://jejjohnson.github.io/gp_model_zoo/software)

I like to keep track of what's going on in the software realm. So I've listed the libraries that I am aware of as well as some things I've noticed about them. I focus on **python** but I do mention some other resources. I will probably get into **Julia** at some point so stay tuned.




  
