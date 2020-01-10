# Gaussian Processes

## Best Resources

This is fairly subjective but I have included the best tutorials that I could find for GPs on the internet. I focused on clarity and detail and I'm partial to resources with lots of plots. The resources range from lectures to blogs so hopefully by the end of this page you will find something that will ease you into understanding Gaussian processes.

---

### Best Lectures

By far the best beginners lecture on GPs (that I have found) is by [Neil Lawrence](https://inverseprobability.com/); a prominent figure that you should know once you enter this field. The lecture video that I have included below are the most recent of his lectures and I personally think it gives the most intuition behind GPs without being too mathy. The blog that's listed are his entire lecture notes in notebook format so you can read more or less verbatim what was said in the lecture; although you might miss out on the nuggets of wisdom he tends to drop during his lectures.

* Neil Lawrence Lecture @ MLSS 2019 - [Blog](http://inverseprobability.com/talks/notes/gaussian-processes.html) | [Lecture](http://inverseprobability.com/talks/notes/gaussian-processes.html) | [Slides](http://inverseprobability.com/talks/notes/gaussian-processes.html)

I would say that the best slides that I have found are by [Marc Deisenroth](https://deisenroth.cc/). Unfortunately, I cannot find the video lectures online. I think these lecture slides are by far the best I've seen. It's fairly math intensive but there arent't too many proofs. These don't have proofs so you'll have to look somewhere else for that. Bishop's book (below) will be able to fill in any gaps. He also has a [distill.pub]() page which has a nice introduction with a few practical tips and tricks to use.

* Foundations of Machine Learning: GPs - Deisenroth (2018-2019) - [Slides](https://deisenroth.co.uk/teaching/2018-19/foundations-of-machine-learning/lecture_gaussian_processes.pdf) | [Practical Guide to GPs](https://drafts.distill.pub/gp/)

---

### Best Visualize Explanations

If you are a visual person (like me) then you will appreciate resources where they go through step-by-step how a Gaussian process is formulated as well as the importance of the kernel and how one can train them.

* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) - Görtler et al. (2019)

---

### Best Books

#### 1. Standard Book

[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) - Rasmussen (2006)

>  This is the standard book that everyone recommends. It gives a fantastic overview with a few different approaches to explaining. However, for details about the more mathy bits, it may not be the best.

#### 2. Better Book

[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Bishop (2006)

> I find this a much better book which highlights a lot of the mathy bits (e.g. being able to fully manipulate joint Gaussian distributions to arrive at the GP).

#### 3. Brief Overview

[Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation-ebook-dp-B00AF1AYTQ/dp/B00AF1AYTQ/ref=mt_kindle?_encoding=UTF8&me=&qid=) - Murphy (2012)

> If you are already familiar with probability and just want a quick overview of GPs, then I would recommend you take a look at Murphy's book. Actually, if you're into probabilistic machine learning in general then I suggest you go through Murphy's book extensively. 

---

### Best Thesis Explanation

Often times the papers that people publish in conferences in Journals don't have enough information in them. Sometimes it's really difficult to go through some of the mathematics that people put  in their articles especially with cryptic explanations like "it's easy to show that..." or "trivially it can be shown that...". For most of us it's not easy nor is it trivial. So I've included a thesis that I found extremely helpful when going step-by-step. It definitely trumps every other thesis that I've read which all assume the reader has some knowledge. The notation is a bit weird at first but once you get used to it, it becomes clearer and clearer.


* Gaussian Process Regression Techniques with Applications to Wind Turbines - Bijl (2016) - [Thesis](https://github.com/HildoBijl/GPRT)

---

### Code Introductions

These resources are the best resources that explain GPs while also walking you through the code. I like going through it step-by-step using code because for some reason, when I have to code things, all of the sudden details start to click. When you can make a program do it from scratch then I think that is where the understanding kicks in. I have also included tutorials where they use python packages if you're interesting in just jumping in.



#### From Scratch

* [Gaussian Processes](http://krasserm.github.io/2018/03/19/gaussian-processes/) - Martin Krasser (2018)

  > Implements GPs from scratch using numpy. I also like the use of functions which breaks things down quite nicely. Quickly glosses over sklearn and GPy.

* 4-Part Gaussian Process Tutorial

  1. [Multivariate Normal Distribution Primer](https://peterroelants.github.io/posts/multivariate-normal-primer/)
  2. [Understanding Gaussian Processes](https://peterroelants.github.io/posts/gaussian-process-tutorial/)
  3. [Fitting a GP](https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/)
  4. [GP Kernels](https://peterroelants.github.io/posts/gaussian-process-kernels/)

  > A nice 4 part tutorial on GPs from scratch. It uses numpy to start and then switches to tensorflow on the 3rth tutorial (not sure why). But it's another explanation of the first tutorial.

* [Gaussian Processes Not Quite for Dummies](https://thegradient.pub/gaussian-process-not-quite-for-dummies/) - Yuge Shi (2019)

  > A more detailed introduction which focuses a bit more on the derivations and the properties of the Gaussian distribution in the context of GPs. Definitely more mathy than the previous posts. It also borrows heavily from a lecture by [Richard Turner](http://cbl.eng.cam.ac.uk/Public/Turner/Turner) (another prominent figure in the GP community).

* [Gaussian Processes](http://efavdb.com/gaussian-processes/) - Jonathan Landy (2017)

  > I like this blog post because it goes over everything in a simple way and also includes some nice nuggets such as acquisition functions for Bayesian Optimization and the derivation of the posterior function which you can find in Bishop. I like the format.

#### Using Libraries

* [Fitting GP models with Python](https://blog.dominodatalab.com/fitting-gaussian-process-models-python/) - Chris Fonnesbeck (2017)

  > This post skips over the 'from-scratch' and goes straight to practical implementations from well-known python libraries sklearn, GPy and PyMC3. Does a nice little comparison of each of the libraries.

* Intro to GPs Demo by Damianou - [Jupyter Notebook](http://adamian.github.io/talks/Damianou_GP_tutorial.html)

  >  Yet another prominent figure in the GP community. He put out a [tweet]() where he expressed how this demo is what he will use when he teaches GPs to another community. So class/demo notes in a nutshell. A valuable tutorial. It almost exclusively uses GPy.

---

### Other Resources


#### Previously Compiled Stuff

So I found a few resources that give papers as well as some codes and lectures that you can look at.

* Into to GPs - [Super Compilation](https://ebonilla.github.io/gaussianprocesses/)
* Deep GPs Nonparametric - [Papers](https://github.com/otokonoko8/deep-Bayesian-nonparametrics-papers/blob/master/README.md)
* GPs for Dynamical System Modeling - [Papers](http://dsc.ijs.si/jus.kocijan/GPdyn/)



