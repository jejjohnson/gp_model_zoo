# Software

Software for Gaussian processes (GPs) have really been improving for quite a while now. It is now a lot easier to not only to actually use the GP models, but also to modify them improve them.



## GPy

The de facto original research library for GPs. They have a lot of little cutting edge goodies and many different GPs. However, code and docs aren’t regularly updated so there are some weird bugs. It’s also not very modular so it’s not easy to dig into the code. 


## GPyTorch (Recommended for scale/research)

If you’re interested in using GPs (with GPUs) at scale, this is the most up-to-date python library available. They do many approximations with matrix-matrix multiplications that are beyond my understanding (see paper) but it’s a great library with good documentation. It uses PyTorch as the backend.


## GPFlow 

The next default GP library for research. It has many little goodies and scales well. There are some weird bugs for memory sometimes but it works really well and it has a large suite of algorithms. There are many other libraries that have used GPflow as the backend. It uses Tensorflow.


## Pyro (easiest to use and modify)

This is my favourite library for being experimenting with a decent amount of freedom to modify. I personally find this really easy to use when I need to put priors on random components within my algorithm. The library is also backed by Uber so it’s regularly updated. It uses PyTorch.
