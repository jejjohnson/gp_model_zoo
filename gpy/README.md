# Algorithms Implemented

I did a light wrapper of the following algorithms:


**Exact GP**

* Linearized Uncertainty Estimates

**Sparse GP**

* Linearized Uncertainty Estimates

**Sparse Variational GP**

* Input Error

---
### Observations

* These are all CPU algorithms. GPy has GPU capabilities but I haven't figured out how to make it work yet.
* I wouldn't use the Variational GP in this setting. It is way too slow to converge in my opinion. Also, it doesn't use natural gradients which makes the optimization also super slow.
* The minibatching isn't faster than the full way for relatively small datasets. And often the results are pretty poor. Not sure why.
* **Note**: I do want to point out that the likelihood is absent in the GP model and you have to instead use the white kernel. This threw me off at first and I made many mistakes with this and tbh I think this should be made clear and fixed.


---
### Inspiration

These are some repos I took inspiration from to write decent wrappers of other algorithms

* GPR with some gradient calculation convenience functions. - [Repo](https://github.com/xingchenwan/wsabi_ratio/blob/master/bayesquad/gps.py)
* New GPy function to account for Heterogeneous MOs - [Repo](https://github.com/pmorenoz/HetMOGP)

