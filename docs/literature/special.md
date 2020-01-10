# Special Models

These are models that I deem 'special' in the sense that they're not exactly what GPs were originally tended for but they have usefulness in other fields. I consider these algorithms in the category of "jacking around" (see Neal Lawrences lecture for the origin of the phrase). In my mind, these are algorithms that tackle a special case problem in the application realm. It's a bit difficult to verify how it does outside of the corner case due to no real comprehensive code-bases or benchmarks of GPs in real-world applications. 

## Derivative Constraints

* [Scaling GPR with Derivatives](https://arxiv.org/abs/1810.12283) - Eriksson et. al. (2018) | [Code]()
  
  > They use MVM methods to enable one to solve for functions and their derivatives at the same time at scale.

## Multi-Output Models 

In regression, this is when we try to learn a function $f$ or multiple functions to predict multiple output $y$ .

<details>
Concretly, we are trying to predict $y\in \mathbb{R}^{N\times P}$ where $N$ is the number of samples and $P$ is the number of outputs.
</details>

#### Terminology 

Firstly, there is some confusion about the terminology. I've heard the following names:

* Multi-Task 
* Multi-Output 
* Multi-Fidelity 

I cannot for the life of me figure out what is the difference between all of them. I have yet to see a paper that is consistent with how these are done. I have broken up each section based off of their name but I won't claim that there is no overlap between terms.


#### Lectures

* GPSS 2017 Summer School - Alvarez - [Video](https://youtu.be/ttgUJtVJthA) | [Slides](http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf)

#### Literature

##### Multi-Task

* Multi-task Gaussian Process prediction - Bonilla et al. 2007 - [paper](https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction) | [GPSS 2008 Slides](http://gpss.cc/bark08/slides/3%20williams.pdf)

##### Multi-Output

* Efficient multioutput Gaussian processes through variational inducing kernels - Alvarez et al. (2011) - [paper](http://proceedings.mlr.press/v9/alvarez10a.html) 
* Remarks on multi-output Gaussian process regression - Liu et. al. (2018) - [pdf](http://memetic-computing.org/publication/journal/MOGP_Remarks.pdf)
* Heterogeneous Multi-output Gaussian Process Prediction - Moreno-Muñez et. al. (2018) - [paper](https://arxiv.org/abs/1805.07633) | [code](https://github.com/pmorenoz/HetMOGP)

##### Multi-Fidelity

* Deep Multi-fidelity Gaussian Processes - Raissi & Karniadakis (2016) - [paper](https://arxiv.org/abs/1604.07484) | [blog](https://maziarraissi.github.io/research/4_multifidelity_modeling/) | 
* Deep Gaussian Processes for Multi-fidelity Modeling - Cutjar et. al. (2019) - [paper](https://arxiv.org/abs/1903.07320) | [notebook](https://github.com/amzn/emukit/tree/master/emukit/examples/multi_fidelity_dgp) | [poster](http://kurtcutajar.com/pres/bdl_poster.pdf)