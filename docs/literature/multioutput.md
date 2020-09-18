# Multi-Output Models 

In regression, this is when we try to learn a function $f$ or multiple functions to predict multiple output $y$ .

!!! details "Details"
    
    Concretly, we have a regression problem and we are trying to predict $y\in \mathbb{R}^{N\times P}$ where $N$ is the number of samples and $P$ is the number of outputs. Now the question is, how do we model all of these outputs jointly? 
    
    There are some questions:
    
    * Are there correlations between the outputs or independent?
    * Is there missing data?
    * Do we actually need to model all of them at the same time and/or jointly?

## Terminology 

Firstly, there is some confusion about the terminology. I've heard the following names:

!!! info "Multi-Task"

    when we have different outputs. BUT possibly with different learning objectives and even different data types. e.g. one output is a regression task, one output is a classification problem, one output is a meta-learning task, etc. (a.k.a. the different tasks). The easiest parallel I can think of is self-driving cars. The objective is to drive, but you need to do many different tasks in order to reach the objective: drive without crashing.

!!! info "Multi-Output"

    typically when we have one task but just more than one output. e.g. multiple outputs for regression or multiple outputs for classification. So a concrete example is if we are predicting temperature and windspeed from some inputs.

!!! info "Multi-Fidelity"

    same as multi-output but catered more to situations where we know one of the outputs is of lower quality. e.g. a regression problem where one of the targets has less resolution or perhaps missing some data.

These definitions come from a discussion I had with the [GPFlow Community](https://gpflow.slack.com/archives/C144SAH60/p1588941862010600). I have yet to see a paper that is consistent with how these are done. I have broken up each section based off of their name but as seen from the names, there is a lot of overlap.


## 🆕 New!

??? info "Scalable Exact Inference in Multi-Output Gaussian Processes - Bruinsma et al (2020)"
    > They show a nice trick where you learn an invertible projection on your output space to reduce the crazy amount of outputs.

    📜 [Paper](https://arxiv.org/abs/1911.06287)

    💻📝 [Code](https://github.com/wesselb/oilmm) | [Julia](https://github.com/willtebbutt/OILMMs.jl)

    📺 [ICML Prezi](https://slideslive.com/38928160/scalable-exact-inference-in-multioutput-gaussian-processes?ref=speaker-37219-latest)


## 📺 Lectures

??? info "GPSS Summer School - Alvarez (2017)"
    
    📺 [Video](https://youtu.be/ttgUJtVJthA)
    
    📋 [Slides](http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf)

---

## 📜 Literature

### Multi-Task

??? info "Multi-task Gaussian Process prediction - Bonilla et al. (2007"
    -> [paper](https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction)
    
    -> [GPSS 2008 Slides](http://gpss.cc/bark08/slides/3%20williams.pdf)

---

### Multi-Output

??? info "Efficient multioutput Gaussian processes through variational inducing kernels - Alvarez et al. (2011)"
    -> [paper](http://proceedings.mlr.press/v9/alvarez10a.html) 

??? info "Remarks on multi-output Gaussian process regression - Liu et. al. (2018)"
    -> [pdf](http://memetic-computing.org/publication/journal/MOGP_Remarks.pdf)

??? info "Heterogeneous Multi-output Gaussian Process Prediction - Moreno-Muñez et. al. (2018)"
    -> [paper](https://arxiv.org/abs/1805.07633)

    -> [code](https://github.com/pmorenoz/HetMOGP)

??? info "Fast Approximate Multi-output Gaussian Processes - Joukov & Kulic"
    -> [paper](https://arxiv.org/abs/2008.09848v1)

---

### Multi-Fidelity

??? info "Deep Multi-fidelity Gaussian Processes - Raissi & Karniadakis (2016)"
    -> [paper](https://arxiv.org/abs/1604.07484)
    
    -> [blog](https://maziarraissi.github.io/research/4_multifidelity_modeling/)

??? info "Deep Gaussian Processes for Multi-fidelity Modeling - Cutjar et. al. (2019)"
    -> [paper](https://arxiv.org/abs/1903.07320)
    
    -> [notebook](https://github.com/amzn/emukit/tree/master/emukit/examples/multi_fidelity_dgp)
    
    -> [poster](http://kurtcutajar.com/pres/bdl_poster.pdf)
