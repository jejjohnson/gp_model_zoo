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

You'll find these below but I wanted to highlight them because they're pretty cool.

??? tip "Scalable Exact Inference in Multi-Output Gaussian Processes - Bruinsma et al (2020)"
    > They show a nice trick where you learn an invertible projection on your output space to reduce the crazy amount of outputs.

    📜 [Paper](https://arxiv.org/abs/1911.06287)

    💻📝 [Code](https://github.com/wesselb/oilmm) | [Julia](https://github.com/willtebbutt/OILMMs.jl)

    📺 [ICML Prezi](https://slideslive.com/38928160/scalable-exact-inference-in-multioutput-gaussian-processes?ref=speaker-37219-latest)


??? tip "A Framework for Interdomain and Multioutput Gaussian Processes - by Van der Wilk et al (2020)"
    > A full framework in GPFlow where they implement a framework that allows maximum flexibility when working with multi-output GPs.
    
    📜 [Paper](https://arxiv.org/abs/2003.01115)

    💻📝 [Demo Notebook](https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html)

## 📺 Lectures

??? info "GPSS Summer School - Alvarez (2017)"
    
    📺 [Video](https://youtu.be/ttgUJtVJthA)
    
    📋 [Slides](http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf)

---

## 📜 Literature

### Multi-Task

Problems like these tend to be when we have a multi-output problem but we don't necessarily have all outputs. We also assume we have **correlated dimensions**.  From (Bonilla & Williams, 2008), known as the **intrinsic model of coregionalization** (ICM), We have the form:

$$
\begin{aligned}
\text{cov}(f_i(X), f_j(X')) &=k(X,X') \cdot B[i,j] \\
\mathbf{B}&=\mathbf{WW^\top} + \text{diag}(k)
\end{aligned}
$$

This is useful for problems with a small number of dimensions because it's quite an expensive method.

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

??? tip "A Framework for Interdomain and Multioutput Gaussian Processes - by Van der Wilk et al (2020)"

    -> [Paper](https://arxiv.org/abs/2003.01115)

    -> [Demo Notebook](https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html)


??? info "Fast Approximate Multi-output Gaussian Processes - Joukov & Kulic (2020)"
    -> [paper](https://arxiv.org/abs/2008.09848v1)

??? tip "Scalable Exact Inference in Multi-Output Gaussian Processes - Bruinsma et al (2020)"
    > They show a nice trick where you learn an invertible projection on your output space to reduce the crazy amount of outputs.

    📜 [Paper](https://arxiv.org/abs/1911.06287)

    💻📝 [Code](https://github.com/wesselb/oilmm) | [Julia](https://github.com/willtebbutt/OILMMs.jl)

    📺 [ICML Prezi](https://slideslive.com/38928160/scalable-exact-inference-in-multioutput-gaussian-processes?ref=speaker-37219-latest)


---

### Multi-Fidelity

??? info "Deep Multi-fidelity Gaussian Processes - Raissi & Karniadakis (2016)"
    -> [paper](https://arxiv.org/abs/1604.07484)
    
    -> [blog](https://maziarraissi.github.io/research/4_multifidelity_modeling/)

??? info "Deep Gaussian Processes for Multi-fidelity Modeling - Cutjar et. al. (2019)"
    -> [paper](https://arxiv.org/abs/1903.07320)
    
    -> [notebook](https://github.com/amzn/emukit/tree/master/emukit/examples/multi_fidelity_dgp)
    
    -> [poster](http://kurtcutajar.com/pres/bdl_poster.pdf)

    -> [Code](https://github.com/apaleyes/emukit/tree/master/emukit/examples/multi_fidelity_dgp)

---
## Software

I decided to include a special section about the software because there is no real go-to library for dealing with multioutput GPs as of now.


#### Exact GP


??? info "**GPFlow**"

    * [Demo Notebook](https://gpflow.readthedocs.io/en/master/notebooks/advanced/coregionalisation.html).
    > Use this if you have correlated outputs with a low number of dimensions and samples.

??? tip "**GPyTorch**"

    * [Coregionalization (Correlated Outputs)](https://docs.gpytorch.ai/en/v1.2.0/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html) 
    > Use this if you have correlated outputs with a low number of dimensions and samples.

    * [Batch Independent MultiOutput GP](https://docs.gpytorch.ai/en/v1.2.0/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html)
    > Use this if you assume independent outputs and the number of inputs and each output have the same size.

    * [ModelList (Multi-Output) GP Regression](https://docs.gpytorch.ai/en/v1.2.0/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html)
    > Use if you have a different independent GP models with no correlation between inputs and outputs.

#### Sparse GP


??? tip "**GPyTorch**"

    * [Coregionalization (Correlated Outputs)](https://docs.gpytorch.ai/en/v1.2.0/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html#Types-of-Variational-Multitask-Models) 
    > Use this if you have correlated outputs with a low number of dimensions and samples. Uses the Linear Model of Coregionalization (LMC).

    * [Batch Independent MultiOutput GP](https://docs.gpytorch.ai/en/v1.2.0/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html#Types-of-Variational-Multitask-Models)
    > Use this if you assume independent outputs and the number of inputs and each output have the same size.

??? tip "**GPFlow**"

    * [Full Framework](https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html)