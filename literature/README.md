# Gaussian Process Literature

* Author: J. Emmanuel Johnson
* Email : jemanjohnson34@gmail.com
* Website: jejjohnson.netlify.com

#### Introduction 

This folder contains lists of papers and code bases that deal with Gaussian processes and applications. I also try to keep up with the literature and show some state-of-the-art (SOTA) algorithms and code bases. I'm quite fond of GPs because they are mathematically beautiful and they have many nice properties. They also work with one of the nicest distributions to work with: the Guassian distribution. They deal with uncertainty and because they are function based, they work well with small (and large datasets) with sophisticated interpolation schemes. The biggest problem of GPs back in the day was that they did not scale but nowadays that isn't the case as there are dozens of sparse methods that allow us to scale GPs to 10K or even 1 million data points.

#### My Perspective

I think that the GP literature is fairly spread over many fields but there is not really much of any compilation of GP resources (not until recently at least). I have been studying GPs for a while now to try to understand the field as well as see how they have been used. My feeling is that there are more methods than actual applications and many of the algorithms seem to have never been applied to many things outside of their "corner case" application. If you look at a standard GP algorithm, it seems to go within one of the following categories:

1. Algorithm
2. Special Kernel
3. Random Features Approximation
4. Heteroscedastic Considerations
5. Sparse Implementation
6. Latent Variable Representation
7. Approximate Inference
   * Variational
   * Expectation Propagation
   * MCMC
8. Apply to MultiOutput (Multi-Fidelity) Cases
9. Deep-ify Algorithm (stack GPs)

In reality, almost all of the methods you'll find in the literature come within these subfields or a combination of a 2 or more. That's not to say that what people do isn't impressive, but it would be nice if we had some better structure to how we classify GP algorithms and improvements we make.

There are a few questions and issues which I think have not been answered but are steadily improving with time. They are individual components of the GP algorithm that I consider a bit weak and I think they can be improved by taking knowledge other fields. A good example would be how the treatment of Matrix-Value-Multiplication (MVM) was for scaling GP algorithms. It fairly small within the main GP community but now it has risen as the most scalable and customizable method to date with a dominant python package. 

#### Special Interest 

Below is a list of methods which are not as prevelant in the literature but that I believe are super important and could possibly greatly improve the methods that we already have. I will try to pay special attention to #1 in particular because it is apart of my research.

1. Input Uncertainty
2. Missing Data
3. Semi-Supervised Learning
4. Parameter Initialization and Priors
5. GP Training procedures
6. Expressive Kernels

---
## Table of Contents

**[Gaussian Processes](gps.md)**

This file some recommended resources as well as some SOTA algorithms and key improvements over the years. It also includes sparse GPs and the treatment of random fourier features. At some point I will do a timeline of some of the most import GP papers within the literature. 

**[Sparse Gaussian Processes](sparse_gps.md)**

This contains some of the main GP algorithms that you need to know in the way of scaling. If your dataset is larger than 2K then you should probably start using one of these sparse methods.

**[Latent Variable Models](latent_variable.md)**

These algorithms are when we assume that the input $X$ is not determinant but instead an unknown. The applications of this range from dimensionality reduction to uncertain inputs and even applications in missing data and semi-supervised learning.

**[Sparse Spectrum](fourier.md)**

These algorithms make use of the Bochner theorem which states that we can represent kernel functions as an infinite series with weights that stem from a Gaussian distribution. These (what are essentially Fourier transformations) are typically known as Sparse Spectrum GPs in the community.

**[Uncertain Inputs](uncertain.md)**

This is directly related to my research so I'll pay special attention to this. I look at the literature spanned from the beginning up until now. This will mostly be about moment-matching algorithms and algorithms that use variational inference.

**[Deep GPs](deep_gps.md)**

I have made this a special section because I think it's quite an interesting topic. It's still a fairly young aspect of GPs (last 7 years or so) so it won't have such a depth of literature like the other topics. I'll also include stuff related to how Deep GPs are related to neural networks.

**[Components](components.md)**

These are key elements of the GP algorithm that have been studied in the 'special interest' listed above, e.g. input uncertainty, training procedures, parameter estimation.

**[Software](software.md)**

The fun part. Here is where I look at all the latest software that one can use to run some of the SOTA algorithms. It will python libraries only because it's the language I personally use to code.  


**[Applications](applications.md)**

This consists of papers which have applied GPs to their respective fields. I'll focus more on Earth observation applications but I'll put up any others if I find them of interest.

**[Special](special.md)**

This will include all of the GP algorithms that I consider 'corner cases', i.e. GPs modified that apply to a specific application where some modification was necessary to make the GP work to a standard. 