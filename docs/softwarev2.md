## What is Deep Learning?

Before we get into the software, I just wanted to quickly define deep learning. A recent debate on [twitter](https://twitter.com/yudapearl/status/1215174538087948288) got me thinking about an appropriate definition and it helped me think about how this definition relates to the software. It gave me perspective.

**Definition 1** by Yann LeCun - [tweet](https://twitter.com/ylecun/status/1215286749477384192) (paraphrased)

> Deep Learning is methodology: building a model by assembling parameterized modules into (possibly dynamic) graphs and optimizing it with gradient-based methods.

**Definition II** by Danilo Rezende - [tweet](https://twitter.com/DeepSpiker/status/1209862283368816641) (paraphrased)

> Deep Learning is a collection of tools to build complex modular differentiable functions.

These definitions are more or less the same: deep learning is a tool to facilitate gradient-based optimization scheme for models. The data we use, the exact way we construct it, and how we train it aren't really in the definition. Most people might think a DL tool is the ensemble of different neural networks like [these](https://pbs.twimg.com/media/EOWJc2KWsAA8xDF?format=jpg&name=4096x4096). But from henceforth, I refer to DL in the terms of facilitating the development of those neural networks, not the network library itself.

So in terms of DL software, we need only a few components:

* Tensor structures
* Automatic differentiation (AutoGrad)
* Model Framework (Layers, etc)
* Optimizers
* Loss Functions

Anything built on top of that can be special cases where we need special structures to create models for special cases. The simple example is a Multi-Layer Perceptron (MLP) model where we need some `weight` parameter, a `bias` parameter and an `activation` function. A library that allows you to train this model using an optimizer and a loss function, I would consider this autograd software (e.g. JAX). A library that has this functionality built-in (a.k.a. a `layer`), I would consider this deep learning software (e.g. TensorFlow, PyTorch). While the only difference is the level of encapsulation, the latter makes it much easier to build '*complex modular*' neural networks whereas the former, not so much. You could still do it with the autograd library but you would have to design your entire model structure from scratch as well. So, there are still a LOT of things we can do with parameters and autograd alone but I wouldn't classify it as DL software. This isn't super important in the grand scheme of things but I think it's important to think about when creating a programming language and/or package and thinking about the target user.

---

## Anatomy of good DL software

Francios Chollet (the creator of `keras`) has been very vocal about the benefits of how TensorFlow caters to a broad audience ranging from applied users and algorithm developers. Both sides of the audience have different needs so building software for both audiences can very, very challenging. Below I have included a really interesting figure which highlights the axis of operations.

<p align="center">

  <img src="https://keras-dev.s3.amazonaws.com/tutorials-img/spectrum-of-workflows.png" alt="drawing" width="800"/>
</p>

**Photo Credit**: Francois Chollet [Tweet](https://twitter.com/fchollet/status/1052228463300493312/photo/1)

As shown, there are two axis which define one way to split the DL software styles: the x-axis covers the **model** construction process and the y-axis covers the **training** process. I am sure that this is just one way to break apart DL software but I find it a good abstract way to look at it because I find that we can classify most use cases somewhere along this graph. I'll briefly outline a few below:

* **Case 1**: All I care about is using a prebuilt model on some new data that my company has given me. I would probably fall somewhere on the upper right corner of the graph with the `Sequential` model and the built-in `training` scheme.
* **Case II**: I need a slightly more complex training scheme because I want to learn two models that share hidden nodes but they're not the same size. I also want to do some sort of cycle training, i.e. train one model first and then train the other. Then I would probably fall somewhere near the middle, and slightly to the right with the `Functional` model and a custom `training` scheme.
* **Case III**: I am a DL researcher and I need to control every single aspect of my model. I belong to the left and on the bottom with the full `subclass` model and completely custom `training` scheme.

So there are many more special cases but by now you can imagine that most general cases can be found on the graph. I would like to stress that designing software to do all of these cases is not easy as these cases require careful design individually. It needs to be flexible.

Maybe I'm old school, but I like the modular way of design. So in essence, I think we should design libraries that focus on one aspect, one audience and do it well. I also like a standard practice and integration so that everything can fit together in the end and we can transfer information or products from one part to another. This is similar to how the Japanese revolutionized building cars by having one machine do one thing at a time and it all fit together via a standard assembly line. So in the end, I want people to be able to mix and match as they see fit. To try to please everyone with "*one DL library that rules them all*" seems a bit silly in my opinion because you're spreading out your resources. But then again, I've never built software from scratch and I'm not a mega coorperation like Google or Facebook, so what do I know? I'm just one user...in a sea of many.

> With great power, comes great responsibility - Uncle Ben

On a side note, when you build popular libraries, you shape how a massive amount of people think about the problem. Just like expressiveness is only as good as your vocabulary and limited by your language, the software you create actively morphs how your users think about framing and solving their problems. Just something to think about.

---
## Convergence of the Libraries

Originally, there was a lot of differences between the deep learning libraries, e.g. `static` v.s. `dynamic`, `Sequential` v.s. `Subclass`. But now they are all starting to converge or at least have similar ways of constructing models and training. Below is a quick example of 4 deep learning libraries. If you know your python DL libraries trivia, try and guess which library do you think it is. Click on the details below to find out the answer.

<p align="center">
  <img src="https://pbs.twimg.com/media/DppB0xJUUAAjGi-?format=jpg&name=4096x4096" alt="drawing" width="800"/>
</p>

**Photo Credit**: Francois Chollet [Tweet](https://twitter.com/fchollet/status/1052228463300493312/photo/1)

??? details "**Answer**"

    |         |            |
    | ------- | ---------- |
    | Gluon   | TensorFlow |
    | PyTorch | Chainer    |

It does begs the question: if all of the libraries are basically the same, why are their multiple libraries? That's a great question and I do not know the answer to that. I think options are good as competition generally stimulates innovation. But at some point, there should be a limit no? But then again, the companies backing each of these languages are quite huge (Google, Microsoft, Uber, Facebook, etc). So I'm sure they have more than enough employees to justify the existence of their own library. But then again, imagine if they all put their efforts into making one great library. It could be an epic success! Or an epic disaster. I guess we will never know.


---
## So what to choose?

There are many schools of thought. Some people suggest [doing things from scratch](https://ericmjl.github.io/blog/2019/10/31/reimplementing-and-testing-deep-learning-models/) while some favour software to allow users to [jumping right in](https://scale.com/interviews/jeremy-howard/transcript). Fortunately, whatever the case may be or where you're at in your ML journey, there is a library to suit your needs. And as seen above, most of them are converging so learning one python package will have be transferable to another. In the end, people are going to choose whatever based on personal factors such as "what is my style" or environmental factors such as "what is my research lab using now?".

I have a personal short list below just from observations, trends and reading but it is by no means concrete. Do whatever works for you!

**Jump Right In** - [fastai](https://docs.fast.ai/)

> If you're interesting in applying your models to new and interesting datasets and are not necessarily interested in development then I suggest you start with fastai. This is a library that simplifies deep learning usage with all of the SOTA tricks built-in so I think it would save the average user a lot of time.

**From Scratch** - [JAX](https://github.com/google/jax)

> If you like to do things from scratch in a very numpy-like way but also want all of the benefits of autograd on CPU/GPU/TPUs, then this is for you.

**Deep Learning Researcher** - [PyTorch](https://pytorch.org/)

> If you're doing research, then I suggest you use PyTorch. It is currently the most popular library for doing ML research. If you're looking at many of the SOTA algorithms, you'll find most of them being written in PyTorch these days. The API is similar to TensorFlow so you can easily transfer your skills to TF if needed.

**Production/Industry** - [TensorFlow](https://www.tensorflow.org/)

> TensorFlow holds the market in production. By far. So if you're looking to go into industry, it's highly likely that you'll be using TensorFlow. There are still a lot of researchers that use TF too. Fortunately, the API is similar to PyTorch if you use the subclass system so the skills are transferable.

!!! warning "**Warning**"
    The machine learning community changes rapidly so any trends you observe are extremely volatile. Just like the machine learning literature, what's popular today can change within 6 months. So don't ever lock yourself in and stay flexible to cope with the changes. But also don't jump on bandwagons either as you'll be jumping every weekend. Keep a good balance and maintain your mental health.
