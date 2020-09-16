
### [Pyro](http://pyro.ai/)
This is my personal defacto library for doing research with GPs with PyTorch. In particular for GPs, I find the library to be super easy to mix and match priors and parameters for my GP models. I'm more comfortable with PyTorch so it was easy for me to Also pyro has a great [forum](https://forum.pyro.ai/) which is very active and the devs are always willing to help. It is backed by Uber and built off of PyTorch so it has a strong dev community. I also talked to the devs at the ICML conference in 2019 and found that they were super open and passionate about the project. 

<!-- tabs:start -->

#### ** Model **

```python
kernel2 = gp.kernels.RBF(
    input_dim=1, 
    variance=torch.tensor(0.1),
    lengthscale=torch.tensor(10.)
)
gpr_model = gp.models.GPRegression(X, y, kernel2, noise=torch.tensor(0.1))
```

#### ** Training **

```python
# define optimizer
optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)

# define loss function
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 1_000

# typical PyTorch boilerplate code
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
```
<!-- tabs:end -->

**[Source](http://pyro.ai/examples/gp.html)**: Pyro Docs

---


### [TensorFlow Probability](https://www.tensorflow.org/probability/)
This library is built into Tensorflow already and they have a few GP modules that allow you to train GP algorithms. In edition, they have a keras-like GP layer which is very useful for using a GP as a final layer in probabilistic neural networks. The GP community is quite small for TFP so I haven't seen too many examples for this.

<!-- tabs:start -->

#### ** Model **

```python
# define kernel function
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()

# Define the model.
model = tfp.layers.VariationalGaussianProcess(
    num_inducing_points=512, 
    kernel_provider=kernel
)
```

#### ** Keras Training **

```python
# Custom Loss Function
loss = lambda y, rv_y: rv_y.variational_loss(
    y, 
    kl_weight=np.array(batch_size, x.dtype) / x.shape[0]
)

# TF2.0 Keras Training Loop
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss)
model.fit(x, y, batch_size=batch_size, epochs=1000, verbose=False)
```

<!-- tabs:end -->

---

### [Edward2](https://github.com/google/edward2)
This is the most exciting one in my opinion because this library will allow GPs (and Deep GPs) to be used for the most novice users and engineers. It features the GP and sparse GP as bayesian keras-like layers. So you can stack as many of them as you want and then call the keras `model.fit()`. With this API, we will be able to prototype very quickly and really start applying GPs out-of-the-box. I think this is a really great feature and will put GPs on the map because it doesn't get any easier than this.

<!-- tabs:start -->

#### ** Model **

```python
# define kernel function
kernel = ExponentiatedQuadratic()

# Define the model.
model = ed.layers.SparseGaussianProcess(3, num_inducing=512, covariance_fn=kernel)
predictions = model(features)
```

#### ** Custom Training **

```python
# Custom Loss Function
def loss_fn(features, labels):
  preds = model(features)
  nll = -tf.reduce_mean(predictions.distribution.log_prob(labels))
  kl = sum(model.losses) / total_dataset_size
  return nll + kl

# TF2.0 Custom Training loop#
num_steps = 1000
for _ in range(num_steps):
  with tf.GradientTape() as tape:
    loss = loss_fn(features, labels)
  gradients = tape.gradient(loss, model.variables)  # use any optimizer here
```

---
