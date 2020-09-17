---
title: 1D Example (Large Scale)
description: Large Scale 1D Example using Different GP Libraries
authors:
    - J. Emmanuel Johnson
path: docs/demos
source: 1d_example_ls.md
---
# Large Scale 1D Example Walk-through

!!! info "Colab Notebooks"

    <center>

    |   Name   |                                                                           Colab Notebook                                                                            |
    | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
    |  PyMC3   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vbDP0vtILN6-FLO_kHOyebSMHeOYR5Y1?usp=sharing) |
    |  GPFlow  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_ip2kWmp344GC76Dj7IX3vYfTZsz-S-S?usp=sharing) |
    | GPyTorch | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15o9-BWW98fP6corLWOew5a0sZq-_3Yvl?usp=sharing) |

    </center>

This post we will go over some of the 1D .

---

## Data

$$
f(x) = \sin(3\pi x) + \frac{1}{3}\sin(9\pi x) + \frac{1}{2} \sin(7 \pi x)
$$

```python
n_samples = 10_000

def f(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

X = np.linspace(-1.1, 1.1, n_samples)
X_plot = np.linspace(- 1.3, 1.3, 100)[:, np.newaxis]
y =  f(X) + 0.2 * rng.randn(X.shape[0])

X, y = X[:, np.newaxis], y[:, np.newaxis]
```

**Source**: [GPFlow - Big Data Tutorial](https://gpflow.readthedocs.io/en/develop/notebooks/advanced/gps_for_big_data.html)

---

## GP Model

=== "GPFlow"

    **Kernel & Mean Function**

    ```python
    from gpflow.mean_functions import Linear
    from gpflow.kernels import RBF

    # define the kernel
    kernel = RBF()

    # define mean function
    mean_f = Linear()
    ```


    **Inducing Points**

    ```python
    from sklearn.cluster import KMeans

    n_inducing = 50
    seed = 123

    # KMeans model
    kmeans_clf = KMeans(n_clusters=n_inducing)
    kmeans_clf.fit(X)

    # get cluster centers as inducing points
    Z = kmeans_clf.cluster_centers_
    ```

    <center>
    ![Placeholder](pics/1d_example_ls/gpflow_fit_ls_z.png){: loading=lazy }
    </center>

    **GP Model**

    ```python
    from gpflow.models import SVGP, SGPR

    # define GP Model
    sgpr_model = SGPR(
        data=(X, y),
        kernel=kernel,
        inducing_variable=Z,
        mean_function=mean_f, 
    )


    # get a nice summary
    print_summary(sgpr_model, )
    ```

    ```bash
    ╒═════════════════════════╤═══════════╤══════════════════╤═════════╤═════════════╤═════════╤═════════╤═════════╕
    │ name                    │ class     │ transform        │ prior   │ trainable   │ shape   │ dtype   │ value   │
    ╞═════════════════════════╪═══════════╪══════════════════╪═════════╪═════════════╪═════════╪═════════╪═════════╡
    │ GPR.mean_function.A     │ Parameter │ Identity         │         │ True        │ (1, 1)  │ float64 │ [[1.]]  │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────┤
    │ GPR.mean_function.b     │ Parameter │ Identity         │         │ True        │ (1,)    │ float64 │ [0.]    │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────┤
    │ GPR.kernel.variance     │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 1.0     │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────┤
    │ GPR.kernel.lengthscales │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 1.0     │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────┤
    │ GPR.likelihood.variance │ Parameter │ Softplus + Shift │         │ True        │ ()      │ float64 │ 1.0     │
    ```

    ??? tip "Numpy 2 Tensor"
        Notice how I didn't do anything about changing from the `np.ndarray` to the `tf.tensor`? Well that's because GPFlow is awesome and does it for you. Little things like that make the coding experience so much better.

=== "GPyTorch"

---

## Training Step

=== "GPFlow"

    ```python
    # define optimizer and params
    minibatch_size = 128

    # turn of training for inducing points
    opt = gpflow.optimizers.Scipy()

    # training loss
    training_loss = sgpr_model.training_loss_closure()
    method = "L-BFGS-B"
    n_iters = 1_000

    # optimize
    opt_logs = opt.minimize(
        sgpr_model.training_loss,
        sgpr_model.trainable_variables, 
        options=dict(maxiter=n_iters),
        method=method,
    )

    # print a summary of the results
    print_summary(sgpr_model)
    ```

    ```bash
    ╒══════════════════════════╤═══════════╤══════════════════╤═════════╤═════════════╤═════════╤═════════╤═════════════════════╕
    │ name                     │ class     │ transform        │ prior   │ trainable   │ shape   │ dtype   │ value               │
    ╞══════════════════════════╪═══════════╪══════════════════╪═════════╪═════════════╪═════════╪═════════╪═════════════════════╡
    │ SGPR.mean_function.A     │ Parameter │ Identity         │         │ True        │ (1, 1)  │ float64 │ [[-0.21]]           │
    ├──────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ SGPR.mean_function.b     │ Parameter │ Identity         │         │ True        │ (1,)    │ float64 │ [0.019]             │
    ├──────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ SGPR.kernel.variance     │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 1.161352313910884   │
    ├──────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ SGPR.kernel.lengthscales │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 0.09574863612699928 │
    ├──────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ SGPR.likelihood.variance │ Parameter │ Softplus + Shift │         │ True        │ ()      │ float64 │ 0.04012048665014923 │
    ├──────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ SGPR.inducing_variable.Z │ Parameter │ Identity         │         │ True        │ (50, 1) │ float64 │ [[-0.786...         │
    ╘══════════════════════════╧═══════════╧══════════════════╧═════════╧═════════════╧═════════╧═════════╧═════════════════════╛
    ```

=== "GPyTorch"

---

## Predictions

=== "GPFlow"

    ```python
    # generate some points for plotting
    X_plot = np.linspace(-1.2, 1.2, 100)[:, np.newaxis]

    # predictive mean and standard deviation
    y_mean, y_var = sgpr_model.predict_y(X_plot)


    # convert to numpy arrays
    y_mean, y_var = y_mean.numpy(), y_var.numpy()

    # confidence intervals
    y_upper = y_mean + 2* np.sqrt(y_var)
    y_lower = y_mean - 2* np.sqrt(y_var)

    # Get learned inducing points
    Z = sgpr_model.inducing_variable.Z.numpy()
    ```

    ??? tip "Tensor 2 Numpy"
        So we do have to convert to numpy arrays from tensors for the predictions. Note, you can plot tensors, but some of the commands might be different.

        E.g. 

        ```python
        y_upper = tf.squeeze(y_mean + 2 * tf.sqrt(y_var))
        ```


=== "GPyTorch"

---

## Visualization


??? info "Plot Function"

    It's the same for all libraries as I first convert everything to numpy arrays and then plot the results.

    ```python
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.scatter(
        X, y, 
        label="Training Data",
        color='Red',
        marker='x',
        alpha=0.2
    )

    ax.plot(
        X_plot, y_mean, 
        color='black', lw=3, label='Predictions'
    )
    plt.fill_between(
        X_plot.squeeze(), 
        y_upper.squeeze(), 
        y_lower.squeeze(), 
        color='lightblue', alpha=0.6,
        label='95% Confidence'
    )
    plt.scatter(
        Z, np.zeros_like(Z), 
        color="black", marker="|", 
        label="Inducing locations"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

=== "GPFlow"

    <center>
    ![Placeholder](pics/1d_example_ls/gpflow_fit_ls.png){: loading=lazy }
    </center>

=== "GPyTorch"

    <center>
    ![Placeholder](pics/1d_example_ls/gpytorch_fit_ls.png){: loading=lazy }
    </center>