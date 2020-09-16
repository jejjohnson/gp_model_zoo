---
title: 1D Example
description: 1D Example using Different GP Libraries
authors:
    - J. Emmanuel Johnson
path: docs/demos
source: 1d_example.md
---
# 1D Example Walk-through

!!! info "Colab Notebooks"

    <center>

    |   Name   |                                                                           Colab Notebook                                                                            |
    | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
    | Sklearn  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vbDP0vtILN6-FLO_kHOyebSMHeOYR5Y1?usp=sharing) |
    |  PyMC3   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vbDP0vtILN6-FLO_kHOyebSMHeOYR5Y1?usp=sharing) |
    |  GPFlow  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_ip2kWmp344GC76Dj7IX3vYfTZsz-S-S?usp=sharing) |
    | GPyTorch | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vbDP0vtILN6-FLO_kHOyebSMHeOYR5Y1?usp=sharing) |

    </center>

This post we will go over some of the 1D .

---

## Data

=== "Scikit-Learn"

=== "PyMC3"

=== "GPFlow"

=== "GPyTorch"

---

## GP Model

=== "Scikit-Learn"

    In `scikit-learn` we just need the `.fit()`, `.predict()`, and `.score()`.

    First we need to define the parameters necessary for the gaussian process regression algorithm

    
    ```python
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    # define the kernel
    kernel = ConstantKernel() * RBF() + WhiteKernel()

    # define the model
    gp_model = GaussianProcessRegressor(
        kernel=kernel,          # Kernel Function
        alpha=1e-5,             # Noise Level
        n_restarts_optimizer=5  # Good Practice
    )
    ```

=== "PyMC3"

=== "GPFlow"

    **Kernel Function**

    ```python
    from gpflow.kernels import RBF

    # define the kernel
    kernel = RBF()

    # get a nice summary
    print_summary(kernel, )
    ```

    ```bash
    ╒═════════════════════════════════╤═══════════╤═════════════╤═════════╤═════════════╤═════════╤═════════╤═════════╕
    │ name                            │ class     │ transform   │ prior   │ trainable   │ shape   │ dtype   │   value │
    ╞═════════════════════════════════╪═══════════╪═════════════╪═════════╪═════════════╪═════════╪═════════╪═════════╡
    │ SquaredExponential.variance     │ Parameter │ Softplus    │         │ True        │ ()      │ float64 │       1 │
    ├─────────────────────────────────┼───────────┼─────────────┼─────────┼─────────────┼─────────┼─────────┼─────────┤
    │ SquaredExponential.lengthscales │ Parameter │ Softplus    │         │ True        │ ()      │ float64 │       1 │
    ╘═════════════════════════════════╧═══════════╧═════════════╧═════════╧═════════════╧═════════╧═════════╧═════════╛
    ```

    **Mean Function**
    ```python
    from gpflow.mean_functions import Linear

    # define mean function
    mean_f = Linear()

    # get a nice summary
    print_summary(mean_f, )
    ```

    **GP Model**

    ```python
    from gpflow.models import GPR

    # define GP Model
    gp_model = GPR(data=(X, y), kernel=kernel, mean_function=mean_f)

    # get a nice summary
    print_summary(gp_model, )
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

=== "Scikit-Learn"

    It doesn't get any simpler than this...

    ```python
    # fit GP model
    gp_model.fit(X, y);
    ```
    
    It's because everything is under the hood within the `.fit` method.

=== "PyMC3"

=== "GPFlow"

    ```python
    # define optimizer and params
    opt = gpflow.optimizers.Scipy()
    method = "L-BFGS-B"
    n_iters = 100

    # optimize
    opt_logs = opt.minimize(
        gp_model.training_loss,
        gp_model.trainable_variables, 
        options=dict(maxiter=n_iters),
        method=method,
    )

    # print a summary of the results
    print_summary(gp_model)
    ```
    ```bash
    ╒═════════════════════════╤═══════════╤══════════════════╤═════════╤═════════════╤═════════╤═════════╤═════════════════════╕
    │ name                    │ class     │ transform        │ prior   │ trainable   │ shape   │ dtype   │ value               │
    ╞═════════════════════════╪═══════════╪══════════════════╪═════════╪═════════════╪═════════╪═════════╪═════════════════════╡
    │ GPR.mean_function.A     │ Parameter │ Identity         │         │ True        │ (1, 1)  │ float64 │ [[-0.053]]          │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ GPR.mean_function.b     │ Parameter │ Identity         │         │ True        │ (1,)    │ float64 │ [-0.09]             │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ GPR.kernel.variance     │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 1.0576019909271173  │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ GPR.kernel.lengthscales │ Parameter │ Softplus         │         │ True        │ ()      │ float64 │ 3.3182129463115735  │
    ├─────────────────────────┼───────────┼──────────────────┼─────────┼─────────────┼─────────┼─────────┼─────────────────────┤
    │ GPR.likelihood.variance │ Parameter │ Softplus + Shift │         │ True        │ ()      │ float64 │ 0.05877109253391096 │
    ╘═════════════════════════╧═══════════╧══════════════════╧═════════╧═════════════╧═════════╧═════════╧═════════════════════╛
    ```

=== "GPyTorch"

---

## Predictions

=== "Scikit-Learn"

    Predictions are also very simple. You simply call the `.predict` method and you will get your predictive mean. If you want the predictive variance, you need to put the `return_std` flag as `True`.

    ```python
    # generate some points for plotting
    X_plot = np.linspace(- 1.2, 1.2, 100)[:, np.newaxis]

    # predictive mean and standard deviation
    y_mean, y_std = gp_model.predict(X_plot, return_std=True)

    # confidence intervals
    y_upper = y_mean.squeeze() + 2* y_std
    y_lower = y_mean.squeeze() - 2* y_std
    ```

    ??? tip "Predictive Standard Deviation"
        There is no way to turn off or no the likelihood noise or not (i.e. the predictive mean `y` of the predictive mean `f`). It is always `y` so the standard deviations will be a little high. To not use the `y`, you will need
        to actually subtract the `WhiteKernel` and the `alpha` from your predictive standard deviation.

=== "PyMC3"

=== "GPFlow"

    ```python
    # generate some points for plotting
    X_plot = np.linspace(-3*np.pi, 3*np.pi, 100)[:, np.newaxis]

    # predictive mean and standard deviation
    y_mean, y_var = gp_model.predict_y(X_plot)

    # convert to numpy arrays
    y_mean, y_var = y_mean.numpy(), y_var.numpy()

    # confidence intervals
    y_upper = y_mean + 2* np.sqrt(y_var)
    y_lower = y_mean - 2* np.sqrt(y_var)
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
    fig, ax = plt.subplots()



    ax.scatter(
        X, y, 
        label="Training Data",
        color='Red'
    )

    ax.plot(
        X_plot, y_mean, 
        color='black', lw=3, label='Predictions'
    )
    plt.fill_between(
        X_plot.squeeze(), 
        y_upper.squeeze(), 
        y_lower.squeeze(), 
        color='darkorange', alpha=0.2,
        label='95% Confidence'
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```


=== "Scikit-Learn"


    <center>
    ![Placeholder](pics/1d_example/sklearn_fit.png){: loading=lazy }
    </center>

=== "PyMC3"

=== "GPFlow"

    <center>
    ![Placeholder](pics/1d_example/gpflow_fit.png){: loading=lazy }
    </center>

=== "GPyTorch"
