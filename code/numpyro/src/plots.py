import matplotlib.pyplot as plt
from uncertainty_toolbox import viz as utviz
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_context(context="talk", font_scale=0.7)


def plot_calibration(y_mu, y_std, y_real):

    fig, ax = plt.subplots()

    return fig, ax


def plot_all_uncertainty(y_pred, y_var, y_true, model, data, log: bool = False):

    y_std = np.sqrt(y_var)
    utviz.plot_parity(y_pred=y_pred.ravel(), y_true=y_true.ravel())
    plt.tight_layout()
    plt.gcf()
    if log:
        wandb.log({f"parity_{model}_{data}": wandb.Image(plt)})
    plt.show()

    utviz.plot_calibration(
        y_pred=y_pred.ravel(), y_std=y_std.ravel(), y_true=y_true.ravel()
    )
    plt.tight_layout()
    plt.gcf()
    if log:
        wandb.log({f"calib_{model}_{data}": wandb.Image(plt)})
    plt.show()

    #     utviz.plot_intervals_ordered(
    #         y_pred=y_pred.ravel(), y_std=y_std.ravel(), y_true=y_true.ravel(), n_subset=100
    #     )
    # #     plt.gcf()
    # #     wandb.log({f"intervals_{data}": wandb.Image(plt)})
    #     plt.show()

    utviz.plot_sharpness(y_std=y_std.ravel(),)
    plt.tight_layout()
    plt.gcf()
    if log:
        wandb.log({f"sharpness_{model}_{data}": wandb.Image(plt)})
    plt.show()


#     utviz.plot_adversarial_group_calibration(
#         y_pred=y_pred.ravel(), y_std=y_std.ravel(), y_true=y_true.ravel(), n_subset=100
#     )
# #     plt.gcf()
# #     wandb.log({f"adverse_{data}": wandb.Image(plt)})
#     plt.show()
