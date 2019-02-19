import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def visualize_loss(
    file_name,
    losses,
    labels,
    x_label='epoch',
    y_label='$\\mathcal{L}$',
    log_scale=False,
    add_total=True,
    total_label='Total',
    show=False
):
    """
    Visualizes training losses.
    Args:
        file_name: File name without extension.
        losses: ndarray (N_epochs, N_losses) with losses.
        labels: list (N_losses, ) with labels for each loss.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    if log_scale:
        ax1.set_yscale('log')
    ax1.grid(linestyle=':')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    T, N = losses.shape
    assert len(labels) == N, "Must provide a label for each loss"
    for n in range(N):
        ax1.plot(np.arange(T), losses[:, n], label=labels[n])
    if add_total:
        ax1.plot(np.arange(T), np.sum(losses, axis=1), label=total_label)

    ax1.legend()
    fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
