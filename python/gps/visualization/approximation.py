import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def visualize_approximation(
    file_name,
    target,
    approx,
    x_label='$t$',
    y_label='$\\mathbf{x}$',
    dim_label_pattern='$\\mathbf{x}_t[%d]$',
    show=False,
    export_data=True
):
    """
    Visualizes approximation ability.
    Args:
        file_name: File name without extension.
        losses: ndarray (N_epochs, N_losses) with losses.
        labels: list (N_losses, ) with labels for each loss.
        show: Display generated plot. This is a blocking operation.
        export_data: Writes a npz file containing the plotted data points.
                     This is useful for later recreation of the plot.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(linestyle=':')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    T, dX = target.shape
    assert target.shape == approx.shape, "Target and approximation must have same shape"

    for dim in range(dX):
        line, = ax1.plot(np.arange(T), target[:, dim], ':')
        c = line.get_color()
        ax1.plot(
            np.arange(T),
            approx[:, dim],
            color=c,
            label=dim_label_pattern % dim if dim_label_pattern is not None else None
        )

    ax1.legend()
    fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
    if export_data:
        np.savez_compressed(file_name, target=target, approx=approx)
