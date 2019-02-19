import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def visualize_linear_model(
    file_name,
    coeff,
    intercept,
    cov,
    x,
    y=None,
    N=100,
    coeff_label='Coefficients',
    intercept_label='Intercept',
    cov_label='Covariance',
    y_label='$\\mathbf{y}$',
    time_label='$t$',
    show=False,
    export_data=True
):
    """
    Creates a figure visualizing a timeseries of linear Gausian models.

    Args:
        file: File to write figure to.
        coeff: Linear coefficients. Shape: (T, dY, dX)
        intercept: Constants. Shape: (T, dY)
        cov: Covariances. Shape: (T, dY, dY)
        x: Shape (T, dX)
        y: Optional. Shape (T, dY)
        N: Number of random samples drawn to visualize variance.
    """
    fig = plt.figure(figsize=(16, 12))

    T, dX = x.shape
    _, dY = intercept.shape

    # Check shapes
    assert coeff.shape == (T, dY, dX)
    assert intercept.shape == (T, dY)
    assert cov.shape == (T, dY, dY)
    assert x.shape == (T, dX)
    if y is not None:
        assert y.shape == (T, dY)

    # Intercept
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel(intercept_label)
    ax1.set_xlabel(time_label)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(linestyle=':')
    for dim in range(dY):
        line, = ax1.plot(np.arange(T), intercept[:, dim])

    # Coefficients
    ax2 = fig.add_subplot(222)
    ax2.set_ylabel(coeff_label)
    ax2.set_xlabel(time_label)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dX):
            line, = ax2.plot(np.arange(T, dtype=int), coeff[:, dim1, dim2])

    # Covariance
    ax3 = fig.add_subplot(223)
    ax3.set_ylabel(cov_label)
    ax3.set_xlabel(time_label)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dY):
            line, = ax3.plot(np.arange(T), cov[:, dim1, dim2])

    # Approximation
    y_ = np.empty((N, T, dY))  # Approx y using the model
    for t in range(T):
        mu = np.dot(coeff[t], x[t]) + intercept[t]
        y_[:, t] = np.random.multivariate_normal(mean=mu, cov=cov[t], size=N)
    y_mean = np.mean(y_, axis=0)
    y_std = np.std(y_, axis=0)

    ax4 = fig.add_subplot(224)
    ax4.set_ylabel(y_label)
    ax4.set_xlabel(time_label)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(linestyle=':')
    for dim in range(dY):
        line, = ax4.plot(np.arange(T), y_mean[:, dim])
        c = line.get_color()
        if y is not None:
            ax4.plot(np.arange(T), y[:, dim], ':')
        ax4.fill_between(
            np.arange(T),
            y_mean[:, dim] - y_std[:, dim],
            y_mean[:, dim] + y_std[:, dim],
            facecolor=c,
            alpha=0.25,
            interpolate=True
        )
    fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
    if export_data:
        np.savez_compressed(file_name, coeff=coeff, intercept=intercept, cov=cov, y=y, y_mean=y_mean, y_std=y_std)
