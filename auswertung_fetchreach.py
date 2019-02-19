from libeval import plot_dist, read_iteration_data, reset_plot_color
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def visualize_fetchreach(file_name, sample_type, show=True, export_pgf=False):
    reset_plot_color()
    if export_pgf:
        plt.rcParams["font.family"] = ["serif"]
        plt.rcParams["pgf.rcfonts"] = False
    fig = plt.figure(figsize=[6.1, 2] if export_pgf else [16, 9])
    plt.suptitle('FetchReach static train/' + sample_type[-6:] + ' test\n$M=4, N=5$,')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Distance to target')
    #ax1.set_ylabel('$||\\mathbf{p}^{ee}_T - \\mathbf{g}||_2$')
    plt.grid(linestyle=':')
    #ax1.set_xscale('log')

    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('success rate')
    plt.grid(linestyle=':')

    target = np.zeros(3)
    threshold = 0.01

    scaler = StandardScaler()
    scaler.mean_ = [6.71023554e-03, -1.16670921e-02, -6.79674174e-03]
    scaler.scale_ = [7.57966860e-02, 1.03377066e-01, 6.24320497e-02]

    def plot(label, experiment, N):
        plot_dist(
            ax1,
            ax2,
            label=label,
            data=read_iteration_data(
                experiment, target, threshold, search=True, sample_type=sample_type, scaler=scaler
            ),
            N=N
        )

    lqr = sample_type[:3] == 'lqr'
    plot(
        "LQR (MDGPS)" if lqr else 'MDGPS',
        'experiments/gym_fetchreach_mdgps/data_files/fetchreach_gps-static-M4-5s-T20-K8-lqr_pol', 4 * 5
    )
    plot(
        "LQR (SRMDGPS)" if lqr else 'SRMDGPS',
        'experiments/gym_fetchreach_srmdgps/data_files/fetchreach_srmdgps-static-M4-5s-T20-K8-lqr_pol', 4 * 5
    )

    ax1.axhline(y=threshold, label='Distance threshold', color='black', linewidth=1, linestyle='--')
    # ax1.set_xscale('log')

    #ax1.legend()
    ax2.legend()
    #plt.tight_layout()
    ax1.set_xlim(right=200),
    ax1.set_ylim(bottom=0),
    ax2.set_ylim(bottom=-0.05, top=1.05),
    plt.subplots_adjust(wspace=0.2)

    if file_name is not None:
        fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0, dpi=300)
        if export_pgf:
            fig.savefig(file_name + '.pgf', bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()

    # figLegend = plt.figure(figsize=(6.1, 1.3))
    # plt.figlegend(*ax1.get_legend_handles_labels(), loc='center', ncol=2, fontsize='smaller')
    # figLegend.savefig(file_name + "-legend.png", bbox_inches='tight', pad_inches=0.05)
    # if export_pgf:
    #     figLegend.savefig(file_name + "-legend.pgf", bbox_inches='tight', pad_inches=0)
    # plt.close(figLegend)

    plt.close(fig)


# visualize_fetchreach(
#     'plots/evaluation_fetchreach_KL-MSE-min_lqr-random', sample_type='lqr-random', show=False, export_pgf=False
# )
# visualize_fetchreach(
#     'plots/evaluation_fetchreach_KL-MSE-min_lqr-static', sample_type='lqr-static', show=False, export_pgf=False
# )
visualize_fetchreach('plots/evaluation_fetchreach_pol-random', sample_type='pol-random', show=False, export_pgf=False)
visualize_fetchreach('plots/evaluation_fetchreach_pol-static', sample_type='pol-static', show=False, export_pgf=False)
