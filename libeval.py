import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from os.path import isfile, isdir


def findExperiments(experiment):
    experiments = []
    seed = 0
    while True and seed < 15:
        if isdir(experiment + '_%d' % seed):
            experiments.append(experiment + '_%d' % seed)
        else:
            break
        seed += 1
    if len(experiments) == 0:
        print("No experiments found for '%s'" % experiment)
        exit(1)
    return experiments


def read_iteration_data(
    experiments, target, threshold, search=False, sample_type='pol', scaler=None, max_iteration=None
):
    if search:
        experiments = findExperiments(experiments)
    itr = 0
    dist = []
    goal_reached = []
    while True:
        valid_experiments = [
            experiment
            for experiment in experiments if isfile(experiment + '/samples_%s_%02d.npz' % (sample_type, itr))
        ]
        if len(valid_experiments) != len(experiments):
            if len(valid_experiments) > 0:
                print("missing", set(experiments).difference(set(valid_experiments)))
            break
        states = np.concatenate(
            [np.load(experiment + '/samples_%s_%02d.npz' % (sample_type, itr))['X'] for experiment in valid_experiments]
        )

        final_eep = states[:, :, -1, -3:].reshape(-1, 3)
        N = final_eep.shape[0]
        if scaler:
            final_eep = scaler.inverse_transform(final_eep)
        dist.append(np.linalg.norm(final_eep - target, axis=1))
        goal_reached.append(np.sum(dist[-1] <= threshold) / float(N))
        itr += 1
    if itr == 0:
        raise ValueError('Experiment does not exist:', experiments[0], sample_type)
    #itr = 15
    #dist = dist[:15]
    #goal_reached = goal_reached[:15]
    return {
        'iterations': itr,
        'mean': np.mean(dist, axis=1),
        'std': np.std(dist, axis=1),
        'median': np.quantile(dist, 0.5, axis=1),
        'quartil1': np.quantile(dist, 0.25, axis=1),
        'quartil3': np.quantile(dist, 0.75, axis=1),
        'min': np.min(dist, axis=1),
        'max': np.max(dist, axis=1),
        # 'responsibilities': responsibilities,
        # 'tac_iterations': [np.sum([len(r) for r in r2])/float(len(r2)) for r2 in responsibilities],
        'goal_reached': np.asarray(goal_reached),
    }


def plot_dist(ax1, ax2, label, data, N, sameColor=False, mean=False):
    if sameColor:
        plot_dist.idx -= 1
    color = 'C%d' % plot_dist.idx
    x = (np.arange(data['iterations']) + 1) * N

    if mean:
        mid = data['mean']
        low, high = mid - data['std'], mid + data['std']
    else:
        mid = data['median']
        low, high = data['quartil1'], data['quartil3']

    #ax1.errorbar(x, mid, label=label, yerr=[mid - low, high - mid], color=color, linewidth=1)
    ax1.plot(x, mid, label=label, color=color, linewidth=1)
    ax1.fill_between(x, low, high, color=color, alpha=0.2, interpolate=True)

    ax2.plot(x, data['goal_reached'], label=label, color=color, linewidth=1)
    plot_dist.idx += 1


def reset_plot_color():
    plot_dist.idx = 0
