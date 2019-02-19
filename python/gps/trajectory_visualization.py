""" This file is used to evaluate Policy Sample Data and print the result as csv."""

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

fig, (ax1, ax2) = plt.subplots(1,2)

def euclidean_distance(vec1, vec2):
    distance = 0
    for i in range(len(vec1)):
        distance += (vec1[i] - vec2[i]) ** 2
    distance = math.sqrt(distance)

    return distance

def plot_trajectory(tx,ty,tz):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(tx, ty, tz, label='parametric curve')
    ax.legend()

def main():
    """
    list with policy sample files (N lqr, N badmm, N gcm)
    load eef trajectories poses
    calc variance
    add them all to a plot
    :return:
    """
    from gps import __file__ as gps_filepath
    BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
    print("base dir: ", BASE_DIR)

    data_logger = DataLogger()
    n_traj = 50

    #lqr samples:
    cond = 0

    #lqr_sample_lists = []
    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../2cond_trajectories/pol_lqr_sample_itr_09.pkl')
    dT = len(lqr_sample_lists[0].get_X()[0])
    lqr_t_x1 = [lqr_sample_lists[0].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y1 = [lqr_sample_lists[0].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z1 = [lqr_sample_lists[0].get_EEF_Position()[0][i][2] for i in range(dT)]

    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../2cond_trajectories/pol_lqr_sample_itr_09.pkl')
    lqr_t_x2 = [lqr_sample_lists[1].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y2 = [lqr_sample_lists[1].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z2 = [lqr_sample_lists[1].get_EEF_Position()[0][i][2] for i in range(dT)]
    """
    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_11.pkl')
    lqr_t_x3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]


    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_12.pkl')
    lqr_t_x4 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y4 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z4 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]


    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_13.pkl')
    lqr_t_x5 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y5 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z5 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]


    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_14.pkl')
    lqr_t_x6 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y6 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z6 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]


    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_12.pkl')
    lqr_t_x2 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y2 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z2 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    lqr_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_lqr_sample_itr_05.pkl')
    lqr_t_x3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    lqr_t_y3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    lqr_t_z3 = [lqr_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    """

    #gcm samples:
    cond = 0
    # gcm_sample_lists = []
    t_x = [None] * n_traj
    t_y = [None] * n_traj
    t_z = [None] * n_traj
    #gcm_sample_lists = data_logger.unpickle('/home/philipp/internal_git/gps_refactored/python/trajectories/pol_mdggcs_sample_itr_08.pkl')
    #print(type(gcm_sample_lists))
    for idx in range(n_traj):
        print(BASE_DIR + ('/../trajectories/%02d_pol_mdggcs_sample_itr_08.pkl' % idx))
        gcm_sample_lists = data_logger.unpickle(BASE_DIR + ('/../2cond_trajectories/%02d_pol_mdgps_sample_itr_08.pkl' % idx))
        #gcm_sample_lists = data_logger.unpickle(BASE_DIR + ('/../trajectories/0_pol_mdggcs_sample_itr_08.pkl'))
        dT = len(gcm_sample_lists[cond].get_X()[0])
        t_x[idx] = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
        t_y[idx] = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
        t_z[idx] = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    print("shape: ", len(t_x[0]))


    """
    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/1_pol_gcm_sample_itr_12.pkl')
    gcm_t_x2 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y2 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z2 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/1_pol_gcm_sample_itr_05.pkl')
    gcm_t_x3 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y3 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z3 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_gcm_sample_itr_13.pkl')
    gcm_t_x4 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y4 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z4 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_gcm_sample_itr_11.pkl')
    gcm_t_x5 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y5 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z5 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_gcm_sample_itr_10.pkl')
    gcm_t_x6 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y6 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z6 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    gcm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_gcm_sample_itr_09.pkl')
    gcm_t_x7 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    gcm_t_y7 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    gcm_t_z7 = [gcm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    """


    # badmm samples:
    cond = 0
    """
    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_14.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x1 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y1 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z1 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_09.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x2 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y2 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z2 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_10.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x3 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y3 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z3 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_11.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x4 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y4 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z4 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_12.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x5 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y5 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z5 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

    badmm_sample_lists = data_logger.unpickle(BASE_DIR + '/../trajectories/pol_badmm_sample_itr_13.pkl')
    dT = len(badmm_sample_lists[cond].get_X()[0])
    badmm_t_x6 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
    badmm_t_y6 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
    badmm_t_z6 = [badmm_sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]
    """
    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
    plt.rcParams.update(pgf_with_rc_fonts)
    plt.rcParams.update({'font.size': 12})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    T = len(t_x[0])-1

    # plot it!
    fig, ax1 = plt.subplots(1, 1)
    ax1 = fig.gca(projection='3d')
    ax1.plot(t_x[0], t_y[0], t_z[0], lw=1, label=r'MDGPS', color='green', marker='o', markerfacecolor='green', markersize=1, alpha=1. / 5)
    ax1.scatter(t_x[0][T], t_y[0][T], t_z[0][T], label=r'endpoint MDGPS', color='magenta', marker='^')

    for i in range(n_traj-1):
        ax1.plot(t_x[i], t_y[i], t_z[i], lw=1, color='green', marker='o', markerfacecolor='green',
                 markersize=1, alpha=1. / 5)
        ax1.scatter(t_x[i][T], t_y[i][T], t_z[i][T], color='magenta', marker='^')

    ax1.plot(lqr_t_x1, lqr_t_y1, lqr_t_z1, lw=2, label=r'mean traj. A', color='blue', marker='o', markerfacecolor='blue',
             markersize=1, alpha=1)
    ax1.plot(lqr_t_x2, lqr_t_y2, lqr_t_z2, lw=2, label=r'mean traj. B', color='red', marker='o', markerfacecolor='red', markersize=1,
             alpha=0.7)

    """
    ax1.plot(lqr_t_x1, lqr_t_y1, lqr_t_z1, lw=1, label=r'iLQR', color='grey', marker='o', markerfacecolor='grey', markersize=3, alpha=0.5)
    ax1.plot(lqr_t_x2, lqr_t_y2, lqr_t_z2, lw=1, color='grey', marker='o', markerfacecolor='grey', markersize=3, alpha=0.5)
    ax1.plot(lqr_t_x3, lqr_t_y3, lqr_t_z3, lw=1, color='grey', marker='o', markerfacecolor='grey',
             markersize=3, alpha=0.5)
    ax1.plot(lqr_t_x4, lqr_t_y4, lqr_t_z4, lw=1, color='grey', marker='o', markerfacecolor='grey', markersize=3, alpha=0.5)
    ax1.plot(lqr_t_x5, lqr_t_y5, lqr_t_z5, lw=1, color='grey', marker='o', markerfacecolor='grey', markersize=3, alpha=0.5)
    ax1.plot(lqr_t_x6, lqr_t_y6, lqr_t_z6, lw=1, color='grey', marker='o', markerfacecolor='grey', markersize=3, alpha=0.5)


    ax1.plot(gcm_t_x2, gcm_t_y2, gcm_t_z2, lw=1, label=r'GCM', color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)
    ax1.plot(gcm_t_x3, gcm_t_y3, gcm_t_z3, lw=1, color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)
    ax1.plot(gcm_t_x4, gcm_t_y4, gcm_t_z4, lw=1, color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)
    ax1.plot(gcm_t_x5, gcm_t_y5, gcm_t_z5, lw=1, color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)
    ax1.plot(gcm_t_x6, gcm_t_y6, gcm_t_z6, lw=1, color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)
    ax1.plot(gcm_t_x7, gcm_t_y7, gcm_t_z7, lw=1, color='blue', marker='o', markerfacecolor='blue', markersize=3, alpha=0.5)

    ax1.plot(badmm_t_x1, badmm_t_y1, badmm_t_z1, lw=1, label=r'GPS', color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    ax1.plot(badmm_t_x2, badmm_t_y2, badmm_t_z2, lw=1, color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    ax1.plot(badmm_t_x3, badmm_t_y3, badmm_t_z3, lw=1, color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    ax1.plot(badmm_t_x4, badmm_t_y4, badmm_t_z4, lw=1, color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    ax1.plot(badmm_t_x5, badmm_t_y5, badmm_t_z5, lw=1, color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    ax1.plot(badmm_t_x6, badmm_t_y6, badmm_t_z6, lw=1, color='red', marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    """


    #ax1.set_title('Robustness Reaching Task')
    ax1.legend(loc='upper center')
    ax1.set_xlabel(r'x')
    ax1.set_ylabel(r'y')
    ax1.set_zlabel(r'z')
    ax1.grid()

    plt.show()


if __name__ == "__main__":
    main()
