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
from mpl_toolkits.mplot3d import Axes3D

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

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
    parser = argparse.ArgumentParser(description='Evaluate Policy Sample Data.')
    parser.add_argument('-lqr', '--lqr_experiment_folder', type=str, help='Name of lqr experiment folder')
    parser.add_argument('-gcm', '--gcm_experiment_folder', type=str, help='Name of gcm experiment folder')
    parser.add_argument('-badmm', '--badmm_experiment_folder', type=str, help='Name of badmm experiment folder')
    parser.add_argument('-i', '--iteration', type=int, help='Iteration to evaluate')
    parser.add_argument('-istart', '--startiteration', type=int, help='Start number of iterations')
    parser.add_argument('-iend', '--enditeration', type=int, help='Start number of iterations')
    args = parser.parse_args()

    from gps import __file__ as gps_filepath
    BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])

    cond = 0

    policies = []
    data_files_dirs = []
    target_states = []

    if args.lqr_experiment_folder is not None:
        policies.append('lqr')
        data_files_dirs.append(BASE_DIR + '/../experiments/' + args.lqr_experiment_folder + '/data_files/')
        lqr_target_filename = BASE_DIR + '/../experiments/' + args.lqr_experiment_folder + '/target.npz'
        lqr_target_pos, _, _ = load_pose_from_npz(lqr_target_filename, 'trial_arm', str(cond), 'target')
        lqr_target_vel = np.zeros(6)
        lqr_target_state = np.zeros(12)
        lqr_target_state[:6] = lqr_target_pos
        lqr_target_state[6:12] = lqr_target_vel
        target_states.append(lqr_target_state)

    if args.gcm_experiment_folder is not None:
        policies.append('mdggcs')
        data_files_dirs.append(BASE_DIR + '/../experiments/' + args.gcm_experiment_folder + '/data_files/')
        gcm_target_filename = BASE_DIR + '/../experiments/' + args.gcm_experiment_folder + '/target.npz'
        gcm_target_pos, _, _ = load_pose_from_npz(gcm_target_filename, 'trial_arm', str(cond), 'target')
        gcm_target_vel = np.zeros(6)
        gcm_target_state = np.zeros(12)
        gcm_target_state[:6] = gcm_target_pos
        gcm_target_state[6:12] = gcm_target_vel
        target_states.append(gcm_target_state)

    if args.badmm_experiment_folder is not None:
        policies.append('mdgps')
        data_files_dirs.append(BASE_DIR + '/../experiments/' + args.badmm_experiment_folder + '/data_files/')
        badmm_target_filename = BASE_DIR + '/../experiments/' + args.badmm_experiment_folder + '/target.npz'
        badmm_target_pos, _, _ = load_pose_from_npz(badmm_target_filename, 'trial_arm', str(cond), 'target')
        badmm_target_vel = np.zeros(6)
        badmm_target_state = np.zeros(12)
        badmm_target_state[:6] = badmm_target_pos
        badmm_target_state[6:12] = badmm_target_vel
        target_states.append(badmm_target_state)

    data_logger = DataLogger()

    description_line = ''
    for pol in policies:
        description_line += ', ' + pol + '_min_distance, ' + pol + '_last_distance, ' + pol + '_summed_distance'
    print(description_line[2:])

    if args.iteration is not None:
        itr_idx = [args.iteration]
    else:
        itr_idx = range(args.startiteration, args.enditeration)

    for itr in itr_idx:
        line = ''
        for p in range(len(policies)):

            sample_lists = data_logger.unpickle(data_files_dirs[p] + 'pol_' + policies[p] + ('_sample_itr_%02d.pkl' % itr))

            dT = len(sample_lists[cond].get_X()[0])

            distances = [euclidean_distance(sample_lists[cond].get_X()[0][i], target_states[p])
                         for i in range(dT)]

            min_distance = min(distances)
            last_distance = distances[-1]
            summed_distance = sum(distances)

            #print("len: ",len(sample_lists[cond].get_EEF_Position()[0]))

            t_x = [sample_lists[cond].get_EEF_Position()[0][i][0] for i in range(dT)]
            t_y = [sample_lists[cond].get_EEF_Position()[0][i][1] for i in range(dT)]
            t_z = [sample_lists[cond].get_EEF_Position()[0][i][2] for i in range(dT)]

            #plot_trajectory(t_x,t_y,t_z)

            line += ', ' + str(min_distance) + ', ' + str(last_distance) + ', ' + str(summed_distance)

        print(line[2:])
    #plt.show()


if __name__ == "__main__":
    main()
