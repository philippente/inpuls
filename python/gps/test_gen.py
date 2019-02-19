""" This file is used to test an LQR implementation """

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

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList


class LQRTestMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        """
        Initialize LQRTestMain
        Args:
            config: Test hyperparameters for experiment
        """
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """

        for itr in range(0, self._hyperparams['iterations']):
            for cond in self._train_idx:
                for i in range(self._hyperparams['num_samples']):
                    self._take_sample(itr, cond, i)

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]
            self._take_iteration(traj_sample_lists)
            #self._log_data(itr, traj_sample_lists, pol_sample_lists)
            if (itr == 3):
                """
                self.data_logger.pickle(
                    self._data_files_dir + 'test_traj_distr.pkl',
                    copy.copy(self.algorithm.prev[0].traj_distr)
                )
                self.data_logger.pickle(
                    self._data_files_dir + 'test_traj_info.pkl',
                    copy.copy(self.algorithm.prev[0].traj_info)
                )
                self.data_logger.pickle(
                    self._data_files_dir + 'test_new_traj_distr.pkl',
                    copy.copy(self.algorithm.prev[0].new_traj_distr)
                )
                self.data_logger.pickle(
                    self._data_files_dir + 'test_final_eta.pkl',
                    copy.copy(self.algorithm.prev[0].eta)
                )
                mu_and_sigma = self.algorithm.forward(self.algorithm.prev[0].new_traj_distr,
                                                      self.algorithm.prev[0].traj_info)
                self.data_logger.pickle(
                    self._data_files_dir + 'test_mu_and_sigma.pkl',
                    copy.copy(mu_and_sigma)
                )
                """
                self.data_logger.pickle(
                    self._data_files_dir + 'test_prior',
                    copy.copy(self.algorithm.prev[0].traj_info.dynamics.get_prior())
                )
                self.data_logger.pickle(
                    self._data_files_dir + 'test_sample_list',
                    copy.copy(self.algorithm.prev[0].sample_list)
                )
                dynamics_data = self.algorithm.prev[0].traj_info.dynamics.Fm, self.algorithm.prev[0].traj_info.dynamics.fv, self.algorithm.prev[0].traj_info.dynamics.dyn_covar
                self.data_logger.pickle(
                    self._data_files_dir + 'test_dynamics',
                    copy.copy(dynamics_data)
                )

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] \
                and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy
        else:
            pol = self.algorithm.cur[cond].traj_distr
        self.agent.sample(
            pol, cond,
            verbose=(i < self._hyperparams['verbose_trials'])
        )

    def _take_iteration(self, sample_lists):
        """
        Take an iteration of the algorithm.
        """
        self.algorithm.iteration(sample_lists)

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

def main():
    """ Main function to be run. """

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/box2d_arm_example/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nCheck for '%s'." %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    import random
    import numpy as np
    import matplotlib.pyplot as plt

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    lqrtest = LQRTestMain(hyperparams.config)
    lqrtest.run()


if __name__ == "__main__":
    main()
