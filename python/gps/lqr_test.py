""" This file is used to test an LQR implementation """

import unittest

import os
import os.path
import sys
import imp
import numpy as np
import random

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger
from gps.proto.gps_pb2 import ACTION


class DummyAgent(object):

    def __init__(self, config):
        self.T = config['T']

        self.dU = config['sensor_dims'][ACTION]

        i = 0
        for sensor in config['state_include']:
            i += config['sensor_dims'][sensor]
        self.dX = i

        i = 0
        for sensor in config['obs_include']:
            i =+ config['sensor_dims'][sensor]
        self.dO = i

        self.x0 = config['x0']


class TestLQR(unittest.TestCase):

    def setUp(self):
        from gps import __file__ as gps_filepath
        gps_filepath = '/'.join(str.split(gps_filepath, '/')[:-1])
        gps_filepath = os.path.abspath(gps_filepath)
        hyperparams_file = gps_filepath + '/test_files/hyperparams.py'
        hyperparams = imp.load_source('hyperparams', hyperparams_file)
        config = hyperparams.config

        seed = config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        config['algorithm']['agent'] = DummyAgent(config['agent'])
        self.algorithm = config['algorithm']['type'](config['algorithm'])

        data_logger = DataLogger()
        self.traj_distr = data_logger.unpickle(
            gps_filepath + '/test_files/traj_distr'
        )
        self.traj_info = data_logger.unpickle(
            gps_filepath + '/test_files/traj_info'
        )
        self.new_traj_distr = data_logger.unpickle(
            gps_filepath + '/test_files/new_traj_distr'
        )
        self.final_eta = data_logger.unpickle(
            gps_filepath + '/test_files/final_eta'
        )
        self.mu, self.sigma = data_logger.unpickle(
            gps_filepath + '/test_files/mu_and_sigma'
        )

        self.algorithm.cur[0].traj_distr = self.traj_distr
        self.algorithm.cur[0].traj_info = self.traj_info

    def test_backward(self):
        traj_distr_to_test = self.algorithm.backward(self.traj_distr,
                                                     self.traj_info,
                                                     self.final_eta)
        self.assertTrue(
            np.array_equal(traj_distr_to_test.K, self.new_traj_distr.K)
        )
        self.assertTrue(
            np.array_equal(traj_distr_to_test.k, self.new_traj_distr.k)
        )
        self.assertTrue(
            np.array_equal(traj_distr_to_test.pol_covar,
                           self.new_traj_distr.pol_covar)
        )
        self.assertTrue(
            np.array_equal(traj_distr_to_test.chol_pol_covar,
                           self.new_traj_distr.chol_pol_covar)
        )
        self.assertTrue(
            np.array_equal(traj_distr_to_test.inv_pol_covar,
                           self.new_traj_distr.inv_pol_covar)
        )

    def test_forward(self):
        mu_to_test, sigma_to_test = self.algorithm.forward(self.new_traj_distr,
                                                           self.traj_info)
        self.assertTrue(
            np.array_equal(mu_to_test, self.mu)
        )
        self.assertTrue(
            np.array_equal(sigma_to_test, self.sigma)
        )

if __name__ == '__main__':
    unittest.main(verbosity = 2)
