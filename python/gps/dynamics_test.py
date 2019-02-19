""" This file is used to test a dynamics implementation """

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


class TestDynamics(unittest.TestCase):

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
        cur_data = data_logger.unpickle(
            gps_filepath + '/test_files/sample_list'
        )
        self.X = cur_data.get_X()
        self.U = cur_data.get_U()
        prior = data_logger.unpickle(
            gps_filepath + '/test_files/prior'
        )
        self.algorithm.cur[0].traj_info.dynamics.prior = prior
        self.Fm, self.fv, self.dyn_covar = data_logger.unpickle(
            gps_filepath + '/test_files/dynamics_data'
        )

    def test_fit(self):
        self.algorithm.cur[0].traj_info.dynamics.fit(self.X, self.U)
        self.assertTrue(
            np.array_equal(self.algorithm.cur[0].traj_info.dynamics.Fm,
                           self.Fm)
        )
        self.assertTrue(
            np.array_equal(self.algorithm.cur[0].traj_info.dynamics.fv,
                           self.fv)
        )
        self.assertTrue(
            np.array_equal(self.algorithm.cur[0].traj_info.dynamics.dyn_covar,
                           self.dyn_covar)
        )

if __name__ == '__main__':
    unittest.main(verbosity = 2)
