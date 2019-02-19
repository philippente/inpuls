""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from os import mkdir
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

from gps import __file__ as gps_filepath
from gps.agent.openai_gym.agent_openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.agent.openai_gym.init_policy import init_gym_pol
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
from gps.algorithm.policy_opt.tf_model_example import example_tf_network

SENSOR_DIMS = {
    'qpos': 13,
    'qvel': 14,
    'cfrc_ext': 6*14,
    ACTION: 8,
}

BASE_DIR = '/'.join(str.split(gps_filepath.replace('\\', '/'), '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/gym_ant_lqr/'


common = {
    'experiment_name': 'gym_ant_lqr' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    # 'train_conditions': [0],
    # 'test_conditions': [0, 1, 2, 3],
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

scaler = StandardScaler()
scaler.mean_ = []
scaler.scale_ = []

agent = {
    'type': AgentOpenAIGym,
    'render': True,
    'T': 200,
    'random_reset': False,
    'x0': [0, 1, 2, 3, 4, 5, 6, 7], # Random seeds for each initial condition
    'dt': 1.0/25,
    'env': 'Ant-v2',
    'sensor_dims': SENSOR_DIMS,
    'target_state': np.zeros(3),  # Target np.zeros(3), 
    'conditions': common['conditions'],
    'state_include': ['qpos', 'qvel'],
    'obs_include': ['qpos', 'qvel'],
#    'scaler': scaler,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
    #'tac_policy': {
    #    'history': 10,
    #},
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.1,
    'env': agent['env'],
    'dt': agent['dt'],
    'stiffness': 0.01,
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        'qvel': {
            'wp': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'target_state': [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1E-3, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 60,
        'min_samples_per_cluster': 40,
        'max_samples': 80,
        'strength': 1,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'num_pol_samples': 5,
    'save_samples': False,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)

param_str = 'fetchreach_lqr'
baseline = True
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-M%d' % config['common']['conditions']
param_str += '-%ds' % config['num_samples']
param_str += '-T%d' % agent['T']
param_str += '-tac_pol' if 'tac_policy' in algorithm else '-lqr_pol'
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])
#mkdir(common['data_files_dir'])
