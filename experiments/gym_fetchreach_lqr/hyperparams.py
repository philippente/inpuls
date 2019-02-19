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
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    'observation': 10,  # FetchReach 10, Fetch 25
    END_EFFECTOR_POINTS: 3,  # 1*3, 15 hand
    ACTION: 4,  #4 , 20 Hand
}

BASE_DIR = '/'.join(str.split(gps_filepath.replace('\\', '/'), '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/gym_fetchreach_lqr/'


common = {
    'experiment_name': 'gym_fetchreach_lqr' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
    # 'train_conditions': [0],
    # 'test_conditions': [0, 1, 2, 3],
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

scaler = StandardScaler()
scaler.mean_ = [
    1.33554446e+00, 7.46866838e-01, 5.25169272e-01, 3.78054915e-06, 1.55438229e-06, -2.66267931e-04, -4.78114576e-05,
    -5.77911271e-05, 4.03657748e-04, 4.67200411e-04, 6.71023554e-03, -1.16670921e-02, -6.79674174e-03
]
scaler.scale_ = [
    1.07650035e-01, 1.29246324e-01, 9.07454956e-02, 2.64638440e-05, 1.08806760e-05, 7.48034005e-03, 7.69515138e-03,
    7.54787489e-03, 1.37551241e-03, 1.52642517e-03, 7.57966860e-02, 1.03377066e-01, 6.24320497e-02
]

agent = {
    'type': AgentOpenAIGym,
    'render': False,
    'T': 20,
    'random_reset': False,
    'x0': [0, 1, 2, 3, 4, 5, 6, 7],  # Random seeds for each initial condition
    'dt': 1.0 / 25,
    'env': 'FetchReach-v1',
    'sensor_dims': SENSOR_DIMS,
    'target_state': scaler.transform([np.zeros(13)])[0, -3:],  # Target np.zeros(3), 
    'conditions': common['conditions'],
    'state_include': ['observation', END_EFFECTOR_POINTS],
    'obs_include': ['observation', END_EFFECTOR_POINTS],
    'actions_include': [ACTION],
    'scaler': scaler,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'init_var_scale': 1.0,
    'env': agent['env'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION]),
}

state_cost = {
    'type': CostState,
    'data_types': {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(3),  # Target size
            'target_state': agent["target_state"],
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
    'prior':
        {
            'type': DynamicsPriorGMM,
            'max_clusters': 8,
            'min_samples_per_cluster': 40,
            'max_samples': 40,
            'strength': 1,
        },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'num_lqr_samples_static': 1,
    'num_lqr_samples_random': 5,
    'num_pol_samples_static': 1,
    'num_pol_samples_random': 20,
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
param_str += '-K%d' % algorithm['dynamics']['prior']['max_clusters']
param_str += '-tac_pol' if 'tac_policy' in algorithm else '-lqr_pol'
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])
mkdir(common['data_files_dir'])
