""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from os import mkdir
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.openai_gym.agent_openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_rfgps import AlgorithmRFGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.agent.openai_gym.init_policy import init_gym_pol
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM

SENSOR_DIMS = {
    'qpos': 22,
    'qvel': 23,
    'cinert': 140, 
    'cvel': 84,
    'qfrc_actuator': 23,
    'cfrc_ext': 84,
    ACTION: 17,
}

BASE_DIR = '/'.join(str.split(gps_filepath.replace('\\', '/'), '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/gym_standup_rfgps/'


common = {
    'experiment_name': 'gym_standup_rfgps' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentOpenAIGym,
    'render': False,
    'T': 50,
    'random_reset': True,
    'x0': np.zeros(25),
    'dt': 1.0/25,
    'env': 'HumanoidStandup-v2',
    'sensor_dims': SENSOR_DIMS,
    'target_state': np.zeros(3), # Target
    'conditions': common['conditions'],
    'state_include': ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_actuator', 'cfrc_ext'],
    'obs_include': ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_actuator', 'cfrc_ext'],
}

algorithm = {
    'type': AlgorithmRFGPS,
    'conditions': common['conditions'],
    'iterations': 100,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
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
        'qpos': {
            'wp': np.concatenate([[1.0], np.zeros(SENSOR_DIMS['qpos']-1)]) * 1.0/0.003,
            'target_state': np.concatenate([[2.0], np.zeros(SENSOR_DIMS['qpos']-1)]),
        },
        'cfrc_ext': {
            'wp': np.ones(SENSOR_DIMS['cfrc_ext']) * 0.5e-6, 
            'target_state': np.zeros(SENSOR_DIMS['cfrc_ext']),
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [0.1, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 60,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
        'strength': 1,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': agent['obs_include'],
        'sensor_dims': SENSOR_DIMS,
    },
    'save_path': common['data_files_dir'],
    'network_model': example_tf_network,
    'iterations': 1500,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'num_pol_samples': 20,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)

algorithm['tac'] = {
    'clusters': 4,
    'min_samples_per_cluster': 3,
    'initial_clustering': 'random',
    'random_resets': 0,
    'T': 50,
    'max_em_iterations': 20,
    'policy_linearization': 'GPS',
    'dyn_covar_smooth': 11,
    'prior_only': False,
    'tac_policy': False,
    'tac_policy_history': 10,
}

if algorithm['tac']['random_resets'] == 0 and algorithm['tac']['max_em_iterations'] == 0 and algorithm['tac']['initial_clustering'] == 'random':
    param_str = 'box2d_mdgps'
    baseline = True
else:
    param_str = 'box2d_rfgps'
    baseline = False
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-%ds-%dc-%dr'%(config['num_samples'], algorithm['tac']['clusters'], algorithm['tac']['random_resets'])
param_str += '-T%d'%agent['T'] if agent['T'] == algorithm['tac']['T'] else '-T%d_%d'%(agent['T'], algorithm['tac']['T'])
if not baseline:
    param_str += '-dynG%d'%(algorithm['tac']['dyn_covar_smooth']) if algorithm['tac']['prior_only'] else '-dynLG%d'%(algorithm['tac']['dyn_covar_smooth'])
param_str += '-tac_pol%d'%(algorithm['tac']['tac_policy_history']) if algorithm['tac']['tac_policy'] else '-gps_pol'
#common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])
#mkdir(common['data_files_dir'])
