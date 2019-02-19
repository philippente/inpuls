""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from os import mkdir
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

from gps import __file__ as gps_filepath
from gps.agent.openai_gym.agent_openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
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
import gps.envs

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    'diff': 6,
    ACTION: 7,
}

BASE_DIR = '/'.join(str.split(gps_filepath.replace('\\', '/'), '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/gym_peg_mdgps/'


common = {
    'experiment_name': 'gym_peg_mdgps' + '_' + \
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


def additional_sensors(sim, sample, t):
    from gps.proto.gps_pb2 import END_EFFECTOR_POINT_JACOBIANS
    jac = np.empty((6, 7))
    jac[:3] = sim.data.get_site_jacp('leg_bottom').reshape((3, -1))
    jac[3:] = sim.data.get_site_jacp('leg_top').reshape((3, -1))
    sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t)


scaler = StandardScaler()
scaler.mean_ = [
    6.269170526989192860e-01, 4.821029797409001616e-01, -1.544978642130366175e+00, -1.176340602904814014e+00,
    5.961119337564275977e-03, -1.430743556156053087e+00, 9.207545508118145094e-02, -4.912633791830883865e-04,
    4.920776339259459904e-03, -5.227700198026216660e-02, -1.211733843546966144e-01, 8.114241896172087742e-02,
    -1.262334659084702604e-01, 3.840017703328454934e-02, 2.194164582447949707e-01, 3.117020534855194369e-01,
    -2.769765942447273699e-01, 2.246371605649680747e-01, 2.734939294257240916e-01, -1.879409311259450099e-01,
    -2.194164582447949707e-01, -1.170205348552046803e-02, -2.230234057552664406e-01, -2.246371605649680747e-01,
    2.650607057427677160e-02, -1.205906887405655374e-02
]
scaler.scale_ = [
    1.175474491141771521e-01, 1.418192829274926015e-01, 7.290010554458773440e-01, 4.690964360306257297e-01,
    1.803339760782196821e+00, 5.855882638525999884e-01, 2.440150963082939661e+00, 7.986024203846138481e-02,
    8.789967075190831258e-02, 7.810188187563219531e-01, 4.007383765581046253e-01, 1.621819389085616736e+00,
    1.115074309447028122e+00, 1.874669395375669012e+00, 2.366287313117650670e-01, 1.671869278950406934e-01,
    1.699603332509740106e-01, 2.305784025129126447e-01, 1.472527037676900352e-01, 1.478939140218039627e-01,
    2.366287313117650670e-01, 1.671869278950406934e-01, 1.699603332509738718e-01, 2.305784025129126447e-01,
    1.472527037676900352e-01, 1.478939140218039905e-01
]

agent = {
    'type': AgentOpenAIGym,
    'render': False,
    'T': 100,
    'random_reset': False,
    'x0': [131, 327, 356, 491, 529, 853, 921, 937],  # Random seeds for each initial condition, 1
    'dt': 1.0/25,
    'env': 'PegInsertion-v0',
    'conditions': common['conditions'],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
    # 'scaler': scaler,
    'additional_sensors': additional_sensors,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 50,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
    'sample_on_policy': False,
    #'tac_policy': {
    #    'history': 10,
    #},
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'init_var_scale': 0.1,
    'env': agent['env'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION])
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, fk_cost],
    'weights': [1e-3, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 5*common['conditions'],
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
    'iterations': 3000,
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
    'num_samples': 5,
    'num_pol_samples': 10,
    'save_samples': False,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)

param_str = 'peg_gps'
baseline = True
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-M%d' % config['common']['conditions']
param_str += '-%ds' % config['num_samples']
param_str += '-T%d' % agent['T']
param_str += '-K%d' % algorithm['dynamics']['prior']['max_clusters']
param_str += '-tac_pol' if 'tac_policy' in algorithm else '-lqr_pol' if not algorithm['sample_on_policy'] else '-gps_pol'
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])
mkdir(common['data_files_dir'])
