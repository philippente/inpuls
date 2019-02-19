""" Hyperparameters for OPC UA experiment. """
from __future__ import division

from datetime import datetime
import os.path
import time
from enum import Enum

import numpy as np
from sklearn.preprocessing import StandardScaler

from gps import __file__ as gps_filepath
from gps.agent.opcua.agent_opcua import AgentOPCUA
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import evallogl2term
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.agent.opcua.init_policy import init_pol
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM


class Sensors(Enum):
    Area3_MotorFCRealCurrent = '3:AZO_KC_Modular_READ,3:Area3_MotorFCRealCurrent'
    Area3_MotorFCRealFrequency = '3:AZO_KC_Modular_READ,3:Area3_MotorFCRealFrequency'
    BlowerArea_AirSpeed_Pump = '3:AZO_KC_Modular_READ,3:BlowerArea_AirSpeed_Pump'
    BlowerArea_Air_Pump = '3:AZO_KC_Modular_READ,3:BlowerArea_Air_Pump'
    BlowerArea_CurrentBlowerHybrid = '3:AZO_KC_Modular_READ,3:BlowerArea_CurrentBlowerHybrid'
    BlowerArea_Pressure = '3:AZO_KC_Modular_READ,3:BlowerArea_Pressure'
    BlowerArea_Pressure_Dynamic = '3:AZO_KC_Modular_READ,3:BlowerArea_Pressure_Dynamic'
    ML_AIR_PUMP = '3:AZO_KC_Modular_READ,3:ML_AIR_PUMP'
    ML_Airspeed_Pump = '3:AZO_KC_Modular_READ,3:ML_Airspeed_Pump'
    MainArea_Humidity = '3:AZO_KC_Modular_READ,3:MainArea_Humidity'
    #MainArea_MobileAnalogInput2 = '3:AZO_KC_Modular_READ,3:MainArea_MobileAnalogInput2'
    MainArea_MobileAnalogInput3 = '3:AZO_KC_Modular_READ,3:MainArea_MobileAnalogInput3'
    MainArea_Pressure = '3:AZO_KC_Modular_READ,3:MainArea_Pressure'
    MainArea_Temperature = '3:AZO_KC_Modular_READ,3:MainArea_Temperature'
    Scale1_ConveyingPerformance = '3:AZO_KC_Modular_READ,3:Scale1_ConveyingPerformance'
    Scale1_DifferencePressureSensor = '3:AZO_KC_Modular_READ,3:Scale1_DifferencePressureSensor'
    # Scale1_GrossWeight = '3:AZO_KC_Modular_READ,3:Scale1_GrossWeight'
    Scale1_Netweight = '3:AZO_KC_Modular_READ,3:Scale1_Netweight'
    # Scale2_GrossWeight = '3:AZO_KC_Modular_READ,3:Scale2_GrossWeight'
    #Scale2_NetWeight = '3:AZO_KC_Modular_READ,3:Scale2_NetWeight'
    XArea10_PressureTransmitters_Pipe_Source = '3:AZO_KC_Modular_READ,3:XArea10_PressureTransmitters_Pipe_Source'
    XArea10_PressureTransmitters_Pipe_5Meter = '5:AZO_KC_SPMAIR_READ,5:XArea10_PressureTransmitters_Pipe_5Meter'
    XArea10_PressureTransmitters_Pipe_15Meter = '5:AZO_KC_SPMAIR_READ,5:XArea10_PressureTransmitters_Pipe_15Meter'
    XArea10_PressureTransmitters_Pipe_25Meter = '5:AZO_KC_SPMAIR_READ,5:XArea10_PressureTransmitters_Pipe_25Meter'
    XArea10_PressureTransmitters_Pipe_35Meter = '5:AZO_KC_SPMAIR_READ,5:XArea10_PressureTransmitters_Pipe_35Meter'
    XArea10_PressureTransmitters_Pipe_Destination = '5:AZO_KC_SPMAIR_READ,5:XArea10_PressureTransmitters_Pipe_Destination'


class Actuators(Enum):
    ML_New_Frequency_Blower = '4:AZO_KC_Modular_WRITE,4:ML_New_Frequency_Blower'
    ML_New_Frequency_DosingEquipment = '4:AZO_KC_Modular_WRITE,4:ML_New_Frequency_DosingEquipment'


class Styles(Enum):
    LAENGE_LEITUNG = 0


SENSOR_DIMS = {
    sensor: 1
    for sensor in (list(Actuators) + list(Sensors))
}  # Every sensor and actuator is assumed to be one dimensional

scaler = StandardScaler()
scaler.mean_ = np.zeros(len(Sensors))  # TODO
scaler.scale_ = np.ones(len(Sensors))  # TODO

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/opcua/'

common = {
    'experiment_name': 'my_experiment' + '_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'cost_log_dir': EXP_DIR + 'cost_log/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    #'train_conditions': [0, 1],
    #'test_conditions': [2, 3],
    'conditions': 1,
    'experiment_ID': '1' + time.ctime(),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type':
        AgentOPCUA,
    'random_reset':
        False,
    'dt':
        2.0,
    'conditions':
        common['conditions'],
    'T':
        45,
    'sensor_dims':
        SENSOR_DIMS,
    'state_include':
        list(Sensors),
    'obs_include': [],
    'actions_include': [Actuators.ML_New_Frequency_DosingEquipment, Actuators.ML_New_Frequency_Blower],
    'opc-ua_server':
        'opc.tcp://inpuls:inpuls@137.226.189.137:4980/DataFeed_UA_InPuLS_Config',
    'scaler':
        scaler,
    'override_actuator':
        [
            {
                'actuator': Actuators.ML_New_Frequency_DosingEquipment,
                'condition': lambda T: T >= 35,
                'value': [0],
            }, {
                'actuator': Actuators.ML_New_Frequency_Blower,
                'condition': lambda T: T >= 44,
                'value': [0],
            }
        ],
    'override_sensor': [{
        'sensor': Sensors.Scale1_Netweight,
        'condition': lambda T: T <= 12,
        'value': [0.0],
    }],
    'send_signal':
        [
            {
                'signal': '4:AZO_KC_Modular_WRITE,4:ML_CYCLE_START',
                'condition': lambda T: T == 0,
            }, {
                'signal': '4:AZO_KC_Modular_WRITE,4:ML_CYCLE_STOP',
                'condition': lambda T: T == 35,
            }, {
                'signal': '4:AZO_KC_Modular_WRITE,4:ML_CYCLE_RESET',
                'condition': lambda T: T == 45,
            }
        ],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_pol,
    'init_const': np.asarray([68.0, 25.0]),
    'init_var': np.diag([25.0, 5.0]),
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': [1.0, 1.0],  # TODO 5e-3 / PR2_GAINS,
}


def blowerArea_pressure_scale(sample):
    return 1.0 / (np.abs(sample.get(Sensors.MainArea_MobileAnalogInput3)) + 0.5)


state_cost = {
    'type': CostState,
    'data_types':
        {
            Sensors.BlowerArea_Pressure:
                {
                    'wp': [6.0],
                    'target_state': [-700.0 / 2000],
                    'scale': blowerArea_pressure_scale,
                },
        },
}

state_cost3 = {
    'type': CostState,
    #'ramp_option': RAMP_LINEAR,
    'data_types': {
        Sensors.Scale1_Netweight: {
            'wp': [2.0],
            'target_state': [1.2],
        },
    },
    'evalnorm': evallogl2term,
    'l1': 1.0,
    'l2': 1.0,
}


def netweight_scale(sample):
    return 1 + np.square(np.asarray([1.0 / 8]) / (np.abs(sample.get(Sensors.ML_Airspeed_Pump)) + 1.0 / 16))


state_cost2 = {
    'type': CostState,
    'data_types': {
        Sensors.Scale1_Netweight: {
            'wp': [0.5],
            'target_state': [1.0],
            'scale': netweight_scale,
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost, state_cost2, state_cost3],
    'weights': [1e-3, 1.0, 1.0, 1.0],
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

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 3,
    'num_lqr_samples_static': 1,
    'num_lqr_samples_random': 0,
    'num_pol_samples_static': 1,
    'num_pol_samples_random': 0,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = (
    'exp_name:   ' + str(common['experiment_name']) + '\n'
    'alg_type:   ' + str(algorithm['type'].__name__) + '\n'
    'alg_dyn:    ' + str(algorithm['dynamics']['type'].__name__) + '\n'
    'alg_cost:   ' + str(algorithm['cost']['type'].__name__) + '\n'
    'iterations: ' + str(config['iterations']) + '\n'
    'conditions: ' + str(algorithm['conditions']) + '\n'
    'samples:    ' + str(config['num_samples']) + '\n'
)
