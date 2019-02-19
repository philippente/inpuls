""" Default configuration and hyperparameters for agent objects. """
import logging
import numpy as np


LOGGER = logging.getLogger(__name__)


# Agent
AGENT = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
}


try:
    import rospkg
    import roslib
    roslib.load_manifest('gps_agent_pkg')


    # AgentROSJACO
    AGENT_ROS_JACO = {
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_position_command',
        'relax_command_topic': 'gps_controller_relax_command',
        'data_request_topic': 'gps_controller_data_request',
        'sample_result_topic': 'gps_controller_report',
        'trial_timeout': 20,    # Give this many seconds for a trial.
        'reset_conditions': [], # Defines reset modes + positions for
                                # trial and auxiliary arms.
        'frequency': 20,
        'end_effector_points': np.array([]),
        'pid_params': np.array([
            120.0, 0.0, 20.0, 4.0,
            100.0, 0.0, 10.0, 4.0,
            70.0, 0.0, 4.0, 4.0,
            30.0, 0.0, 6.0, 2.0,
            30.0, 0.0, 4.0, 2.0,
            30.0, 0.0, 2.0, 2.0
        ]),
    }

except ImportError as e:
    AGENT_ROS_JACO = {}
    LOGGER.debug('No ROS enabled: %s', e)
except rospkg.common.ResourceNotFound as e:
    AGENT_ROS_JACO = {}
    LOGGER.debug('No gps_agent_pkg: %s', e)

AGENT_BOX2D = {
    'render': True,
}
