""" This file defines an agent for the Kinova Jaco2 ROS environment. """
import copy
import time
import numpy as np
from random import random
from math import pi


import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Timer
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ROS_JACO
from gps.agent.ros_jaco.ros_utils import TimeoutException, ServiceEmulator, msg_to_sample, \
        policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
from gps_agent_pkg.msg import TrialCommand, SampleResult, PositionCommand, \
        RelaxCommand, DataRequest, TfActionCommand, TfObsData, DataType
from gps.utility.perpetual_timer import PerpetualTimer

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
    from gps.algorithm.agmp.agmp_controller import AGMP_Controller
    from gps.algorithm.gcm.gcm_controller import GCMController
    from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
    from gps.proto.gps_pb2 import RGB_IMAGE, END_EFFECTOR_POINTS, END_EFFECTOR_POSITIONS, END_EFFECTOR_ROTATIONS, JOINT_ANGLES, JOINT_SPACE
except ImportError:  # user does not have tf installed.
    TfPolicy = None


class AgentROSJACO(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(AGENT_ROS_JACO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_ros_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        conditions = self._hyperparams['conditions']

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        self.x0 = self._hyperparams['x0']
        #self.x_tgt = self._hyperparams['exp_x_tgts']
        self.target_state = np.zeros(self.dX)
        self.dt = self._hyperparams['dt']

        self.gui = None

        self.condition = None
        self.policy = None
        r = rospy.Rate(1)
        r.sleep()

        self.stf_policy = None
        self.init_tf = False
        self.use_tf = False
        self.observations_stale = True
        self.noise = None
        self.cur_timestep = None
        self.vision_enabled = False
        img_width = self._hyperparams['rgb_shape'][0]
        img_height = self._hyperparams['rgb_shape'][1]
        img_channel = self._hyperparams['rgb_shape'][2]
        self.rgb_image = np.empty([img_height, img_width, img_channel])
        self.rgb_image_seq = np.empty([self.T, img_height, img_width, img_channel], dtype=np.uint8)

        self.sample_processing = False
        self.sample_save = False

        self._cv_bridge = CvBridge()

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(
            self._hyperparams['trial_command_topic'], TrialCommand,
        self._hyperparams['sample_result_topic'], SampleResult
        )
        self._reset_service = ServiceEmulator(
            self._hyperparams['reset_command_topic'], PositionCommand,
            self._hyperparams['sample_result_topic'], SampleResult
        )
        self._relax_service = ServiceEmulator(
            self._hyperparams['relax_command_topic'], RelaxCommand,
            self._hyperparams['sample_result_topic'], SampleResult
        )
        self._data_service = ServiceEmulator(
            self._hyperparams['data_request_topic'], DataRequest,
            self._hyperparams['sample_result_topic'], SampleResult
        )

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2 ** 30)
        return self._seq_id

    def get_data(self, arm=TRIAL_ARM):
        """
        Request for the most recent value for data/sensor readings.
        Returns entire sample report (all available data) in sample.
        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM.
        """
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = arm
        request.stamp = rospy.get_rostime()
        result_msg = self._data_service.publish_and_wait(request)
        # TODO - Make IDs match, assert that they match elsewhere here.
        sample = msg_to_sample(result_msg, self)
        return sample

    # TODO - The following could be more general by being relax_actuator
    #        and reset_actuator.
    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
        """
        relax_command = RelaxCommand()
        relax_command.id = self._get_next_seq_id()
        relax_command.stamp = rospy.get_rostime()
        relax_command.arm = arm
        self._relax_service.publish_and_wait(relax_command)

    def reset_arm(self, arm, mode, data):
        """
        Issues a position command to an arm.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
            mode: An integer code (defined in gps_pb2).
            data: An array of floats.
        """
        reset_command = PositionCommand()
        reset_command.mode = mode
        reset_command.data = data
        reset_command.pd_gains = self._hyperparams['pid_params']
        reset_command.arm = arm
        timeout = self._hyperparams['trial_timeout']
        reset_command.id = self._get_next_seq_id()
        try:
            self._reset_service.publish_and_wait(reset_command, timeout=timeout)
        except TimeoutException:
            self.relax_arm(arm)
            wait = raw_input('The robot arm seems to be stuck. Unstuck it and press <ENTER> to continue.')
            self.reset_arm(arm, mode, data)

        #TODO: Maybe verify that you reset to the correct position.

    def reset_arm_rnd(self, arm, mode, data):
        """
        Issues a position command to an arm.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
            mode: An integer code (defined in gps_pb2).
            data: An array of floats.
        """
        reset_command = PositionCommand()
        reset_command.mode = mode
        # Reset to uniform random state
        # Joints 1, 4, 5 and 6 have a range of -10,000 to +10,000 degrees. Joint 2 has a range of +42 to +318 degrees. Joint 3 has a range of +17 to +343 degrees. (see http://wiki.ros.org/jaco)
        reset_command.data = [
            (random() * 2 - 1) * pi,
            pi + (random() * 2 - 1) * pi / 2,
            # Limit elbow joints to 180 +/-90 degrees to prevent getting stuck in the ground
            pi + (random() * 2 - 1) * pi / 2,
            (random() * 2 - 1) * pi,
            (random() * 2 - 1) * pi,
            (random() * 2 - 1) * pi]
        reset_command.pd_gains = self._hyperparams['pid_params']
        reset_command.arm = arm
        timeout = self._hyperparams['trial_timeout']
        reset_command.id = self._get_next_seq_id()
        try:
            self._reset_service.publish_and_wait(reset_command, timeout=timeout)
        except TimeoutException:
            self.relax_arm(arm)
            wait = raw_input('The robot arm seems to be stuck. Unstuck it and press <ENTER> to continue.')
            self.reset_arm(arm, mode, data)


    def reset(self, condition, rnd=None):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        self.condition = condition
        #print("condition: ", condition)
        condition_data = self._hyperparams['reset_conditions'][condition]
        #print("condition data: ", condition_data)
        #print("rnd: ", rnd)
        if rnd:
            self.reset_arm_rnd(TRIAL_ARM, condition_data[TRIAL_ARM]['mode'],
                               condition_data[TRIAL_ARM]['data'])
        else:
            self.reset_arm(TRIAL_ARM, condition_data[TRIAL_ARM]['mode'],
                           condition_data[TRIAL_ARM]['data'])
        time.sleep(0.2)  # useful for the real robot, so it stops completely


    def independent_sample(self, policy, condition, verbose=True, save=True, noisy=True,
               use_TfController=False, first_itr=False):
        """
                Reset and execute a policy and collect a sample.
                Args:
                    policy: A Policy object.
                    condition: Which condition setup to run.
                    verbose: Unused for this agent.
                    save: Whether or not to store the trial into the samples.
                    noisy: Whether or not to use noise during sampling.
                    use_TfController: Whether to use the syncronous TfController
                Returns:
                    sample: A Sample object.
                """

        if use_TfController:
            self._init_tf(policy, policy.dU)
            self.use_tf = True
            self.cur_timestep = 0
            self.sample_save = save
            self.active = True

        #self.condition = condition

        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
            self.noise = noise
        else:
            self.noise = None

        # Fill in trial command
        trial_command = TrialCommand()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = \
            policy_to_msg(policy, noise, use_TfController=use_TfController)
        trial_command.T = self.T
        trial_command.id = self._get_next_seq_id()
        trial_command.frequency = self._hyperparams['frequency']
        ee_points = self._hyperparams['end_effector_points']
        trial_command.ee_points = ee_points.reshape(ee_points.size).tolist()
        trial_command.ee_points_tgt = \
            self._hyperparams['ee_points_tgt'][condition].tolist()
        trial_command.state_datatypes = self._hyperparams['state_include']
        trial_command.obs_datatypes = self._hyperparams['state_include']

        # Execute trial.
        sample_msg = self._trial_service.publish_and_wait(
            trial_command, timeout=self._hyperparams['trial_timeout']
        )
        if self.vision_enabled:
            sample_msg = self.add_rgb_stream_to_sample(sample_msg)
        sample = msg_to_sample(sample_msg, self)
        # sample = self.replace_samplestates_with_errorstates(sample, self.x_tgt[condition])
        if save:
            self._samples[condition].append(sample)
        self.active = False
        return sample

    def sample(self, policy, condition, verbose=True, save=True, noisy=True,
               use_TfController=False, first_itr=False, timeout=None, reset=True, rnd=None):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            use_TfController: Whether to use the syncronous TfController
        Returns:
            sample: A Sample object.
        """
        if use_TfController:
            self._init_tf(policy, policy.dU)
            self.use_tf = True
            self.cur_timestep = 0
            self.sample_save = save
            self.active = True

        self.policy = policy

        if reset:
            self.reset(condition, rnd=rnd)
            self.condition = condition


        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
            self.noise = noise
        else:
            noise = np.zeros((self.T, self.dU))
            self.noise = None

        # Fill in trial command
        trial_command = TrialCommand()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = \
                policy_to_msg(policy, noise, use_TfController=use_TfController)
        if timeout is not None:
            trial_command.T = timeout
        else:
            trial_command.T = self.T
        trial_command.id = self._get_next_seq_id()
        trial_command.frequency = self._hyperparams['frequency']
        ee_points = self._hyperparams['end_effector_points']
        trial_command.ee_points = ee_points.reshape(ee_points.size).tolist()
        trial_command.ee_points_tgt = \
                self._hyperparams['ee_points_tgt'][self.condition].tolist()
        trial_command.state_datatypes = self._hyperparams['state_include']
        trial_command.obs_datatypes = self._hyperparams['state_include']

        # Execute trial.
        sample_msg = self._trial_service.publish_and_wait(
            trial_command, timeout=(trial_command.T + self._hyperparams['trial_timeout'])
        )
        if self.vision_enabled:
            sample_msg = self.add_rgb_stream_to_sample(sample_msg)
        sample = msg_to_sample(sample_msg, self)
        #sample = self.replace_samplestates_with_errorstates(sample, self.x_tgt[condition])
        if save:
            self._samples[condition].append(sample)
        self.active = False
        return sample

    def add_rgb_stream_to_sample(self, sample_msg):
        img_data = DataType()
        img_data.data_type = 11
        img_data.shape = self.rgb_image_seq.shape
        #print("img_data: ", img_data.shape)
        #print("rgb_seq0: ", self.rgb_image_seq[0,:,:,:].shape)
        #cv2.imshow('dst_rt', self.rgb_image_seq[0,:,:,:])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        img_data.data = np.array(self.rgb_image_seq).reshape(-1)
        sample_msg.sensor_data.append(img_data)
        return sample_msg

    def _get_new_action(self, policy, obs, t=None):
        return policy.act(obs, obs, t, self.noise)

    def _tf_callback(self, message):
        obs = tf_obs_msg_to_numpy(message)
        obs = self.extend_state_space(obs)
        #ja_tgt = self._hyperparams['exp_x_tgts'][self.condition][0:6]
        #obs = np.append(obs, ja_tgt)
        if self.vision_enabled:
            self.rgb_image_seq[self.current_action_id, :, :, :] = self.rgb_image
            # self.stf_policy.update_task_context(obs, self.rgb_image)
        if self.current_action_id > (self.T - 1):
            print("self.T: ", self.T)
            return
        action_msg = \
                tf_policy_to_action_msg(self.dU,
                                        self._get_new_action(self.stf_policy,
                                                             obs,
                                                             self.current_action_id),
                                        self.current_action_id + 1)
        self._tf_publish(action_msg)
        self.current_action_id += 1


    def _cv_callback(self, image_msg):
        self.rgb_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.vision_enabled = True


    def _tf_publish(self, pub_msg):
        """ Publish a message without waiting for response. """
        self._pub.publish(pub_msg)

    def _init_tf(self, policy, dU):
        if self.init_tf is False:  # init pub and sub if this init has not been called before.
            self._pub = rospy.Publisher('/gps_controller_sent_robot_action_tf', TfActionCommand)
            self._sub = rospy.Subscriber('/gps_obs_tf', TfObsData, self._tf_callback)
            self._sub = rospy.Subscriber(self._hyperparams['rgb_topic'], Image, self._cv_callback)
            r = rospy.Rate(0.5)  # wait for publisher/subscriber to kick on.
            r.sleep()
            self.init_tf = True
        self.stf_policy = policy
        self.dU = dU
        self.current_action_id = 0

