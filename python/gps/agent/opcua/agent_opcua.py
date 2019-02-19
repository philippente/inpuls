""" This file defines an agent for the Kinova Jaco2 ROS environment. """
import _thread as thread
import numpy as np
import logging
import time
from multiprocessing.pool import ThreadPool

from opcua import ua, Client

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION


class AgentOPCUA(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """

    def __init__(self, hyperparams):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        Agent.__init__(self, hyperparams)
        self.x0 = None

        self.dt = self._hyperparams['dt']
        self.scaler = self._hyperparams['scaler']
        self.actuator_overrides = self._hyperparams['override_actuator']
        self.sensor_overrides = self._hyperparams['override_sensor']
        self.signals = self._hyperparams['send_signal']
        self.sensors = self.x_data_types
        self.actuators = self.u_data_types
        #self.__init_opcua()
        self.client = None
        self.pool = ThreadPool(len(self.sensors))
        self.debug = True

    def __init_opcua(self):
        if self.client is not None:  # Close existing connection
            try:
                self.client.disconnect()
            except ua.uaerrors.UaError:
                pass
        self.client = Client(self._hyperparams['opc-ua_server'])
        self.client.connect()
        try:
            # Get Nodes
            self.opcua_vars = {}

            for sensor in (self.sensors + self.actuators):
                self.opcua_vars[sensor] = self.__get_node(sensor.value)
            for signal in self.signals:
                self.opcua_vars[signal['signal']] = self.__get_node(signal['signal'])
        except Exception:  # Close connection in case of error
            self.close()
            raise

    def __get_node(self, sensor):
        browse_name = sensor.split(',')
        try:
            return self.client.get_objects_node().get_child(browse_name)
        except ua.uaerrors.BadNoMatch:
            logging.critical("Node not found: '%s'" % (sensor))
            raise

    def transform(self, sensor, sensor_data):
        """
        Scales sates according to the scaler.
        """
        data_idx = self._x_data_idx[sensor]
        return (sensor_data - self.scaler.mean_[data_idx]) / self.scaler.scale_[data_idx]

    def read_sensor(self, sensor):
        sensor_data = self.opcua_vars[sensor].get_value()
        if np.isscalar(sensor_data):  # Wrap scalars into single element arrays
            sensor_data = [sensor_data]
        return self.transform(sensor, np.asarray(sensor_data))

    def write_actuator(self, actuator, data):
        if len(data) == 1:
            data = data.item()  # TODO Maybe keep original dimension?
        data = ua.Variant(data, ua.VariantType.Float)
        try:
            self.opcua_vars[actuator].set_data_value(data)
        except ua.uaerrors._auto.BadTypeMismatch:
            logging.critical(
                "Failed to write %r to %r of type %r" % (data, actuator, self.opcua_vars[actuator].get_data_type())
            )
            raise

    def reset(self, cond):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        if not self.debug:
            input('Press Enter to confirm reset')

    def sample(self, policy, condition, save=True, noisy=True, reset_cond=None, **kwargs):
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
        # Get a new sample
        sample = Sample(self)
        sample_ok = False
        while not sample_ok:
            if not self.debug:
                self.reset(reset_cond)

            self.__init_opcua()

            if noisy:
                noise = generate_noise(self.T, self.dU, self._hyperparams)
            else:
                noise = np.zeros((self.T, self.dU))

            # Execute policy over a time period of [0,T]
            start = time.time()
            for t in range(self.T):
                # Read sensors and store sensor data in sample
                def store_sensor(sensor):
                    sample.set(sensor, self.read_sensor(sensor), t)

                self.pool.map(store_sensor, self.sensors)
                # Override sensors
                for override in self.sensor_overrides:
                    if override['condition'](t):
                        sensor = override['sensor']
                        sample.set(sensor, np.copy(override['value']), t)

                print('X_%02d' % t, sample.get_X(t))

                # Get action
                U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise)

                # Override actuators
                for override in self.actuator_overrides:
                    if override['condition'](t):
                        actuator = override['actuator']
                        U_t[self._u_data_idx[actuator]] = np.copy(override['value'])

                # Send signals
                self.send_signals(t)

                # Perform action
                for actuator in self._u_data_idx:
                    self.write_actuator(actuator, U_t[self._u_data_idx[actuator]])
                sample.set(ACTION, U_t, t)

                print('U_%02d' % t, U_t)

                # Check if agent is keeping up
                sleep_time = start + (t + 1) * self.dt - time.time()
                if sleep_time < 0:
                    logging.critical("Agent can't keep up. %fs bedind." % sleep_time)
                elif sleep_time < self.dt / 2:
                    logging.warning(
                        "Agent may not keep up (%.0f percent busy)" % (((self.dt - sleep_time) / self.dt) * 100)
                    )

                # Wait for next timestep
                if sleep_time > 0 and not self.debug:
                    time.sleep(sleep_time)
            if save:
                self._samples[condition].append(sample)
            self.active = False
            self.finalize_sample()

            sample_ok = input('Continue?') == 'y'
            if not sample_ok:
                print('Repeating')
        return sample

    def finalize_sample(self):
        if not self.debug:
            time.sleep(10)
        self.send_signals(self.T)

    def send_signals(self, t):
        if not self.debug:
            for signal in self.signals:
                if signal['condition'](t):
                    self.send_signal(signal, True)
                    thread.start_new_thread(self.send_signal, (signal, False, 2))

    def send_signal(self, signal, value, sleep=0):
        if sleep > 0:
            time.sleep(sleep)

        opcua_var = self.opcua_vars[signal['signal']]
        print("Signal", signal['signal'], value)
        opcua_var.set_data_value(ua.Variant(value, ua.VariantType.Boolean))

    def close(self):
        """ Releases any resources the agent may hold. """
        if self.client is not None:
            self.client.disconnect()
