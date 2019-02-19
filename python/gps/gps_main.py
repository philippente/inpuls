""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import os
import os.path
import sys
import copy
import time
import shutil

import numpy as np

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__.replace('\\', '/'), '/')[:-2]))  # Replace backslashes for windows compability
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

from gps.algorithm.algorithm import Timer
from gps.algorithm.algorithm_badmm import AlgorithmBADMM

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.policy.policy_tac import PolicyTAC
from tqdm import trange

class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False, no_algorithm=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']
        config['agent']['data_files_dir'] = self._data_files_dir
        config['algorithm']['data_files_dir'] = self._data_files_dir

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = None
        if config['gui_on']:
            from gps.gui.gps_training_gui import GPSTrainingGUI  # Only import if neccessary
            self.gui = GPSTrainingGUI(config['common'])
        self.mode = None

        config['algorithm']['agent'] = self.agent
        if not no_algorithm:
            self.algorithm = config['algorithm']['type'](config['algorithm'])
            self.algorithm._data_files_dir = self._data_files_dir
            if hasattr(self.algorithm, 'policy_opt'):
                self.algorithm.policy_opt._data_files_dir = self._data_files_dir

        self.session_id = None

    def run(self, session_id, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        itr_start = self._initialize(itr_load)
        if session_id is not None:
            self.session_id = session_id

        for itr in range(itr_start, self._hyperparams['iterations']):
            self.iteration_count = itr
            if hasattr(self.algorithm, 'policy_opt'):
                self.algorithm.policy_opt.iteration_count = itr

            print("*** Iteration %02d ***" % itr)
            # Take trajectory samples
            with Timer(self.algorithm.timers, 'sampling'):
                for cond in self._train_idx:
                    for i in trange(self._hyperparams['num_samples'], desc='Taking samples'):
                        self._take_sample(itr, cond, i)
            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples']) for cond in self._train_idx
            ]
            self.export_samples(traj_sample_lists)

            # Iteration
            with Timer(self.algorithm.timers, 'iteration'):
                self.algorithm.iteration(traj_sample_lists, itr)
            self.export_dynamics()
            self.export_controllers()
            self.export_times()

            # Sample learned policies for visualization

            # LQR policies static resets
            if self._hyperparams['num_lqr_samples_static'] > 0:
                self.export_samples(
                    self._take_policy_samples(N=self._hyperparams['num_lqr_samples_static'], pol=None, rnd=False),
                    '_lqr-static'
                )

            # LQR policies random resets
            if self._hyperparams['num_lqr_samples_random'] > 0:
                self.export_samples(
                    self._take_policy_samples(N=self._hyperparams['num_lqr_samples_random'], pol=None, rnd=True),
                    '_lqr-random'
                )

            if hasattr(self.algorithm, 'policy_opt'):
                # Global policy static resets
                if self._hyperparams['num_pol_samples_static'] > 0:
                    self.export_samples(
                        self._take_policy_samples(
                            N=self._hyperparams['num_pol_samples_static'],
                            pol=self.algorithm.policy_opt.policy,
                            rnd=False
                        ), '_pol-static'
                    )

                # Global policy static resets
                if self._hyperparams['num_pol_samples_random'] > 0:
                    self.export_samples(
                        self._take_policy_samples(
                            N=self._hyperparams['num_pol_samples_random'],
                            pol=self.algorithm.policy_opt.policy,
                            rnd=True
                        ), '_pol-random'
                    )

        self._end()

    def test_policy(self, itr, N, reset_cond=None):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit()
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N, reset_cond=reset_cond)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text((
                    'Took %d policy sample(s) from ' +
                    'algorithm state at iteration %d.\n' +
                    'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1)  # called instead of sys.exit(), since this is in a thread

            if self.gui:
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' + 'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if 'tac_policy' in self.algorithm._hyperparams and self.algorithm.iteration_count > 0:
            pol = PolicyTAC(self.algorithm, self.algorithm._hyperparams['tac_policy']['history'])
            if 'tac' in self.algorithm._hyperparams:
                self.agent.T = self.algorithm._hyperparams['tac']['T'] # Use possibly larger T for on-policy sampling
        elif self.algorithm._hyperparams['sample_on_policy'] and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy
            if 'tac' in self.algorithm._hyperparams:
                self.agent.T = self.algorithm._hyperparams['tac']['T'] # Use possibly larger T for on-policy sampling
        else:
            pol = self.algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'test':
                        self.mode = 'test'
                        self._take_policy_samples()
                        self.gui.request = 'stop'
                    if self.gui.request == 'go':
                        self.mode = 'go'
                    if self.gui.request == 'gcm':
                        self.mode = 'gcm'
                    if self.gui.request == 'stop':
                        self.mode = 'stop'

                    if self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials']), noisy=True, use_TfController=True,
                    rnd=self.agent._hyperparams['random_reset']
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials']), noisy=True, use_TfController=True,
                reset_cond=None if self.agent._hyperparams['random_reset'] else cond
            )

    def _take_policy_samples(self, N, pol, rnd=False):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
            pol: Policy to sample. None for LQR policies.
        Returns: None
        """
        if pol is None:
            pol_samples = [[None] * N] * len(self._test_idx)
            for i, cond in enumerate(self._test_idx, 0):
                for n in trange(N, desc='Taking LQR-policy samples m=%d, cond=%s' % (cond, 'rnd' if rnd else cond)):
                    pol_samples[i][n] = self.agent.sample(
                        self.algorithm.cur[cond].traj_distr,
                        None,
                        verbose=None,
                        save=False,
                        noisy=False,
                        reset_cond=None if rnd else cond,
                        record=False
                    )
            return [SampleList(samples) for samples in pol_samples]
        else:
            conds = self._test_idx if not rnd else [None]
            # stores where the policy has lead to
            pol_samples = [[None] * N] * len(conds)
            for i, cond in enumerate(conds):
                for n in trange(
                    N, desc='Taking %s policy samples cond=%s' % (type(pol).__name__, 'rnd' if rnd else cond)
                ):
                    pol_samples[i][n] = self.agent.sample(
                        pol, None, verbose=None, save=False, noisy=False, reset_cond=cond, record=n < 0
                    )
            return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None, controller=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        #print("log 0")
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            #self.gui.update(itr, self.algorithm, self.agent,
            #                copy.copy(traj_sample_lists), pol_sample_lists)
            #self.gui.save_figure(
            #    self._data_files_dir + ('figure_itr_%02d.png' % itr)
            #    )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        #print("log 1")
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        #print("log 2")
        self.data_logger.pickle(
            self._data_files_dir + ('%s_samplelist_itr%02d.pkl' % (self.session_id, itr)),
            copy.copy(traj_sample_lists)
        )
        if controller:
            self.data_logger.pickle(
                self._data_files_dir + ('%s_controller_itr%02d.pkl' % (self.session_id, itr)),
                copy.copy(controller)
            )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_lqr_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

    def export_samples(self, traj_sample_lists, sample_type=''):
        """
        Exports trajectoy samples in a compressed numpy file.
        """
        M, N, T, dX, dU = len(traj_sample_lists), len(traj_sample_lists[0]), self.agent.T, self.agent.dX, self.agent.dU
        X = np.empty((M, N, T, dX))
        U = np.empty((M, N, T, dU))

        for m in range(M):
            sample_list = traj_sample_lists[m]
            for n in range(N):
                sample = sample_list[n]
                X[m, n] = sample.get_X()
                U[m, n] = sample.get_U()

        np.savez_compressed(
            self._data_files_dir + 'samples%s_%02d' % (sample_type, self.iteration_count),
            X=X,
            U=U,
        )

    def export_dynamics(self):
        """
        Exports the local dynamics data in a compressed numpy file.
        """
        M, T, dX, dU = self.algorithm.M, self.agent.T, self.agent.dX, self.agent.dU
        Fm = np.empty((M, T - 1, dX, dX + dU))
        fv = np.empty((M, T - 1, dX))
        dyn_covar = np.empty((M, T - 1, dX, dX))

        for m in range(M):
            dynamics = self.algorithm.cur[m].traj_info.dynamics
            Fm[m] = dynamics.Fm[:-1]
            fv[m] = dynamics.fv[:-1]
            dyn_covar[m] = dynamics.dyn_covar[:-1]

        np.savez_compressed(
            self._data_files_dir + 'dyn_%02d' % self.iteration_count,
            Fm=Fm,
            fv=fv,
            dyn_covar=dyn_covar,
        )

    def export_controllers(self):
        """
        Exports the local controller data in a compressed numpy file.
        """
        M, T, dX, dU = self.algorithm.M, self.agent.T, self.agent.dX, self.agent.dU
        K = np.empty((M, T - 1, dU, dX))
        k = np.empty((M, T - 1, dU))
        prc = np.empty((M, T - 1, dU, dU))

        traj_mu = np.empty((M, T, dX + dU))
        traj_sigma = np.empty((M, T, dX + dU, dX + dU))

        for m in range(M):
            traj = self.algorithm.cur[m].traj_distr
            K[m] = traj.K[:-1]
            k[m] = traj.k[:-1]
            prc[m] = traj.inv_pol_covar[:-1]
            traj_mu[m] = self.algorithm.new_mu[m]
            traj_sigma[m] = self.algorithm.new_sigma[m]

        np.savez_compressed(
            self._data_files_dir + 'ctr_%02d' % self.iteration_count,
            K=K,
            k=k,
            prc=prc,
            traj_mu=traj_mu,
            traj_sigma=traj_sigma,
        )

    def export_times(self):
        """
        Exports timer values into a csv file by appending a line for each iteration.
        """
        header = ','.join(self.algorithm.timers.keys()) if self.iteration_count == 0 else ''
        with open(self._data_files_dir + 'timers.csv', 'ab') as out_file:
            np.savetxt(out_file, np.asarray([np.asarray([f for f in self.algorithm.timers.values()])]), header=header)
