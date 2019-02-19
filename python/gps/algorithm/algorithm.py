""" This file defines the base algorithm class. """

import abc
from collections import OrderedDict
import copy
import logging
import time

import numpy as np

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition

LOGGER = logging.getLogger(__name__)


class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config
        self.timers = OrderedDict()

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['tac']['clusters'] if 'tac' in hyperparams else hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO
        #self.dX = self._hyperparams['agent']['dtgtX'] + self.dX

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        self.new_mu = [None] * self.M
        self.new_sigma = [None] * self.M
        dynamics = self._hyperparams['dynamics']
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[0] # TODO Global x0
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        #self.traj_opt = hyperparams['traj_opt']['type'](
        #    hyperparams['traj_opt']
        #)
        self.cost = [
            hyperparams['cost']['type'](hyperparams['cost'])
            for _ in range(self.M)
        ]
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, sample_list, itr, train_gcm=False):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self, update_prior=True, prior_only=False):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        with Timer(self.timers, 'dynamics_fit'):
            for m in range(self.M):
                cur_data = self.cur[m].sample_list
                X = cur_data.get_X()
                U = cur_data.get_U()

                # Update prior and fit dynamics.
                if update_prior:
                    self.cur[m].traj_info.dynamics.update_prior(cur_data)
                self.cur[m].traj_info.dynamics.fit(X, U, prior_only=prior_only)

                # Update mean and covariance
                mu = np.concatenate((X[:, :, :], U[:, :, :]), axis=2)
                #print("shape mu: ", mu.shape)
                self.cur[m].traj_info.xmu = np.mean(mu, axis=0)
                self.cur[m].traj_info.xmusigma = np.mean(X[:, :, :], axis=0)

                # Fit x0mu/x0sigma.
                x0 = X[:, 0, :]
                x0mu = np.mean(x0, axis=0)
                self.cur[m].traj_info.x0mu = x0mu
                self.cur[m].traj_info.x0sigma = np.diag(
                    np.maximum(np.var(x0, axis=0), self._hyperparams['initial_state_var'])
                )

                prior = self.cur[m].traj_info.dynamics.get_prior()
                if prior:
                    mu0, Phi, priorm, n0 = prior.initial_state()
                    N = len(cur_data)
                    self.cur[m].traj_info.x0sigma += Phi + (N * priorm) / (N +
                                                                           priorm) * np.outer(x0mu - mu0,
                                                                                              x0mu - mu0) / (N + n0)

        self.visualize_dynamics(0)

    def _update_trajectories(self, itr=None):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        with Timer(self.timers, 'traj_opt'):
            for cond in range(self.M):
                self.new_traj_distr[cond], self.cur[cond].eta, self.new_mu[cond], self.new_sigma[cond] = \
                    self.traj_opt.update(cond, self)

        self.visualize_local_policy(0)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

    def visualize_dynamics(self, m):
        from gps.visualization import visualize_linear_model

        traj_info = self.cur[m].traj_info
        dynamics = traj_info.dynamics

        visualize_linear_model(
            file_name=self._data_files_dir + 'plot_dynamics_m%d-%02d' % (m, self.iteration_count),
            coeff=dynamics.Fm[:-1],
            intercept=dynamics.fv[:-1],
            cov=dynamics.dyn_covar[:-1],
            x=traj_info.xmu[:-1],
            y=traj_info.xmu[1:, :self.dX],
            coeff_label='$f_{\\mathbf{x}\\mathbf{u} t}$',
            intercept_label='$f_{\\mathbf{c} t}$',
            cov_label='$\\mathbf{F}_t$',
            y_label='$\\mathbf{x}_{t+1}$'
        )

    def visualize_local_policy(self, m, title='pol_lqr'):
        from gps.visualization import visualize_linear_model

        traj_info = self.cur[m].traj_info
        traj_distr = self.new_traj_distr[m]

        visualize_linear_model(
            file_name=self._data_files_dir + 'plot_%s-m%d-%02d' % (title, m, self.iteration_count),
            coeff=traj_distr.K,
            intercept=traj_distr.k,
            cov=traj_distr.pol_covar,
            x=traj_info.xmu[:, :self.dX],
            y=None,
            coeff_label='$\\mathbf{K}_t$',
            intercept_label='$\\mathbf{k}_t$',
            cov_label='$\\Sigma_t$',
            y_label='$\\mathbf{u}_t$'
        )

    def visualize_policy_linearization(self, m, title='pol_lin'):
        from gps.visualization import visualize_linear_model

        pol_info = self.cur[m].pol_info
        traj_info = self.cur[m].traj_info

        visualize_linear_model(
            file_name=self._data_files_dir + 'plot_%s-m%d-%02d' % (title, m, self.iteration_count),
            coeff=pol_info.pol_K,
            intercept=pol_info.pol_k,
            cov=pol_info.pol_S,
            x=traj_info.xmu[:, :self.dX],
            y=None,
            coeff_label='$\\bar{\\mathbf{K}}_t$',
            intercept_label='$\\bar{\\mathbf{k}}_t$',
            cov_label='$\\bar{\\Sigma}_t$',
            y_label='$\\bar{\\mathbf{u}}_t$'
        )


class Timer:
    """
    Timer context to measure elapsed time
    """
    def __init__(self, timers, name):
        self.timers = timers
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.timers[self.name] = time.time() - self.start
