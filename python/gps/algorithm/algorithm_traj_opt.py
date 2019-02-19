""" This file defines the iLQG-based trajectory optimization method. """

import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Timer
from gps.algorithm.algorithm_NN import Algorithm_NN

LOGGER = logging.getLogger(__name__)


class AlgorithmTrajOpt(Algorithm_NN):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams):
        super(AlgorithmTrajOpt, self).__init__(hyperparams)

    def _update_trajectories(self, _):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        with Timer(self.timers, 'traj_opt'):
            for cond in range(self.M):
                self.new_traj_distr[cond], self.cur[cond].eta, self.new_mu[cond], self.new_sigma[cond], _ = \
                        self.traj_opt_update(cond)

        self.visualize_local_policy(0)

    def backward(self, prev_traj_distr, traj_info, eta):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Lagrange dual variable.
        Returns:
            traj_distr: A new linear Gaussian policy.
        """
        # Constants.
        T = prev_traj_distr.T
        dimU = prev_traj_distr.dU
        dimX = prev_traj_distr.dX

        index_x = slice(dimX)
        index_u = slice(dimX, dimX + dimU)

        # Get quadratic expansion of the extended cost function
        Cm_ext, cv_ext = self.compute_extended_costs(eta, traj_info,
                                                     prev_traj_distr)
        self.Cm_ext = Cm_ext
        self.cv_ext = cv_ext

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Allocate.
        Vm = np.zeros((T, dimX, dimX))
        vv = np.zeros((T, dimX))

        traj_distr = prev_traj_distr.nans_like()

        # Compute state-action-state function at each time step.
        for t in range(T - 1, -1, -1):
            # Add in the cost.
            Qm = Cm_ext[t, :, :]  # (X+U) x (X+U)
            qv = cv_ext[t, :]  # (X+U) x 1

            # Add in the value function from the next time step.
            if t < T - 1:
                Qm += Fm[t, :, :].T.dot(Vm[t+1, :, :]).dot(Fm[t, :, :])
                qv += Fm[t, :, :].T.dot(vv[t+1, :] + Vm[t+1, :, :].dot(fv[t, :]))

            # Symmetrize quadratic component to counter numerical errors.
            Qm = 0.5 * (Qm + Qm.T)

            # Compute Cholesky decomposition of Q function action
            # component.
            U = sp.linalg.cholesky(Qm[index_u, index_u])
            L = U.T

            # Compute mean terms.
            traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, Qm[index_u, index_x],
                                              lower=True)
            )
            traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, qv[index_u], lower=True)
            )

            # Store conditional covariance, inverse, and Cholesky.
            traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, np.eye(dimU), lower=True)
            )
            traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                traj_distr.pol_covar[t, :, :]
            )
            traj_distr.inv_pol_covar[t, :, :] = Qm[index_u, index_u]

            # Compute value function.
            Vm[t, :, :] = Qm[index_x, index_x] + \
                    Qm[index_x, index_u].dot(traj_distr.K[t, :, :])
            # Symmetrize quadratic component to counter numerical errors.
            Vm[t, :, :] = 0.5 * (Vm[t, :, :] + Vm[t, :, :].T)
            vv[t, :] = qv[index_x] + Qm[index_x, index_u].dot(traj_distr.k[t, :])

            traj_distr.Qm[t, :, :] = Qm
            traj_distr.qv[t, :] = qv

        return traj_distr

    def forward(self, traj_distr, traj_info):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: T x (dX + dU) mean state + action vector.
            sigma: T x (dX + dU) x (dX + dU) state + action covariance matrix.
        """
        # Constants.
        T = traj_distr.T
        dimU = traj_distr.dU
        dimX = traj_distr.dX

        index_x = slice(dimX)
        index_u = slice(dimX, dimX + dimU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Allocate space.
        sigma = np.zeros((T, dimX + dimU, dimX + dimU))
        mu = np.zeros((T, dimX + dimU))

        # Set initial mean and covariance
        mu[0, index_x] = traj_info.x0mu
        sigma[0, index_x, index_x] = traj_info.x0sigma

        for t in range(T):
            mu[t, index_u] = traj_distr.K[t, :, :].dot(mu[t, index_x]) + \
                             traj_distr.k[t, :]

            sigma[t, index_x, index_u] = \
                sigma[t, index_x, index_x].dot(traj_distr.K[t, :, :].T)

            sigma[t, index_u, index_x] = \
                traj_distr.K[t, :, :].dot(sigma[t, index_x, index_x])

            sigma[t, index_u, index_u] = \
                traj_distr.K[t, :, :].dot(sigma[t, index_x, index_x]).dot(
                    traj_distr.K[t, :, :].T
                ) + traj_distr.pol_covar[t, :, :]

            if t < T - 1:
                mu[t + 1, index_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

                sigma[t + 1, index_x, index_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
        return mu, sigma
