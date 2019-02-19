from gps.algorithm.algorithm import Algorithm
import copy
import logging
import numpy as np
import scipy as sp
from gps.algorithm.traj_opt.traj_opt_utils import calc_traj_distr_kl, DGD_MAX_ITER
from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
import abc


LOGGER = logging.getLogger(__name__)


class Algorithm_NN(Algorithm):
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        super(Algorithm_NN, self).__init__(hyperparams)
        traj_opt_config = copy.deepcopy(TRAJ_OPT_LQR)
        self.traj_opt_hyperparams = traj_opt_config
        self.itr = None
        self.init_step_mult = []

    @abc.abstractmethod
    def _update_trajectories(self, train):
        """
        Compute new linear Gaussian controllers.
        """
        raise NotImplementedError("Must be implemented in subclass")

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
                Qm += Fm[t, :, :].T.dot(Vm[t + 1, :, :]).dot(Fm[t, :, :])
                qv += Fm[t, :, :].T.dot(vv[t + 1, :] + Vm[t + 1, :, :].dot(fv[t, :]))

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

    def iteration(self, sample_lists, itr=None, train_gcm = False):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        self.itr = itr
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        self._update_dynamics()

        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        self.init_step_mult = []
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)
            self.init_step_mult.append(copy.deepcopy(self.cur[m].step_mult))

        self._update_trajectories(train_gcm)

        self._advance_iteration_variables()

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = self.estimate_cost(
            self.cur[m].traj_distr, self.cur[m].traj_info
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def estimate_cost(self, traj_distr, traj_info):
        """ Compute Laplace approximation to expected cost. """
        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because
        # traj_info may have different dynamics from the ones that were
        # used to compute the distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * \
                    np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + 0.5 * \
                    mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                    mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

    def traj_opt_update(self, m):
        """ Run dual gradient decent to optimize trajectories. """
        T = self.T
        eta = self.cur[m].eta
        print("eta_0: ", eta)
        step_mult = self.cur[m].step_mult
        traj_info = self.cur[m].traj_info
        print("step_mult: ", step_mult)

        prev_traj_distr = self.cur[m].traj_distr

        # Set KL-divergence step size (epsilon).
        kl_step = T * self.base_kl_step * step_mult

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        min_eta = self.traj_opt_hyperparams['min_eta']
        max_eta = self.traj_opt_hyperparams['max_eta']

        LOGGER.debug("Running DGD for trajectory %d, eta: %f", m, eta)
        for itr in range(DGD_MAX_ITER):

            LOGGER.debug("Iteration %i, bracket: (%.2e , %.2e , %.2e)",
                    itr, min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # NOTE: we can just ignore case when the new eta is larger.
            traj_distr = self.backward(prev_traj_distr, traj_info,
                                                eta)
            new_mu, new_sigma = self.forward(traj_distr, traj_info)

            # Compute KL divergence constraint violation.
            kl_div, kl_div_t = calc_traj_distr_kl(new_mu, new_sigma,
                                   traj_distr, prev_traj_distr)
            con = kl_div - kl_step

            print("kl_div - kl_step: ", con)
            # Convergence check - constraint satisfaction.
            if (abs(con) < 0.1*kl_step):
                LOGGER.debug("KL: %f / %f, converged iteration %i",
                        kl_div, kl_step, itr)
                break

            # Choose new eta (bisect bracket or multiply by constant)
            if con < 0: # Eta was too big.
                max_eta = eta
                geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                new_eta = max(geom, 0.1*max_eta)
                LOGGER.debug("KL: %f / %f, eta too big, new eta: %f",
                        kl_div, kl_step, new_eta)
            else: # Eta was too small.
                min_eta = eta
                geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                new_eta = min(geom, 10.0*min_eta)
                LOGGER.debug("KL: %f / %f, eta too small, new eta: %f",
                        kl_div, kl_step, new_eta)

            # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
            eta = new_eta

            print("eta_1: ", eta)
            print("kl_step: ", kl_step)

        if kl_div > kl_step and abs(kl_div - kl_step) > 0.1*kl_step:
            LOGGER.warning(
                "Final KL divergence after DGD convergence is too high."
            )

        return traj_distr, eta, new_mu, new_sigma, kl_div_t

    def compute_extended_costs(self, eta, traj_info, traj_distr):
        """ Compute expansion of extended cost used in the LQR backward pass.
            The extended cost function is 1/eta * c(x, u) - log p(u | x)
            with eta being the lagrange dual variable and
            p being the previous trajectory distribution
        """
        Cm_ext, cv_ext = traj_info.Cm / eta, traj_info.cv / eta
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            Cm_ext[t, :, :] += np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            cv_ext[t, :] += np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return Cm_ext, cv_ext
