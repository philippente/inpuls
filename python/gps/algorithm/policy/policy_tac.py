import numpy as np
from scipy.stats import multivariate_normal as mvn

from gps.algorithm.policy.policy import Policy


class PolicyTAC(Policy):
    def __init__(self, algorithm, history_length):
        Policy.__init__(self)
        self.dU = algorithm.dU  # This should be in super class

        self.algorithm = algorithm
        self.X = np.empty([algorithm.T, algorithm.dX])
        self.U = np.empty([algorithm.T, algorithm.dU])
        self.history_length = history_length

    def act(self, x, obs, t, noise=None):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        self.X[t, :] = x
        logpdfs = np.empty(self.algorithm.M)
        for m in range(self.algorithm.M):  # TODO Parallelize
            logpdfs[m] = self._logpdf_traj(self.X, self.U, m, prev=True, r=range(max(0, t-(self.history_length+1)), max(0, t-1)))
        m = np.argmax(logpdfs)

        # Choose action from closest cluster
        u = self.algorithm.cur[m].traj_distr.act(x, obs, t, noise)
        self.U[t, :] = u
        return u
    
    def _logpdf_traj(self, X, U, m, prev=False, r=None):
        """
        Calculates the liklihood of a sample being drawn from a cluster.

        Args:
            X, U: The sample
            m: The cluster
            prev: Match against clusters of previous iteration (for initial distribution and off-policy sampling)
            r: Range of steps to include (for off-policy sampling)
        """
        # By default include all time steps
        if r is None:
            r = range(self.algorithm.T-1)

        # Select trajectory distribution of current or previous iteration
        traj_info = self.algorithm.cur[m].traj_info if not prev else self.algorithm.prev[m].traj_info
        #pol_info = self.cur[m].pol_info if not prev else self.prev[m].pol_info

        # Initial state distribution
        logpdf_x0 = mvn.logpdf(X[0], traj_info.x0mu, traj_info.x0sigma)
        if r:
            logpdf_pol, logpdf_dyn = np.zeros(len(r)), np.empty(len(r))
            for t in r:

                ## Policy
                #if self.pol_lin == 'GPS':
                #    K, k, pol_covar = pol_info.pol_K[t], pol_info.pol_k[t], pol_info.pol_S[t]
                #elif self.pol_lin == 'GMR':
                #    K, k, pol_covar = self.policy_opt.policy.linearization(traj_info.xmu[t, :self.dX], traj_info.xmu[t, :self.dX])
                #else:
                #    raise ValueError('Unknown policy linearization: %s'%self.pol_lin)
                #logpdf_pol[t-r[0]] = mvn.logpdf(
                #    U[t],
                #    mean=K.dot(X[t]) + k,
                #    cov=pol_covar)

                # Dynamics
                Fm = traj_info.dynamics.Fm[t]
                fv = traj_info.dynamics.fv[t]
                dyn_covar = traj_info.dynamics.dyn_covar[t]
                logpdf_dyn[t-r[0]] = mvn.logpdf(
                    X[t+1],
                    mean=Fm[:, :self.algorithm.dX].dot(X[t]) + Fm[:, self.algorithm.dX:].dot(U[t]) + fv,
                    cov=dyn_covar)
        else:
            logpdf_pol, logpdf_dyn = np.zeros(0), np.zeros(0)

        logpdf = logpdf_x0 + np.sum(logpdf_pol) + np.sum(logpdf_dyn)
        #print("logpdf", logpdf, "x0", logpdf_x0, "pol", np.sum(logpdf_pol), "dyn", np.sum(logpdf_dyn))
        return logpdf
