""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics


class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
                self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    def fit(self, X, U, prior_only=False):
        """ Fit dynamics. """
        # Constants
        N, T, dimX = X.shape
        dimU = U.shape[2]

        index_xu = slice(dimX + dimU)
        index_x = slice(dimX + dimU, dimX + dimU + dimX)

        sig_reg = np.zeros((dimX + dimU + dimX, dimX + dimU + dimX))
        sig_reg[index_xu, index_xu] = self._hyperparams['regularization']

        # Weights used in computing sample mean and sample covariance.
        dwts = (1.0 / N) * np.ones(N)
        D = np.diag((1.0 / (N - 1)) * np.ones(N))

        # Allocate 
        self.Fm = np.zeros([T, dimX, dimX + dimU])
        self.fv = np.zeros([T, dimX])
        self.dyn_covar = np.zeros([T, dimX, dimX])

        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]

            # Obtain normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dimX, dimU, Ys)

            # Compute empirical mean and covariance.
            empmu = np.sum((Ys.T * dwts).T, axis=0)
            diff = Ys - empmu
            empsig = diff.T.dot(D).dot(diff)
            # Symmetrize empsig to counter numerical errors.
            empsig = 0.5 * (empsig + empsig.T)

            # Compute posterior estimates of mean and covariance.
            mu = empmu if not prior_only else mu0
            sigma = (Phi + (N - 1) * empsig + (N * mm) / (N + mm) *
                     np.outer(empmu - mu0, empmu - mu0)) / (N + n0) if not prior_only else Phi
            # Symmetrize sigma to counter numerical errors.
            sigma = 0.5 * (sigma + sigma.T)
            # Add sigma regularization.
            sigma += sig_reg

            # Conditioning to get dynamics.
            Fm = np.linalg.solve(sigma[index_xu, index_xu],
                                 sigma[index_xu, index_x]).T
            fv = mu[index_x] - Fm.dot(mu[index_xu])
            dyn_covar = sigma[index_x, index_x] - Fm.dot(sigma[index_xu, index_xu]).dot(Fm.T)
            # Symmetrize dyn_covar to counter numerical errors.
            dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar

        return self.Fm, self.fv, self.dyn_covar

    def fit_gcmdistr(self, ref_mu, gcm_mu, dX, dU):
        """ Fit dynamics. """
        # ('shape X: ', (80, 12))
        # ('shape U: ', (80, 6))
        X = [ref_mu[:, :dX].tolist(), gcm_mu[:, :dX].tolist()]
        U = [ref_mu[:, dX:dX + dU].tolist(), gcm_mu[:, dX:dX + dU].tolist()]
        X = np.array(X)
        U = np.array(U)

        # print("X shape: ", X.shape)
        # print("U shape: ", U.shape)
        # Constants
        N, T, dimX = X.shape
        dimU = U.shape[2]

        index_xu = slice(dimX + dimU)
        index_x = slice(dimX + dimU, dimX + dimU + dimX)

        sig_reg = np.zeros((dimX + dimU + dimX, dimX + dimU + dimX))
        sig_reg[index_xu, index_xu] = self._hyperparams['regularization']

        # Weights used in computing sample mean and sample covariance.
        dwts = (1.0 / N) * np.ones(N)
        # dwts = np.ones(1)
        D = np.diag((1.0 / (N - 1)) * np.ones(N))

        # Allocate
        self.Fm = np.zeros([T, dimX, dimX + dimU])
        self.fv = np.zeros([T, dimX])
        self.dyn_covar = np.zeros([T, dimX, dimX])
        # self.next_x = np.zeros([T, dimX])

        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]
            Ys_ref = np.r_[X[0, t, :], U[0, t, :], X[0, t + 1, :]]
            Ys_gcm = np.r_[X[1, t, :], U[1, t, :], X[1, t + 1, :]]
            Ys_ref = np.array([Ys_ref])
            Ys_gcm = np.array([Ys_gcm])

            # Compute empirical mean and covariance.
            empmu = np.sum((Ys_ref.T * np.ones(1)).T, axis=0)
            diff = Ys_gcm - Ys_ref
            empsig = diff.T.dot(diff)
            # Symmetrize empsig to counter numerical errors.
            gcm_sig = 0.5 * (empsig + empsig.T)

            # ref_mu = np.sum((Ys_ref.T * np.ones(1)).T, axis=0)
            # diff = Ys - Ys_ref
            # print("shape Ys: ", Ys.shape)
            # print("shape Ys_ref", Ys_ref.shape)
            # print("shape Ys_gcm", Ys_gcm.shape)#

            # print("shape D: ", D.shape)
            # print("shape diff: ", diff.shape)

            # empsig = diff.T.dot(D).dot(diff)
            # empsig = diff.T.dot(diff)
            # print("shape empgsig: ", empsig.shape)
            # raw_input("wait")
            # Symmetrize empsig to counter numerical errors.
            # gcm_sig = 0.5 * (empsig + empsig.T)

            # Obtain normal-inverse-Wishart prior.
            # print("shape Ys_gcm: ", Ys_gcm.shape)
            # print("shape Ys: ", Ys.shape)
            mu0, Phi, mm, n0 = self.prior.eval(dimX, dimU, Ys_gcm)

            # Compute posterior estimates of mean and covariance.
            # mu = empmu
            # mu = ref_mu
            mu = (mm * mu0 + n0 * empmu) / (mm + n0)
            sigma = (Phi + (N - 1) * gcm_sig + (N * mm) / (N + mm) *
                     np.outer(empmu - mu0, empmu - mu0)) / (N + n0)
            # Symmetrize sigma to counter numerical errors.
            sigma = 0.5 * (sigma + sigma.T)
            # Add sigma regularization.
            sigma += sig_reg

            # Conditioning to get dynamics.
            Fm = np.linalg.solve(sigma[index_xu, index_xu],
                                 sigma[index_xu, index_x]).T
            fv = mu[index_x] - Fm.dot(mu[index_xu])
            dyn_covar = sigma[index_x, index_x] - Fm.dot(sigma[index_xu, index_xu]).dot(Fm.T)
            # Symmetrize dyn_covar to counter numerical errors.
            dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
            # self.next_x[t, :] = mu[index_x]
            # print("shape mu: ", mu.shape)
            # print("shape mu[index_x]: ", mu[index_x].shape)
            # raw_input("wait")

        return self.Fm, self.fv, self.dyn_covar

