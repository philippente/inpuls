import numpy as np

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy


def init_pol(hyperparams):
    dU, dX = hyperparams['dU'], hyperparams['dX']
    T = hyperparams['T']

    K = np.zeros((T, dU, dX))
    k = np.empty((T, dU))
    PSig = np.empty((T, dU, dU))
    cholPSig = np.empty((T, dU, dU))
    inv_pol_covar = np.empty((T, dU, dU))

    for t in range(T):
        k[t] = hyperparams['init_const']
        PSig[t] = hyperparams['init_var']
        cholPSig[t] = np.linalg.cholesky(PSig[t])
        inv_pol_covar[t] = np.linalg.inv(PSig[t])

    return LinearGaussianPolicy(K, k, PSig, cholPSig, inv_pol_covar)
