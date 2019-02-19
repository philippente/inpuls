import numpy as np
import gym

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.agent.openai_gym.agent_openai_gym import is_goal_based


def init_gym_pol(hyperparams):
    env = gym.make(hyperparams['env'])

    if is_goal_based(env):
        dX = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    else:
        dX = env.observation_space.shape[0]
    dU = env.action_space.shape[0]
    T = hyperparams['T']

    low = np.asarray(env.action_space.low)
    high = np.array(env.action_space.high)

    K = np.zeros((T, dU, dX))
    k = np.empty((T, dU))
    PSig = np.empty((T, dU, dU))
    cholPSig = np.empty((T, dU, dU))
    inv_pol_covar = np.empty((T, dU, dU))

    for t in range(T):
        k[t] = (low+high)/2
        PSig[t] = np.diag(np.square(high - low)/12) * hyperparams['init_var_scale']
        cholPSig[t] = np.linalg.cholesky(PSig[t])
        inv_pol_covar[t] = np.linalg.inv(PSig[t])

    return LinearGaussianPolicy(K, k, PSig, cholPSig, inv_pol_covar)
