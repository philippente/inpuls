"""
RFGPS = MDGPS + TAC

"""

import math

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.ndimage.filters import maximum_filter1d
from scipy.special import logsumexp

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS

class AlgorithmRFGPS(AlgorithmMDGPS):

    def __init__(self, hyperparams):
        config = hyperparams
        AlgorithmMDGPS.__init__(self, config)
        self.min_samples_per_cluster = config['tac']['min_samples_per_cluster']
        self.initial_clustering = config['tac']['initial_clustering']
        self.random_resets = config['tac']['random_resets']
        self.max_em_iterations = config['tac']['max_em_iterations']
        self.pol_lin = config['tac']['policy_linearization']
        self.init_policy = True
        self.prior_only = config['tac']['prior_only']
        self.dyn_covar_smooth = config['tac']['dyn_covar_smooth']

        # Global dynamics and policy priors
        self.__share_global_priors()

    def iteration(self, sample_lists, itr, train_gcm=False):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
            _: to match parent class
        """
        # Get all samples
        samples = [sample
                   for i in range(len(sample_lists))
                   for sample in sample_lists[i].get_samples()]

        # Split longer trajectories in shorter segements
        if samples[0].T > self.T:
            assert samples[0].T % self.T == 0
            samples[0].agent.T = self.T # Fake new T
            new_samples = []
            for sample in samples:
                for i in range(samples[0].T/self.T):
                    new_sample = Sample(sample.agent)
                    for sensor in sample._data: # Split data
                        new_sample._data[sensor] = sample._data[sensor][i*self.T:(i+1)*self.T]
                    new_samples.append(new_sample)
            samples = new_samples

        self.N = len(samples)
        print("itr", itr, "N: ", self.N, "M: ", self.M)
        assert self.min_samples_per_cluster * self.M <= self.N

        X = np.asarray([sample.get_X() for sample in samples])
        U = np.asarray([sample.get_U() for sample in samples])

        # Update global dynamics prior
        self.dynamics_prior.update(X, U)

        # Store end effector points for visualization
        self.eeps = [s.get(END_EFFECTOR_POINTS) for s in samples]

        # Cluster samples
        clusterings = self.tac(samples, self.initial_clustering)
        for i in range(self.random_resets):
            clusterings.extend(self.tac(samples, 'random'))
        self.responsibilitieses = [c[0] for c in clusterings] # Store for export
        # Select clustering with maximal likelihood
        self._assign_samples(samples, clusterings[np.argmax([c[1] for c in clusterings])][0])
        self.m_step(for_tac=False) #Fit linearizations again, but this time also using the local trajectories

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _advance_iteration_variables(self):
        super(AlgorithmRFGPS, self)._advance_iteration_variables()
        
        # This step is repeated each iteration as deep copy of traj_info/pol_info object breaks sharing
        self.__share_global_priors()

    def __share_global_priors(self):
        """
        Replaces priors for each condition with shared global dynamics and policy priors
        """
        self.dynamics_prior = self.cur[0].traj_info.dynamics.prior # Use prior of first condition
        #self.policy_prior = self.cur[0].pol_info.policy_prior      # Use prior of first condition
        for m in range(1, self.M):                                 # Overwrite priors of other conditions
            self.cur[m].traj_info.dynamics.prior = self.dynamics_prior
            #self.cur[m].pol_info.policy_prior = self.policy_prior

    def _assign_samples(self, samples, responsibilities):
        """
        Assigns samples to clusters by their responsibilities.

        """
        for m in range(self.M):
            self.cur[m].sample_list = SampleList([samples[i] for i in range(self.N) if responsibilities[i] == m])

    def m_step(self, for_tac=False):
        """
        Performs local dynamics and policy fitting.

        Args:
            responsibilities: Assignment of samples to clusters
        """
        # Evaluate the costs.
        for m in range(self.M):
            self._eval_cost(m)

        # Update dynamics linearizations.
        if for_tac:
            self._update_dynamics(update_prior=False, prior_only=self.prior_only)
            self.trace = np.sum([np.trace(self.cur[m].traj_info.dynamics.dyn_covar[t]) for m in range(self.M) for t in range(self.T-1)]) / (self.M * self.T)
            if self.dyn_covar_smooth > 1: # Smooth dyn_covar
                for m in range(self.M):
                    smooth_covars(self.cur[m].traj_info.dynamics.dyn_covar[:-1], self.dyn_covar_smooth)
        else:
            self._update_dynamics(update_prior=False, prior_only=False)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.init_policy:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy()
            self.init_policy = False

        # Update policy linearizations.
        if not for_tac:
            for m in range(self.M):
                self._update_policy_fit(m)

    def tac(self, samples, initial_clustering):
        logpdfs = None
        if self.iteration_count == 0 or initial_clustering == 'random':
            # Assign samples randomly
            responsibilities = self._random_responsibilities()
        elif initial_clustering == 'prev_clusters':
            # Assign to most likely cluster of prev iteration
            logpdfs = self._logpdfs(samples, prev=True)
            responsibilities = np.argmax(logpdfs, axis=1)
        elif initial_clustering == 'fixed':
            responsibilities = np.repeat(np.arange(self.M), math.ceil(float(self.N) / self.M))
        else:
            raise NotImplementedError(initial_clustering)

        # Perform EM
        iterations = 0
        results = [] # Also return intermediate clusterings as due to the empty cluster prevention the likelihood increase may not be monotonic
        prev_responsibilities = set()
        while True:
            # Check for empty clusters
            cluster_sizes = np.asarray([np.count_nonzero(responsibilities == m) for m in range(self.M)])
            undersized_clusters = cluster_sizes < self.min_samples_per_cluster
            if np.any(undersized_clusters):
                oversized_clusters = cluster_sizes > self.min_samples_per_cluster
                available = [i for i in range(self.N) if oversized_clusters[responsibilities[i]]]
                if logpdfs is None:  # Assign random trajectory to empty cluster
                    # By choosng only from oversized clusters, this can't cause an other cluster to become undersized
                    choosen = np.random.choice(available)
                else: # Assign sample with lowest likelihood in current cluster to empty cluster
                    cur_logpdfs = [logpdfs[i, responsibilities[i]] for i in range(self.N)]
                    cur_logpdfs_sorted = np.sort([cur_logpdfs[i] for i in available])
                    # Give n tickets to the lowest likelihood, n-1 to the second lowest, etc.
                    p = len(available) - np.asarray([np.where(cur_logpdfs_sorted == cur_logpdfs[a])[0][0] for a in available], dtype=float) 
                    p /= np.sum(p) # Normalize probabilities
                    choosen = np.random.choice(available, p=p)
                    # TODO Add cycle check here
                responsibilities[choosen] = np.arange(self.M)[undersized_clusters][0]
                continue # Restart loop to ensure no cluster is undersized
                
            # Check if resposibilities were already tried before (cycle detection)
            if tuple(responsibilities) in prev_responsibilities:
                print("TAC did not converge: Cycle detected")
                break # EM did not converge, but best result can be used anyway
            prev_responsibilities.add(tuple(responsibilities))

            self._assign_samples(samples, responsibilities)

            ### M-step ###
            self.m_step(for_tac=True) # Use only prior so the local dynamics/policy don't overfit to the samples

            ### E-step ###
            logpdfs = self._logpdfs(samples)
            new_responsibilities = np.argmax(logpdfs, axis=1)
            ll = self._logpdf_clustering(responsibilities, logpdfs)
            new_ll = self._logpdf_clustering(new_responsibilities, logpdfs)
            changed = not np.array_equal(new_responsibilities, responsibilities)

            results.append([responsibilities, ll])

            print("changed", changed, "ll", ll, " => ", new_ll, "tr", self.trace)
            if not changed:
                break # EM converged

            iterations += 1
            if iterations >= self.max_em_iterations:
                print("TAC did not converge: Too many iterations")
                break # EM did not converge, but best result can be used anyway
            responsibilities = new_responsibilities
        return results

    def _logpdfs(self, samples, prev=False):
        logpdfs = np.empty([len(samples), self.M], dtype='longdouble')
        for i, sample in enumerate(samples):
            X = sample.get_X()
            U = sample.get_U()
            for m in range(self.M):
                logpdfs[i, m] = self._logpdf_traj(X, U, m, prev)
            #print("--------------------------------------")
        return logpdfs

    def _logpdf_clustering(self, responsibilities, logpdfs):
        return np.sum([logpdfs[n, responsibilities[n]]-logsumexp(logpdfs[n]) for n in range(self.N)]) / self.N

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
            r = range(self.T-1)

        # Select trajectory distribution of current or previous iteration
        traj_info = self.cur[m].traj_info if not prev else self.prev[m].traj_info
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
                    mean=Fm[:, :self.dX].dot(X[t]) + Fm[:, self.dX:].dot(U[t]) + fv,
                    cov=dyn_covar)
        else:
            logpdf_pol, logpdf_dyn = np.zeros(0), np.zeros(0)

        logpdf = logpdf_x0 + np.sum(logpdf_pol) + np.sum(logpdf_dyn)
        #print("logpdf", logpdf, "x0", logpdf_x0, "pol", np.sum(logpdf_pol), "dyn", np.sum(logpdf_dyn))
        return logpdf

    def _random_responsibilities(self):
        return np.random.permutation(np.repeat(np.arange(self.M), math.ceil(float(self.N) / self.M)))[:self.N]

def smooth_covars(cov, size):
    T = cov.shape[0]
    dX = cov.shape[1]
    traces = np.empty([T])
    for t in range(T):
        traces[t] = np.trace(cov[t])
    smoothed_traces = maximum_filter1d(traces, size)
    for t in range(T):
        if smoothed_traces[t] > traces[t]:
            cov[t] += np.eye(dX) * (smoothed_traces[t] - traces[t]) / dX
