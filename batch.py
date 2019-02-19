from os import system
from sys import argv

experiment = argv[1]
itr = int(argv[2])


def set_seed(seed):
    system('sed -i "/random_seed\':/c\\ \\ \\ \\ \'random_seed\': %d," experiments/%s/hyperparams.py' % (seed, experiment))


def set_initial_clustering(initial_clustering):
    system('sed -i "/\'initial_clustering\':/c\\ \\ \\ \\ \'initial_clustering\': \'%s\'," experiments/%s/hyperparams.py' % (initial_clustering, experiment))


def set_em_iterations(em_iterations):
    system('sed -i "/\'max_em_iterations\':/c\\ \\ \\ \\ \'max_em_iterations\': %d," experiments/%s/hyperparams.py' % (em_iterations, experiment))


def set_prior_only(prior_only):
    system('sed -i "/\'prior_only\':/c\\ \\ \\ \\ \'prior_only\': %r," experiments/%s/hyperparams.py' % (prior_only, experiment))


def set_tac_policy(tac_policy):
    system('sed -i "/\'tac_policy\':/c\\ \\ \\ \\ \'tac_policy\': %r," experiments/%s/hyperparams.py' % (tac_policy, experiment))


def run():
    system('python python/gps/main.py %s -s' % experiment)


# Baseline
set_initial_clustering('random')
set_em_iterations(0)
set_tac_policy(False)
set_prior_only(False)
for i in range(itr):
    set_seed(i)
    run()

# LocalGlobal
set_initial_clustering('prev_clusters')
set_em_iterations(20)
for i in range(itr):
    set_seed(i)
    run()

# LocalGlobal + Tac Pol
set_tac_policy(True)
for i in range(itr):
    set_seed(i)
    run()

# Global
set_tac_policy(False)
set_prior_only(True)
for i in range(itr):
    set_seed(i)
    run()

# Global + Tac pol
set_tac_policy(True)
for i in range(itr):
    set_seed(i)
    run()
