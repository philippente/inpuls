import numpy as np
import sys

sys.path.append('python')

from gps.visualization.latent_space import visualize_latent_space_tsne

beta_kl = 0
data = np.load(
    'experiments/gym_fetchreach_mdggcs/data_files/fetchreach_gmrgps-static-initial_step0.1-M4-5s-T20-K8-kl%r-gmr_pol_lin-dropout0.1-lqr_pol_669/plot_vae_latent_space-00.npz'
    % beta_kl
)

x_train = data['x_train']
z_mean_train = data['z_mean_train']
z_std_train = data['z_std_train']
x_test = data['x_test']
z_mean_test = data['z_mean_test']
z_std_test = data['z_std_test']

S = 25
N_train, dZ = z_mean_train.shape
N_test, _ = z_mean_test.shape

z_train = np.empty((N_train, S, dZ))
for n in range(N_train):
    z_train[n] = np.random.multivariate_normal(z_mean_train[n], np.diag(np.square(z_std_train[n])), S)

z_test = np.empty((N_test, S, dZ))
for n in range(N_test):
    z_test[n] = np.random.multivariate_normal(z_mean_test[n], np.diag(np.square(z_std_test[n])), S)

visualize_latent_space_tsne('plots/vae_latent_space-kl%r' % beta_kl, x_train, z_train, x_test, z_test, show=True)
