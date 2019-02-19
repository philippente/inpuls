LQR Forward pass
================

Notation
--------

In this part we consider a normal distribution over ${\bold x}$ and ${\bold u}$:

$$ \begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}\ \sim\ {\mathscr N}({\bold \mu}, {\bold \Sigma}) $$

where

$$ {\bold \mu}=\begin{pmatrix}{\bold \mu}_{\bold x}\\{\bold \mu}_{\bold u}\end{pmatrix} $$
$$ $$
$$ {\bold \Sigma}=\begin{pmatrix}{\bold \Sigma}_{{\bold x},{\bold x}}&{\bold \Sigma}_{{\bold x},{\bold u}}\\{\bold \Sigma}_{{\bold u},{\bold x}}&{\bold \Sigma}_{{\bold u},{\bold u}}\end{pmatrix} $$

Code
----

The forward recursion function takes as arguments
+ `traj_distr`: The policy object containing the linear gaussian policy
+ `traj_info`: This object contains the dynamics

```python
def forward(self, traj_distr, traj_info):
```

We get the number of timesteps and the dimensions of the states and the actions from the policy object:

```python
T = traj_distr.T
dimU = traj_distr.dU
dimX = traj_distr.dX
```

We use slice syntax so that `sigma[index_x, index_u]` means ${\bold \Sigma}_{{\bold x},{\bold u}}$ etc.

```python
index_x = slice(dimX)
index_u = slice(dimX, dimX + dimU)
```

We allocate space for ${\bold \mu}$ and ${\bold \Sigma}$ and set the initial values for ${\bold \mu}_{0,{\bold x}}$ and ${\bold \Sigma}_{0,{\bold x}{\bold x}}$:

```python
sigma = np.zeros((T, dimX + dimU, dimX + dimU))
mu = np.zeros((T, dimX + dimU))

mu[0, index_x] = traj_info.x0mu
sigma[0, index_x, index_x] = traj_info.x0sigma
```

We iterate over $t$ and compute:

$$ {\bold \mu}_{t,{\bold u}} = {\bold K}_t{\bold \mu}_{t,{\bold x}}+{\bold k}_t $$
$$ {\bold \Sigma}_{t,{\bold x},{\bold u}} = {\bold \Sigma}_{t,{\bold x},{\bold x}}{\bold K}_t^T $$
$$ {\bold \Sigma}_{t,{\bold u},{\bold x}} = {\bold K}_t{\bold \Sigma}_{t,{\bold x},{\bold x}} $$
$$ {\bold \Sigma}_{t,{\bold u}{\bold u}} = {\bold K}_t{\bold \Sigma}_{t,{\bold x},{\bold x}}{\bold K}_t^T+{\bold \Sigma_{pol,t}}$$

```python
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
```

for $t<T$ we compute:

$$ {\bold \mu}_{t+1,{\bold x}}={\bold F}_t{\bold \mu}_t+{\bold f}_t $$
$$ {\bold \Sigma}_{t+1,{\bold x},{\bold x}}={\bold F}_t{\bold \Sigma}_t{\bold F}_t^T+{\bold \Sigma}_{dyn} $$

```python
if t < T - 1:
    mu[t+1, index_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

    sigma[t+1, index_x, index_x] = \
        Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \ 
        dyn_covar[t, :, :]
```

After that the loop ends and we return `mu` and `sigma`:

```python
return mu, sigma
```