LQR Forward pass
================

Notation
--------

We use the following quadratic expansion of the costs, value-function and Q-function:

$$ c({\bold x}, {\bold u})=\frac{1}{2}\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}^T{\bold C}\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}+\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}^T{\bold c}+const. $$
$$ $$
$$ V({\bold x})=\frac{1}{2}{\bold x}^T{\bold V}{\bold x}+{\bold x}^T{\bold v}+const. $$
$$ $$
$$ Q({\bold x}, {\bold u})=\frac{1}{2}\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}^T{\bold Q}\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}+\begin{pmatrix}{\bold x}\\{\bold u}\end{pmatrix}^T{\bold q}+const. $$

where

$$ {\bold Q}=\begin{pmatrix}{\bold Q}_{{\bold x},{\bold x}}&{\bold Q}_{{\bold x},{\bold u}}\\{\bold Q}_{{\bold u},{\bold x}}&{\bold Q}_{{\bold u},{\bold u}}\end{pmatrix} $$
$$ $$
$$ {\bold q}=\begin{pmatrix}{\bold q}_{\bold x}\\{\bold q}_{\bold u}\end{pmatrix} $$

The `m` and `v` in `Vm` and `vv` etc. stand for 'matrix' and 'vector'.

Code
----

The backward recursion function takes as arguments
+ `prev_traj_distr`: The policy object of the previous iteration, used get the dimensions of the new policy object
+ `traj_info`: This object contains the dynamics
+ `eta`: Lagrange dual variable; needed to compute the extended cost function

```python
def backward(self, prev_traj_distr, traj_info, eta):
```

We get the number of timesteps and the dimensions of the states and the actions from the previous policy object:

```python
T = prev_traj_distr.T
dimU = prev_traj_distr.dU
dimX = prev_traj_distr.dX
```

We get the quadratic expansion of the cost function and the dynamics:

```python
Cm_ext, cv_ext = self.compute_extended_costs(eta, traj_info, prev_traj_distr)

Fm = traj_info.dynamics.Fm
fv = traj_info.dynamics.fv
```

We use slice syntax so that `Qm[index_x, index_u]` means ${\bold Q}_{{\bold x},{\bold u}}$ etc.

```python
index_x = slice(dimX)
index_u = slice(dimX, dimX + dimU)
```

We allocate space for the Value-function and initialize the new policy object:

```python
Vm = np.zeros((T, dimX, dimX))
vv = np.zeros((T, dimX))

traj_distr = prev_traj_distr.nans_like()
```

We iterate over $t$, starting at $T - 1$ (because the last index of an array with $T$ entries is $T - 1$) and going backward:

```python
for t in range(T - 1, -1, -1):
```

At each timestep, we compute the quadratic and the linear coefficients of the Q-Function:

$$ {\bold Q}_t = {\bold C}_t + {\bold F}_t^T{\bold V}_{t+1}{\bold F}_t $$
$$ {\bold q}_t = {\bold c}_t + {\bold F}_t^T({\bold V}_{t+1}{\bold f}_t + {\bold v}_{t+1}) $$

At $t = T$ we have no Value Function available, so ${\bold Q}_t = {\bold C}_t$ and ${\bold q}_t = {\bold c}_t$.

```python
Qm = Cm_ext[t, :, :]
qv = cv_ext[t, :]

if t < T - 1:
    Qm += Fm[t, :, :].T.dot(Vm[t + 1, :, :]).dot(Fm[t, :, :])
    qv += Fm[t, :, :].T.dot(Vm[t + 1, :, :].dot(fv[t, :]) + vv[t + 1, :])

Qm = 0.5 * (Qm + Qm.T)
```

${\bold Q}_t$ is a symmetric matrix, but numerical errors lead to `Qm` being not quite symmetric. To counter these numerical errors we symmetrize `Qm`.

Instead of directly computing ${\bold Q}_{t,{\bold u},{\bold u}}^{-1}$ we use Cholesky decomposition:

```python
U = sp.linalg.cholesky(Qm[index_u, index_u])
L = U.T
```

We calculate ${\bold K}_t = -{\bold Q}_{t,{\bold u},{\bold u}}^{-1}{\bold Q}_{t,{\bold u},{\bold x}}$ and ${\bold k}_t = -{\bold Q}_{t,{\bold u},{\bold u}}^{-1}{\bold q}_{t,{\bold u}}$ and store them in the new `traj_distr` object:

```python
traj_distr.K[t, :, :] = - sp.linalg.solve_triangular(
    U, sp.linalg.solve_triangular(L, Qm[index_u, index_x], lower = True)
)
traj_distr.k[t, :] = - sp.linalg.solve_triangular(
    U, sp.linalg.solve_triangular(L, qv[index_u], lower=True)
)
```

We store the covariance ${\bold \Sigma} = {\bold Q}_{t,{\bold u},{\bold u}}^{-1}$ and also its Cholesky decomposition and its inverse ${\bold \Sigma}^{-1} = {\bold Q}_{t,{\bold u},{\bold u}}$ in the `traj_distr` object:

```python
traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
    U, sp.linalg.solve_triangular(L, np.eye(dimU), lower=True)
)
traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
    traj_distr.pol_covar[t, :, :]
)
traj_distr.inv_pol_covar[t, :, :] = Qm[index_u, index_u]
```

We calculate the quadratic and the linear coefficients of the Value-function, which are used in the next iteration:

$$ {\bold V}_t = {\bold Q}_{t,{\bold x},{\bold x}} + {\bold Q}_{t,{\bold x},{\bold u}}{\bold K}_t $$
$$ {\bold v}_t = {\bold q}_{t,{\bold x}} + {\bold Q}_{t,{\bold x},{\bold u}}{\bold k}_t $$

```python
Vm[t, :, :] = Qm[index_x, index_x] + \
              Qm[index_x, index_u].dot(traj_distr.K[t, :, :])
Vm[t, :, :] = 0.5 * (Vm[t, :, :] + Vm[t, :, :].T)
vv[t, :] = qv[index_x] + Qm[index_x, index_u].dot(traj_distr.k[t, :])
```

After that, the loop ends.

Finally, we return the new `traj_distr` object:

```python
return traj_distr
```