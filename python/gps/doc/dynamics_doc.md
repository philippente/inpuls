Dynamics fitting
================

Notation
--------

In this part we consider a normal distribution over

$$ {\bold y}=\begin{pmatrix}{\bold x}\\{\bold u}\\{\bold x}'\end{pmatrix} $$

where ${\bold x}'$ is the state that results from taking the action ${\bold u}$ in state ${\bold x}$. We divide the parameters of the normal distribution ${\bold y}\ \sim\ {\mathscr N}({\bold \mu}, {\bold \Sigma})$ as follows:

$$ {\bold \mu}=\begin{pmatrix}{\bold \mu}_{{\bold x}{\bold u}}\\{\bold \mu}_{{\bold x}'}\end{pmatrix} $$
$$ $$
$$ {\bold \Sigma}=\begin{pmatrix}{\bold \Sigma}_{{\bold x}{\bold u},{\bold x}{\bold u}}&{\bold \Sigma}_{{\bold x}{\bold u},{\bold x}'}\\{\bold \Sigma}_{{\bold x}',{\bold x}{\bold u}}&{\bold \Sigma}_{{\bold x}',{\bold x}'}\end{pmatrix} $$

Code
----

The fitting function takes as arguments:
+ `X`: States of the trajectory samples from the previous policy
+ `U`: Actions of the trajectory samples from the previous policy

```python
def fit(self, X, U):
```

We get the number of samples, the number of timesteps and the dimensions of the states and the actions from the policy object:

```python
N, T, dimX = X.shape
dimU = U.shape[2]
```

We use slice syntax so that `sigma[index_xu, index_x]` means ${\bold \Sigma}_{{\bold x}{\bold u},{\bold x}'}$ etc.

```python
index_xu = slice(dimX + dimU)
index_x = slice(dimX + dimU, dimX + dimU + dimX)
```

We obtain the regularization term for the covariance:

```python
sig_reg = np.zeros((dimX + dimU + dimX, dimX + dimU + dimX))
sig_reg[index_xu, index_xu] = self._hyperparams['regularization']
```

We compute the weight vector and matrix, used to compute sample mean and sample covariance:

```python
dwts = (1.0 / N) * np.ones(N)
D = np.diag((1.0 / (N - 1)) * np.ones(N))
```

We allocate space for ${\bold F}$, ${\bold f}$ and ${\bold \Sigma}_{dyn}$:

```python
self.Fm = np.zeros([T, dimX, dimX + dimU])
self.fv = np.zeros([T, dimX]) 
self.dyn_covar = np.zeros([T, dimX, dimX])
```

We iterate over $t$ and assemble

$$ {\bold y}_t^n=\begin{pmatrix}{\bold x}_t^n\\{\bold u}_t^n\\{\bold x}_{t+1}^n\end{pmatrix} $$

where the superscript $n$ denotes the number of the sample.

```python
for t in range(T - 1): 
    Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]
```

We obtain the hyperparameters of the normal-inverse-Wishart prior $NIW(\mu_0,\Phi,m,n_0)$

```python
mu0, Phi, mm, n0 = self.prior.eval(dimX, dimU, Ys)
```

We compute the empirical mean and empirical covariance

$$ {\bold \mu}_{emp,t}=\frac{1}{N}\sum_{n=1}^N{\bold y}_t^n $$
$$ {\bold \Sigma}_{emp,t}=\frac{1}{N-1}\sum_{n=1}^N({\bold y}_t^n-{\bold \mu}_{emp,t})({\bold y}_t^n-{\bold \mu}_{emp,t})^T $$

```python
empmu = np.sum((Ys.T * dwts).T, axis=0)
diff = Ys - empmu
empsig = diff.T.dot(D).dot(diff)                                       
empsig = 0.5 * (empsig + empsig.T)
```

We use the empirical mean as our estimated mean and use the normal-inverse-Wishart posterior to get the estimate for the covariance:

$$ {\bold \mu}_t={\bold \mu}_{emp,t} $$
$$ $$
$$ {\bold \Sigma}_t=\frac{{\bold \Phi}+(N-1){\bold \Sigma}_{emp,t}+\frac{Nm}{N+m}({\bold \mu}_{emp,t}-{\bold \mu}_0)({\bold \mu}_{emp,t}-{\bold \mu}_0)^T}{N+n_0} $$

```python
mu = empmu 
sigma = (Phi + (N - 1) * empsig + (N * mm) / (N + mm) *
    np.outer(empmu - mu0, empmu - mu0)) / (N + n0) 
sigma = 0.5 * (sigma + sigma.T)
sigma += sig_reg
```

${\bold \Sigma}_t$ can contain singularities so that its inverse contains infinities. To prevent that we add a small regularization term.

Now we condition the gaussian on $x$ and $u$:

$$ {\bold F} = {\bold \Sigma}_{x',xu}{\bold \Sigma}_{xu,xu}^{-1} = ({\bold \Sigma}_{xu,xu}^{-1}{\bold \Sigma}_{xu,x'})^T $$
$$ {\bold f} = {\bold \mu}_x-{\bold F}{\bold \mu}_{xu} $$
$$ {\bold \Sigma}_{dyn} = {\bold \Sigma}_{x',x'}-{\bold F}{\bold \Sigma}_{xu,xu}{\bold F}^T $$

```python
Fm = np.linalg.solve(sigma[index_xu, index_xu],
                     sigma[index_xu, index_x]).T
fv = mu[index_x] - Fm.dot(mu[index_xu])
dyn_covar = sigma[index_x, index_x] - Fm.dot(sigma[index_xu, index_xu]).dot(Fm.T)
dyn_covar = 0.5 * (dyn_covar + dyn_covar.T) 
```

We store ${\bold F}$, ${\bold f}$ and ${\bold \Sigma}_{dyn}$:

```python
self.Fm[t, :, :] = Fm
self.fv[t, :] = fv
self.dyn_covar[t, :, :] = dyn_covar
```

After that, the loop ends and we return ${\bold F}$, ${\bold f}$ and ${\bold \Sigma}_{dyn}$:

```python
return self.Fm, self.fv, self.dyn_covar
```