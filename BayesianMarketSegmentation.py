"""
Variational Inference Tool for Bayesian Nonparametric Market Segmentation
"""

import numpy as np

from scipy import stats, special

from tqdm import tqdm_notebook as tqdm
import copy


class BaseDirichletMixture:
    """
    Base class of Dirichlet Process Mixture Models for Variational Inference
    """

    def __init__(self, s=np.array([1e-5, 1e-5])):
        """
        params:
        s : array-like, shape=(2, )
            hyperparameters of hyper (Gamma) prior for Dirichlet Process concentration parameter α
        """
        self.s = np.array(s)
        self.τ = self._Params()

    class _Params:
        def __init__(self):
            pass

    def VI(
        self, T, maxiter=100, tol=1e-3, limits=1e-10, τ=None, γ=None, w=None, leave=True
    ):
        """
        Mean-field Variational Inference with Coordinate Ascent Algorithm (Blei and Jordan 2006)
        -----------------------
        params:
        T : int
            truncation level
        maxiter : int
            number of iteration
        tol : float
            convergence threshold
        limits : float
            lower bound to treat as zero
        """
        self.T = int(T)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.limits = float(limits)

        self.ELBO = np.zeros(int(maxiter))
        self._init(τ, γ, w)

        with tqdm(total=self.maxiter, leave=leave) as pbar:
            for i in range(self.maxiter):
                # update params
                self._updateφ()
                self._updateτ()
                self._updateγ()
                self._updatew()

                # calc ELBO
                self.ELBO[i] = self._calc_ELBO()

                pbar.update(1)
                pbar.set_postfix(ELBO=self.ELBO[i])

        self._calc_π()
        if self.ELBO[-1] - self.ELBO[-2] < self.tol:
            self.converge = True
        else:
            self.converge = False

    def VI_II(
        self,
        T,
        maxiter=100,
        burn_in=10,
        burn_in_iter=10,
        tol=1e-3,
        limits=1e-10,
        τ=None,
        γ=None,
        w=None,
        leave=True,
    ):
        """
        Mean-field Variational Inference with Coordinate Ascent Algorithm with Incremental Initialization (Blei and Jordan 2006)
        -----------------------
        params:
        T : int
            truncation level
        maxiter : int
            number of iteration
        burn_in : int
            number of incrementation for initialization
        burn_in_iter : int
            number of iteration for each increment step
        tol : float
            convergence threshold
        limits : float
            lower bound to treat as zero
        """
        self.T = int(T)
        self.maxiter = int(maxiter)
        self.burn_in = int(burn_in)
        self.burn_in_iter = int(burn_in_iter)
        self.tol = float(tol)
        self.limits = float(limits)

        self.ELBO = np.zeros(int(maxiter))
        self._init(τ, γ, w)

        with tqdm(
            total=self.burn_in * self.burn_in_iter + self.maxiter,
            desc="initialize",
            leave=leave,
        ) as pbar:
            for i in range(self.burn_in):
                submodel = self.__class__(
                    *self._choose_subsample(n_size=int(self.N / self.burn_in * (i + 1)))
                )
                submodel.s = self.s
                submodel.τ = self.τ

                submodel.T = self.T
                submodel.limits = self.limits
                submodel._init(τ, γ, w)
                for j in range(self.burn_in_iter):
                    # update params
                    submodel._updateφ()
                    submodel._updateτ()
                    submodel._updateγ()
                    submodel._updatew()

                    pbar.update(1)

                τ = submodel.τ
                self.τ = τ
                γ = submodel.γ
                w = submodel.w

            pbar.set_description("inference")
            self._init(τ, γ, w)
            for i in range(self.maxiter):
                # update params
                self._updateφ()
                self._updateτ()
                self._updateγ()
                self._updatew()

                # calc ELBO
                self.ELBO[i] = self._calc_ELBO()

                pbar.update(1)
                pbar.set_postfix(ELBO=self.ELBO[i])

        self._calc_π()
        if self.ELBO[-1] - self.ELBO[-2] < self.tol:
            self.converge = True
        else:
            self.converge = False

    def _choose_subsample(self, n_size):
        """
        return subsamples
        """
        idx = np.random.choice(np.arange(self.N, dtype=int), size=n_size)
        return (self.X[idx, :, 0],)  # rewrite if it is needed

    def _calc_ELBO(self):
        """
        calculate Evidence Lower Bound
        """
        elbo = 0
        mask = self._z_ > self.limits  # treat responsibility as zero

        # logp(X|)
        elbo += np.sum(self._z_[mask] * self._logx_[mask])
        # logp(Z|)
        _z_ = np.zeros_like(self._z_)
        _z_[mask] = self._z_[mask]
        elbo += np.sum(
            [
                np.sum(_z_[:, i + 1 :], axis=1) * self._log1_v_[i]
                + _z_[:, i] * self._logv_[i]
                for i in range(self.T - 1)
            ]
        )
        # logp(v|)
        elbo += np.sum((self._α_ - 1) * self._log1_v_ + self._logα_)
        # logp(η|)
        elbo += self._calc_evidence_pη()
        # logp(α|)
        elbo += (
            (self.s[0] - 1) * self._logα_
            - self.s[1] * self._α_
            + self.s[0] * np.log(self.s[1])
            - special.loggamma(self.s[0])
        )

        # logq(Z)
        elbo -= np.sum(self._z_[mask] * np.log(self.φ[mask]))
        # logq(η)
        elbo -= self._calc_evidence_qη()
        # logq(v)
        elbo -= np.sum(
            (self.γ[:, 0] - 1) * self._logv_
            + (self.γ[:, 1] - 1) * self._log1_v_
            + special.loggamma(self.γ.sum(axis=1))
            - special.loggamma(self.γ[:, 0])
            - special.loggamma(self.γ[:, 1])
        )
        # logq(α)
        elbo -= (
            (self.w[0] - 1) * self._logα_
            - self.w[1] * self._α_
            + self.w[0] * np.log(self.w[1])
            - special.loggamma(self.w[0])
        )

        return elbo

    def _init(self, τ=None, γ=None, w=None):
        """
        initialize variational parameters
        """
        self._initτ(τ)

        if γ is None:
            self.γ = np.random.uniform(size=(self.T - 1, 2))
        else:
            self.γ = γ
        if w is None:
            self.w = np.random.uniform(size=2)
        else:
            self.w = w

        # mean statistics
        self._v_ = self.γ[:, 0] / np.sum(self.γ, axis=1)
        self._logv_ = special.digamma(self.γ[:, 0]) - special.digamma(
            np.sum(self.γ, axis=1)
        )
        self._log1_v_ = special.digamma(self.γ[:, 1]) - special.digamma(
            np.sum(self.γ, axis=1)
        )

        self._α_ = self.w[0] / self.w[1]
        self._logα_ = special.digamma(self.w[0]) - np.log(self.w[1])

    def _initτ(self):
        """
        initialize variational parameters of each components
        """
        # write for each models

    def _updateφ(self):
        """
        update variational parameters for categorical distribution (responsibility)
        """
        self.φ = self._responsibility_train()

        # mean parameters
        self._z_ = self.φ.copy()

    def _updateτ(self, τ):
        """
        update variational parameters for observation distribution
        """
        # write for each models
        if τ is None:
            pass
        else:
            self.τ = τ

        # mean parameters

        self._logx_ = self._calc_log_likelihood(self.X)

    def _updateγ(self):
        """
        update variational parameters for beta distribution (prior of π)
        """
        self.γ[:, 0] = np.sum(self._z_, axis=0)[:-1] + 1
        for t in range(self.T - 1):
            self.γ[t, 1] = self._α_ + np.sum(self._z_[:, t + 1 :])

        # mean paramters
        self._v_ = self.γ[:, 0] / np.sum(self.γ, axis=1)
        self._logv_ = special.digamma(self.γ[:, 0]) - special.digamma(
            np.sum(self.γ, axis=1)
        )
        self._log1_v_ = special.digamma(self.γ[:, 1]) - special.digamma(
            np.sum(self.γ, axis=1)
        )

    def _updatew(self):
        """
        update variational parameters for gamma distribution (prior of α)
        """
        self.w[0] = self.s[0] + self.T - 1
        self.w[1] = self.s[1] - np.sum(self._log1_v_)

        # mean parameters
        self._α_ = self.w[0] / self.w[1]
        self._logα_ = special.digamma(self.w[0]) - np.log(self.w[1])

    def _calc_log_likelihood(self, X):
        """
        calculate log likelihood of each datum on each cluster
        """
        # write for each models

    def _calc_evidence_pη(self):
        """
        calculate E[log p(η)]
        """
        # write for each models

    def _calc_evidence_qη(self):
        """
        calculate E[log q(η)]
        """
        # write for each models

    def _responsibility_train(self):
        """
        calc responsibility of each training datum
        """
        φ = self._logx_.copy()
        φ[:, :-1] += self._logv_
        φ[:, 1:] += np.cumsum(self._log1_v_)
        φ -= special.logsumexp(φ, axis=1, keepdims=True)
        φ = np.exp(φ)
        return φ

    def responsibility(self, X):
        """
        calc responsibility of each datum
        """
        φ = self._calc_log_likelihood(X)  # rewrite if it is needed
        φ[:, :-1] += self._logv_
        φ[:, 1:] += np.cumsum(self._log1_v_)
        φ -= special.logsumexp(φ, axis=1, keepdims=True)
        φ = np.exp(φ)
        return φ

    def _calc_π(self):
        self.π = np.ones(self.T)
        self.π[:-1] = self._v_
        self.π[1:] *= np.cumprod(1 - self._v_)

    def _inverse(self, A):
        """
        calculate inversed matrix of A with cholesky decomposition
        """
        L = np.linalg.inv(np.linalg.cholesky(A))
        return L.T @ L


def IterativeVI(
    model, trial, T, maxiter=100, tol=1e-3, limits=1e-10, τ=None, γ=None, w=None
):
    score = -np.inf
    with tqdm(total=trial) as pbar:
        for i in range(trial):
            m = copy.deepcopy(model)
            m.VI(T, maxiter, tol, limits, τ, γ, w, leave=False)
            if m.ELBO[-1] > score:
                best = m
                score = m.ELBO[-1]
            pbar.set_postfix(ELBO=score)
            pbar.update(1)
    return best


def IterativeVI_II(
    model,
    trial,
    T,
    maxiter=100,
    burn_in=10,
    burn_in_iter=10,
    tol=1e-3,
    limits=1e-10,
    τ=None,
    γ=None,
    w=None,
):
    score = -np.inf
    with tqdm(total=trial) as pbar:
        for i in range(trial):
            m = copy.deepcopy(model)
            m.VI_II(
                T, maxiter, burn_in, burn_in_iter, tol, limits, τ, γ, w, leave=False
            )
            if m.ELBO[-1] > score:
                best = m
                score = m.ELBO[-1]
            pbar.set_postfix(ELBO=score)
            pbar.update(1)
    return best


class IGRMM(BaseDirichletMixture):
    """
    Infinite Gaussian Regression Mixture Model (IGRMM)
    """

    def __init__(
        self,
        X,
        Exog,
        Y,
        s=np.array([1e-5, 1e-5]),
        m=None,
        β=1e-5,
        ν=None,
        W=None,
        a=np.array([1e-2, 1e-4]),
        b=np.array([1e-2, 1e-4]),
    ):
        """
        params:
        # Data
        X : array-like, shape=(n_samples, n_features)
            input data (multivariate gaussian)
        Exog : array-like, shape=(n_samples, n_features)
            exogenous variables
        Y : array-like, shape=(n_samples,)
            endogenous variable

        # Dirichlet Hyperprior
        s : array-like, shape=(2,)
            hyperparameters of hyper (Gamma) prior for Dirichlet Process concentration parameter α

        # Gaussian Hyperprior
        m : array-like, shape=(n_features, )
            hyperparmeter of mean parameter μ
        β : float (positive)
            hyperparmeter of mean parameter μ
            NOTE: it should be small to make prior vague
        ν : float (larger than (n_features - 1))
            hyperparameter of precision parameter Λ
            NOTE: the smaller nu is, the closer to maximum likelihood the model is
        W : array-like, shape=(n_features, n_features) (positive definite)
            hyperparameter of precision parameter Λ
            NOTE: it should be large to make prior vague

        # Regression Hyperprior
        a : array-like (positive), shape=(2,)
            hyperparameters of noise precision parameter lambda
            shape and rate parameters of gamma distribution
        b : array-like (positive), shape=(2,)
            hyperparameters of ADR prior
            shape and rate parameters of gamma distribution
        """
        super().__init__(s)

        self.N, self.D = X.shape
        self.X = np.array(X).reshape((self.N, self.D, 1))
        self.Y = np.array(Y)
        self.D_exog = Exog.shape[1]
        self.Exog = np.array(Exog).reshape((self.N, self.D_exog, 1))

        # for components distribution
        self.τ.Gauss = super()._Params()
        if m is None:
            self.τ.Gauss.m = np.mean(self.X, axis=0)
        else:
            self.τ.Gauss.m = np.array(m).reshape((self.D, 1))
        self.τ.Gauss.β = float(β)
        if ν is None:
            self.τ.Gauss.ν = self.D
        else:
            self.τ.Gauss.ν = float(ν)
        if W is None:
            self.τ.Gauss.W = (
                super()._inverse(np.cov(X.T).reshape((self.D, self.D)))
                / self.τ.Gauss.ν
                * 1e3
            )
        else:
            self.τ.Gauss.W = np.array(W.reshape((self.D, self.D)))
        self.τ.Gauss.W_inv = super()._inverse(self.τ.Gauss.W)

        # for components regression
        self.τ.Regre = super()._Params()
        self.τ.Regre.a = np.array(a)
        self.τ.Regre.b = np.array(b)

    def _choose_subsample(self, n_size):
        """
        return subsamples
        """
        idx = np.random.choice(np.arange(self.N, dtype=int), size=n_size)
        return (
            self.X[idx, :, 0],
            self.Exog[idx, :, 0],
            self.Y[idx],
        )  # rewrite if it is needed

    def _initτ(self, τ):
        """
        initialize variational parameters of each components
        """
        if τ is None:
            select = np.random.choice(np.arange(self.N), size=self.T)
            # Gaussian
            self.τ.Gauss.β_ = np.array([self.τ.Gauss.β] * self.T) + 1
            self.τ.Gauss.m_ = np.array(
                [
                    (self.X[select[t]] + self.τ.Gauss.β * self.τ.Gauss.m)
                    / self.τ.Gauss.β_[t]
                    for t in range(self.T)
                ]
            )
            self.τ.Gauss.ν_ = np.array([self.τ.Gauss.ν] * self.T) + 1
            self.τ.Gauss.W_inv_ = np.array(
                [
                    self.X[select[t]] @ self.X[select[t]].T
                    + self.τ.Gauss.β * self.τ.Gauss.m @ self.τ.Gauss.m.T
                    - self.τ.Gauss.β_[t] * self.τ.Gauss.m_[t] @ self.τ.Gauss.m_[t].T
                    + self.τ.Gauss.W_inv
                    for t in range(self.T)
                ]
            )
            self.τ.Gauss.W_ = np.linalg.inv(self.τ.Gauss.W_inv_)

            # Regression
            self.τ.Regre.C_ = (
                self.Exog[select] @ self.Exog[select].swapaxes(1, 2)
                + (np.eye(self.D_exog) * self.τ.Regre.b[1] / self.τ.Regre.b[0])[
                    np.newaxis
                ]
            )
            self.τ.Regre.C_inv_ = np.linalg.inv(self.τ.Regre.C_)
            self.τ.Regre.ω_ = np.array(
                [
                    self.τ.Regre.C_inv_[t] @ (self.Y[select[t]] * self.Exog[select[t]])
                    for t in range(self.T)
                ]
            )

            self.τ.Regre.a_ = np.empty((self.T, 2))
            self.τ.Regre.a_[:, 0] = np.array([self.τ.Regre.a[0] + 0.5] * self.T)
            self.τ.Regre.a_[:, 1] = np.array(
                [
                    self.τ.Regre.a[1]
                    + 0.5
                    * (
                        self.Y[select[t]] ** 2
                        - self.τ.Regre.ω_[t].T @ self.τ.Regre.C_[t] @ self.τ.Regre.ω_[t]
                    )
                    for t in range(self.T)
                ]
            ).flatten()

            self.τ.Regre.b_ = np.empty((self.T, self.D_exog, 2))
            self.τ.Regre.b_[:, :, 0] = self.τ.Regre.b[0] + 0.5
            self.τ.Regre.b_[:, :, 1] = self.τ.Regre.b[1] + 0.5 * (
                (self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1])[:, np.newaxis]
                * self.τ.Regre.ω_[:, :, 0] ** 2
                + np.array([np.diag(self.τ.Regre.C_inv_[t]) for t in range(self.T)])
            )
        else:
            self.τ = τ

        # mean parameters
        # Gaussian
        self._Λ_ = self.τ.Gauss.ν_[:, np.newaxis, np.newaxis] * self.τ.Gauss.W_
        self._logΛ_ = (
            np.sum(
                np.array(
                    [
                        special.digamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
                        for d in range(1, self.D + 1)
                    ]
                ),
                axis=0,
            )
            + self.D * np.log(2 * np.pi)
            + np.linalg.slogdet(self.τ.Gauss.W_)[1]
        )
        self._Λμ_ = np.array(
            [
                self.τ.Gauss.ν_[t] * self.τ.Gauss.W_[t] @ self.τ.Gauss.m_[t]
                for t in range(self.T)
            ]
        )
        self._μTΛμ_ = np.array(
            [
                self.τ.Gauss.ν_[t]
                * self.τ.Gauss.m_[t].T
                @ self.τ.Gauss.W_[t]
                @ self.τ.Gauss.m_[t]
                + self.D / self.τ.Gauss.β_[t]
                for t in range(self.T)
            ]
        )

        # Regression
        self._λh2_ = np.array(
            [
                self.τ.Regre.a_[t, 0] / self.τ.Regre.a_[t, 1] * self.τ.Regre.ω_[t] ** 2
                + np.diag(self.τ.Regre.C_inv_[t])[:, np.newaxis]
                for t in range(self.T)
            ]
        )
        self._λhhT_ = np.array(
            [
                self.τ.Regre.a_[t, 0]
                / self.τ.Regre.a_[t, 1]
                * self.τ.Regre.ω_[t]
                @ self.τ.Regre.ω_[t].T
                + self.τ.Regre.C_inv_[t]
                for t in range(self.T)
            ]
        )
        self._h_ = self.τ.Regre.ω_.copy()
        self._λ_ = self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1]
        self._logλ_ = special.digamma(self.τ.Regre.a_[:, 0]) - np.log(
            self.τ.Regre.a_[:, 1]
        )

        self._c_ = self.τ.Regre.b_[:, :, 0] / self.τ.Regre.b_[:, :, 1]
        self._logc_ = special.digamma(self.τ.Regre.b_[:, :, 0]) - np.log(
            self.τ.Regre.b_[:, :, 1]
        )

        self._logx_ = self._calc_log_likelihood(self.X, self.Exog, self.Y)

    def _updateτ(self):
        """
        update variational parameters for observation distribution
        """
        # Gaussian
        self.τ.Gauss.β_ = np.sum(self._z_, axis=0) + self.τ.Gauss.β
        self.τ.Gauss.ν_ = np.sum(self._z_, axis=0) + self.τ.Gauss.ν
        for t in range(self.T):
            mask = self._z_[:, t] > self.limits
            self.τ.Gauss.m_[t] = (
                np.sum(self._z_[mask, t, np.newaxis, np.newaxis] * self.X[mask], axis=0)
                + self.τ.Gauss.β * self.τ.Gauss.m
            ) / self.τ.Gauss.β_[t]
            self.τ.Gauss.W_inv_[t] = (
                np.sum(
                    self._z_[mask, t, np.newaxis, np.newaxis]
                    * self.X[mask]
                    @ self.X[mask].swapaxes(1, 2),
                    axis=0,
                )
                + self.τ.Gauss.β * self.τ.Gauss.m @ self.τ.Gauss.m.T
                - self.τ.Gauss.β_[t] * self.τ.Gauss.m_[t] @ self.τ.Gauss.m_[t].T
                + self.τ.Gauss.W_inv
            )
        self.τ.Gauss.W_ = np.linalg.inv(self.τ.Gauss.W_inv_)

        # mean parameters
        self._Λ_ = self.τ.Gauss.ν_[:, np.newaxis, np.newaxis] * self.τ.Gauss.W_
        digamma = np.zeros(self.T)
        for d in range(1, self.D + 1):
            digamma += special.digamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
        self._logΛ_ = (
            digamma + self.D * np.log(2 * np.pi) + np.linalg.slogdet(self.τ.Gauss.W_)[1]
        )
        for t in range(self.T):
            self._Λμ_[t] = self.τ.Gauss.ν_[t] * self.τ.Gauss.W_[t] @ self.τ.Gauss.m_[t]
            self._μTΛμ_[t] = (
                self.τ.Gauss.ν_[t]
                * self.τ.Gauss.m_[t].T
                @ self.τ.Gauss.W_[t]
                @ self.τ.Gauss.m_[t]
                + self.D / self.τ.Gauss.β_[t]
            )

        # Regression
        for t in range(self.T):
            # coefficients
            mask = self._z_[:, t] > self.limits
            self.τ.Regre.C_[t] = np.sum(
                self._z_[mask, t, np.newaxis, np.newaxis]
                * (self.Exog[mask] @ self.Exog[mask].swapaxes(1, 2)),
                axis=0,
            ) + np.diag(self._c_[t])
            self.τ.Regre.C_inv_[t] = super()._inverse(self.τ.Regre.C_[t])
            self.τ.Regre.ω_[t] = self.τ.Regre.C_inv_[t] @ np.sum(
                (self._z_[mask, t] * self.Y[mask])[:, np.newaxis, np.newaxis]
                * self.Exog[mask],
                axis=0,
            )

            # precision
            self.τ.Regre.a_[t, 0] = self.τ.Regre.a[0] + 0.5 * np.sum(self._z_[:, t])
            self.τ.Regre.a_[t, 1] = self.τ.Regre.a[1] + 0.5 * (
                np.sum(self._z_[mask, t] * self.Y[mask] ** 2)
                - self.τ.Regre.ω_[t].T @ self.τ.Regre.C_[t] @ self.τ.Regre.ω_[t]
            )

            self._λh2_[t] = (
                self.τ.Regre.a_[t, 0] / self.τ.Regre.a_[t, 1] * self.τ.Regre.ω_[t] ** 2
                + np.diag(self.τ.Regre.C_inv_[t])[:, np.newaxis]
            )
            self._λhhT_[t] = (
                self.τ.Regre.a_[t, 0]
                / self.τ.Regre.a_[t, 1]
                * self.τ.Regre.ω_[t]
                @ self.τ.Regre.ω_[t].T
                + self.τ.Regre.C_inv_[t]
            )
        self._h_ = self.τ.Regre.ω_.copy()
        self._λ_ = self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1]
        self._logλ_ = special.digamma(self.τ.Regre.a_[:, 0]) - np.log(
            self.τ.Regre.a_[:, 1]
        )

        # ADR
        self.τ.Regre.b_[:, :, 0] = self.τ.Regre.b[0] + 0.5
        self.τ.Regre.b_[:, :, 1] = self.τ.Regre.b[1] + 0.5 * self._λh2_.reshape(
            (self.T, self.D_exog)
        )

        self._c_ = self.τ.Regre.b_[:, :, 0] / self.τ.Regre.b_[:, :, 1]
        self._logc_ = special.digamma(self.τ.Regre.b_[:, :, 0]) - np.log(
            self.τ.Regre.b_[:, :, 1]
        )

        self._logx_ = self._calc_log_likelihood(self.X, self.Exog, self.Y)

    def _calc_log_likelihood(self, X, Exog=None, Y=None):
        """
        calculate log likelihood of each datum on each cluster
        """
        N = len(X)
        logx = np.empty((N, self.T))
        if Y is None:
            for n in range(N):
                for t in range(self.T):
                    # Gaussian
                    logx[n, t] = -0.5 * (
                        X[n].T @ self._Λ_[t] @ X[n]
                        - 2 * X[n].T @ self._Λμ_[t]
                        + self._μTΛμ_[t]
                        - self._logΛ_[t]
                        + self.D * np.log(2 * np.pi)
                    )
        else:
            for n in range(N):
                for t in range(self.T):
                    # Gaussian
                    logx[n, t] = -0.5 * (
                        X[n].T @ self._Λ_[t] @ X[n]
                        - 2 * X[n].T @ self._Λμ_[t]
                        + self._μTΛμ_[t]
                        - self._logΛ_[t]
                        + self.D * np.log(2 * np.pi)
                    )

                    # Regression
                    logx[n, t] += -0.5 * (
                        self._λ_[t] * Y[n] ** 2
                        - 2 * Y[n] * (Exog[n].T @ (self._λ_[t] * self._h_[t]))
                        + Exog[n].T @ self._λhhT_[t] @ Exog[n]
                        - self._logλ_[t]
                        + np.log(2 * np.pi)
                    )
        return logx

    def responsibility(self, X, Exog=None, Y=None):
        """
        calc responsibility of each datum
        """
        φ = self._calc_log_likelihood(X, Exog, Y)  # rewrite if it is needed
        φ[:, :-1] += self._logv_
        φ[:, 1:] += np.cumsum(self._log1_v_)
        φ -= special.logsumexp(φ, axis=1, keepdims=True)
        φ = np.exp(φ)
        return φ

    def _calc_evidence_pη(self):
        """
        calculate E[log p(η)]
        """
        l = 0
        for t in range(self.T):
            # Gaussian
            l += -0.5 * (
                self.τ.Gauss.β
                * (
                    self._μTΛμ_[t]
                    - 2 * self.τ.Gauss.m.T @ self._Λμ_[t]
                    + self.τ.Gauss.m.T @ self._Λ_[t] @ self.τ.Gauss.m
                )
                - self.D * np.log(self.τ.Gauss.β)
                - self._logΛ_[t]
                + self.D * np.log(2 * np.pi)
            )
            l += 0.5 * (self.τ.Gauss.ν + self.D - 1) * self._logΛ_[t] - 0.5 * np.trace(
                self.τ.Gauss.W_inv @ self._Λ_[t]
            )
            l += -0.5 * (
                self.τ.Gauss.ν * np.linalg.slogdet(self.τ.Gauss.W)[1]
                - self.τ.Gauss.ν * self.D * np.log(2)
                - self.D * (self.D - 1) * np.log(np.pi)
                - self.D * special.loggamma(0.5 * (self.τ.Gauss.ν + self.D - 1))
            )

            # Regression
            l += -0.5 * (
                np.trace(np.diag(self._c_[t]) @ self._λhhT_[t])
                - self.D * self._logλ_[t]
                - np.sum(self._logc_[t])
                + self.D * np.log(2 * np.pi)
            )
            l += (
                (self.τ.Regre.a[0] - 1) * self._logλ_[t]
                - self.τ.Regre.a[1] * self._λ_[t]
                + self.τ.Regre.a[0] * np.log(self.τ.Regre.a[1])
                - special.loggamma(self.τ.Regre.a[0])
            )
            l += np.sum(
                (self.τ.Regre.b[0] - 1) * self._logc_[t]
                - self.τ.Regre.b[1] * self._c_[t]
                + self.τ.Regre.b[0] * np.log(self.τ.Regre.b[1])
                - special.loggamma(self.τ.Regre.b[0])
            )
        return float(l)

    def _calc_evidence_qη(self):
        """
        calculate E[log q(η)]
        """
        # Gaussian
        l = 0.5 * (
            self.D * np.log(self.τ.Gauss.β_)
            + self._logΛ_
            - self.D * (np.log(2 * np.pi) + 1)
        )
        l += (
            0.5 * (self.τ.Gauss.ν_ - self.D - 1) * self._logΛ_
            - 0.5 * self.τ.Gauss.ν_ * self.D
        )
        l += (
            -0.5 * self.τ.Gauss.ν_ * np.linalg.slogdet(self.τ.Gauss.W_)[1]
            - 0.5 * self.τ.Gauss.ν_ * self.D * np.log(2)
            - 0.25 * self.D * (self.D - 1) * np.log(np.pi)
        )
        l += -np.sum(
            [
                special.loggamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
                for d in range(1, self.D + 1)
            ],
            axis=0,
        )

        # Regression
        l += (
            self.D * 0.5 * (self._logλ_ - np.log(2 * np.pi) - 1)
            + 0.5 * np.linalg.slogdet(self.τ.Regre.C_)[1]
        )
        l += (
            (self.τ.Regre.a_[:, 0] - 1) * special.digamma(self.τ.Regre.a_[:, 0])
            + np.log(self.τ.Regre.a_[:, 1])
            - self.τ.Regre.a_[:, 0]
            - special.loggamma(self.τ.Regre.a_[:, 0])
        )
        l += np.sum(
            (self.τ.Regre.b_[:, :, 0] - 1) * special.digamma(self.τ.Regre.b_[:, :, 0])
            + np.log(self.τ.Regre.b_[:, :, 1])
            - self.τ.Regre.b_[:, :, 0]
            - special.loggamma(self.τ.Regre.b_[:, :, 0]),
            axis=1,
        )
        return np.sum(l)


class IGBRMM(BaseDirichletMixture):
    """
    Infinite Gaussian Bernoulli Regression Mixture Model (IGBRMM)
    """

    def __init__(
        self,
        X,
        dummy,
        Exog,
        Y,
        s=np.array([1e-5, 1e-5]),
        m=None,
        β=1e-5,
        ν=None,
        W=None,
        ζ=np.ones(2),
        a=np.array([1e-2, 1e-4]),
        b=np.array([1e-2, 1e-4]),
    ):
        """
        params:
        # Data
        X : array-like, shape=(n_samples, n_features)
            input data (multivariate gaussian)
        dummy : array-like, shape=(n_samples, n_features)
            dummy input data (bernoulli)
        Exog : array-like, shape=(n_samples, n_features)
            exogenous variables
        Y : array-like, shape=(n_samples,)
            endogenous variable

        # Dirichlet Hyperprior
        s : array-like, shape=(2,)
            hyperparameters of hyper (Gamma) prior for Dirichlet Process concentration parameter α

        # Gaussian Hyperprior
        m : array-like, shape=(n_features, )
            hyperparmeter of mean parameter μ
        β : float (positive)
            hyperparmeter of mean parameter μ
            NOTE: it should be small to make prior vague
        ν : float (larger than (n_features - 1))
            hyperparameter of precision parameter Λ
            NOTE: the smaller nu is, the closer to maximum likelihood the model is
        W : array-like, shape=(n_features, n_features) (positive definite)
            hyperparameter of precision parameter Λ
            NOTE: it should be large to make prior vague

        # Bernoulli Hyperprior
        ζ : array-like, shape=(n_features,2)
            hyperparameter of dummy mean parameters (shape parameters of beta distribution)
            if ζ=1, hyperprior is uniform distribution

        # Regression Hyperprior
        a : array-like (positive), shape=(2,)
            hyperparameters of noise precision parameter lambda
            shape and rate parameters of gamma distribution
        b : array-like (positive), shape=(2,)
            hyperparameters of ADR prior
            shape and rate parameters of gamma distribution
        """
        super().__init__(s)

        self.N, self.D = X.shape
        self.X = np.array(X).reshape((self.N, self.D, 1))
        if len(dummy.shape) == 1:
            self.D_dummy = 1
        else:
            self.D_dummy = dummy.shape[1]
        self.dummy = np.array(dummy).reshape((self.N, self.D_dummy))
        self.Y = np.array(Y)
        self.D_exog = Exog.shape[1]
        self.Exog = np.array(Exog).reshape((self.N, self.D_exog, 1))

        # for components distribution
        # Gaussian
        self.τ.Gauss = super()._Params()
        if m is None:
            self.τ.Gauss.m = np.mean(self.X, axis=0)
        else:
            self.τ.Gauss.m = np.array(m).reshape((self.D, 1))
        self.τ.Gauss.β = float(β)
        if ν is None:
            self.τ.Gauss.ν = self.D
        else:
            self.τ.Gauss.ν = float(ν)
        if W is None:
            self.τ.Gauss.W = (
                super()._inverse(np.cov(X.T).reshape((self.D, self.D)))
                / self.τ.Gauss.ν
                * 1e3
            )
        else:
            self.τ.Gauss.W = np.array(W.reshape((self.D, self.D)))
        self.τ.Gauss.W_inv = super()._inverse(self.τ.Gauss.W)

        # Bernoulli
        self.τ.Bern = super()._Params()
        self.τ.Bern.ζ = ζ

        # for components regression
        self.τ.Regre = super()._Params()
        self.τ.Regre.a = np.array(a)
        self.τ.Regre.b = np.array(b)

    def _choose_subsample(self, n_size):
        """
        return subsamples
        """
        idx = np.random.choice(np.arange(self.N, dtype=int), size=n_size)
        return (
            self.X[idx, :, 0],
            self.dummy[idx, :],
            self.Exog[idx, :, 0],
            self.Y[idx],
        )  # rewrite if it is needed

    def _initτ(self, τ):
        """
        initialize variational parameters of each components
        """
        if τ is None:
            select = np.random.choice(np.arange(self.N), size=self.T)
            # Gaussian
            self.τ.Gauss.β_ = np.array([self.τ.Gauss.β] * self.T) + 1
            self.τ.Gauss.m_ = np.array(
                [
                    (self.X[select[t]] + self.τ.Gauss.β * self.τ.Gauss.m)
                    / self.τ.Gauss.β_[t]
                    for t in range(self.T)
                ]
            )
            self.τ.Gauss.ν_ = np.array([self.τ.Gauss.ν] * self.T) + 1
            self.τ.Gauss.W_inv_ = np.array(
                [
                    self.X[select[t]] @ self.X[select[t]].T
                    + self.τ.Gauss.β * self.τ.Gauss.m @ self.τ.Gauss.m.T
                    - self.τ.Gauss.β_[t] * self.τ.Gauss.m_[t] @ self.τ.Gauss.m_[t].T
                    + self.τ.Gauss.W_inv
                    for t in range(self.T)
                ]
            )
            self.τ.Gauss.W_ = np.linalg.inv(self.τ.Gauss.W_inv_)

            # Bernoulli
            self.τ.Bern.ζ_ = np.empty((self.T, self.D_dummy, 2))
            self.τ.Bern.ζ_[:, :, 0] = self.τ.Bern.ζ[0] + self.dummy[select]
            self.τ.Bern.ζ_[:, :, 1] = self.τ.Bern.ζ[1] + 1 - self.dummy[select]

            # Regression
            self.τ.Regre.C_ = (
                self.Exog[select] @ self.Exog[select].swapaxes(1, 2)
                + (np.eye(self.D_exog) * self.τ.Regre.b[1] / self.τ.Regre.b[0])[
                    np.newaxis
                ]
            )
            self.τ.Regre.C_inv_ = np.linalg.inv(self.τ.Regre.C_)
            self.τ.Regre.ω_ = np.array(
                [
                    self.τ.Regre.C_inv_[t] @ (self.Y[select[t]] * self.Exog[select[t]])
                    for t in range(self.T)
                ]
            )

            self.τ.Regre.a_ = np.empty((self.T, 2))
            self.τ.Regre.a_[:, 0] = np.array([self.τ.Regre.a[0] + 0.5] * self.T)
            self.τ.Regre.a_[:, 1] = np.array(
                [
                    self.τ.Regre.a[1]
                    + 0.5
                    * (
                        self.Y[select[t]] ** 2
                        - self.τ.Regre.ω_[t].T @ self.τ.Regre.C_[t] @ self.τ.Regre.ω_[t]
                    )
                    for t in range(self.T)
                ]
            ).flatten()

            self.τ.Regre.b_ = np.empty((self.T, self.D_exog, 2))
            self.τ.Regre.b_[:, :, 0] = self.τ.Regre.b[0] + 0.5
            self.τ.Regre.b_[:, :, 1] = self.τ.Regre.b[1] + 0.5 * (
                (self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1])[:, np.newaxis]
                * self.τ.Regre.ω_[:, :, 0] ** 2
                + np.array([np.diag(self.τ.Regre.C_inv_[t]) for t in range(self.T)])
            )
        else:
            self.τ = τ

        # mean parameters
        # Gaussian
        self._Λ_ = self.τ.Gauss.ν_[:, np.newaxis, np.newaxis] * self.τ.Gauss.W_
        self._logΛ_ = (
            np.sum(
                np.array(
                    [
                        special.digamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
                        for d in range(1, self.D + 1)
                    ]
                ),
                axis=0,
            )
            + self.D * np.log(2 * np.pi)
            + np.linalg.slogdet(self.τ.Gauss.W_)[1]
        )
        self._Λμ_ = np.array(
            [
                self.τ.Gauss.ν_[t] * self.τ.Gauss.W_[t] @ self.τ.Gauss.m_[t]
                for t in range(self.T)
            ]
        )
        self._μTΛμ_ = np.array(
            [
                self.τ.Gauss.ν_[t]
                * self.τ.Gauss.m_[t].T
                @ self.τ.Gauss.W_[t]
                @ self.τ.Gauss.m_[t]
                + self.D / self.τ.Gauss.β_[t]
                for t in range(self.T)
            ]
        )

        # Bernoulli
        self._ρ_ = self.τ.Bern.ζ_[:, :, 0] / np.sum(self.τ.Bern.ζ_, axis=2)
        self._logρ_ = special.digamma(self.τ.Bern.ζ_[:, :, 0]) - special.digamma(
            np.sum(self.τ.Bern.ζ_, axis=2)
        )
        self._log1_ρ_ = special.digamma(self.τ.Bern.ζ_[:, :, 1]) - special.digamma(
            np.sum(self.τ.Bern.ζ_, axis=2)
        )

        # Regression
        self._λh2_ = np.array(
            [
                self.τ.Regre.a_[t, 0] / self.τ.Regre.a_[t, 1] * self.τ.Regre.ω_[t] ** 2
                + np.diag(self.τ.Regre.C_inv_[t])[:, np.newaxis]
                for t in range(self.T)
            ]
        )
        self._λhhT_ = np.array(
            [
                self.τ.Regre.a_[t, 0]
                / self.τ.Regre.a_[t, 1]
                * self.τ.Regre.ω_[t]
                @ self.τ.Regre.ω_[t].T
                + self.τ.Regre.C_inv_[t]
                for t in range(self.T)
            ]
        )
        self._h_ = self.τ.Regre.ω_.copy()
        self._λ_ = self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1]
        self._logλ_ = special.digamma(self.τ.Regre.a_[:, 0]) - np.log(
            self.τ.Regre.a_[:, 1]
        )

        self._c_ = self.τ.Regre.b_[:, :, 0] / self.τ.Regre.b_[:, :, 1]
        self._logc_ = special.digamma(self.τ.Regre.b_[:, :, 0]) - np.log(
            self.τ.Regre.b_[:, :, 1]
        )

        self._logx_ = self._calc_log_likelihood(self.X, self.Exog, self.Y)

    def _updateτ(self):
        """
        update variational parameters for observation distribution
        """
        # Gaussian
        self.τ.Gauss.β_ = np.sum(self._z_, axis=0) + self.τ.Gauss.β
        self.τ.Gauss.ν_ = np.sum(self._z_, axis=0) + self.τ.Gauss.ν
        for t in range(self.T):
            mask = self._z_[:, t] > self.limits
            self.τ.Gauss.m_[t] = (
                np.sum(self._z_[mask, t, np.newaxis, np.newaxis] * self.X[mask], axis=0)
                + self.τ.Gauss.β * self.τ.Gauss.m
            ) / self.τ.Gauss.β_[t]
            self.τ.Gauss.W_inv_[t] = (
                np.sum(
                    self._z_[mask, t, np.newaxis, np.newaxis]
                    * self.X[mask]
                    @ self.X[mask].swapaxes(1, 2),
                    axis=0,
                )
                + self.τ.Gauss.β * self.τ.Gauss.m @ self.τ.Gauss.m.T
                - self.τ.Gauss.β_[t] * self.τ.Gauss.m_[t] @ self.τ.Gauss.m_[t].T
                + self.τ.Gauss.W_inv
            )
        self.τ.Gauss.W_ = np.linalg.inv(self.τ.Gauss.W_inv_)

        # mean parameters
        self._Λ_ = self.τ.Gauss.ν_[:, np.newaxis, np.newaxis] * self.τ.Gauss.W_
        digamma = np.zeros(self.T)
        for d in range(1, self.D + 1):
            digamma += special.digamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
        self._logΛ_ = (
            digamma + self.D * np.log(2 * np.pi) + np.linalg.slogdet(self.τ.Gauss.W_)[1]
        )
        for t in range(self.T):
            self._Λμ_[t] = self.τ.Gauss.ν_[t] * self.τ.Gauss.W_[t] @ self.τ.Gauss.m_[t]
            self._μTΛμ_[t] = (
                self.τ.Gauss.ν_[t]
                * self.τ.Gauss.m_[t].T
                @ self.τ.Gauss.W_[t]
                @ self.τ.Gauss.m_[t]
                + self.D / self.τ.Gauss.β_[t]
            )

        # Bernoulli
        zx = self._z_.T @ self.dummy
        self.τ.Bern.ζ_[:, :, 0] = self.τ.Bern.ζ[0] + zx
        self.τ.Bern.ζ_[:, :, 1] = (self.τ.Bern.ζ[1] + np.sum(self._z_, axis=0))[
            :, np.newaxis
        ] - zx

        # mean parameters
        self._ρ_ = self.τ.Bern.ζ_[:, :, 0] / np.sum(self.τ.Bern.ζ_, axis=2)
        self._logρ_ = special.digamma(self.τ.Bern.ζ_[:, :, 0]) - special.digamma(
            np.sum(self.τ.Bern.ζ_, axis=2)
        )
        self._log1_ρ_ = special.digamma(self.τ.Bern.ζ_[:, :, 1]) - special.digamma(
            np.sum(self.τ.Bern.ζ_, axis=2)
        )

        # Regression
        for t in range(self.T):
            # coefficients
            mask = self._z_[:, t] > self.limits
            self.τ.Regre.C_[t] = np.sum(
                self._z_[mask, t, np.newaxis, np.newaxis]
                * (self.Exog[mask] @ self.Exog[mask].swapaxes(1, 2)),
                axis=0,
            ) + np.diag(self._c_[t])
            self.τ.Regre.C_inv_[t] = super()._inverse(self.τ.Regre.C_[t])
            self.τ.Regre.ω_[t] = self.τ.Regre.C_inv_[t] @ np.sum(
                (self._z_[mask, t] * self.Y[mask])[:, np.newaxis, np.newaxis]
                * self.Exog[mask],
                axis=0,
            )

            # precision
            self.τ.Regre.a_[t, 0] = self.τ.Regre.a[0] + 0.5 * np.sum(self._z_[:, t])
            self.τ.Regre.a_[t, 1] = self.τ.Regre.a[1] + 0.5 * (
                np.sum(self._z_[mask, t] * self.Y[mask] ** 2)
                - self.τ.Regre.ω_[t].T @ self.τ.Regre.C_[t] @ self.τ.Regre.ω_[t]
            )

            self._λh2_[t] = (
                self.τ.Regre.a_[t, 0] / self.τ.Regre.a_[t, 1] * self.τ.Regre.ω_[t] ** 2
                + np.diag(self.τ.Regre.C_inv_[t])[:, np.newaxis]
            )
            self._λhhT_[t] = (
                self.τ.Regre.a_[t, 0]
                / self.τ.Regre.a_[t, 1]
                * self.τ.Regre.ω_[t]
                @ self.τ.Regre.ω_[t].T
                + self.τ.Regre.C_inv_[t]
            )
        self._h_ = self.τ.Regre.ω_.copy()
        self._λ_ = self.τ.Regre.a_[:, 0] / self.τ.Regre.a_[:, 1]
        self._logλ_ = special.digamma(self.τ.Regre.a_[:, 0]) - np.log(
            self.τ.Regre.a_[:, 1]
        )

        # ADR
        self.τ.Regre.b_[:, :, 0] = self.τ.Regre.b[0] + 0.5
        self.τ.Regre.b_[:, :, 1] = self.τ.Regre.b[1] + 0.5 * self._λh2_.reshape(
            (self.T, self.D_exog)
        )

        self._c_ = self.τ.Regre.b_[:, :, 0] / self.τ.Regre.b_[:, :, 1]
        self._logc_ = special.digamma(self.τ.Regre.b_[:, :, 0]) - np.log(
            self.τ.Regre.b_[:, :, 1]
        )

        self._logx_ = self._calc_log_likelihood(self.X, self.dummy, self.Exog, self.Y)

    def _calc_log_likelihood(self, X, dummy, Exog=None, Y=None):
        """
        calculate log likelihood of each datum on each cluster
        """
        N = len(X)
        logx = np.empty((N, self.T))
        if Y is None:
            for n in range(N):
                for t in range(self.T):
                    # Gaussian
                    logx[n, t] = -0.5 * (
                        X[n].T @ self._Λ_[t] @ X[n]
                        - 2 * X[n].T @ self._Λμ_[t]
                        + self._μTΛμ_[t]
                        - self._logΛ_[t]
                        + self.D * np.log(2 * np.pi)
                    )
                    # Bernoulli
                    logx[n, t] += np.sum(
                        dummy[n] * self._logρ_[t] + (1 - dummy[n]) * self._log1_ρ_[t]
                    )
        else:
            for n in range(N):
                for t in range(self.T):
                    # Gaussian
                    logx[n, t] = -0.5 * (
                        X[n].T @ self._Λ_[t] @ X[n]
                        - 2 * X[n].T @ self._Λμ_[t]
                        + self._μTΛμ_[t]
                        - self._logΛ_[t]
                        + self.D * np.log(2 * np.pi)
                    )
                    # Bernoulli
                    logx[n, t] += np.sum(
                        dummy[n] * self._logρ_[t] + (1 - dummy[n]) * self._log1_ρ_[t]
                    )
                    # Regression
                    logx[n, t] += -0.5 * (
                        self._λ_[t] * Y[n] ** 2
                        - 2 * Y[n] * (Exog[n].T @ (self._λ_[t] * self._h_[t]))
                        + Exog[n].T @ self._λhhT_[t] @ Exog[n]
                        - self._logλ_[t]
                        + np.log(2 * np.pi)
                    )
        return logx

    def responsibility(self, X, dummy, Exog=None, Y=None):
        """
        calc responsibility of each datum
        """
        φ = self._calc_log_likelihood(X, dummy, Exog, Y)  # rewrite if it is needed
        φ[:, :-1] += self._logv_
        φ[:, 1:] += np.cumsum(self._log1_v_)
        φ -= special.logsumexp(φ, axis=1, keepdims=True)
        φ = np.exp(φ)
        return φ

    def _calc_evidence_pη(self):
        """
        calculate E[log p(η)]
        """
        l = 0
        for t in range(self.T):
            # Gaussian
            l += -0.5 * (
                self.τ.Gauss.β
                * (
                    self._μTΛμ_[t]
                    - 2 * self.τ.Gauss.m.T @ self._Λμ_[t]
                    + self.τ.Gauss.m.T @ self._Λ_[t] @ self.τ.Gauss.m
                )
                - self.D * np.log(self.τ.Gauss.β)
                - self._logΛ_[t]
                + self.D * np.log(2 * np.pi)
            )
            l += 0.5 * (self.τ.Gauss.ν + self.D - 1) * self._logΛ_[t] - 0.5 * np.trace(
                self.τ.Gauss.W_inv @ self._Λ_[t]
            )
            l += -0.5 * (
                self.τ.Gauss.ν * np.linalg.slogdet(self.τ.Gauss.W)[1]
                - self.τ.Gauss.ν * self.D * np.log(2)
                - self.D * (self.D - 1) * np.log(np.pi)
                - self.D * special.loggamma(0.5 * (self.τ.Gauss.ν + self.D - 1))
            )

            # Bernoulli
            l += np.sum(
                (self.τ.Bern.ζ[0] - 1)[np.newaxis] * self._logρ_[t]
                + (self.τ.Bern.ζ[1] - 1)[np.newaxis] * self._log1_ρ_[t]
            )
            l += (
                special.loggamma(np.sum(self.τ.Bern.ζ))
                - special.loggamma(self.τ.Bern.ζ[0])
                - special.loggamma(self.τ.Bern.ζ[1])
            )

            # Regression
            l += -0.5 * (
                np.trace(np.diag(self._c_[t]) @ self._λhhT_[t])
                - self.D * self._logλ_[t]
                - np.sum(self._logc_[t])
                + self.D * np.log(2 * np.pi)
            )
            l += (
                (self.τ.Regre.a[0] - 1) * self._logλ_[t]
                - self.τ.Regre.a[1] * self._λ_[t]
                + self.τ.Regre.a[0] * np.log(self.τ.Regre.a[1])
                - special.loggamma(self.τ.Regre.a[0])
            )
            l += np.sum(
                (self.τ.Regre.b[0] - 1) * self._logc_[t]
                - self.τ.Regre.b[1] * self._c_[t]
                + self.τ.Regre.b[0] * np.log(self.τ.Regre.b[1])
                - special.loggamma(self.τ.Regre.b[0])
            )
        return float(l)

    def _calc_evidence_qη(self):
        """
        calculate E[log q(η)]
        """
        # Gaussian
        l = 0.5 * (
            self.D * np.log(self.τ.Gauss.β_)
            + self._logΛ_
            - self.D * (np.log(2 * np.pi) + 1)
        )
        l += (
            0.5 * (self.τ.Gauss.ν_ - self.D - 1) * self._logΛ_
            - 0.5 * self.τ.Gauss.ν_ * self.D
        )
        l += (
            -0.5 * self.τ.Gauss.ν_ * np.linalg.slogdet(self.τ.Gauss.W_)[1]
            - 0.5 * self.τ.Gauss.ν_ * self.D * np.log(2)
            - 0.25 * self.D * (self.D - 1) * np.log(np.pi)
        )
        l += -np.sum(
            [
                special.loggamma(0.5 * (self.τ.Gauss.ν_ + 1 - d))
                for d in range(1, self.D + 1)
            ],
            axis=0,
        )

        # Bernoulli
        l += np.sum(
            (self.τ.Bern.ζ_[:, :, 0] - 1) * special.digamma(self.τ.Bern.ζ_[:, :, 0])
            + (self.τ.Bern.ζ_[:, :, 1] - 1) * special.digamma(self.τ.Bern.ζ_[:, :, 1]),
            axis=1,
        )
        l += -np.sum(
            (np.sum(self.τ.Bern.ζ_, axis=2) - 2)
            * special.digamma(np.sum(self.τ.Bern.ζ_, axis=2)),
            axis=1,
        )
        l += np.sum(
            special.loggamma(np.sum(self.τ.Bern.ζ_, axis=2))
            - special.loggamma(self.τ.Bern.ζ_[:, :, 0])
            - special.loggamma(self.τ.Bern.ζ_[:, :, 1]),
            axis=1,
        )

        # Regression
        l += (
            self.D * 0.5 * (self._logλ_ - np.log(2 * np.pi) - 1)
            + 0.5 * np.linalg.slogdet(self.τ.Regre.C_)[1]
        )
        l += (
            (self.τ.Regre.a_[:, 0] - 1) * special.digamma(self.τ.Regre.a_[:, 0])
            + np.log(self.τ.Regre.a_[:, 1])
            - self.τ.Regre.a_[:, 0]
            - special.loggamma(self.τ.Regre.a_[:, 0])
        )
        l += np.sum(
            (self.τ.Regre.b_[:, :, 0] - 1) * special.digamma(self.τ.Regre.b_[:, :, 0])
            + np.log(self.τ.Regre.b_[:, :, 1])
            - self.τ.Regre.b_[:, :, 0]
            - special.loggamma(self.τ.Regre.b_[:, :, 0]),
            axis=1,
        )
        return np.sum(l)


NISHI = IGBRMM
