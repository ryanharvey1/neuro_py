import numpy as np
from scipy.linalg import LinAlgError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis
from sklearn.utils.validation import check_array, check_is_fitted


class mDLAG(BaseEstimator, TransformerMixin):
    """
    Multi-group Delayed Latents Across Groups (mDLAG) model.

    A dimensionality reduction framework for characterizing the multi-dimensional,
    concurrent flow of signals across multiple groups of time series data.

    Parameters
    ----------
    n_factors : int, default=10
        Number of latent factors per group
    max_delay : int, default=5
        Maximum delay (in time steps) to consider for cross-group interactions
    n_iter : int, default=100
        Maximum number of iterations for EM algorithm
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, RandomState instance or None, default=None
        Random state for reproducible results
    verbose : bool, default=False
        Whether to print convergence information
    reg_lambda : float, default=1e-6
        Regularization parameter for numerical stability

    Attributes
    ----------
    n_groups_ : int
        Number of groups in the data
    n_features_ : list of int
        Number of features per group
    n_timesteps_ : int
        Number of time steps in the data
    loading_matrices_ : list of ndarray
        Loading matrices for each group
    delay_matrices_ : list of list of ndarray
        Delay interaction matrices between groups
    noise_variances_ : list of ndarray
        Noise variances for each group
    latent_factors_ : ndarray
        Estimated latent factors
    log_likelihood_ : float
        Log-likelihood of the fitted model
    """

    def __init__(
        self,
        n_factors=10,
        max_delay=5,
        n_iter=100,
        tol=1e-4,
        random_state=None,
        verbose=False,
        reg_lambda=1e-6,
    ):
        self.n_factors = n_factors
        self.max_delay = max_delay
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.reg_lambda = reg_lambda

    def _validate_input(self, X):
        """Validate input data format."""
        if not isinstance(X, list):
            raise ValueError("X must be a list of arrays, one for each group")

        if len(X) == 0:
            raise ValueError("X cannot be empty")

        # Check that all arrays have the same number of time steps
        n_timesteps = X[0].shape[0]
        for i, x in enumerate(X):
            x = check_array(x, dtype=np.float64)
            if x.shape[0] != n_timesteps:
                raise ValueError(
                    f"All groups must have the same number of timesteps. "
                    f"Group 0 has {n_timesteps}, group {i} has {x.shape[0]}"
                )
            X[i] = x

        return X

    def _initialize_parameters(self, X):
        """Initialize model parameters."""
        np.random.seed(self.random_state)

        self.n_groups_ = len(X)
        self.n_features_ = [x.shape[1] for x in X]
        self.n_timesteps_ = X[0].shape[0]

        # Initialize loading matrices using factor analysis
        self.loading_matrices_ = []
        for i, x in enumerate(X):
            fa = FactorAnalysis(
                n_components=self.n_factors, random_state=self.random_state
            )
            fa.fit(x)
            # Loading matrix should be (n_features, n_factors)
            self.loading_matrices_.append(fa.components_.T)

        # Initialize delay matrices (interactions between groups)
        self.delay_matrices_ = []
        for i in range(self.n_groups_):
            group_delays = []
            for j in range(self.n_groups_):
                if i != j:  # No self-interactions with delay
                    delays = []
                    for d in range(1, self.max_delay + 1):
                        # Small random initialization
                        delay_matrix = 0.1 * np.random.randn(
                            self.n_factors, self.n_factors
                        )
                        delays.append(delay_matrix)
                    group_delays.append(delays)
                else:
                    group_delays.append([])
            self.delay_matrices_.append(group_delays)

        # Initialize noise variances
        self.noise_variances_ = []
        for i, x in enumerate(X):
            # Compute residuals after factor analysis initialization
            fa_reconstruction = (
                x @ self.loading_matrices_[i] @ self.loading_matrices_[i].T
            )
            residuals = x - fa_reconstruction
            noise_var = np.var(residuals, axis=0)
            noise_var = np.maximum(noise_var, self.reg_lambda)  # Ensure positive
            self.noise_variances_.append(noise_var)

        # Initialize latent factors
        self.latent_factors_ = np.random.randn(
            self.n_timesteps_, self.n_groups_ * self.n_factors
        )

    def _e_step(self, X):
        """E-step: Estimate latent factors given parameters."""
        T = self.n_timesteps_
        K = self.n_factors
        G = self.n_groups_

        # Build observation model
        C = np.zeros((sum(self.n_features_), G * K))
        R_inv = np.zeros((sum(self.n_features_), sum(self.n_features_)))

        start_idx = 0
        for i in range(G):
            end_idx = start_idx + self.n_features_[i]
            factor_start = i * K
            factor_end = (i + 1) * K

            C[start_idx:end_idx, factor_start:factor_end] = self.loading_matrices_[i]
            R_inv[start_idx:end_idx, start_idx:end_idx] = np.diag(
                1.0 / self.noise_variances_[i]
            )
            start_idx = end_idx

        # Posterior inference for latent factors
        new_factors = np.zeros_like(self.latent_factors_)

        for t in range(T):
            # Observation at time t
            y_t = np.concatenate([X[i][t] for i in range(G)])

            # Base precision matrix (identity for prior)
            prior_precision = np.eye(G * K)
            prior_mean = np.zeros(G * K)

            # Add delayed interactions as prior information
            for i in range(G):
                for j in range(G):
                    if i != j:
                        for d, delay_matrix in enumerate(self.delay_matrices_[i][j]):
                            tau = d + 1  # delay
                            if t >= tau:  # Only if we have enough history
                                i_idx = slice(i * K, (i + 1) * K)
                                j_idx = slice(j * K, (j + 1) * K)

                                # Add to prior precision
                                prior_precision[i_idx, i_idx] += (
                                    delay_matrix.T @ delay_matrix
                                )

                                # Add to prior mean
                                prior_mean[i_idx] += (
                                    delay_matrix.T
                                    @ delay_matrix
                                    @ delay_matrix
                                    @ self.latent_factors_[t - tau, j_idx]
                                )

            # Posterior precision and mean
            post_precision = prior_precision + C.T @ R_inv @ C
            post_mean_part = C.T @ R_inv @ y_t + prior_mean

            try:
                # Solve for posterior mean
                new_factors[t] = np.linalg.solve(post_precision, post_mean_part)
            except LinAlgError:
                # Fallback with regularization
                post_precision += self.reg_lambda * np.eye(G * K)
                new_factors[t] = np.linalg.solve(post_precision, post_mean_part)

        self.latent_factors_ = new_factors
        return new_factors

    def _m_step(self, X):
        """M-step: Update parameters given latent factors."""
        T = self.n_timesteps_
        K = self.n_factors
        G = self.n_groups_

        # Update loading matrices
        for i in range(G):
            factor_idx = slice(i * K, (i + 1) * K)
            X_i = X[i]
            Z_i = self.latent_factors_[:, factor_idx]

            # Regularized least squares: X_i = Z_i @ C_i.T + noise
            # So C_i = (Z_i.T @ Z_i + reg*I)^(-1) @ Z_i.T @ X_i
            ZTZ = Z_i.T @ Z_i + self.reg_lambda * np.eye(K)
            ZTX = Z_i.T @ X_i
            self.loading_matrices_[i] = np.linalg.solve(ZTZ, ZTX).T

        # Update delay matrices
        for i in range(G):
            for j in range(G):
                if i != j:
                    i_idx = slice(i * K, (i + 1) * K)
                    j_idx = slice(j * K, (j + 1) * K)

                    for d in range(len(self.delay_matrices_[i][j])):
                        tau = d + 1

                        # Collect data for this delay
                        Z_i_delayed = []
                        Z_j_source = []

                        for t in range(tau, T):
                            Z_i_delayed.append(self.latent_factors_[t, i_idx])
                            Z_j_source.append(self.latent_factors_[t - tau, j_idx])

                        if len(Z_i_delayed) > 0:
                            Z_i_delayed = np.array(Z_i_delayed)
                            Z_j_source = np.array(Z_j_source)

                            # Regularized least squares
                            ZjTZj = (
                                Z_j_source.T @ Z_j_source + self.reg_lambda * np.eye(K)
                            )
                            ZjTZi = Z_j_source.T @ Z_i_delayed
                            self.delay_matrices_[i][j][d] = np.linalg.solve(
                                ZjTZj, ZjTZi
                            ).T

        # Update noise variances
        for i in range(G):
            factor_idx = slice(i * K, (i + 1) * K)
            X_i = X[i]
            Z_i = self.latent_factors_[:, factor_idx]

            # Reconstruction: X_i = Z_i @ C_i.T
            reconstruction = Z_i @ self.loading_matrices_[i].T
            residuals = X_i - reconstruction
            self.noise_variances_[i] = np.var(residuals, axis=0)
            self.noise_variances_[i] = np.maximum(
                self.noise_variances_[i], self.reg_lambda
            )

    def _compute_log_likelihood(self, X):
        """Compute log-likelihood of the data given current parameters."""
        log_likelihood = 0.0
        T = self.n_timesteps_

        for t in range(T):
            for i, x_i in enumerate(X):
                factor_idx = slice(i * self.n_factors, (i + 1) * self.n_factors)
                z_i = self.latent_factors_[t, factor_idx]

                # Reconstruction: x_i[t] = C_i @ z_i + noise
                mean = self.loading_matrices_[i] @ z_i
                residual = x_i[t] - mean

                # Log-likelihood contribution
                log_likelihood -= 0.5 * np.sum(residual**2 / self.noise_variances_[i])
                log_likelihood -= 0.5 * np.sum(
                    np.log(2 * np.pi * self.noise_variances_[i])
                )

        # Add prior on latent factors
        log_likelihood -= 0.5 * np.sum(self.latent_factors_**2)
        log_likelihood -= 0.5 * self.latent_factors_.shape[1] * T * np.log(2 * np.pi)

        return log_likelihood

    def fit(self, X, y=None):
        """
        Fit the mDLAG model to data.

        Parameters
        ----------
        X : list of array-like, shape = [n_timesteps, n_features_i]
            List of data matrices, one for each group
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
        """
        X = self._validate_input(X)
        self._initialize_parameters(X)

        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            self._e_step(X)

            # M-step
            self._m_step(X)

            # Compute log-likelihood
            current_log_likelihood = self._compute_log_likelihood(X)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}: Log-likelihood = {current_log_likelihood:.6f}"
                )

            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_log_likelihood = current_log_likelihood

        self.log_likelihood_ = current_log_likelihood
        return self

    def transform(self, X):
        """
        Transform data to latent space.

        Parameters
        ----------
        X : list of array-like, shape = [n_timesteps, n_features_i]
            List of data matrices to transform

        Returns
        -------
        X_transformed : ndarray, shape = [n_timesteps, n_groups * n_factors]
            Transformed data in latent space
        """
        check_is_fitted(self)
        X = self._validate_input(X)

        # Store original state
        original_factors = self.latent_factors_.copy()
        original_timesteps = self.n_timesteps_

        # Update for new data dimensions
        self.n_timesteps_ = X[0].shape[0]

        # Initialize latent factors for new data
        self.latent_factors_ = np.zeros(
            (self.n_timesteps_, self.n_groups_ * self.n_factors)
        )

        # Use E-step to infer latent factors for new data
        transformed = self._e_step(X)

        # Restore original state
        self.latent_factors_ = original_factors
        self.n_timesteps_ = original_timesteps

        return transformed

    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data.

        Parameters
        ----------
        X : list of array-like, shape = [n_timesteps, n_features_i]
            List of data matrices
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        X_transformed : ndarray, shape = [n_timesteps, n_groups * n_factors]
            Transformed data in latent space
        """
        return self.fit(X).latent_factors_

    def inverse_transform(self, X_transformed):
        """
        Transform data back from latent space to original space.

        Parameters
        ----------
        X_transformed : array-like, shape = [n_timesteps, n_groups * n_factors]
            Data in latent space

        Returns
        -------
        X_reconstructed : list of ndarray
            Reconstructed data for each group
        """
        check_is_fitted(self)
        X_transformed = check_array(X_transformed)

        X_reconstructed = []
        for i in range(self.n_groups_):
            factor_idx = slice(i * self.n_factors, (i + 1) * self.n_factors)
            factors_i = X_transformed[:, factor_idx]
            reconstructed_i = factors_i @ self.loading_matrices_[i].T
            X_reconstructed.append(reconstructed_i)

        return X_reconstructed

    def get_delay_interactions(self):
        """
        Get the delay interaction matrices between groups.

        Returns
        -------
        delay_interactions : dict
            Dictionary with keys (source_group, target_group, delay)
            and values as interaction matrices
        """
        check_is_fitted(self)

        interactions = {}
        for i in range(self.n_groups_):
            for j in range(self.n_groups_):
                if i != j:
                    for d, delay_matrix in enumerate(self.delay_matrices_[i][j]):
                        interactions[(j, i, d + 1)] = delay_matrix.copy()

        return interactions

    def predict_cross_group_influence(self, source_group, target_group, delay):
        """
        Predict the influence of one group on another at a specific delay.

        Parameters
        ----------
        source_group : int
            Index of the source group
        target_group : int
            Index of the target group
        delay : int
            Delay in timesteps

        Returns
        -------
        influence_matrix : ndarray
            Matrix representing the influence
        """
        check_is_fitted(self)

        if delay < 1 or delay > self.max_delay:
            raise ValueError(f"Delay must be between 1 and {self.max_delay}")

        if source_group == target_group:
            raise ValueError("Source and target groups must be different")

        delay_idx = delay - 1
        return self.delay_matrices_[target_group][source_group][delay_idx].copy()


# Example usage and utility functions
def generate_synthetic_mdlag_data(
    n_timesteps=200,
    n_groups=3,
    n_features=[20, 15, 25],
    n_factors=5,
    max_delay=3,
    noise_level=0.1,
    random_state=42,
):
    """
    Generate synthetic data for testing mDLAG.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps
    n_groups : int
        Number of groups
    n_features : list of int
        Number of features per group
    n_factors : int
        Number of latent factors per group
    max_delay : int
        Maximum delay for interactions
    noise_level : float
        Level of observation noise
    random_state : int
        Random seed

    Returns
    -------
    X : list of ndarray
        Generated data
    true_factors : ndarray
        True latent factors
    """
    np.random.seed(random_state)

    # Generate true latent factors
    true_factors = np.random.randn(n_timesteps, n_groups * n_factors)

    # Add delay interactions
    for t in range(max_delay, n_timesteps):
        for i in range(n_groups):
            for j in range(n_groups):
                if i != j:
                    for delay in range(1, max_delay + 1):
                        if t - delay >= 0:
                            # Random interaction strength
                            interaction = 0.3 * np.random.randn(n_factors, n_factors)
                            i_idx = slice(i * n_factors, (i + 1) * n_factors)
                            j_idx = slice(j * n_factors, (j + 1) * n_factors)
                            true_factors[t, i_idx] += (
                                interaction @ true_factors[t - delay, j_idx]
                            )

    # Generate observations
    X = []
    for i in range(n_groups):
        # Random loading matrix
        loading = np.random.randn(n_features[i], n_factors)
        factor_idx = slice(i * n_factors, (i + 1) * n_factors)

        # Generate observations with noise
        observations = true_factors[:, factor_idx] @ loading.T
        observations += noise_level * np.random.randn(n_timesteps, n_features[i])
        X.append(observations)

    return X, true_factors


# Demo usage
# if __name__ == "__main__":
#     # Generate synthetic data
#     X_synthetic, true_factors = generate_synthetic_mdlag_data(
#         n_timesteps=100, n_groups=3, n_features=[10, 8, 12],
#         n_factors=4, max_delay=2, random_state=42
#     )

#     print("Generated synthetic data:")
#     for i, x in enumerate(X_synthetic):
#         print(f"Group {i}: {x.shape}")

#     # Fit mDLAG model
#     mdlag = mDLAG(n_factors=4, max_delay=2, n_iter=50, verbose=True, random_state=42)
#     X_transformed = mdlag.fit_transform(X_synthetic)

#     print(f"\nTransformed data shape: {X_transformed.shape}")
#     print(f"Final log-likelihood: {mdlag.log_likelihood_:.4f}")

#     # Get delay interactions
#     interactions = mdlag.get_delay_interactions()
#     print(f"\nFound {len(interactions)} delay interactions")

#     # Reconstruct data
#     X_reconstructed = mdlag.inverse_transform(X_transformed)

#     # Compute reconstruction error
#     reconstruction_errors = []
#     for i, (original, reconstructed) in enumerate(zip(X_synthetic, X_reconstructed)):
#         error = np.mean((original - reconstructed)**2)
#         reconstruction_errors.append(error)
#         print(f"Group {i} reconstruction MSE: {error:.6f}")
