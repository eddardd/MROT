import ot
import numpy as np
import typing as t
import xgboost as xgb
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator


def exclusion_zone_cost(
    C: np.array, k: int = 5
):
    """Engineers the base-ground cost matrix C to prevent
    OT from sending mass from a point to its k-nearest neighbors.

    Args:
        C (np.array): The base cost matrix
        k (int): the number of nearest neighbors
    """
    C_tilde = C.copy()

    neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
    neigh.fit(C_tilde)
    _, indices = neigh.kneighbors(C_tilde)

    for i in range(C_tilde.shape[0]):
        C_tilde[i, indices[i]] = C_tilde[i, :].max()

    return C_tilde


class XGBoost:
    """A wrapper for XGBoost model to make it compatible
    with scikit-learn API."""
    def __init__(
        self,
        max_depth: int,
        eta: float,
        objective='binary:logistic',
        num_round: int = 100
    ):
        self.params = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': objective
        }
        self.num_round = num_round
        self.xgb = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.xgb = xgb.train(self.params, dtrain, self.num_round)
        return self

    def predict(self, X):
        assert self.xgb, "Trying to predict on unfitted model."
        dtest = xgb.DMatrix(X)
        return self.xgb.predict(dtest)


class MassRepulsiveOptimalTransport:
    """Mass Repulsive Optimal Transport (MROT).

    This class implements our algorithm, which has 4 steps:

    1. Computes the MROT plan using samples from a measure P,
    2. Compute the transportation effort for each sample
    3. Estimate the density of the transportation efforts
    4. Fit a regression model on the map between each sample and the CDF of
    its effort.

    Args:
        k (int): The number of nearest neighbors to exclude from the cost
        matrix.
        reg_e (float): Regularization parameter for the transportation effort.
        n_bins (int): Number of bins for the histogram density estimation.
        regressor (BaseEstimator): A scikit-learn compatible regressor to fit
        the map.
        density_estimator (str): Type of density estimator to use, either
        "kde" or "histogram".
    """
    def __init__(
        self,
        k: int = 5,
        reg_e: float = 0.1,
        n_bins: int = 1000,
        regressor: BaseEstimator = None,
        density_estimator: t.Literal["kde", "histogram"] = "kde"
    ):
        self.k = k
        self.reg_e = reg_e
        self.n_bins = n_bins

        if regressor is None:
            self.regressor = XGBoost(
                max_depth=20,
                eta=0.1,
                objective="binary:logistic",
                num_round=100)
        else:
            self.regressor = regressor

        assert density_estimator.lower() in ["kde", "histogram"], \
            "Density estimator must be either 'kde' or 'histogram'."
        self.density_estimator = density_estimator

    def fit(self, X: np.array):
        # --- Step 1: Compute the MROT plan ---
        a = np.ones(len(X)) / len(X)
        C = cdist(X, X, metric='sqeuclidean')
        C_tilde = exclusion_zone_cost(C, self.k)

        if self.reg_e > 0.0:
            ot_plan = ot.sinkhorn(
                a, a, C_tilde / C_tilde.max(), reg=self.reg_e)
        else:
            ot_plan = ot.emd(a, a, C_tilde)

        # --- Step 2: Compute the transportation effort for each sample ---
        scores = (ot_plan * C_tilde).sum(axis=1)

        # --- Step 3: Estimate the density of the transportation efforts ---
        if self.density_estimator.lower() == "kde":
            # Statistical Modeling
            kde = gaussian_kde(scores, bw_method='scott')
            s_grid = np.linspace(min(scores), max(scores), self.n_bins)
            ds = np.diff(s_grid)[0]

            # Defining the density over the grid
            density = kde(s_grid)
            density /= density.sum() * ds
        elif self.density_estimator.lower() == "hist":
            density, s_grid = np.histogram(
                scores, bins=self.n_bins, density=True)
            ds = np.diff(s_grid)[0]
            s_grid = 0.5 * (s_grid[1:] + s_grid[:-1])
        else:
            raise ValueError(
                "Invalid density estimation method:"
                f" '{self.density_estimator}'"
            )

        # Defining the CDF
        cdf = np.cumsum(density) * ds

        # Defining the values to regress
        cdf_values = np.interp(x=scores, xp=s_grid, fp=cdf)

        # --- Step 4: Fit the regression model ---
        self.regressor = self.regressor.fit(X, cdf_values)

    def predict(self, X: np.array):
        y_pred = self.regressor.predict(X)

        return y_pred


class RespulsiveCostOptimalTransport:
    """Repulsive Optimal Transport.

    This class implements our algorithm, which has 4 steps:

    1. Computes the Repulsive OT plan using samples from a measure P,
    2. Compute the transportation effort for each sample
    3. Estimate the density of the transportation efforts
    4. Fit a regression model on the map between each sample and the CDF of
    its effort.

    Args:
        k (int): The number of nearest neighbors to exclude from the cost
        matrix.
        reg_e (float): Regularization parameter for the transportation effort.
        n_bins (int): Number of bins for the histogram density estimation.
        regressor (BaseEstimator): A scikit-learn compatible regressor to fit
        the map.
        density_estimator (str): Type of density estimator to use, either
        "gaussian" or "histogram".
    """
    def __init__(
        self,
        k: int = 5,
        reg_e: float = 0.1,
        n_bins: int = 1000,
        regressor: BaseEstimator = None,
        density_estimator: t.Literal["kde", "histogram"] = "gaussian"
    ):
        self.k = k
        self.reg_e = reg_e
        self.n_bins = n_bins

        if regressor is None:
            self.regressor = XGBoost(
                max_depth=20,
                eta=0.1,
                objective="binary:logistic",
                num_round=100)
        else:
            self.regressor = regressor

        assert density_estimator.lower() in ["kde", "histogram"], \
            "Density estimator must be either 'kde' or 'histogram'."
        self.density_estimator = density_estimator

    def fit(self, X: np.array):
        # --- Step 1: Compute the ROT plan ---
        a = np.ones(len(X)) / len(X)
        C = 1 / (cdist(X, X, metric='euclidean') + 1)

        if self.reg_e > 0.0:
            ot_plan = ot.sinkhorn(
                a, a, C / C.max(), reg=self.reg_e)
        else:
            ot_plan = ot.emd(a, a, C)

        # --- Step 2: Compute the transportation effort for each sample ---
        scores = (ot_plan * C).sum(axis=1)

        # --- Step 3: Estimate the density of the transportation efforts ---
        if self.density_estimator.lower() == "kde":
            # Statistical Modeling
            kde = gaussian_kde(scores, bw_method='scott')
            s_grid = np.linspace(min(scores), max(scores), self.n_bins)
            ds = np.diff(s_grid)[0]

            # Defining the density over the grid
            density = kde(s_grid)
            density /= density.sum() * ds
        elif self.density_estimator.lower() == "hist":
            density, s_grid = np.histogram(
                scores, bins=self.n_bins, density=True)
            ds = np.diff(s_grid)[0]
            s_grid = 0.5 * (s_grid[1:] + s_grid[:-1])
        else:
            raise ValueError("Invalid density estimation method")

        # Defining the CDF
        cdf = np.cumsum(density) * ds

        # Defining the values to regress
        cdf_values = np.interp(x=scores, xp=s_grid, fp=cdf)

        # --- Step 4: Fit the regression model ---
        self.regressor = self.regressor.fit(X, cdf_values)

    def predict(self, X: np.array):
        # NOTE: The prediction is the inverse of the regression model
        # because ROT actually assigns higher scores to normal points
        # and lower scores to outliers.
        y_pred = 1 - self.regressor.predict(X)

        return y_pred
