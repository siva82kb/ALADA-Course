"""Module containing classes, functions, and other data structures related to 
topic covered in Chapter 01.

Author: Sivakuar Balasubramanian
Date: 09 July 2024
"""

import numpy as np
import polars as pl

# k-mean clustering algorithm
class KMeans(object):

    def __init__(self, X: np.array, k: int) -> None:
        """Initialize the k-means model.
        
        Parameters
        ----------
        X : np.array
            Input data to cluster. N x M array with N corresponding to the 
            number of samples, and M corresponding to the number of features.
        k : int
            Number of clusters to form.
        """
        self.k = k
        self.X = X
        self.centroids = None
        self.clustassign = None
        self.J = None
    
    def _get_cluster_assignment(self) -> np.array:
        """Assign each data point to the nearest cluster.

        Returns
        -------
        np.array
            Array containing the cluster assignment for each data point.
        """
        return np.array([
            np.argmin(np.linalg.norm(self.centroids - _x, axis=1))
            for _x in self.X
        ])
    
    def _get_cluster_mean(self) -> np.array:
        """Get the mean of each cluster.
        
        Returns
        -------
        np.array
            Array containing the mean of each cluster.
        """
        return np.array([
            np.mean(self.X[self.clustassign == _c, :], axis=0)
            for _c in np.unique(self.clustassign)
        ])

    def _get_j_clust(self) -> float:
        """Get the cost of the clustering.
        
        Returns
        -------
        float
            Cost of the clustering.
        """
        # cluster indices
        _cinx = [self.clustassign == _c for _c in range(self.k)]
        return np.sum([
            np.sum(np.linalg.norm(self.X[_ci, :] - self.centroids[i, :], axis=1) ** 2)
            for i, _ci in enumerate(_cinx)
        ])

    def fit(self, max_iter: int = 100, cost_change_th: float = 1) -> tuple[np.array, np.array]:
        """Fit the k-means model to the data.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations to run the algorithm, by default 100.
        cost_change_th : float, optional
            Cost change threshold to stop the algorithm, by default 1%. A change 
            in the cost less thatn 1% will terminate the iteration.
        """
        # reset centroids, cluster assignments, and cost
        self.centroids = None
        self.clustassign = None
        self.J = None
        # initialize centroids
        _inx = np.random.choice(self.X.shape[0], self.k, replace=False)
        self.centroids = self.X[_inx, :]
        # get cluster assignment
        self.clustassign = self._get_cluster_assignment()
        # cost
        self.J = self._get_j_clust()
        # temporary cluster means, assignments, and cost
        _cmean = [self.centroids]
        _cassign = [self.clustassign]
        _J = [self.J]
        for i in range(max_iter):
            # update centroids
            self.centroids = self._get_cluster_mean()
            # update cluster assignments
            self.clustassign = self._get_cluster_assignment()
            # update cost
            self.J = self._get_j_clust()

            # update temporary cluster means, assignments, and cost
            _cmean.append(self.centroids)
            _cassign.append(self.clustassign)
            _J.append(self.J)

            # check for convergence
            if 100 * np.abs(_J[-1] - _J[-2]) / _J[-2] < cost_change_th:
                break
        return np.array(_cmean), np.array(_cassign), np.array(_J)