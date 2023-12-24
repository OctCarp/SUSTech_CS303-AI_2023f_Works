import numpy as np
from tqdm import tqdm


class NNS:
    def __init__(self, k=5):
        """
        Initialize the NNS with a specified value of k.

        Parameters:
        - k: Number of neighbors (default is 5).
        """
        self.k = k

    def fit(self, X_train):
        """
        Fit the NNS to the repository data.

        Parameters:
        - X_train: Repository data.
        """

        self.X_train = X_train

    def predict(self, X_test):
        """
        Find the IDs of the k repository data points that are closest to the test sample points.
        
        Parameters:
        - X_test: Repository data.

        Returns:
        - y_pred: IDs of the k repository data points that are closest to the test sample points.
        """
        y_pred = []
        for x in tqdm(X_test):
            k_indices = self._predict(x)
            y_pred.append(k_indices)
        return np.array(y_pred)

    def _predict(self, x):
        """
        Find the IDs of the k repository data points that are closest to the single test data point.

        Parameters:
        - x: Test data point.

        Returns:
        - k_indices: IDs of the k repository data points that are closest to the test sample point.
        """
        distances = [np.sum(((x - x_train) ** 2) ** 0.4) for x_train in self.X_train]
        # distances = [np.sum(((x - x_train + 1) ** 2 + 1) ** 0.3) for x_train in self.X_train]
        # distances = [np.sum((x - x_train)**2) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        return k_indices
