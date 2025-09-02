from abc import abstractmethod

import numpy as np


class Transformer:
    """Represents general prediction model implementations. Each prediction model should provide a fit and a predict
    method."""

    @abstractmethod
    def fit(self, X: list[np.ndarray], Y: np.ndarray) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def fit_transform(self, X: list[np.ndarray], Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, X: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        """
        Clones the prediction model using the same hyper parameters but not fitted.

        :return: An unfitted clone of the prediction model.
        """
        raise NotImplementedError
