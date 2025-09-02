from collections.abc import Callable
import numpy as np

from dowhy.gcm.causal_mechanisms import (
    ConditionalStochasticModel,
    StochasticModel,
)
from dowhy.gcm.util.general import shape_into_2d

class DefinedConditionalStochasticModel(ConditionalStochasticModel):
    def __init__(
        self,
        relation: Callable[
            [np.ndarray],
            np.ndarray,
        ],
        noise: Callable[[], float] = lambda: 0,
    ) -> None:
        self.relation = relation
        self.noise = noise

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fits the model according to the data."""
        pass

    def evaluate(
        self,
        parent_samples: np.ndarray,
        noise_samples: np.ndarray,
    ) -> np.ndarray:
        parent_samples, noise_samples = shape_into_2d(parent_samples, noise_samples)
        predictions = shape_into_2d(self.relation(parent_samples))
        return predictions + noise_samples

    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        """Draws samples for the fitted model."""
        return self.evaluate(
            parent_samples, self.draw_noise_samples(parent_samples.shape[0])
        )

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return np.array([self.noise() for _ in range(num_samples)])

    def clone(self):
        return DefinedConditionalStochasticModel(
            relation=self.relation, noise=self.noise
        )


class DefinedStochasticModel(StochasticModel):
    def __init__(
        self,
        distribution: Callable[
            [],
            float,
        ],
    ) -> None:
        self.distribution = distribution

    def fit(self, X: np.ndarray) -> None:
        """Fits the model according to the data."""

    def draw_samples(self, num_samples: int) -> np.ndarray:
        """Draws samples for the fitted model."""
        return np.array([self.distribution() for _ in range(num_samples)])

    def clone(self):
        return DefinedStochasticModel(distribution=self.distribution)


class RelationIndexer:
    def __init__(self, relation: Callable[[np.ndarray], np.ndarray], i: int) -> None:
        self.relation = relation
        self.i: int = i

    def __call__(self, x: np.ndarray) -> np.ndarray:
        a = self.relation(x)
        return a[:, self.i]



