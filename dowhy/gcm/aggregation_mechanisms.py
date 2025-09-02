from typing import Optional
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, StochasticModel
from dowhy.gcm.defined_causal_mechanisms import DefinedConditionalStochasticModel
from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.util.general import shape_into_2d


import numpy as np
import pandas as pd


from collections.abc import Callable

from dowhy.gcm.util.transformer import Transformer


class DefinedAggregationMechanism(DefinedConditionalStochasticModel):
    """A mechanism that aggregates samples based on a specified aggregation function. The aggregation is based on the index in the first column of the parent samples."""

    def __init__(
        self,
        aggregation_function: Callable[[np.ndarray], float],
    ):
        super().__init__(
            aggregation_function,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def evaluate(
        self,
        parent_samples: np.ndarray,
        noise_samples: np.ndarray,
    ) -> np.ndarray:
        aggregation_column = parent_samples[:, 0]
        parent_samples = parent_samples[:, 1:]  # Remove the aggregation column

        parent_samples, noise_samples = shape_into_2d(parent_samples, noise_samples)
        # TODO: how to integrate noise samples
        samples_df = pd.DataFrame(
            data=parent_samples,
            index=aggregation_column,
            dtype=np.float64,
        )
        samples_agg: pd.DataFrame = samples_df.groupby(by=samples_df.index).apply(
            self.relation
        )

        return samples_agg.to_numpy()

    def clone(self):
        return DefinedAggregationMechanism(self.relation)


class AggregationMechanism(AdditiveNoiseModel):
    def __init__(
        self,
        preprocess_transformer: Optional[Transformer ],
        transformer: Transformer,
        prediction_model: PredictionModel,
        noise_model: Optional[StochasticModel] = None,
    ) -> None:
        if noise_model is None:
            from dowhy.gcm.stochastic_models import EmpiricalDistribution

            noise_model = EmpiricalDistribution()
        self.preprocess_transformer = preprocess_transformer
        self.transformer = transformer

        from dowhy.gcm.ml.regression import InvertibleIdentityFunction

        super(AdditiveNoiseModel, self).__init__(
            prediction_model=prediction_model,
            noise_model=noise_model,
            invertible_function=InvertibleIdentityFunction(),
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        aggregation_column = X[:, 0]
        X = X[:, 1:]
        # reshape
        X_agg = shape_into_3d(X, aggregation_column)

        # Resample
        if self.preprocess_transformer is not None:
            X_agg = self.preprocess_transformer.fit_transform(X_agg)

        X_agg = self.transformer.fit_transform(X_agg, Y)

        self._prediction_model.fit(X=X_agg, Y=self._invertible_function.evaluate_inverse(Y))
        self._noise_model.fit(X=self.estimate_noise(Y, X_agg))

    def evaluate(
        self,
        parent_samples: np.ndarray,
        noise_samples: np.ndarray,
    ) -> np.ndarray:
        aggregation_column = parent_samples[:, 0]
        parent_samples = parent_samples[:, 1:]  # Remove the aggregation column

        parent_samples, noise_samples = shape_into_2d(parent_samples, noise_samples)

        # shape into 3D
        agg_parent_samples = shape_into_3d(parent_samples, aggregation_column)

        # Resample
        if self.preprocess_transformer is not None:
            agg_parent_samples = self.preprocess_transformer.transform(
                agg_parent_samples
            )

        transformed_parent_samples = self.transformer.transform(agg_parent_samples)

        predictions = shape_into_2d(
            self._prediction_model.predict(transformed_parent_samples)
        )

        return self._invertible_function.evaluate(predictions + noise_samples[:len(predictions)])


def shape_into_3d(samples: np.ndarray, aggregation_column:np.ndarray):
    samples_df = pd.DataFrame(
        data=samples,
        index=aggregation_column,
        dtype=np.float64,
    )
    agg_parent_samples: list[np.ndarray] = (
        samples_df.groupby(level=0).apply(pd.DataFrame.to_numpy).tolist()
    )
    agg_parent_samples = [agg_parent_sample.T for agg_parent_sample in agg_parent_samples]
    return agg_parent_samples
