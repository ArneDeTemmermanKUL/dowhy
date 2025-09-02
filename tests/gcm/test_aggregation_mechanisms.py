import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm.aggregation_mechanisms import AggregationMechanism
from pytest import fixture,mark

from aeon.transformations.collection.unequal_length import Padder
from aeon.transformations.collection.feature_based import SevenNumberSummary

@fixture
def data(n: int = 1000, aggregation_lengths: list[int] = [15,5]):

    aggregation = np.repeat(range(int(n * len(aggregation_lengths)  / sum(aggregation_lengths))), np.tile(aggregation_lengths,int(n /(sum(aggregation_lengths)))))

    X = np.random.normal(loc=0, scale=1, size=n)
    X_df = pd.DataFrame({"X":X},index=aggregation)
    X_agg = (
        X_df
        .groupby(
            level=0
        )
        .sum()
        .reset_index()
    )
    Y_df = X_agg[["X"]]

    return X_df,Y_df

def test_defined_aggregation_mechanism(data):
    pass

def test_aggregation_mechanism(data):

    X,Y = data

    transformer = SevenNumberSummary()

    aam = AggregationMechanism(preprocess_transformer=None,transformer=transformer,
    prediction_model=gcm.ml.create_linear_regressor())

    aam.fit(X.reset_index().to_numpy(),Y.to_numpy().squeeze())

    samples = aam.draw_samples(X.reset_index().to_numpy())


def test_aggregation_mechanism_with_uneven_data(data):

    X,Y = data

    padder = Padder()
    transformer = SevenNumberSummary()
    
    aam = AggregationMechanism(preprocess_transformer=padder,transformer=transformer,
    prediction_model=gcm.ml.create_linear_regressor())

    aam.fit(X.reset_index().to_numpy(),Y.to_numpy().squeeze())

    samples = aam.draw_samples(X.reset_index().to_numpy())