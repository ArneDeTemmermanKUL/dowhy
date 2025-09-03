import numpy as np
import numpy.typing as npt
from pytest import approx
from dowhy.gcm.defined_causal_mechanisms import (
    DefinedConditionalStochasticModel,
    DefinedStochasticModel,
    RelationIndexer,
)


def test_defined_stochastic_model():
    def distribution():
        return np.random.normal(loc=0, scale=1)

    dsm = DefinedStochasticModel(distribution=distribution)
    samples = dsm.draw_samples(num_samples=1000)

    assert np.mean(samples) == approx(0, abs=0.2)
    assert np.std(samples) == approx(1, abs=0.2)


def test_defined_conditional_stochastic_model():
    def relation(a: npt.NDArray):
        return dict(b=2 * a)

    def noise():
        return np.random.normal(loc=0, scale=1)

    dsm = DefinedConditionalStochasticModel(relation=relation, noise=noise)

    parent_samples = np.full(shape=(1000), fill_value=4.0)
    samples = dsm.draw_samples(parent_samples)

    assert np.mean(samples) == approx(8, abs=0.2)
    assert np.std(samples) == approx(1, abs=0.2)


def test_relation_indexer():

    def relation(a: npt.NDArray):
        return dict(b=a,c=3*a)

    def noise():
        return np.random.normal(loc=0, scale=1)

    partial_relation = RelationIndexer(relation=relation, node="c")

    dsm = DefinedConditionalStochasticModel(relation=partial_relation, noise=noise)

    parent_samples = np.full(shape=(1000), fill_value=4.0)
    samples = dsm.draw_samples(parent_samples)

    assert np.mean(samples) == approx(12, abs=0.2)
    assert np.std(samples) == approx(1, abs=0.2)


