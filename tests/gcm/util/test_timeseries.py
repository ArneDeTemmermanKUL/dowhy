from flaky import flaky
from networkx import DiGraph
import numpy as np
from pytest import fixture
import networkx as nx
import pandas as pd


from dowhy import gcm
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel
from dowhy.gcm.causal_models import StructuralCausalModel
from dowhy.gcm.ml.regression import create_linear_regressor
from dowhy.gcm.stochastic_models import EmpiricalDistribution
from dowhy.gcm.util.timeseries import (
    draw_samples_incremental,
    strongly_connected_components_generations,
    strongly_connected_components_sort,
)




@fixture
def temporal_cyclic_causal_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge("A", "B", time_lag=(0,))
    graph.add_edge("B", "C", time_lag=(0,))
    graph.add_edge("B", "D", time_lag=(1,))
    graph.add_edge("D", "B", time_lag=(0,))
    graph.add_edge("C", "E", time_lag=(2,))
    graph.add_edge("D", "E", time_lag=(0,))
    return graph




@fixture
def data(n=1000) -> pd.DataFrame:
    data = np.empty((n, 5))


    data[:2, :] = np.zeros(shape=(2, 5))


    for i in range(2, n):
        data[i, 0] = np.random.randint(0,5)  # A
        data[i, 3] = data[i - 1, 1] - 2  # D
        data[i, 1] = data[i, 0] + data[i, 3]  # B
        data[i, 2] = data[i, 1] * 2  # C


        data[i, 4] = data[i - 2, 2] + data[i, 3]  # E


    return pd.DataFrame(data=data, columns=["A", "B", "C", "D", "E"])




@fixture
def temporal_cyclic_graphical_causal_model(
    temporal_cyclic_causal_graph: DiGraph, data: pd.DataFrame
) -> StructuralCausalModel:
    scm = StructuralCausalModel(temporal_cyclic_causal_graph)


    scm.set_causal_mechanism("A", EmpiricalDistribution())
    scm.set_causal_mechanism(
        "B", AdditiveNoiseModel(prediction_model=create_linear_regressor())
    )
    scm.set_causal_mechanism(
        "C", AdditiveNoiseModel(prediction_model=create_linear_regressor())
    )
    scm.set_causal_mechanism(
        "D", AdditiveNoiseModel(prediction_model=create_linear_regressor())
    )
    scm.set_causal_mechanism(
        "E", AdditiveNoiseModel(prediction_model=create_linear_regressor())
    )




    es = gcm.fit(causal_model=scm, data=data,return_evaluation_summary=True)
    print(es)
    return scm




def test_temporal_topological_generations(temporal_cyclic_causal_graph: DiGraph):
    generations = list(strongly_connected_components_sort(temporal_cyclic_causal_graph))


    assert len(generations) == 3
    assert generations[0] == {"A"}
    assert generations[1] == {"B", "C", "D"}
    assert generations[2] == {"E"}




def test_strongly_connected_components_generations(
    temporal_cyclic_causal_graph: DiGraph,
):
    generations = strongly_connected_components_generations(
        temporal_cyclic_causal_graph
    )
    assert len(generations) == 3
    assert generations[0] == [{"A"}]
    assert generations[1] == [{"B", "C", "D"}]
    assert generations[2] == [{"E"}]


def test_draw_samples_incremental(
    temporal_cyclic_graphical_causal_model: StructuralCausalModel, data: pd.DataFrame,num_samples=10
):
    samples = draw_samples_incremental(
        temporal_cyclic_graphical_causal_model, num_samples, observed_data=[data.head(2),data[["A"]].head(10)]
    )
    np.testing.assert_allclose(samples[data.columns].values,data.head(num_samples).values,atol=0.1,rtol=0.05) 