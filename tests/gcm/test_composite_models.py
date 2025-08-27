import numpy as np
import pandas as pd
from pytest import fixture
import networkx as nx
from dowhy import gcm
from dowhy.gcm.defined_causal_mechanisms import AggregationMechanism


@fixture
def graphs() -> list[nx.DiGraph]:
    graphs = []

    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B")])
    graphs.append(graph)

    graph = nx.DiGraph()
    graph.add_edges_from([("C", "D")])
    graphs.append(graph)
    return graphs


@fixture
def datasets(n: int = 1000, n_aggregations: int = 20) -> list[pd.DataFrame]:
    datasets: list[pd.DataFrame] = []

    A = np.random.normal(loc=0, scale=1, size=n)
    B = 2 * A

    aggregation1 = np.repeat(range(n_aggregations), int(n / n_aggregations))

    datasets.append(pd.DataFrame(data=dict(AAA_aggregation=aggregation1, A=A, B=B)))

    data_agg = (
        datasets[0]
        .groupby(
            by="AAA_aggregation",
        )
        .sum()
        .reset_index()
    )
    C = data_agg["B"]
    aggregation2 = data_agg["AAA_aggregation"]

    D = 4 * C

    datasets.append(pd.DataFrame(data=dict(AAA_aggregation=aggregation2, C=C, D=D)))

    return datasets


@fixture
def links():
    return {(frozenset(["B"]), "C", "AAA_aggregation"): "sum"}


@fixture
def graphical_causal_models(
    graphs: list[nx.DiGraph], datasets: list[pd.DataFrame]
) -> list[gcm.StructuralCausalModel]:
    causal_models = []

    for graph, dataset in zip(graphs, datasets):
        causal_model = gcm.StructuralCausalModel(graph=graph)
        gcm.auto.assign_causal_mechanisms(causal_model, dataset, override_models=False)
        gcm.fit(
            causal_model,
            dataset,
            return_evaluation_summary=False,
        )
        causal_models.append(causal_model)

    return causal_models


@fixture
def composite_gcm(graphical_causal_models: list[gcm.StructuralCausalModel], links):
    composite = gcm.StructuralCausalModelComposite(
        models=graphical_causal_models, links=links
    )
    return composite


def test_interventional_samples(composite_gcm, datasets):
    observed_data = datasets[0][["AAA_aggregation", "A"]]

    samples = gcm.interventional_samples(
        causal_model=composite_gcm,
        interventions={"A": lambda a: a},
        observed_data=observed_data,
    )

    assert all([node in samples for node in composite_gcm.graph.nodes])

    for dataset in datasets:
        for column in dataset:
            if column == "AAA_aggregation":
                continue

            np.testing.assert_allclose(
                samples[column], dataset[column].values, atol=1.0, rtol=0.0
            )
