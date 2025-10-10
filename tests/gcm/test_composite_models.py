import numpy as np
import pandas as pd
from pytest import fixture
import networkx as nx
from dowhy import gcm
from dowhy.gcm.aggregation_mechanisms import AggregationRegressionMechanism, DefinedAggregationMechanism


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
def datasets(n: int = 1000, aggregation_lengths: list[int] = [15,5]) -> list[pd.DataFrame]:
    datasets: list[pd.DataFrame] = []

    A = np.random.normal(loc=0, scale=1, size=n)
    B = 2 * A

    aggregation1 = np.repeat(range(int(n * len(aggregation_lengths)  / sum(aggregation_lengths))), np.tile(aggregation_lengths,int(n /(sum(aggregation_lengths)))))

    datasets.append(pd.DataFrame(data={"!index":aggregation1, "A":A, "B":B}))

    data_agg = (
        datasets[0]
        .groupby(
            by="!index",
        )
        .sum()
        .reset_index()
    )
    C = data_agg["B"]
    aggregation2 = data_agg["!index"]

    D = 4 * C

    datasets.append(pd.DataFrame(data={"!index":aggregation2, "C":C, "D":D}))

    return datasets


@fixture
def links():
    return {(frozenset(["B"]), "C", "!index"): DefinedAggregationMechanism("sum")}


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



def test_interventional_samples(graphical_causal_models: list[gcm.StructuralCausalModel], links, datasets):
    observed_data = datasets[0][["!index", "A"]]

    composite_gcm = gcm.StructuralCausalModelComposite(
        models=graphical_causal_models, links=links
    )

    samples = gcm.interventional_samples(
        causal_model=composite_gcm,
        interventions={"A": lambda a: a},
        observed_data=observed_data,
    )

    assert all([node in samples for node in composite_gcm.graph.nodes])

    for dataset in datasets:
        for column in dataset:
            if column == "!index":
                continue

            np.testing.assert_allclose(
                samples[column], dataset[column].values, atol=1.0, rtol=0.0
            )


def test_aggregation_composite_with_unequal_length_transformer(graphical_causal_models: list[gcm.StructuralCausalModel], datasets: list[pd.DataFrame]):

    from aeon.transformations.collection.feature_based import SevenNumberSummary


    transformer = SevenNumberSummary()

    aam = AggregationRegressionMechanism(preprocess_transformer=None,transformer=transformer,
    prediction_model=gcm.ml.create_linear_regressor())

    X =  datasets[0][["!index","B"]].to_numpy()
    Y = datasets[1]["C"].to_numpy().squeeze()

    aam.fit(X,Y)

    links= {(frozenset(["B"]), "C", "!index"): aam}

    composite_gcm = gcm.StructuralCausalModelComposite(
        models=graphical_causal_models, links=links
    )

    samples = gcm.interventional_samples(composite_gcm,interventions={'A':lambda a: a},observed_data=datasets[0][["A","!index"]])


def test_aggregation_composite_with_padding(graphical_causal_models: list[gcm.StructuralCausalModel], datasets: list[pd.DataFrame]):

    from aeon.transformations.collection import Padder
    from aeon.transformations.collection.feature_based import SevenNumberSummary

    padder = Padder()
    transformer = SevenNumberSummary()


    aam = AggregationRegressionMechanism(preprocess_transformer=padder,transformer=transformer,
    prediction_model=gcm.ml.create_linear_regressor())

    X =  datasets[0][["!index","B"]].to_numpy()
    Y = datasets[1]["C"].to_numpy().squeeze()

    aam.fit(X,Y)

    links= {(frozenset(["B"]), "C", "!index"): aam}

    composite_gcm = gcm.StructuralCausalModelComposite(
        models=graphical_causal_models, links=links
    )

    samples = gcm.interventional_samples(composite_gcm,interventions={'A':lambda a: a},observed_data=datasets[0][["A","!index"]])

