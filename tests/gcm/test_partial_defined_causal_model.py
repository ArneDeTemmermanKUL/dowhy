import numpy as np
import numpy.typing as npt
import pandas as pd
from dowhy.gcm.partial_defined_causal_models import PartialDefinedStructuralCausalModel
from flaky import flaky
import networkx as nx
from pytest import approx

from scipy import stats

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ScipyDistribution,
    draw_samples,
    fit,
)
from dowhy.gcm.auto import assign_causal_mechanisms
from dowhy.gcm.divergence import estimate_kl_divergence_continuous_clf
from dowhy.gcm.ml import (
    create_linear_regressor,
)


def _generate_data_with_categorical_input():
    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)

    data = pd.DataFrame(
        {
            "X0": X0,
            "X1": X1,
        }
    )

    data["X2"] = x2_relation(data["X0"].to_numpy(),data["X1"].to_numpy())["X2"]

    return data


def x2_relation(X0: np.ndarray, X1: np.ndarray):
    X2 = []
    for i in range(len(X0)):
        X0i, X1i = X0[i], X1[i]
        tmp_value = 2 * X0i
        if X1i == "0":
            tmp_value -= 5
        elif X1i == "1":
            tmp_value += 10
        else:
            tmp_value += 5
        X2.append(tmp_value)
    return dict(X2=np.array(X2))


@flaky(max_runs=3)
def test_given_categorical_input_data_when_draw_from_fitted_causal_graph_with_linear_anm_then_generates_correct_marginal_distribution():
    training_data = _generate_data_with_categorical_input()

    scm = PartialDefinedStructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))

    scm.add_known_mappings(
        {(frozenset(["X0", "X1"]), frozenset(["X2"])): (x2_relation, lambda: (0.0,))}
    )
    assign_causal_mechanisms(scm, training_data)

    fit(scm, data=training_data)

    assert scm.causal_mechanism("X2").evaluate(
        np.array([[2, "1"]], dtype=object), np.array([0])
    ) == approx(14)

    test_data = training_data.to_numpy()
    assert scm.causal_mechanism("X2").evaluate(
        test_data[:, :2], np.array([0] * 1)
    ) == approx(test_data[:, 2].astype(float).reshape(-1, 1))

    assert estimate_kl_divergence_continuous_clf(
        test_data[:, 2], draw_samples(scm, 1000)["X2"].to_numpy()
    ) == approx(0, abs=0.05)
