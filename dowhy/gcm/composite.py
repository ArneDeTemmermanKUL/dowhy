import networkx as nx
import numpy as np
from dowhy.gcm.causal_models import (
    PARENTS_DURING_FIT,
    CAUSAL_MECHANISM,
    ProbabilisticCausalModel,
)
from dowhy.gcm.defined_causal_mechanisms import (
    AggregationMechanism,
    DefinedStochasticModel,
)


class StructuralCausalModelComposite(ProbabilisticCausalModel):
    def __init__(
        self,
        models: list[ProbabilisticCausalModel],
        links: dict[tuple[dict[str], str, str], AggregationMechanism] = {},
    ):
        self.graph = self._graph(models, links)

        for (u, v, aggregation_column), mechanism in links.items():
            self.set_causal_mechanism(v, mechanism=mechanism)
            self.graph.nodes[v][PARENTS_DURING_FIT] = [aggregation_column] + sorted(
                list(u)
            )
            if CAUSAL_MECHANISM not in self.graph.nodes[aggregation_column]:
                self.set_causal_mechanism(
                    node=aggregation_column,
                    mechanism=DefinedStochasticModel(lambda: np.nan),
                )
                self.graph.nodes[aggregation_column][PARENTS_DURING_FIT] = []

    def _graph(self, models, links: list[tuple[str, str]]):
        total_graph = models[0].graph
        for i in range(1, len(models)):
            total_graph = nx.compose(total_graph, H=models[i].graph)
        for u, v, aggregation in links:
            for ui in u:
                total_graph.add_edge(ui, v, time_lag=(0,))
            total_graph.add_edge(aggregation, v, time_lag=(0,))

        return total_graph

    def clone(self):
        """Clones the causal model, but keeps causal mechanisms untrained."""
        return StructuralCausalModelComposite([model.clone() for model in self.models])
