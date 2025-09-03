import logging
from collections.abc import Callable
from typing import Union

import numpy as np
import pandas as pd
from dowhy.gcm.causal_models import PARENTS_DURING_FIT, StructuralCausalModel

from .defined_causal_mechanisms import (
    DefinedConditionalStochasticModel,
    DefinedStochasticModel,
    RelationIndexer,
)

logger = logging.getLogger(__name__)


class PartialDefinedStructuralCausalModel(StructuralCausalModel):
    def add_known_mappings(
        self,
        known_mappings: dict[
            tuple[frozenset[str], frozenset[str]],
            tuple[Callable[[np.ndarray], dict[str, np.ndarray]], Callable[[], float]],
        ],
    ) -> None:
        for (u, v), mapping_noise in known_mappings.items():
            if isinstance(mapping_noise, tuple):
                mapping, noise = mapping_noise
            else:
                mapping = mapping_noise
                noise = lambda: 0  # noqa: E731

            for vi in v:
                if vi not in self.graph.nodes:
                    continue

                if set(u) != set(self.graph.predecessors(vi)):
                    logger.warning(
                        f"Predecessors {set(self.graph.predecessors(vi))} of node {vi} do not correspond to the input columns {u}"  # noqa: E501, G004
                    )
                    continue
                if len(v) <= 1:
                    partial_mapping = mapping
                else:
                    partial_mapping = RelationIndexer(mapping, node=vi)
                if len(u) == 0:
                    mechanism = DefinedStochasticModel(partial_mapping)
                else:
                    mechanism = DefinedConditionalStochasticModel(
                        relation=partial_mapping, noise=noise
                    )
                self.set_causal_mechanism(vi, mechanism=mechanism)
                self.graph.nodes[vi][PARENTS_DURING_FIT] = list(u)

    def is_known_mapping(self, start_node: Union[str, None], end_node: str) -> bool:
        try:
            causal_mechanism = self.causal_mechanism(node=end_node)
        except KeyError:
            return False

        if not isinstance(causal_mechanism, DefinedConditionalStochasticModel):
            return False

        if start_node is None:
            return True

        return start_node in self.graph.nodes[end_node][PARENTS_DURING_FIT]

    def fill_data_based_on_known_mappings(self, data: pd.DataFrame) -> None:
        while True:
            latent_nodes = set(self.graph.nodes) - set(data.columns)

            if not latent_nodes:
                break

            added_columns = 0

            for node in latent_nodes:
                try:
                    causal_mechanism = self.causal_mechanism(node=node)
                except KeyError:
                    continue

                if isinstance(causal_mechanism, DefinedConditionalStochasticModel):
                    predecessors: list[str] = self.graph.nodes[node][PARENTS_DURING_FIT]
                    if not set(predecessors).issubset(data.columns):
                        continue
                    resulting_values = causal_mechanism.draw_samples(
                        data[predecessors].values
                    )
                elif isinstance(causal_mechanism, DefinedStochasticModel):
                    resulting_values = causal_mechanism.draw_samples(data.shape[0])
                else:
                    continue

                data[node] = (
                    resulting_values[:, None]
                    if resulting_values.ndim == 1
                    else resulting_values
                )
                added_columns += 1

            if added_columns == 0:
                break
