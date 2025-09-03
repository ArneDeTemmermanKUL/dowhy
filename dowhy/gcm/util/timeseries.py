from collections import defaultdict
from queue import Queue

import numpy as np
from typing import Any
from dowhy import gcm
from dowhy.gcm.causal_models import  PARENTS_DURING_FIT, ProbabilisticCausalModel, validate_causal_graph
from dowhy.graph import get_ordered_predecessors, is_root_node

import networkx as nx
import pandas as pd


def is_self_referential(graph: nx.DiGraph, node: str) -> bool:
    """Checks if the given node is self-referential in the causal model."""
    return node in graph.predecessors(node)


def timelag_data(causal_model, node:str, based_on: dict[np.ndarray]) -> pd.DataFrame:
    ordered_predecessors = get_ordered_predecessors(causal_model.graph, node)
    predecessors_data = {}
    # lag the data to match the time lags of the predecessors
    for ordered_predecessor in ordered_predecessors:
        if all(pd.isna(based_on[ordered_predecessor])):
            continue
        edge_data = causal_model.graph.get_edge_data(ordered_predecessor, node)
        if "time_lag" in edge_data:
            parent_time_lag = edge_data["time_lag"]
            if not isinstance(parent_time_lag, tuple):
                parent_time_lag = (parent_time_lag,)
            for lag in parent_time_lag:

                # shift by 1) roll and 2) mask
                predecessors_data[f"{ordered_predecessor}_{-lag}"] = np.roll( based_on[
                    ordered_predecessor
                ],lag)

                # mask the shifted values that went over the zero index
                if lag > 0:
                    predecessors_data[f"{ordered_predecessor}_{-lag}"][:lag] = np.nan
                elif lag < 0:
                    predecessors_data[f"{ordered_predecessor}_{-lag}"][lag:] = np.nan

        else:
            predecessors_data[ordered_predecessor] = based_on[ordered_predecessor]
    return pd.DataFrame(predecessors_data)


def extract_lag_level_graph(G: nx.DiGraph, max_lag: int) -> nx.DiGraph:
    """Extracts a graph with only edges that have time lag less than or equal to the maximum lag."""
    current_graph = G.copy()

    for lag in range(0, max_lag + 1):
        for u, v, data in G.edges(data=True):
            if "time_lag" not in data:
                timelag = (0,)
            else:
                timelag = data["time_lag"]
            if not isinstance(timelag, tuple):
                timelag = (timelag,)

            if all(t > lag for t in timelag):
                current_graph.remove_edge(u, v)
    return current_graph


def temporal_topological_sort(G: nx.DiGraph):
    """Performs a topological sort on the graph, considering time lags. G must be a directed acyclic graph (DAG) with edges having a 'time_lag' attribute."""

    expanded_G = lagged_graph(G)
    expanded_order = list(nx.topological_sort(expanded_G))
    order = list(reversed(dict.fromkeys(reversed([n for n, _ in expanded_order]))))

    return order


def strongly_connected_components_sort(G: nx.DiGraph):
    for generation in strongly_connected_components_generations(G):
        yield from generation


def strongly_connected_components_generations(G: nx.DiGraph) -> list[list[list[str]]]:
    scc: list[set[str]] = list(nx.strongly_connected_components(G))

    contracted_G = G.copy()
    contracted_G.remove_edges_from(nx.selfloop_edges(contracted_G))

    for s in scc:
        contracted_node_set(contracted_G, s, self_loops=False)

    if nx.algorithms.dag.has_cycle(contracted_G):
        raise ValueError("Graph still contains a cycle")

    generations: list[set[str]] = nx.topological_generations(contracted_G)

    expanded_generations: list[list[list[str]]] = []
    for generation in generations:
        expanded_generation: list[list[str]] = []
        for node in generation:
            sccs_with_node = [s for s in scc if node in s]
            if len(sccs_with_node) != 1:
                raise ValueError(f"Node {node} is not part of a single SCC")
            scc_with_node = sccs_with_node[0]
            expanded_generation.append(scc_with_node)

        expanded_generations.append(expanded_generation)

    return expanded_generations


def contracted_node_set(G: nx.DiGraph, nodes: set[str], self_loops: bool = True):
    nodes = nodes.copy()
    start_node = nodes.pop()
    for node in nodes:
        nx.contracted_nodes(G, start_node, node, self_loops=self_loops, copy=False)


def lagged_graph(G: nx.DiGraph, max_lag=10):
    expanded_G = nx.DiGraph()

    for t in range(max_lag):
        for u, v, data in G.edges(data=True):
            lag_tuple: tuple[int] = data.get("time_lag", (0,))
            for lag in lag_tuple:
                source_time = t - lag
                target_time = t
                if source_time >= 0:
                    expanded_G.add_edge((u, source_time), (v, target_time))

    return expanded_G


def _parent_samples_of(
    node: Any, scm: ProbabilisticCausalModel, samples: dict[np.ndarray]
) -> np.ndarray:
    predecessors_data = timelag_data(scm, node, samples)
    return predecessors_data[scm.graph.nodes[node][PARENTS_DURING_FIT]].to_numpy()


def draw_samples_incremental(
    causal_model: ProbabilisticCausalModel,
    num_samples: int,
    observed_datas: list[pd.DataFrame]= [],
) -> dict[str,np.ndarray]:
    """Draws new joint samples from the given graphical causal model. This is done by first generating random samples
    from root nodes and then propagating causal downstream effects through the graph.
    :param causal_model: New samples are generated based on the given causal model.
    :param num_samples: Number of samples to draw.
    :return: A pandas data frame where columns correspond to the nodes in the graph and rows to the drawn joint samples.
    """
    validate_causal_graph(causal_model.graph)
    if any(not isinstance(od, pd.DataFrame) for od in observed_datas):
        raise ValueError("observed_data must be a list of pandas dataframes")

    sorted_nodes = temporal_topological_sort(causal_model.graph)
    current_region_length = num_samples
    drawn_samples: dict[str, np.ndarray] = {
    }

    # each generation must be evaluated in parallel
    generations = strongly_connected_components_sort(causal_model.graph)

    for generation in generations:
        sG = nx.subgraph(causal_model.graph, generation)
        is_cyclic = nx.algorithms.dag.has_cycle(sG)
        gen_nodes = sorted(list(generation), key=lambda x: sorted_nodes.index(x))

        if is_cyclic:
            # iterative approach creates empty arrays upfront
            #create empty arrays
            for node in gen_nodes:
                drawn_samples[node] = np.full(current_region_length,np.nan)

            # fill with observed data
            for observed_data in observed_datas:
                for col in observed_data.columns:
                    if col in gen_nodes:
                        drawn_samples[col][:len(observed_data[col])] = observed_data[col].to_numpy()

            for iteration in range(current_region_length):
                for node in gen_nodes:

                    if not np.isnan(drawn_samples[node][iteration]):
                        print(f"Using intial sample for {node} at row {iteration}.")
                        continue

                    causal_mechanism = causal_model.causal_mechanism(node)

                    if is_root_node(causal_model.graph, node):
                        drawn_samples[node][iteration] = (
                            causal_mechanism.draw_samples(1).squeeze()
                        )
                    else:
                        _parent_samples = _parent_samples_of(
                            node, causal_model, drawn_samples
                        )[: iteration + 1,]
                       
                        x_nan_mask = pd.isna(_parent_samples).any(axis=1)
                        _parent_samples[x_nan_mask] = 0.0
                        drawn_samples[node][iteration] = (
                            causal_mechanism.draw_samples(_parent_samples).reshape(
                                (-1)
                            )[-1]
                        )

        else:
            for node in gen_nodes:
                causal_mechanism = causal_model.causal_mechanism(node)

                observed_data_node = np.array([])
                # fill with observed data
                for observed_data in observed_datas:
                    for col in observed_data.columns:
                        if col in gen_nodes:
                            observed_data_node = observed_data[col].to_numpy()


                n_filled_rows: int = len(observed_data_node)

                if is_root_node(causal_model.graph, node):

                    samples = causal_mechanism.draw_samples(current_region_length - n_filled_rows).squeeze()

                else:
                    parent_samples = _parent_samples_of(node, causal_model, drawn_samples)[n_filled_rows:]
                    samples = causal_mechanism.draw_samples(parent_samples).squeeze()

                    #shortening for aggregation
                    if len(samples)< len(parent_samples):
                        current_region_length = len(samples) + n_filled_rows

                drawn_samples[node] = np.concatenate([observed_data_node,samples])



    return drawn_samples
