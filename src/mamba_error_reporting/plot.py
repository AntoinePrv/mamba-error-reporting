import collections
from typing import TypeVar

import libmambapy
import matplotlib.pyplot as plt
import networkx as nx

import mamba_error_reporting as mer

N = TypeVar("N")
E = TypeVar("E")


def plot_dag(
    graph: nx.DiGraph, node_labels: dict[N, str] | None = None, edge_labels: dict[E, str] | None = None
) -> None:
    plt.figure(figsize=(10, 6), dpi=300)

    # Position using levels
    pos = {}
    for level, nodes in enumerate(nx.topological_generations(graph)):
        nodes = sorted(nodes, key=lambda n: graph[n].get("name", "None"))
        length = max(len(nodes) - 1, 1)
        pos.update({node: (j / length, -level - 0.2 * (j % 2)) for j, node in enumerate(nodes)})

    options = {"node_size": 800, "alpha": 0.5}
    nx.draw_networkx_nodes(graph, pos, node_color="blue", **options)
    nx.draw_networkx_edges(graph, pos, **options)

    if node_labels is not None:
        nx.draw_networkx_labels(
            graph, pos, collections.defaultdict(lambda _: "unknown", node_labels), font_size=7
        )
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(
            graph, pos, collections.defaultdict(lambda _: "", edge_labels), font_size=7
        )

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def plot_solvable_dag(pb_data: mer.algorithm.ProblemData) -> None:
    def repr_pkg_info(p: libmambapy.PackageInfo) -> str:
        return f"{p.name}-{p.version}-{p.build_number}"

    plot_dag(
        pb_data.graph, node_labels={n: repr_pkg_info(pb_data.package_info[n]) for n in pb_data.graph.nodes}
    )


def plot_group_dag(pb_data: mer.algorithm.ProblemData, cp_data: mer.algorithm.CompressionData) -> None:
    names = mer.algorithm.Names(pb_data=pb_data, cp_data=cp_data)
    node_labels = {
        group_id: "{name}-[{versions}]".format(
            name=names.group_name(group_id),
            versions=names.solv_group_versions_trunc(group_id),
        )
        for group_id in cp_data.graph.nodes
    }

    edge_labels = {}
    for e, attr in cp_data.graph.edges.items():
        dep_names = {names.dependency_name(dep_id) for dep_id in attr["dependency_ids"]}
        if len(dep_names) == 1:
            edge_labels[e] = next(iter(dep_names))
        else:
            prefix = mer.utils.common_prefix(dep_names).strip()
            edge_labels[e] = "{prefix}-[{variants}]".format(
                    prefix=prefix, variants="|".join(v.removeprefix(prefix).strip() for v in dep_names)
            )
    plot_dag(cp_data.graph, node_labels=node_labels, edge_labels=edge_labels)
