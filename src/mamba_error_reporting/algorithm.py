from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
from typing import Callable, Iterable, NewType, Sequence, TypeVar

import libmambapy
import networkx as nx

DependencyId = NewType("DependencyId", int)
SolvableId = NewType("SolvableId", int)
GroupId = NewType("GroupId", int)
NodeType = TypeVar("NodeType")
EdgeType = TypeVar("EdgeType")


@dataclasses.dataclass
class ProblemData:
    problems_by_type: dict[libmambapy.SolverRuleinfo, list[libmambapy.SolverProblem]]
    dependency_names: dict[DependencyId, str]
    package_info: dict[SolvableId, libmambapy.PackageInfo]
    graph: nx.DiGraph

    @staticmethod
    def from_libsolv(solver: libmambapy.Solver, pool: libmambapy.Pool) -> ProblemData:
        graph = nx.DiGraph()
        dependency_names = {}
        package_info = {}
        problems_by_type = {}

        def add_solvable(id, pkg_info=None):
            graph.add_node(id)
            package_info[id] = pkg_info if pkg_info is not None else pool.id2pkginfo(id)

        def add_dependency(source_id, dep_id, dep_name):
            dependency_names[p.dep_id] = dep_name
            solvs = pool.select_solvables(dep_id)
            for s in solvs:
                add_solvable(s, pool.id2pkginfo(s))
                graph.add_edge(source_id, s, dependency_id=dep_id)

        for p in solver.all_problems_structured():
            problems_by_type.setdefault(p.type, []).append(p)

            # Root dependencies are in JOB with source not useful (except 0)
            if p.type == libmambapy.SolverRuleinfo.SOLVER_RULE_JOB:
                if p.source_id == 0:
                    add_solvable(0, libmambapy.PackageInfo("problem", "", "", 0))
                    add_dependency(0, p.dep_id, p.dep())
                else:
                    # FIXME hope that's not taken
                    add_solvable(-1, libmambapy.PackageInfo("installed", "", "", 0))
                    add_dependency(-1, p.dep_id, p.dep())
            elif p.type == libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_REQUIRES:
                add_solvable(p.source_id)
                add_dependency(p.source_id, p.dep_id, p.dep())
            elif p.type == libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_CONSTRAINS:
                # We do not add dependencies because they loop back to installed packages,
                # making it difficut to explain problems in the graph.
                add_solvable(p.source_id)
                add_solvable(p.target_id)
            else:
                if p.source() is not None:
                    add_solvable(p.source_id)
                if p.target() is not None:
                    add_solvable(p.target_id)
                if p.dep() is not None:
                    # Assuming source is valid
                    add_dependency(p.source_id, p.dep_id, p.dep())

        return ProblemData(
            graph=graph,
            dependency_names=dependency_names,
            package_info=package_info,
            problems_by_type=problems_by_type,
        )

    @functools.cached_property
    def pkg_conflicts(self) -> set[tuple[SolvableId, SolvableId]]:
        out = {
            (p.source_id, p.target_id)
            for rule_info in (
                libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_SAME_NAME,
                libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_CONSTRAINS,
            )
            for p in self.problems_by_type.get(rule_info, [])
        }
        out = out.union({(b, a) for (a, b) in out})
        return out

    @functools.cached_property
    def pkg_nothing_provides(self) -> dict[SolvableId, set[DependencyId]]:
        out = {}
        for p in self.problems_by_type.get(
            libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_NOTHING_PROVIDES_DEP, []
        ):
            out.setdefault(p.source_id, set()).add(p.dep_id)
        return out


@dataclasses.dataclass
class Counter:
    cnt: int = 0

    def __call__(self) -> int:
        old = self.cnt
        self.cnt += 1
        return old


@dataclasses.dataclass
class SolvableGroups:
    solv_to_group: dict[SolvableId, GroupId] = dataclasses.field(default_factory=dict)
    group_to_solv: dict[GroupId, set[SolvableId]] = dataclasses.field(default_factory=dict)

    def set_solvables(self, group_id: GroupId, solvs: Sequence[SolvableId]) -> None:
        for s in self.group_to_solv.get(group_id, set()):
            del self.solv_to_group[s]
        self.solv_to_group.update({s: group_id for s in solvs})
        self.group_to_solv[group_id] = set(solvs)


def solvable_by_pkg_name(
    package_info: dict[SolvableId, libmambapy.PackageInfo]
) -> dict[str, set(SolvableId)]:
    out = {}
    for solv_id, pkg_info in package_info.items():
        out.setdefault(pkg_info.name, set()).add(solv_id)
    return out


def compatibility_graph(
    nodes: Sequence[SolvableId], compatible: Callable[[SolvableId, SolvableId], bool]
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((n1, n2) for n1, n2 in itertools.combinations(nodes, 2) if compatible(n1, n2))
    return graph


def greedy_clique_partition(graph: nx.Graph) -> list[list[NodeType]]:
    graph = graph.copy()
    cliques = []
    while len(graph) > 0:
        max_clique = max(nx.find_cliques(graph), key=len)
        cliques.append(max_clique)
        graph.remove_nodes_from(max_clique)
    return cliques


def compress_solvables(pb_data: ProblemData) -> SolvableGroups:
    def same_children(n1: SolvableId, n2: SolvableId) -> bool:
        return set(pb_data.graph.successors(n1)) == set(pb_data.graph.successors(n2))

    def compatible(n1: SolvableId, n2: SolvableId) -> bool:
        return (
            # Packages must not be in conflict
            ((n1, n2) not in pb_data.pkg_conflicts)
            # Packages must have same missing dependencies (when that is the case)
            and pb_data.pkg_nothing_provides.get(n1) == pb_data.pkg_nothing_provides.get(n2)
            # Packages must have the same successors
            and same_children(n1, n2)
        )

    groups = SolvableGroups()
    counter = Counter()

    for solvs in solvable_by_pkg_name(pb_data.package_info).values():
        cliques = greedy_clique_partition(compatibility_graph(solvs, compatible=compatible))
        for c in cliques:
            groups.set_solvables(counter(), c)
    return groups


def compress_graph(pb_data: ProblemData) -> tuple[nx.DiGraph, SolvableGroups]:
    groups = compress_solvables(pb_data)
    compressed_graph = nx.DiGraph()
    compressed_graph.add_edges_from(
        (groups.solv_to_group[a], groups.solv_to_group[b], attr)
        for (a, b), attr in pb_data.graph.edges.items()
    )
    return compressed_graph, groups


def ancestors_subgraph(graph: nx.DiGraph, node: NodeType) -> nx.DiGraph:
    ancestors = set(nx.ancestors(graph, node))
    ancestors.add(node)
    return nx.subgraph_view(graph, filter_node=lambda n: (n in ancestors))


def successors_subgraph(graph: nx.DiGraph, node: NodeType) -> nx.DiGraph:
    successors = set(nx.successors(graph, node))
    successors.add(node)
    return nx.subgraph_view(graph, filter_node=lambda n: (n in successors))


def find_root(graph: nx.DiGraph, node: NodeType) -> NodeType:
    visited = set()
    while graph.in_degree(node) > 0:
        if node in visited:
            raise RuntimeError("Cycle detected")
        visited.add(node)
        node = next(graph.predecessors(node))
    return node


def find_leaves(graph: nx.DiGraph, node: NodeType) -> list[NodeType]:
    leaves = []
    to_visit = [node]
    visited = set()
    while len(to_visit) > 0:
        node = to_visit.pop()
        visited.add(node)
        if graph.out_degree(node) > 0:
            to_visit += [n for n in graph.successors(node) if n not in visited]
        else:
            leaves.append(node)
    return leaves


class ExplanationType(enum.Enum):
    leaf = "leaf"
    visited = "visited"
    split = "split"
    diving = "diving"


def explanation_path(
    graph: nx.DiGraph,
    root: NodeType,
    visited: set[GroupId],
    is_multi: dict[GroupId, bool],
    explore_all: bool = False,
) -> Iterable[int, DependencyId, GroupId, ExplanationType, bool]:
    visited_multi = set()
    to_visit = [(None, None, root, 0)]
    while len(to_visit) > 0:
        dep_id_from, old_node, node, depth = to_visit.pop()

        successors: dict[DependencyId, list[GroupId]] = {}
        for s in graph.successors(node):
            successors.setdefault(graph.edges[(node, s)]["dependency_id"], []).append(s)

        # Check if the node is part of a dependency split by versions
        node_is_in_split = (old_node is not None) and (
            is_multi[graph.edges[(old_node, node)]["dependency_id"]]
        )
        # If the node is the first being visited in a version split
        if node_is_in_split and (dep_id_from not in visited_multi):
            visited_multi.add(dep_id_from)
            yield (depth, dep_id_from, node, ExplanationType.split, node_is_in_split)
        depth += node_is_in_split

        if len(successors) == 0:
            visited.add(node)
            yield (depth, dep_id_from, node, ExplanationType.leaf, node_is_in_split)
        elif node in visited:
            yield (depth, dep_id_from, node, ExplanationType.visited, node_is_in_split)
        else:
            visited.add(node)

            if explore_all:
                dep_ids = list(successors.keys())
            else:
                # If we only explore one dep_id, we choose the one with minimum splits
                dep_ids = [min(successors, key=lambda id: len(successors[id]))]

            to_visit += [(dep_id, node, s, depth + 1) for dep_id in dep_ids for s in successors[dep_id]]

            yield (depth, dep_id_from, node, ExplanationType.diving, node_is_in_split)
