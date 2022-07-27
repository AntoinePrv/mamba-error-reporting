from __future__ import annotations

import dataclasses
import itertools
from typing import List, Sequence

import libmambapy
import networkx as nx

DependencyId = int
SolvableId = int


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

        for p in solver.all_problems_structured():
            # Root dependencies are in JOB with source not useful
            if p.type == libmambapy.SolverRuleinfo.SOLVER_RULE_JOB:
                p.source_id = 0
                package_info[p.source_id] = libmambapy.PackageInfo("root", "", "", 0)

            problems_by_type.setdefault(p.type, []).append(p)
            if p.source() is not None:
                package_info[p.source_id] = pool.id2pkginfo(p.source_id)
            if p.target() is not None:
                package_info[p.target_id] = pool.id2pkginfo(p.target_id)
            if p.dep() is not None:
                solvs = pool.select_solvables(p.dep_id)
                dependency_names[p.dep_id] = p.dep()
                package_info.update({s: pool.id2pkginfo(s) for s in solvs})
                graph.add_edges_from(((p.source_id, s) for s in solvs), dependency_id=p.dep_id)

        return ProblemData(
            graph=graph,
            dependency_names=dependency_names,
            package_info=package_info,
            problems_by_type=problems_by_type,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GroupId:
    """Strongly typed solvable set ID.

    Used to make new dependency-like ids without number conflict.
    """

    id: int

    @classmethod
    def new(cls) -> GroupId:
        if not hasattr(cls, "count"):
            cls.count = -1
        cls.count += 1
        return GroupId(cls.count)


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


def package_same_name_edge(
    problems_by_type: dict[libmambapy.SolverRuleinfo, list[libmambapy.SolverProblem]], symetric: bool = True
) -> List[tuple[SolvableId, SolvableId]]:
    out = [
        (p.source_id, p.target_id)
        for p in problems_by_type.get(libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_SAME_NAME, [])
    ]
    if symetric:
        out += [(b, a) for (a, b) in out]
    return out


def compressable_solvable_graph(
    graph: nx.DiGraph, nodes: Sequence[SolvableId], children: dict[SolvableId, set[SolvableId]]
) -> nx.Graph:
    compatibles = nx.Graph()
    compatibles.add_nodes_from(nodes)
    compatibles.add_edges_from(
        (n1, n2) for n1, n2 in itertools.combinations(nodes, 2) if children[n1] == children[n2]
    )
    return compatibles


def greedy_clique_partition(graph: nx.Graph) -> List[List[SolvableId]]:
    graph = graph.copy()
    cliques = []
    while len(graph) > 0:
        max_clique = max(nx.find_cliques(graph), key=len)
        cliques.append(max_clique)
        graph.remove_nodes_from(max_clique)
    return cliques


def compress_solvables(pb_data: ProblemData) -> SolvableGroups:
    groups = SolvableGroups()

    # Add Conflicts edges to avoid merging the nodes
    graph = pb_data.graph.copy()
    graph.add_edges_from(package_same_name_edge(pb_data.problems_by_type))

    for solvs in solvable_by_pkg_name(pb_data.package_info).values():
        cliques = greedy_clique_partition(
            compressable_solvable_graph(graph, solvs, children={n: set(graph.successors(n)) for n in solvs})
        )
        for c in cliques:
            groups.set_solvables(GroupId.new(), c)
    return groups


def compress_graph(pb_data: ProblemData) -> tuple[nx.DiGraph, SolvableGroups]:
    groups = compress_solvables(pb_data)
    compressed_graph = nx.DiGraph()
    compressed_graph.add_edges_from(
        (groups.solv_to_group[a], groups.solv_to_group[b], attr)
        for (a, b), attr in pb_data.graph.edges.items()
    )
    return compressed_graph, groups
