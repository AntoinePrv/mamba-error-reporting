from __future__ import annotations

import dataclasses
import itertools
from typing import List, Sequence, Any

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

        def add_solvable(id, pkg_info = None):
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

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, GroupId):
            return self.id.__lt__(other.id)
        return NotImplemented


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
