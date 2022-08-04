from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
from typing import Callable, Iterable, NewType, Sequence, TypeVar

import libmambapy
import networkx as nx
import packaging.version

import mamba_error_reporting as mer

DependencyId = NewType("DependencyId", int)
SolvableId = NewType("SolvableId", int)
GroupId = NewType("GroupId", int)
NodeType = TypeVar("NodeType")
EdgeType = TypeVar("EdgeType")


############################
#  Initial data available  #
############################


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
    def package_conflicts(self) -> set[tuple[SolvableId, SolvableId]]:
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
    def package_nothing_provides(self) -> dict[SolvableId, set[DependencyId]]:
        out = {}
        for p in self.problems_by_type.get(
            libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_NOTHING_PROVIDES_DEP, []
        ):
            out.setdefault(p.source_id, set()).add(p.dep_id)
        return out

    @functools.cached_property
    def solvable_by_package_name(self) -> dict[str, set[SolvableId]]:
        out = {}
        for solv_id, pkg_info in self.package_info.items():
            out.setdefault(pkg_info.name, set()).add(solv_id)
        return out


#################################
#  Graph compression algorithm  #
#################################


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
            ((n1, n2) not in pb_data.package_conflicts)
            # Packages must have same missing dependencies (when that is the case)
            and pb_data.package_nothing_provides.get(n1) == pb_data.package_nothing_provides.get(n2)
            # Packages must have the same successors
            and same_children(n1, n2)
        )

    groups = SolvableGroups()
    counter = Counter()

    for solvs in pb_data.solvable_by_package_name.values():
        cliques = greedy_clique_partition(compatibility_graph(solvs, compatible=compatible))
        for c in cliques:
            groups.set_solvables(counter(), c)
    return groups


@dataclasses.dataclass
class CompressionData:
    graph: nx.DiGraph
    groups: SolvableGroups


def compress_graph(pb_data: ProblemData) -> CompressionData:
    groups = compress_solvables(pb_data)
    compressed_graph = nx.DiGraph()
    compressed_graph.add_edges_from(
        (groups.solv_to_group[a], groups.solv_to_group[b], attr)
        for (a, b), attr in pb_data.graph.edges.items()
    )
    return CompressionData(graph=compressed_graph, groups=groups)


############################
#  Generic graph routines  #
############################


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


############################
#  Error message crafting  #
############################


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


@dataclasses.dataclass
class Names:
    pb_data: ProblemData
    cp_data: CompressionData

    def group_name(self, group_id: GroupId) -> str:
        sample_solv_id = next(iter(self.cp_data.groups.group_to_solv[group_id]))
        return self.pb_data.package_info[sample_solv_id].name

    def group_versions(self, group_id: GroupId) -> list[str]:
        return list(
            set([self.pb_data.package_info[s].version for s in self.cp_data.groups.group_to_solv[group_id]])
        )

    def group_versions_trunc(
        self, group_id: GroupId, trunc_threshold: int = 5, trunc_show: int = 2, trunc_str: str = "..."
    ) -> list[str]:
        versions = self.group_versions(group_id)
        if len(versions) > trunc_threshold:
            return versions[:trunc_show] + [trunc_str] + versions[-trunc_show:]
        return versions

    def group_versions_range(self, group_id: GroupId, range_sep: str = "->") -> list[str]:
        parsed_versions = [packaging.version.parse(v) for v in self.group_versions(group_id)]
        return [str(min(parsed_versions)), range_sep, str(max(parsed_versions))]

    def dependency_name(self, dep_id: DependencyId) -> str:
        return self.pb_data.dependency_names[dep_id]


@dataclasses.dataclass
class LeafDescriptor:
    pb_data: ProblemData
    cp_data: CompressionData

    @functools.cached_property
    def conflicting_groups(self) -> dict[GroupId, GroupId]:
        return {
            self.cp_data.groups.solv_to_group[s1]: self.cp_data.groups.solv_to_group[s2]
            for s1, s2 in self.pb_data.package_conflicts
        }

    def leaf_has_conflict(self, group_id: GroupId) -> bool:
        return group_id in self.conflicting_groups

    def leaf_conflict(self, group_id: GroupId) -> GroupId:
        return self.conflicting_groups[group_id]

    @functools.cached_property
    def nothing_provides_groups(self) -> dict[GroupId, set[DependencyId]]:
        out = {}
        for solv_id, dep_ids in self.pb_data.package_nothing_provides.items():
            out.setdefault(self.cp_data.groups.solv_to_group[solv_id], set()).update(dep_ids)
        return out

    def leaf_has_problem(self, group_id: GroupId) -> bool:
        return group_id in self.nothing_provides_groups

    def leaf_problem(self, group_id: GroupId) -> DependencyId:
        # If there are more type of problem return an enum/union along the dep name
        dep_ids = self.nothing_provides_groups[group_id]
        # Picking only one of the missing dependencies
        return next(iter(dep_ids))


@dataclasses.dataclass
class Color:
    @staticmethod
    def available(msg: str) -> str:
        return mer.color.color(msg, fg="green", style="bold")

    @staticmethod
    def unavailable(msg: str) -> str:
        return mer.color.color(msg, fg="red", style="bold")


@dataclasses.dataclass
class Explainer:
    names: Names
    leaf_descriptor: LeafDescriptor
    indent: str = "  "
    color: type = Color

    def explain(self, path: Iterable[int, DependencyId, GroupId, ExplanationType, bool]) -> str:
        message: list[str] = []
        for (self.depth, self.dep_id_from, self.group_id, type, self.node_is_in_split), (
            next_depth,
            *_,
        ) in mer.utils.pairwise(path, last=(1,)):
            if self.dep_id_from is None:
                continue
            if next_depth == 1:
                term = "."
            elif type == ExplanationType.leaf:
                term = ";"
            else:
                term = ""
            message += [
                self.indent * self.depth,
                "".join(getattr(self, f"explain_{type.value}")()),
                term,
                "\n",
            ]

        message.pop()  # Last line break
        return "".join(message)

    @property
    def pkg_name(self) -> str:
        return self.names.group_name(self.group_id)

    @property
    def dep_name(self) -> str:
        return self.names.dependency_name(self.dep_id_from)

    @property
    def pkg_repr(self) -> str:
        if self.node_is_in_split:
            return "{name} [{versions}]".format(
                name=self.pkg_name, versions="|".join(self.names.group_versions_trunc(self.group_id))
            )
        return self.dep_name


class ProblemExplainer(Explainer):
    def explain_diving(self) -> tuple[str]:
        return (
            "This implies that " if self.depth <= 1 else "",
            self.pkg_repr,
            " cannot be installed because it requires" if self.depth <= 1 else ", which requires",
        )

    def explain_split(self) -> tuple[str]:
        return (
            "This implies that " if self.depth <= 1 else "",
            self.dep_name,
            " cannot be installed because" if self.depth <= 1 else " for which",
            " none of the following versions can be installed",
        )

    def explain_leaf(self) -> tuple[str]:
        message = ("This implies that " if self.depth <= 1 else "", self.pkg_repr)

        if self.leaf_descriptor.leaf_has_conflict(self.group_id):
            conflict_id = self.leaf_descriptor.leaf_conflict(self.group_id)
            conflict_name = self.names.group_name(conflict_id)
            return message + (
                " cannot be installed because it" if self.depth <= 1 else ", which",
                f" conflicts with any possible versions of {self.color.unavailable(conflict_name)}",
            )
        elif self.leaf_descriptor.leaf_has_problem(self.group_id):
            missing_dep_id = self.leaf_descriptor.leaf_problem(self.group_id)
            missing_dep_name = self.names.dependency_name(missing_dep_id)
            return message + (
                " cannot be installed because it" if self.depth <= 1 else ", which",
                f" requires the missing package {self.color.unavailable(missing_dep_name)}",
            )
        else:
            return message + (
                "" if self.depth <= 1 else ", which",
                " cannot be installed for an unknown reason",
            )

    def explain_visited(self) -> tuple[str]:
        return (
            self.pkg_repr,
            "" if self.depth <= 1 else ", which",
            " cannot be installed (as previously explained)",
        )


class InstallExplainer(Explainer):
    def explain_diving(self) -> tuple[str]:
        return (
            "A package satisfying " if self.depth <= 1 else "",
            self.pkg_repr,
            " is requested, and it requires" if self.depth <= 1 else ", which requires",
        )

    def explain_split(self) -> tuple[str]:
        return (
            "A package satisfying " if self.depth <= 1 else "",
            self.dep_name,
            " is requested, so one of the following versions must be installed",
        )

    def explain_leaf(self) -> tuple[str]:
        message = ("A package satisfying " if self.depth <= 1 else "",)
        if self.leaf_descriptor.leaf_has_problem(self.group_id):
            missing_dep_id = self.leaf_descriptor.leaf_problem(self.group_id)
            missing_dep_name = self.names.dependency_name(missing_dep_id)
            return message + (
                self.pkg_repr,
                "" if self.depth <= 1 else ", which",
                f" requires the non-existent package {self.color.unavailable(missing_dep_name)}",
            )
        return (
            self.color.available(self.pkg_repr),
            " is requested and could be installed" if self.depth <= 1 else ", which could be installed",
        )

    def explain_visited(self) -> tuple[str]:
        return (
            self.pkg_repr,
            "" if self.depth <= 1 else ", which",
            " could be installed (as previously explained)",
        )


# Groups may be superset of the dependencies
def make_dep_id_to_groups(graph: nx.DiGraph) -> dict[DependencyId, set[GroupId]]:
    groups: dict[DependencyId, set[GroupId]] = {}
    for (_, s), attr in graph.edges.items():
        groups.setdefault(attr["dependency_id"], set()).add(s)
    return groups


def header_message(pb_data: ProblemData, cp_data: CompressionData, color: type = Color) -> str:
    problem_root = cp_data.groups.solv_to_group[0]
    # This won't work if there are mutliple problem dependencies
    sample_edge = next(iter(cp_data.graph.out_edges(problem_root)))
    problem_name = pb_data.dependency_names[cp_data.graph.edges[sample_edge]["dependency_id"]]
    problem_name = color.unavailable(problem_name)
    return f"Error: Could not install any versions from requested package {problem_name}."


def explain_graph(pb_data: ProblemData, cp_data: CompressionData) -> str:
    names = Names(pb_data, cp_data)
    leaf_descriptor = LeafDescriptor(pb_data, cp_data)
    inst_explainer = InstallExplainer(names, leaf_descriptor)
    pb_explainer = ProblemExplainer(names, leaf_descriptor)

    install_root = cp_data.groups.solv_to_group[-1]
    problem_root = cp_data.groups.solv_to_group[0]

    dep_id_to_groups = make_dep_id_to_groups(cp_data.graph)
    is_multi = {dep_id: len(group) > 1 for dep_id, group in dep_id_to_groups.items()}

    header_msg = header_message(pb_data, cp_data)
    install_msg = inst_explainer.explain(
        mer.algorithm.explanation_path(cp_data.graph, install_root, set(), is_multi, explore_all=True)
    )
    problem_msg = pb_explainer.explain(
        mer.algorithm.explanation_path(cp_data.graph, problem_root, set(), is_multi, explore_all=False)
    )
    return f"{header_msg}\n\n{install_msg}\n\n{problem_msg}"
