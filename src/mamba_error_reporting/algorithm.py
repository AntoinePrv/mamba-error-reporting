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
    dependency_solvables: dict[DependencyId, list[SolvableId]]
    package_info: dict[SolvableId, libmambapy.PackageInfo]
    graph: nx.DiGraph

    @staticmethod
    def from_libsolv(solver: libmambapy.Solver, pool: libmambapy.Pool) -> ProblemData:
        graph = nx.DiGraph()
        dependency_names = {}
        dependency_solvables = {}
        package_info = {}
        problems_by_type = {}

        def add_solvable(id, pkg_info=None):
            graph.add_node(id)
            package_info[id] = pkg_info if pkg_info is not None else pool.id2pkginfo(id)

        def add_dependency(source_id, dep_id, dep_name):
            dependency_names[p.dep_id] = dep_name
            dependency_solvables[dep_id] = pool.select_solvables(dep_id)
            for s in dependency_solvables[dep_id]:
                add_solvable(s, pool.id2pkginfo(s))
                graph.add_edge(source_id, s, dependency_id=dep_id)

        for p in solver.all_problems_structured():
            problems_by_type.setdefault(p.type, []).append(p)

            # Root dependencies are in JOB with source and target not useful (except 0)
            if (p.type == libmambapy.SolverRuleinfo.SOLVER_RULE_JOB):
                # FIXME hope -1 is not taken
                source_id = 0 if p.source() is None else -1
                add_solvable(source_id, libmambapy.PackageInfo(f"root-{source_id}", "", "", 0))
                add_dependency(source_id, p.dep_id, p.dep())
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
            dependency_solvables=dependency_solvables,
            package_info=package_info,
            problems_by_type=problems_by_type,
        )

    @functools.cached_property
    def package_conflicts(self) -> set[tuple[SolvableId, SolvableId]]:
        # PKG_SAME_NAME are in direct conflict
        out = {
            (p.source_id, p.target_id)
            for p in self.problems_by_type.get(libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_SAME_NAME, [])
        }
        for p in self.problems_by_type.get(libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_CONSTRAINS, []):
            solvs = self.dependency_solvables[p.dep_id]
            # Writting PKG_CONSTRAINS as in conlfict with resolved dependency solvables
            # yields better strucutre (pkg in conflicts have the same name)
            if len(solvs) > 1:
                out.update({(p.target_id, s) for s in solvs})
            # However sometimes there are no such pacakges (e.g. with virtual pacakges) so we
            # use target_id to avoid dropping the conflict
            else:
                out.add((p.source_id, p.target_id))
        # Make the output symetric
        out.update({(b, a) for (a, b) in out})
        return out

    @functools.cached_property
    def package_missing(self) -> dict[SolvableId, set[DependencyId]]:
        nothing_rules = [
            libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_NOTHING_PROVIDES_DEP,
            libmambapy.SolverRuleinfo.SOLVER_RULE_JOB_NOTHING_PROVIDES_DEP,
        ]
        out = {}
        for rule in nothing_rules:
            for p in self.problems_by_type.get(rule, []):
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
            and pb_data.package_missing.get(n1) == pb_data.package_missing.get(n2)
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
    graph = nx.DiGraph()
    # Add all nodes even if they don't have edges
    graph.add_nodes_from(groups.group_to_solv.keys())
    # Add edges and group dependency ids
    for (a, b), attr in pb_data.graph.edges.items():
        grp_a, grp_b = groups.solv_to_group[a], groups.solv_to_group[b]
        if not graph.has_edge(grp_a, grp_b):
            graph.add_edge(grp_a, grp_b, dependency_ids=set())
        graph.edges[(grp_a, grp_b)]["dependency_ids"].add(attr["dependency_id"])
    return CompressionData(graph=graph, groups=groups)


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
    # A single node with no ancestors or successors.
    standalone = enum.auto()
    # A root node with no ancestors and at least one successor.
    root = enum.auto()
    # A leaf node with at least one ancestors and no successor.
    leaf = enum.auto()
    # A node that has already been visited, in a DAG in must have at least one ancestor.
    visited = enum.auto()
    # Indicate the begining of a dependency split (multiple edges with same dep_id).
    split = enum.auto()
    # A regular node with at least one ancestor and at least one successor.
    diving = enum.auto()


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
            # A group can be a successor of multiple dep_id, even for a single node
            for dep_id in graph.edges[(node, s)]["dependency_ids"]:
                successors.setdefault(dep_id, []).append(s)

        # Check if the node is part of a dependency split by versions
        node_is_in_split = (old_node is not None) and is_multi[dep_id_from]

        # If the node is the first being visited in a version split
        if node_is_in_split and (dep_id_from not in visited_multi):
            visited_multi.add(dep_id_from)
            yield (depth, dep_id_from, node, ExplanationType.split, node_is_in_split)
        depth += node_is_in_split

        if len(successors) == 0:
            visited.add(node)
            yield (
                depth,
                dep_id_from,
                node,
                ExplanationType.standalone if old_node is None else ExplanationType.leaf,
                node_is_in_split,
            )
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

            yield (
                depth,
                dep_id_from,
                node,
                ExplanationType.root if old_node is None else ExplanationType.diving,
                node_is_in_split,
            )


@dataclasses.dataclass
class Names:
    pb_data: ProblemData
    cp_data: CompressionData

    def group_name(self, group_id: GroupId) -> str:
        sample_solv_id = next(iter(self.cp_data.groups.group_to_solv[group_id]))
        return self.pb_data.package_info[sample_solv_id].name

    def group_versions(self, group_id: GroupId) -> list[str]:
        unique_versions = set(
            [self.pb_data.package_info[s].version for s in self.cp_data.groups.group_to_solv[group_id]]
        )
        return sorted(unique_versions, key=packaging.version.parse)

    def group_versions_trunc(self, group_id: GroupId) -> str:
        return mer.utils.repr_trunc(self.group_versions(group_id), sep="|")

    def dependency_name(self, dep_id: DependencyId) -> str:
        return self.pb_data.dependency_names[dep_id]


@dataclasses.dataclass
class LeafDescriptor:
    pb_data: ProblemData
    cp_data: CompressionData

    @functools.cached_property
    def conflicting_groups(self) -> dict[GroupId, set[GroupId]]:
        out = {}
        for s1, s2 in self.pb_data.package_conflicts:
            g1 = self.cp_data.groups.solv_to_group[s1]
            g2 = self.cp_data.groups.solv_to_group[s2]
            out.setdefault(g1, set()).add(g2)
        return out

    def leaf_has_conflict(self, group_id: GroupId) -> bool:
        return group_id in self.conflicting_groups

    def leaf_conflict(self, group_id: GroupId) -> set[GroupId]:
        return self.conflicting_groups[group_id]

    @functools.cached_property
    def nothing_provides_groups(self) -> dict[GroupId, set[DependencyId]]:
        out = {}
        for solv_id, dep_ids in self.pb_data.package_missing.items():
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
        path = list(path)
        message: list[str] = []
        for i, (self.depth, self.dep_id_from, self.group_id, type, self.node_is_in_split) in enumerate(path):
            if i == len(path) - 1:
                term = "."
            elif type in [ExplanationType.leaf, ExplanationType.visited]:
                term = ";"
            else:
                term = ""
            message += [
                self.indent * self.depth,
                *getattr(self, f"explain_{type.name}")(),
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
                name=self.pkg_name, versions=self.names.group_versions_trunc(self.group_id)
            )
        return self.dep_name


class ProblemExplainer(Explainer):
    def explain_standalone(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_problem(self.group_id):
            missing_dep_id = self.leaf_descriptor.leaf_problem(self.group_id)
            missing_dep_name = self.names.dependency_name(missing_dep_id)
            return (
                "The environment could not be satisfied because it requires the missing package ",
                self.color.unavailable(missing_dep_name),
            )
        return ("The environment could not be satisfied.",)

    def explain_root(self) -> tuple[str]:
        return ("The following packages could not be installed:",)

    def explain_diving(self) -> tuple[str]:
        return (self.pkg_repr, " which requires")

    def explain_split(self) -> tuple[str]:
        return (self.dep_name, " for which none of the following versions can be installed")

    def explain_leaf(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_conflict(self.group_id):
            conflict_ids = self.leaf_descriptor.leaf_conflict(self.group_id)
            conflict_names = ", ".join(self.color.unavailable(self.names.group_name(g)) for g in conflict_ids)
            return (self.pkg_repr, ", which conflicts with any installable versions of ", conflict_names)
        elif self.leaf_descriptor.leaf_has_problem(self.group_id):
            missing_dep_id = self.leaf_descriptor.leaf_problem(self.group_id)
            missing_dep_name = self.names.dependency_name(missing_dep_id)
            return (
                self.pkg_repr,
                ", which requires the missing package ",
                self.color.unavailable(missing_dep_name),
            )
        else:
            return (", which cannot be installed for an unknown reason",)

    def explain_visited(self) -> tuple[str]:
        return (self.pkg_repr, ", which cannot be installed (as previously explained)")


class InstallExplainer(Explainer):
    def explain_standalone(self) -> tuple[str]:
        return tuple()

    def explain_root(self) -> tuple[str]:
        return ("The following requirements limit the allowed versions of downstream packages:",)

    def explain_diving(self) -> tuple[str]:
        return (self.pkg_repr, " is requested, and it requires" if self.depth == 1 else ", which requires")

    def explain_split(self) -> tuple[str]:
        return (self.dep_name, " is requested," if self.depth == 1 else "", " with the potential options")

    def explain_leaf(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_problem(self.group_id):
            missing_dep_id = self.leaf_descriptor.leaf_problem(self.group_id)
            missing_dep_name = self.names.dependency_name(missing_dep_id)
            return (
                self.pkg_repr,
                ", which requires the non-existent package ",
                self.color.unavailable(missing_dep_name),
            )
        return (
            self.color.available(self.pkg_repr),
            " is directly requested" if self.depth == 1 else ", which can be installed",
        )

    def explain_visited(self) -> tuple[str]:
        return (self.pkg_repr, ", which can be installed (as previously explained)")


# Groups may be superset of the dependencies
def make_dep_id_to_groups(graph: nx.DiGraph) -> dict[DependencyId, set[GroupId]]:
    groups: dict[DependencyId, set[GroupId]] = {}
    for (_, s), attr in graph.edges.items():
        for d in attr["dependency_ids"]:
            groups.setdefault(d, set()).add(s)
    return groups


def header_message(pb_data: ProblemData, color: type = Color) -> str | None:
    deps = {
        pb_data.dependency_names[pb_data.graph.edges[e]["dependency_id"]] for e in pb_data.graph.out_edges(0)
    }
    if len(deps) == 0:
        return None
    return "Could not find any installable versions for requested package{s} {pkgs}.".format(
        s=("s" if len(deps) > 1 else ""), pkgs=", ".join(color.unavailable(d) for d in deps)
    )


def explain_graph(pb_data: ProblemData, cp_data: CompressionData) -> str:
    names = Names(pb_data, cp_data)
    leaf_descriptor = LeafDescriptor(pb_data, cp_data)

    dep_id_to_groups = make_dep_id_to_groups(cp_data.graph)
    is_multi = {dep_id: len(group) > 1 for dep_id, group in dep_id_to_groups.items()}

    header_msg = header_message(pb_data)

    if (install_root := cp_data.groups.solv_to_group.get(-1)) is not None:
        inst_explainer = InstallExplainer(names, leaf_descriptor)
        install_msg = inst_explainer.explain(
            mer.algorithm.explanation_path(cp_data.graph, install_root, set(), is_multi, explore_all=True)
        )
    else:
        install_msg = None

    problem_root = cp_data.groups.solv_to_group[0]
    pb_explainer = ProblemExplainer(names, leaf_descriptor)
    problem_msg = pb_explainer.explain(
        mer.algorithm.explanation_path(cp_data.graph, problem_root, set(), is_multi, explore_all=False)
    )

    return "Error: " + "\n\n".join([m for m in [header_msg, install_msg, problem_msg] if m is not None])
