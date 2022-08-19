from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
import re
from typing import Callable, Iterable, NewType, Sequence, TypeVar

import libmambapy
import networkx as nx
import packaging.version

import mamba_error_reporting as mer

SolvableId = NewType("SolvableId", int)
SolvableGroupId = NewType("SolvableGroupId", int)
DependencyId = NewType("DependencyId", int)
DependencyGroupId = NewType("DependencyGroupId", int)

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
            job_rules = [
                libmambapy.SolverRuleinfo.SOLVER_RULE_JOB,
                libmambapy.SolverRuleinfo.SOLVER_RULE_JOB_NOTHING_PROVIDES_DEP,
            ]
            if p.type in job_rules:
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

    @property
    def problems(self) -> Iterable[libmambapy.PackageInfo]:
        return itertools.chain.from_iterable(self.problems_by_type.values())

    @functools.cached_property
    def package_conflicts(self) -> set[tuple[SolvableId, SolvableId]]:
        # PKG_SAME_NAME are in direct conflict
        name_rule = libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_SAME_NAME
        out = {(p.source_id, p.target_id) for p in self.problems_by_type.get(name_rule, [])}
        # PKG_CONSTRAINS are in conlfict with resolved dependency solvables.
        # See package_missing for cases where the solvables resolves to an empty list.
        cons_rule = libmambapy.SolverRuleinfo.SOLVER_RULE_PKG_CONSTRAINS
        out.update(
            {
                (p.target_id, s)
                for p in self.problems_by_type.get(cons_rule, [])
                for s in self.dependency_solvables[p.dep_id]
            }
        )
        # Make the output symetric
        out.update({(b, a) for (a, b) in out})
        return out

    @functools.cached_property
    def package_missing(self) -> dict[SolvableId, set[DependencyId]]:
        """return packages that have a dependency with an empty set solvables."""
        # This include PKG_CONSTRAINS type of conlicts, likely a virtual package in that case.
        # It's not perfect but it deeply simplify their handling.
        out = {}
        for p in self.problems:
            if (
                (p.dep_id in self.dependency_solvables)  # Guard against dependency ids we excluded
                and (p.source_id in self.package_info)  # Guard against solvable ids we excluded
                and len(self.dependency_solvables[p.dep_id]) == 0  # Actual logic
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
    solv_to_group: dict[SolvableId, SolvableGroupId] = dataclasses.field(default_factory=dict)
    group_to_solv: dict[SolvableGroupId, set[SolvableId]] = dataclasses.field(default_factory=dict)

    group_counter: Counter = dataclasses.field(default_factory=Counter, init=False)

    def add(self, solvs: Sequence[SolvableId]) -> SolvableGroupId:
        grp_id = self.group_counter()
        self.solv_to_group.update({s: grp_id for s in solvs})
        self.group_to_solv[grp_id] = set(solvs)
        return grp_id


@dataclasses.dataclass
class DependencyGroups:
    dep_to_groups: dict[DependencyId, set[DependencyGroupId]] = dataclasses.field(default_factory=dict)
    group_to_deps: dict[DependencyGroupId, set[DependencyId]] = dataclasses.field(default_factory=dict)

    group_counter: Counter = dataclasses.field(default_factory=Counter, init=False)

    def search_grp_id(self, deps: Sequence[DependencyId]) -> DependencyGroupId | None:
        # Maybe not great in term of complexity
        for dep_grp_id, exisiting_deps in self.group_to_deps.items():
            if set(deps) == exisiting_deps:
                return dep_grp_id
        return None

    def add(self, deps: Sequence[DependencyId]) -> DependencyGroupId:
        # Search if there is a group id that has the same dependencies
        if (grp_id := self.search_grp_id(deps)) is not None:
            return grp_id

        # Otherwise create new one
        grp_id = self.group_counter()
        for d in deps:
            self.dep_to_groups.setdefault(d, set()).add(grp_id)
        self.group_to_deps[grp_id] = set(deps)
        return grp_id


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

    for solvs in pb_data.solvable_by_package_name.values():
        cliques = greedy_clique_partition(compatibility_graph(solvs, compatible=compatible))
        for c in cliques:
            groups.add(c)
    return groups


@dataclasses.dataclass
class CompressionData:
    graph: nx.DiGraph
    solv_groups: SolvableGroups
    dep_groups: DependencyGroups
    conflicting_groups: dict[SolvableGroupId, set[SolvableGroupId]]
    missing_groups: dict[SolvableGroupId, DependencyId]


def compress_graph(pb_data: ProblemData) -> CompressionData:
    solv_groups = compress_solvables(pb_data)

    # Compute conflicting groups
    conflicting_groups: dict[SolvableGroupId, set[SolvableGroupId]] = {}
    for s1, s2 in pb_data.package_conflicts:
        conflicting_groups.setdefault(solv_groups.solv_to_group[s1], set()).add(solv_groups.solv_to_group[s2])

    # From previous edges and new groups, comute new edges and theis set of dependencies.
    edges_deps: dict[tuple[SolvableGroupId, SolvableGroupId], set[DependencyId]] = {}
    for (a, b), attr in pb_data.graph.edges.items():
        grp_a, grp_b = solv_groups.solv_to_group[a], solv_groups.solv_to_group[b]
        edges_deps.setdefault((grp_a, grp_b), set()).add(attr["dependency_id"])

    graph = nx.DiGraph()
    dep_groups = DependencyGroups()

    # Add all nodes even if they don't have edges.
    graph.add_nodes_from(solv_groups.group_to_solv.keys())
    # Add all edges between group with a new dependecy group id.
    for (grp_a, grp_b), deps in edges_deps.items():
        graph.add_edge(grp_a, grp_b, dependency_group_id=dep_groups.add(deps))

    # Compute missing groups
    missing_deps: dict[SolvableGroupId, set[DependencyId]] = {}
    for solv_id, dep_ids in pb_data.package_missing.items():
        missing_deps.setdefault(solv_groups.solv_to_group[solv_id], set()).update(dep_ids)
    # FIXME Later we assume there is only a single package name in a dependency group but here
    # we are unsure if that is the case with pacakge_missing
    missing_groups: dict[SolvableGroupId, DependencyGroupId] = {
        solv_grp_id: dep_groups.add(dep_ids) for solv_grp_id, dep_ids in missing_deps.items()
    }

    return CompressionData(
        graph=graph,
        solv_groups=solv_groups,
        dep_groups=dep_groups,
        conflicting_groups=conflicting_groups,
        missing_groups=missing_groups,
    )


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


@dataclasses.dataclass
class ExplanationNode:
    solv_grp_id: SolvableGroupId | None
    solv_grp_id_from: SolvableGroupId | None
    dep_grp_id_from: DependencyGroupId | None
    depth: int
    type: ExplanationType
    in_split: bool

    @property
    def is_root(self) -> bool:
        return self.depth == 0


def explanation_path(
    graph: nx.DiGraph,
    root: NodeType,
    leaf_status: Callable[[ExplanationType], bool],
) -> list[ExplanationNode]:
    visited_nodes: set[SolvableGroupId] = set()

    def visit(
        solv_grp_id: SolvableGroupId, solv_grp_id_from: SolvableGroupId | None, depth: int, in_split: bool
    ) -> tuple[list[ExplanationNode], bool]:
        successors: dict[DependencyGroupId, list[SolvableGroupId]] = {}
        for s in graph.successors(solv_grp_id):
            successors.setdefault(graph.edges[solv_grp_id, s]["dependency_group_id"], []).append(s)

        # Type of node we are encountering
        if len(successors) == 0:
            explanation = ExplanationType.standalone if depth == 0 else ExplanationType.leaf
        elif solv_grp_id in visited_nodes:
            explanation = ExplanationType.visited
        else:
            explanation = ExplanationType.root if depth == 0 else ExplanationType.diving

        if depth == 0:
            dep_grp_id_from = None
        else:
            dep_grp_id_from = graph.edges[solv_grp_id_from, solv_grp_id]["dependency_group_id"]
        current = ExplanationNode(
            solv_grp_id=solv_grp_id,
            solv_grp_id_from=solv_grp_id_from,
            dep_grp_id_from=dep_grp_id_from,
            depth=depth,
            type=explanation,
            in_split=in_split,
        )
        visited_nodes.add(solv_grp_id)

        if (len(successors) == 0) or (explanation == ExplanationType.visited):
            return [current], leaf_status(current)

        children_paths: list[list[ExplanationNode]] = []
        children_status: list[bool] = []
        for dep_grp_id, children in successors.items():
            children_are_split: bool = len(children) > 1
            # Get the path and success for all children
            dep_children_path: list[ExplanationNode] = []
            dep_children_status: bool = False
            if children_are_split:
                # Split node is injected dynamically
                dep_children_path.append(
                    ExplanationNode(
                        solv_grp_id=None,
                        solv_grp_id_from=solv_grp_id,
                        dep_grp_id_from=dep_grp_id,
                        depth=(depth + 1),
                        type=ExplanationType.split,
                        in_split=children_are_split,
                    )
                )
            for c in children:
                path, status = visit(
                    solv_grp_id=c,
                    solv_grp_id_from=solv_grp_id,
                    depth=(depth + 1 + children_are_split),
                    in_split=children_are_split,
                )
                dep_children_path += path
                dep_children_status |= status

            # If there are any positive status downstream of path, the status of split
            # is considered positive
            children_paths.append(dep_children_path)
            children_status.append(dep_children_status)

        # Looking for smallest path with negative status (or None)
        min_path = min(
            (p for p, s in zip(children_paths, children_status) if not s),
            key=len,
            default=None,
        )
        # There is a negative status in the children (split have previously been merged).
        # That is enough to justify current node as negative.
        if min_path is not None:
            return [current]  + min_path, False
        # Otherwise all path are needed to explain positive status
        return [current] + list(itertools.chain.from_iterable(children_paths)), True

    path, _ = visit(solv_grp_id=root, solv_grp_id_from=None, depth=0, in_split=False)
    return path


@dataclasses.dataclass
class Names:
    pb_data: ProblemData
    cp_data: CompressionData

    package_name: re.Pattern = dataclasses.field(default=re.compile(r"\w[\w-]*"), init=False)

    def solv_group_name(self, solv_grp_id: SolvableGroupId) -> str:
        # All sovables in a node should have the same package name
        sample_solv_id = next(iter(self.cp_data.solv_groups.group_to_solv[solv_grp_id]))
        return self.pb_data.package_info[sample_solv_id].name

    def solv_group_versions(self, solv_grp_id: SolvableGroupId) -> list[str]:
        unique_versions = {
            self.pb_data.package_info[s].version for s in self.cp_data.solv_groups.group_to_solv[solv_grp_id]
        }
        return sorted(unique_versions, key=packaging.version.parse)

    def solv_group_versions_trunc(self, solv_grp_id: SolvableGroupId) -> str:
        return mer.utils.repr_trunc(self.solv_group_versions(solv_grp_id))

    def solv_group_repr(self, solv_grp_id: SolvableGroupId) -> str:
        versions = self.solv_group_versions(solv_grp_id)
        if len(versions) == 1:
            return "{name} {version}".format(name=self.solv_group_name(solv_grp_id), version=versions[0])
        return "{name} [{versions}]".format(
            name=self.solv_group_name(solv_grp_id), versions=mer.utils.repr_trunc(versions, sep="|")
        )

    def dep_group_name(self, dep_grp_id: DependencyGroupId) -> str:
        # All deps in an edge should have the same package name
        sample_dep_id = next(iter(self.cp_data.dep_groups.group_to_deps[dep_grp_id]))
        sample_dep = self.pb_data.dependency_names[sample_dep_id]
        if (match := self.package_name.match(sample_dep)) is not None:
            return match.group()
        return ""

    def dep_group_versions(self, dep_grp_id: DependencyGroupId) -> list[str]:
        dep_name = self.dep_group_name(dep_grp_id)
        unique_names = {
            self.pb_data.dependency_names[d].removeprefix(dep_name).strip()
            for d in self.cp_data.dep_groups.group_to_deps[dep_grp_id]
        }
        if "" in unique_names:
            unique_names.remove("")
        # This won't yield anything meaningful since all the packages should already the same
        # (only the version range changes) but at least it gives an arbitrary order.
        return sorted(unique_names)

    def dep_group_versions_trunc(self, dep_grp_id: DependencyGroupId) -> str:
        return mer.utils.repr_trunc(self.dep_group_versions(dep_grp_id), sep="|")

    def dep_group_repr(self, dep_grp_id: DependencyGroupId) -> str:
        versions = self.dep_group_versions(dep_grp_id)
        if len(versions) == 0:
            return self.dep_group_name(dep_grp_id)
        if len(versions) == 1:
            return "{name} {version}".format(name=self.dep_group_name(dep_grp_id), version=versions[0])
        return "{name} [{versions}]".format(
            name=self.dep_group_name(dep_grp_id), versions=mer.utils.repr_trunc(versions, sep="|")
        )


@dataclasses.dataclass
class LeafDescriptor:
    pb_data: ProblemData
    cp_data: CompressionData

    def leaf_has_conflict(self, solv_grp_id: SolvableGroupId) -> bool:
        return solv_grp_id in self.cp_data.conflicting_groups

    def leaf_conflict(self, solv_grp_id: SolvableGroupId) -> set[SolvableGroupId]:
        return self.cp_data.conflicting_groups[solv_grp_id]

    def leaf_has_problem(self, solv_grp_id: SolvableGroupId) -> bool:
        return solv_grp_id in self.cp_data.missing_groups

    def leaf_problem(self, solv_grp_id: SolvableGroupId) -> DependencyGroupId:
        # If there are more type of problem return an enum/union along the dep name
        return self.cp_data.missing_groups[solv_grp_id]


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

    def explain(self, path: Sequence[int, DependencyGroupId, SolvableGroupId, ExplanationType, bool]) -> str:
        message: list[str] = []
        for i, self.node in enumerate(path):
            if i == len(path) - 1:
                term = "."
            elif self.node.type in [ExplanationType.leaf, ExplanationType.visited]:
                term = ";"
            else:
                term = ""
            message += [
                self.indent * self.node.depth,
                *getattr(self, f"explain_{self.node.type.name}")(),
                term,
                "\n",
            ]

        message.pop()  # Last line break
        return "".join(message)

    @property
    def pkg_name(self) -> str:
        return self.names.solv_group_name(self.node.solv_grp_id)

    @property
    def dep_repr(self) -> str:
        return self.names.dep_group_repr(self.node.dep_grp_id_from)

    @property
    def solv_repr(self) -> str:
        return self.names.solv_group_repr(self.node.solv_grp_id)

    @property
    def pkg_repr(self) -> str:
        return self.solv_repr if self.node.in_split else self.dep_repr


class ProblemExplainer(Explainer):
    def explain_standalone(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_problem(self.node.solv_grp_id):
            missing_dep_grp_id = self.leaf_descriptor.leaf_problem(self.node.solv_grp_id)
            missing_dep_name = self.names.dep_group_repr(missing_dep_grp_id)
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
        return (self.dep_repr, " for which none of the following versions can be installed")

    def explain_leaf(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_conflict(self.node.solv_grp_id):
            conflict_ids = self.leaf_descriptor.leaf_conflict(self.node.solv_grp_id)
            conflict_names = ", ".join(
                self.color.unavailable(self.names.solv_group_name(g)) for g in conflict_ids
            )
            return (self.pkg_repr, ", which conflicts with any installable versions of ", conflict_names)
        elif self.leaf_descriptor.leaf_has_problem(self.node.solv_grp_id):
            missing_dep_grp_id = self.leaf_descriptor.leaf_problem(self.node.solv_grp_id)
            missing_dep_name = self.names.dep_group_repr(missing_dep_grp_id)
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
        return (
            self.pkg_repr,
            " is requested, and it requires" if self.node.depth == 1 else ", which requires",
        )

    def explain_split(self) -> tuple[str]:
        return (
            self.dep_repr,
            " is requested," if self.node.depth == 1 else "",
            " with the potential options",
        )

    def explain_leaf(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_problem(self.node.solv_grp_id):
            missing_dep_grp_id = self.leaf_descriptor.leaf_problem(self.node.solv_grp_id)
            missing_dep_name = self.names.dep_group_repr(missing_dep_grp_id)
            return (
                self.pkg_repr,
                ", which requires the missing package ",
                self.color.unavailable(missing_dep_name),
            )
        return (
            self.color.available(self.pkg_repr),
            " is directly requested" if self.node.depth == 1 else ", which can be installed",
        )

    def explain_visited(self) -> tuple[str]:
        return (self.pkg_repr, ", which can be installed (as previously explained)")


# Groups may be superset of the dependencies
def make_dep_id_to_groups(graph: nx.DiGraph) -> dict[DependencyGroupId, set[SolvableGroupId]]:
    groups: dict[DependencyGroupId, set[SolvableGroupId]] = {}
    for (_, s), attr in graph.edges.items():
        groups.setdefault(attr["dependency_group_id"], set()).add(s)
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

    header_msg = header_message(pb_data)

    if (install_root := cp_data.solv_groups.solv_to_group.get(-1)) is not None:
        inst_explainer = InstallExplainer(names, leaf_descriptor)
        install_msg = inst_explainer.explain(
            explanation_path(cp_data.graph, install_root, leaf_status=lambda _: True)
        )
    else:
        install_msg = None

    problem_root = cp_data.solv_groups.solv_to_group[0]
    pb_explainer = ProblemExplainer(names, leaf_descriptor)
    problem_msg = pb_explainer.explain(
        mer.algorithm.explanation_path(cp_data.graph, problem_root, leaf_status=lambda _: False)
    )

    return "Error: " + "\n\n".join([m for m in [header_msg, install_msg, problem_msg] if m is not None])
