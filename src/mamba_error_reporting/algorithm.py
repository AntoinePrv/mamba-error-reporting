from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
import re
import typing as T

import libmambapy
import networkx as nx
import packaging.version

import mamba_error_reporting as mer

SolvableId = T.NewType("SolvableId", int)
SolvableGroupId = T.NewType("SolvableGroupId", int)
DependencyId = T.NewType("DependencyId", int)
DependencyGroupId = T.NewType("DependencyGroupId", int)

NodeType = T.TypeVar("NodeType")
EdgeType = T.TypeVar("EdgeType")


############################
#  Initial data available  #
############################


@dataclasses.dataclass(frozen=True)
class DependencyInfo:
    dep_re: T.ClassVar[re.Pattern] = re.compile(r"\s*(\w[\w-]*)\s*(.*)\s*")

    name: str
    range: str

    @classmethod
    def parse(cls, repr: str) -> DependencyInfo:
        match = cls.dep_re.match(repr)
        return cls(*match.groups())

    def __str__(self) -> str:
        return f"{self.name} {self.range}"


@dataclasses.dataclass
class ProblemData:
    problems_by_type: dict[libmambapy.SolverRuleinfo, list[libmambapy.SolverProblem]]
    dependency_info: dict[DependencyId, DependencyInfo]
    dependency_solvables: dict[DependencyId, list[SolvableId]]
    package_info: dict[SolvableId, libmambapy.PackageInfo]
    graph: nx.DiGraph

    @staticmethod
    def from_libsolv(solver: libmambapy.Solver, pool: libmambapy.Pool) -> ProblemData:
        graph = nx.DiGraph()
        dependency_info = {}
        dependency_solvables = {}
        package_info = {}
        problems_by_type = {}

        def add_solvable(id, pkg_info=None):
            graph.add_node(id)
            package_info[id] = pkg_info if pkg_info is not None else pool.id2pkginfo(id)

        def add_dependency(source_id, dep_id, dep_repr):
            dependency_info[p.dep_id] = DependencyInfo.parse(dep_repr)
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
                p.source_id = 0
                add_solvable(p.source_id, libmambapy.PackageInfo("root", "", "", 0))
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
            dependency_info=dependency_info,
            dependency_solvables=dependency_solvables,
            package_info=package_info,
            problems_by_type=problems_by_type,
        )

    @property
    def problems(self) -> T.Iterable[libmambapy.PackageInfo]:
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

    def add(self, solvs: T.Sequence[SolvableId]) -> SolvableGroupId:
        grp_id = self.group_counter()
        self.solv_to_group.update({s: grp_id for s in solvs})
        self.group_to_solv[grp_id] = set(solvs)
        return grp_id


@dataclasses.dataclass
class DependencyGroups:
    dep_to_groups: dict[DependencyId, set[DependencyGroupId]] = dataclasses.field(default_factory=dict)
    group_to_deps: dict[DependencyGroupId, set[DependencyId]] = dataclasses.field(default_factory=dict)

    group_counter: Counter = dataclasses.field(default_factory=Counter, init=False)

    def search_grp_id(self, deps: T.Sequence[DependencyId]) -> DependencyGroupId | None:
        # Maybe not great in term of complexity
        for dep_grp_id, exisiting_deps in self.group_to_deps.items():
            if set(deps) == exisiting_deps:
                return dep_grp_id
        return None

    def add(self, deps: T.Sequence[DependencyId]) -> DependencyGroupId:
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
    nodes: T.Sequence[SolvableId], compatible: T.Callable[[SolvableId, SolvableId], bool]
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
    def same_parents(n1: SolvableId, n2: SolvableId) -> bool:
        return set(pb_data.graph.predecessors(n1)) == set(pb_data.graph.predecessors(n2))

    def same_children(n1: SolvableId, n2: SolvableId) -> bool:
        return set(pb_data.graph.successors(n1)) == set(pb_data.graph.successors(n2))

    def same_missing_name(n1: SolvableId, n2: SolvableId) -> bool:
        s1 = {pb_data.dependency_info[d].name for d in pb_data.package_missing.get(n1, {})}
        s2 = {pb_data.dependency_info[d].name for d in pb_data.package_missing.get(n2, {})}
        return s1 == s2

    def is_leaf(n1: SolvableId) -> bool:
        return len(set(pb_data.graph.successors(n1))) == 0

    def compatible(n1: SolvableId, n2: SolvableId) -> bool:
        return (
            # Packages must not be in conflict
            ((n1, n2) not in pb_data.package_conflicts)
            # Packages must have same missing dependencies name (when that is the case)
            and same_missing_name(n1, n2)
            # Packages must have the same successors
            and same_children(n1, n2)
            # Non-leaf packages must have the same predecessors
            and (is_leaf(n1) and is_leaf(n2) or same_parents(n1, n2))
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

    @classmethod
    def from_position(cls, has_predecessors: bool, has_successors: bool, is_visited: bool) -> ExplanationType:
        if not has_successors:
            return cls.leaf if has_predecessors else cls.standalone
        elif is_visited:
            return cls.visited
        else:
            return cls.diving if has_predecessors else cls.root


@dataclasses.dataclass
class ExplanationNode:
    solv_grp_id: SolvableGroupId | None
    solv_grp_id_from: SolvableGroupId | None
    dep_grp_id_from: DependencyGroupId | None
    type: ExplanationType
    in_split: bool
    status: bool
    tree_position: list[bool]

    @property
    def depth(self) -> int:
        return len(self.tree_position)

    @property
    def is_root(self) -> bool:
        return self.depth == 0


@dataclasses.dataclass
class GraphWalker:
    graph: nx.DiGraph
    leaf_status: T.Callable[[SolvableGroupId], bool]
    split_sort_key: T.Callable[[SolvableGroupId], T.Any] = lambda s: s
    dep_sort_key: T.Callable[[DependencyGroupId], T.Any] = lambda d: d

    def successors_per_dep(
        self, solv_grp_id: SolvableGroupId
    ) -> dict[DependencyGroupId, list[SolvableGroupId]]:
        successors: dict[DependencyGroupId, list[SolvableGroupId]] = {}
        for s in self.graph.successors(solv_grp_id):
            successors.setdefault(self.graph.edges[solv_grp_id, s]["dependency_group_id"], []).append(s)
        return successors

    def visit(self, root: SolvableGroupId) -> list[ExplanationNode]:
        # TODO these list operations are more expensive than they need to be.
        # TODO a recursive form is no longer required.
        return self.visit_node(
            solv_grp_id=root,
            solv_grp_id_from=None,
            tree_position=[],
            in_split=False,
            visited_nodes={},
        )

    def visit_split(
        self,
        solv_grp_id_from: SolvableGroupId,
        dep_grp_id_from: DependencyGroupId,
        children: list[SolvableGroupId],
        tree_position: list[bool],
        visited_nodes: dict[SolvableGroupId, bool],
    ) -> list[ExplanationNode]:
        children.sort(key=self.split_sort_key)
        # Split node is prepended dynamically
        path: list[ExplanationNode] = [
            ExplanationNode(
                solv_grp_id=None,
                solv_grp_id_from=solv_grp_id_from,
                dep_grp_id_from=dep_grp_id_from,
                tree_position=tree_position,
                type=ExplanationType.split,
                in_split=True,
                status=False,  # Placeholder
            )
        ]
        for i, c in enumerate(children):
            child_path = self.visit_node(
                solv_grp_id=c,
                solv_grp_id_from=solv_grp_id_from,
                tree_position=tree_position + [(i == len(children) - 1)],
                in_split=True,
                visited_nodes=visited_nodes,
            )
            path.extend(child_path)
            # if there are any valid option in the split, the split is iself valid
            path[0].status |= child_path[0].status
        return path

    def visit_node(
        self,
        solv_grp_id: SolvableGroupId,
        solv_grp_id_from: SolvableGroupId | None,
        tree_position: list[bool],
        in_split: bool,
        visited_nodes: dict[SolvableGroupId, bool] = {},
    ) -> list[ExplanationNode]:
        successors = self.successors_per_dep(solv_grp_id)

        # Type of node we are encountering
        depth = len(tree_position)

        explanation = ExplanationType.from_position(
            has_predecessors=(depth > 0),
            has_successors=(len(successors) > 0),
            is_visited=(solv_grp_id in visited_nodes),
        )

        if depth == 0:
            dep_grp_id_from = None
        else:
            dep_grp_id_from = self.graph.edges[solv_grp_id_from, solv_grp_id]["dependency_group_id"]
        current = ExplanationNode(
            solv_grp_id=solv_grp_id,
            solv_grp_id_from=solv_grp_id_from,
            dep_grp_id_from=dep_grp_id_from,
            type=explanation,
            in_split=in_split,
            tree_position=tree_position,
            status=True,  # Placeholder, modified later
        )

        if len(successors) == 0:
            current.status = self.leaf_status(current.solv_grp_id)
            visited_nodes[solv_grp_id] = current.status
            return [current]
        if solv_grp_id in visited_nodes:
            current.status = visited_nodes[solv_grp_id]
            return [current]

        path: list[ExplanationNode] = [current]
        for i, dep_grp_id in enumerate(sorted(successors.keys(), key=self.dep_sort_key)):
            children = successors[dep_grp_id]
            if len(children) > 1:
                child_path = self.visit_split(
                    children=children,
                    solv_grp_id_from=solv_grp_id,
                    dep_grp_id_from=dep_grp_id,
                    tree_position=tree_position + [i == len(successors) - 1],
                    visited_nodes=visited_nodes,
                )
            else:
                child_path = self.visit_node(
                    solv_grp_id=children[0],
                    solv_grp_id_from=solv_grp_id,
                    tree_position=tree_position + [i == len(successors) - 1],
                    in_split=False,
                    visited_nodes=visited_nodes,
                )
            # All dependencies need to be valid for a parent to be valid
            current.status &= child_path[0].status
            path.extend(child_path)

        visited_nodes[solv_grp_id] = current.status
        return path


@dataclasses.dataclass
class Names:
    pb_data: ProblemData
    cp_data: CompressionData

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
        return self.pb_data.dependency_info[sample_dep_id].name

    def dep_group_versions(self, dep_grp_id: DependencyGroupId) -> list[str]:
        unique_names = {
            self.pb_data.dependency_info[d].range for d in self.cp_data.dep_groups.group_to_deps[dep_grp_id]
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
class ColorSet:
    @staticmethod
    def available(msg: str) -> str:
        return mer.color.color(msg, fg="green", style="bold")

    @staticmethod
    def unavailable(msg: str) -> str:
        return mer.color.color(msg, fg="red", style="bold")


@dataclasses.dataclass
class ProblemExplainer:
    names: Names
    leaf_descriptor: LeafDescriptor
    indent: tuple[tuple[str, str]] = (("│  ", "   "), ("├─ ", "└─ "))
    color_set: type = ColorSet

    def explain(
        self, path: T.Sequence[int, DependencyGroupId, SolvableGroupId, ExplanationType, bool]
    ) -> str:
        message: list[str] = []
        for i, self.node in enumerate(path):
            if i == len(path) - 1:
                term = "."
            elif self.node.type in [ExplanationType.leaf, ExplanationType.visited]:
                term = ";"
            else:
                term = ""

            if self.node.depth > 0:
                pos = self.node.tree_position
                message += [self.indent[j == len(pos) - 1][is_last] for j, is_last in enumerate(pos)]
            message += [
                *getattr(self, f"explain_{self.node.type.name}")(),
                term,
                "\n",
            ]

        message.pop()  # Last line break
        return "".join(message)

    def color(self, msg: str) -> str:
        return self.color_set.available(msg) if self.node.status else self.color_set.unavailable(msg)

    @property
    def pkg_name(self) -> str:
        return self.names.solv_group_name(self.node.solv_grp_id)

    @property
    def dep_repr(self) -> str:
        return self.color(self.names.dep_group_repr(self.node.dep_grp_id_from))

    @property
    def solv_repr(self) -> str:
        return self.color(self.names.solv_group_repr(self.node.solv_grp_id))

    @property
    def pkg_repr(self) -> str:
        return self.color(self.solv_repr if self.node.in_split else self.dep_repr)

    def explain_standalone(self) -> tuple[str]:
        if self.leaf_descriptor.leaf_has_problem(self.node.solv_grp_id):
            missing_dep_grp_id = self.leaf_descriptor.leaf_problem(self.node.solv_grp_id)
            missing_dep_name = self.names.dep_group_repr(missing_dep_grp_id)
            return (
                "The environment could not be satisfied because it requires the missing package ",
                self.color_set.unavailable(missing_dep_name),
            )
        return ("The environment could not be satisfied",)

    def explain_root(self) -> tuple[str]:
        return ("The following packages conflict with one another",)

    def explain_diving(self) -> tuple[str]:
        if self.node.depth == 1:
            return (
                self.pkg_repr,
                " is installable and" if self.node.status else "is uninstallable because",
                "  it requires",
            )
        return (self.pkg_repr, ", which requires")

    def explain_split(self) -> tuple[str]:
        if self.node.depth == 1:
            return (
                self.dep_repr,
                " is installable with the potential"
                if self.node.status == 1
                else " is uninstallable with no viable",
                " options",
            )
        return (
            self.dep_repr,
            " with ",
            "the potential" if self.node.status else "no viable",
            " options",
        )

    def explain_leaf(self) -> tuple[str]:
        if self.node.status:
            return (
                self.color(self.pkg_repr),
                " is requested and" if self.node.depth == 1 else ", which",
                " can be installed",
            )
        elif self.leaf_descriptor.leaf_has_conflict(self.node.solv_grp_id):
            conflict_names = {
                self.names.solv_group_name(g)
                for g in self.leaf_descriptor.leaf_conflict(self.node.solv_grp_id)
            }
            conflict_repr = ", ".join(self.color_set.unavailable(n) for n in conflict_names)
            return (
                self.pkg_repr,
                " is uninstallable because it" if self.node.depth == 1 else ", which",
                " conflicts with any installable versions of ",
                conflict_repr,
            )
        elif self.leaf_descriptor.leaf_has_problem(self.node.solv_grp_id):
            missing_dep_grp_id = self.leaf_descriptor.leaf_problem(self.node.solv_grp_id)
            missing_dep_name = self.names.dep_group_repr(missing_dep_grp_id)
            return (
                self.pkg_repr,
                " is uninstallable because it" if self.node.depth == 1 else ", which",
                " requires the missing package ",
                self.color_set.unavailable(missing_dep_name),
            )
        else:
            return (self.color(self.pkg_repr), ", which cannot be installed for an unknown reason")

    def explain_visited(self) -> tuple[str]:
        return (
            self.pkg_repr,
            ", which ",
            "can" if self.node.status else "cannot",
            " be installed (as previously explained)",
        )


def header_message(pb_data: ProblemData, color: type = ColorSet) -> str | None:
    deps = {
        pb_data.dependency_info[pb_data.graph.edges[e]["dependency_id"]] for e in pb_data.graph.out_edges(0)
    }
    if len(deps) == 0:
        return None
    return "Could not find any compatible versions for requested package{s} {pkgs}.".format(
        s=("s" if len(deps) > 1 else ""), pkgs=", ".join(color.unavailable(d) for d in deps)
    )


def explain_graph(pb_data: ProblemData, cp_data: CompressionData) -> str:
    header_msg = header_message(pb_data)

    names = Names(pb_data, cp_data)
    leaf_descriptor = LeafDescriptor(pb_data, cp_data)

    # Packages are considered installable the first time thay appear in a conflict, the next
    # time it is considered a conflict.
    installables: set[SolvableGroupId] = set()

    def leaf_status(solv_grp_id: SolvableGroupId) -> bool:
        if leaf_descriptor.leaf_has_problem(solv_grp_id):
            return False
        if leaf_descriptor.leaf_has_conflict(solv_grp_id):
            if any(c in installables for c in leaf_descriptor.leaf_conflict(solv_grp_id)):
                return False
            installables.add(solv_grp_id)
        return True

    def split_sort_key(
        solv_grp_id: SolvableGroupId,
    ) -> tuple[packaging.version.Version, packaging.version.Version]:
        versions = [packaging.version.parse(v) for v in names.solv_group_versions(solv_grp_id)]
        return min(versions), max(versions)

    problem_root = cp_data.solv_groups.solv_to_group[0]
    pb_explainer = ProblemExplainer(names, leaf_descriptor)
    problem_msg = pb_explainer.explain(
        GraphWalker(
            graph=cp_data.graph,
            leaf_status=leaf_status,
            dep_sort_key=lambda dep: dep,
            split_sort_key=split_sort_key,
        ).visit(problem_root)
    )

    return "Error: " + "\n\n".join([m for m in [header_msg, problem_msg] if m is not None])
