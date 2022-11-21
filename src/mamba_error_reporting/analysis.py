from typing import Sequence

import libmambapy
import pandas as pd


def all_problems_structured_df(solver: libmambapy.Solver) -> pd.DataFrame:
    problems = []
    for p in solver.all_problems_structured():
        problems.append(
            {
                "type": p.type,
                "source_id": p.source_id,
                "source_is_pkg": p.source is not None,
                "dependency": p.dep,
                "dependency_id": p.dep_id,
                "target_id": p.target_id,
                "target_is_pkg": p.target() is not None,
                "explanation": str(p),
            }
        )
    return pd.DataFrame(problems)


def id_to_pkg_info_df(pool: libmambapy.Pool, pkg_ids: Sequence[int]) -> pd.DataFrame:
    pkgs = []
    for pkg_id in pkg_ids:
        pkg = pool.id2pkginfo(pkg_id)
        pkgs.append(
            {
                "id": pkg_id,
                "name": pkg.name,
                "version": pkg.version,
                "build_string": pkg.build_string,
                "build_number": pkg.build_number,
            }
        )
    return pd.DataFrame(pkgs)


def select_solvables_df(pool: libmambapy.Pool, dep_ids: Sequence[int]) -> pd.DataFrame:
    solvables = []
    for dep in dep_ids:
        for sol in pool.select_solvables(dep):
            solvables.append(
                {
                    "dependency_id": dep,
                    "solvable_id": sol,
                }
            )
    return pd.DataFrame(solvables)
