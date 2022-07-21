import json
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union

import libmambapy
import mamba.utils


def _create_package(
    name: str,
    version: str,
    dependencies: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    build_number: int = 0,
    build_string: str = "bstring",
) -> Dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "build_number": build_number,
        "build_string": build_string,
        "version": version,
        "depends": dependencies if dependencies is not None else [],
        "constraints": constraints if constraints is not None else [],
    }


def _create_repodata(
    directory: Union[str, pathlib.Path], packages: Sequence[str]
) -> None:
    dierctory = pathlib.Path(directory)
    (dierctory / "noarch").mkdir()
    repodata_file = dierctory / "noarch" / "repodata.json"
    repodata = {}
    repodata["packages"] = {}
    for p in packages:
        repodata["packages"][
            f"{p['name']}-{p['version']}-{p['build_string']}.tar.bz2"
        ] = p
    repodata_file.write_text(json.dumps(repodata))


def create_basic_conflict():
    repos = []
    pool = libmambapy.Pool()

    with tempfile.TemporaryDirectory() as d:
        _create_repodata(
            d,
            [
                _create_package("A", "0.1.0"),
                _create_package("A", "0.2.0"),
                _create_package("A", "0.3.0"),
            ],
        )

        # change this to point where you cloned mamba
        channels = [f"file:///{d}"]

        mamba.utils.load_channels(
            pool, channels, repos, prepend=False, platform="linux-64"
        )
    specs = ["A=0.4.0"]

    solver_options = [(libmambapy.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
    solver = libmambapy.Solver(pool, solver_options)

    solver.add_jobs(specs, libmambapy.SOLVER_INSTALL)
    return solver, pool


def create_pubgrub():
    repos = []
    pool = libmambapy.Pool()

    with tempfile.TemporaryDirectory() as d:
        _create_repodata(
            d,
            [
                _create_package("menu", "1.5.0", dependencies=["dropdown=2.*"]),
                _create_package("menu", "1.4.0", dependencies=["dropdown=2.*"]),
                _create_package("menu", "1.3.0", dependencies=["dropdown=2.*"]),
                _create_package("menu", "1.2.0", dependencies=["dropdown=2.*"]),
                _create_package("menu", "1.1.0", dependencies=["dropdown=2.*"]),
                _create_package("menu", "1.0.0", dependencies=["dropdown=1.*"]),
                _create_package("dropdown", "2.3.0", dependencies=["icons=2.*"]),
                _create_package("dropdown", "2.2.0", dependencies=["icons=2.*"]),
                _create_package("dropdown", "2.1.0", dependencies=["icons=2.*"]),
                _create_package("dropdown", "2.0.0", dependencies=["icons=2.*"]),
                _create_package(
                    "dropdown", "1.8.0", dependencies=["icons=1.*", "intl=3.*"]
                ),
                # create_package("icons", "2.1.0"),  # Not original
                _create_package("icons", "2.0.0"),
                # create_package("icons", "1.2.0"),  # Not original
                _create_package("icons", "1.0.0"),
                _create_package("intl", "5.0.0"),
                _create_package("intl", "4.0.0"),
                _create_package("intl", "3.0.0"),
            ],
        )

        # change this to point where you cloned mamba
        channels = [f"file:///{d}"]

        mamba.utils.load_channels(
            pool, channels, repos, prepend=False, platform="linux-64"
        )
    specs = ["menu", "icons=1.*", "intl=5.*"]
    # specs = ["menu", "icons=1.*", "intl>=4.0"]  # Not original

    solver_options = [(libmambapy.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
    solver = libmambapy.Solver(pool, solver_options)

    solver.add_jobs(specs, libmambapy.SOLVER_INSTALL)
    return solver, pool


def create_conda_forge(specs):
    repos = []
    pool = libmambapy.Pool()
    channels = ["conda-forge"]
    mamba.utils.load_channels(pool, channels, repos, prepend=False, platform="linux-64")
    solver_options = [(libmambapy.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
    solver = libmambapy.Solver(pool, solver_options)
    solver.add_jobs(specs, libmambapy.SOLVER_INSTALL)
    return solver, pool


def create_pytorch():
    return create_conda_forge(["python=2.7", "pytorch"])


def create_r_base():
    return create_conda_forge(
        ["r-base=3.5.* ", "pandas=0", "numpy<1.20.0", "matplotlib=2", "r-matchit=4.*"]
    )
