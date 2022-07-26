import json
import pathlib
import tempfile
from typing import Any, Optional, Sequence, Union

import libmambapy
import mamba.utils


def _create_package(
    name: str,
    version: str,
    dependencies: Optional[list[str]] = None,
    constrains: Optional[list[str]] = None,
    build_number: int = 0,
    build_string: str = "bstring",
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "build_number": build_number,
        "build_string": build_string,
        "version": version,
        "depends": dependencies if dependencies is not None else [],
        "constrains": constrains if constrains is not None else [],
    }


def _create_repodata(directory: Union[str, pathlib.Path], packages: Sequence[dict[str, Any]]) -> None:
    dierctory = pathlib.Path(directory)
    (dierctory / "noarch").mkdir()
    repodata_file = dierctory / "noarch" / "repodata.json"
    repodata: dict[str, dict[str, str]] = {
        "packages": {f"{p['name']}-{p['version']}-{p['build_string']}.tar.bz2": p for p in packages}
    }
    repodata_file.write_text(json.dumps(repodata))


def _create_problem_manual(
    packages: Sequence[dict[str, Any]], specs: Sequence[str]
) -> tuple[libmambapy.Solver, libmambapy.Pool]:
    repos: list[str] = []
    pool = libmambapy.Pool()

    with tempfile.TemporaryDirectory() as dir:
        _create_repodata(dir, [_create_package(**pkg) for pkg in packages])

        # change this to point where you cloned mamba
        channels = [f"file:///{dir}"]

        mamba.utils.load_channels(pool, channels, repos, prepend=False, platform="linux-64")

    solver_options = [(libmambapy.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
    solver = libmambapy.Solver(pool, solver_options)

    solver.add_jobs(specs, libmambapy.SOLVER_INSTALL)
    return solver, pool


def create_basic_conflict() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return _create_problem_manual(
        packages=[
            {"name": "A", "version": "0.1.0"},
            {"name": "A", "version": "0.2.0"},
            {"name": "A", "version": "0.3.0"},
        ],
        specs=["A=0.4.0"],
    )


def create_pubgrub() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    """Create the PubGrub blog post example.

    The example given by Natalie Weizenbaum (credits https://nex3.medium.com/pubgrub-2fb6470504f).
    """
    return _create_problem_manual(
        packages=[
            {"name": "menu", "version": "1.5.0", "dependencies": ["dropdown=2.*"]},
            {"name": "menu", "version": "1.4.0", "dependencies": ["dropdown=2.*"]},
            {"name": "menu", "version": "1.3.0", "dependencies": ["dropdown=2.*"]},
            {"name": "menu", "version": "1.2.0", "dependencies": ["dropdown=2.*"]},
            {"name": "menu", "version": "1.1.0", "dependencies": ["dropdown=2.*"]},
            {"name": "menu", "version": "1.0.0", "dependencies": ["dropdown=1.*"]},
            {"name": "dropdown", "version": "2.3.0", "dependencies": ["icons=2.*"]},
            {"name": "dropdown", "version": "2.2.0", "dependencies": ["icons=2.*"]},
            {"name": "dropdown", "version": "2.1.0", "dependencies": ["icons=2.*"]},
            {"name": "dropdown", "version": "2.0.0", "dependencies": ["icons=2.*"]},
            {"name": "dropdown", "version": "1.8.0", "dependencies": ["icons=1.*", "intl=3.*"]},
            {"name": "icons", "version": "2.0.0"},
            {"name": "icons", "version": "1.0.0"},
            {"name": "intl", "version": "5.0.0"},
            {"name": "intl", "version": "4.0.0"},
            {"name": "intl", "version": "3.0.0"},
        ],
        specs=["menu", "icons=1.*", "intl=5.*"],
    )


def create_pubgrub_hard(missing_package: bool = False) -> tuple[libmambapy.Solver, libmambapy.Pool]:
    packages: list[dict[str, Any]] = [
        {"name": "menu", "version": "2.1.0", "dependencies": ["dropdown>=2.1", "emoji"]},
        {"name": "menu", "version": "2.0.1", "dependencies": ["dropdown>=2", "emoji"]},
        {"name": "menu", "version": "2.0.0", "dependencies": ["dropdown>=2", "emoji"]},
        {"name": "menu", "version": "1.5.0", "dependencies": ["dropdown=2.*", "emoji"]},
        {"name": "menu", "version": "1.4.0", "dependencies": ["dropdown=2.*", "emoji"]},
        {"name": "menu", "version": "1.3.0", "dependencies": ["dropdown=2.*"]},
        {"name": "menu", "version": "1.2.0", "dependencies": ["dropdown=2.*"]},
        {"name": "menu", "version": "1.1.0", "dependencies": ["dropdown=1.*"]},
        {"name": "menu", "version": "1.0.0", "dependencies": ["dropdown=1.*"]},
        {"name": "emoji", "version": "1.1.0", "dependencies": ["libicons=2.*"]},
        {"name": "emoji", "version": "1.0.0", "dependencies": ["libicons=2.*"]},
        {"name": "dropdown", "version": "2.3.0", "dependencies": ["libicons=2.*"]},
        {"name": "dropdown", "version": "2.2.0", "dependencies": ["libicons=2.*"]},
        {"name": "dropdown", "version": "2.1.0", "dependencies": ["libicons=2.*"]},
        {"name": "dropdown", "version": "2.0.0", "dependencies": ["libicons=2.*"]},
        {"name": "dropdown", "version": "1.8.0", "dependencies": ["libicons=1.*", "intl=3.*"]},
        {"name": "dropdown", "version": "1.7.0", "dependencies": ["libicons=1.*", "intl=3.*"]},
        {"name": "dropdown", "version": "1.6.0", "dependencies": ["libicons=1.*", "intl=3.*"]},
        {"name": "pyicons", "version": "2.0.0", "dependencies": ["libicons=2.*"]},
        {"name": "pyicons", "version": "1.1.0", "dependencies": ["libicons=1.2.*"]},
        {"name": "pyicons", "version": "1.0.0", "dependencies": ["libicons=1.*"]},
        {"name": "pretty", "version": "1.1.0", "dependencies": ["pyicons=1.1.*"]},
        {"name": "pretty", "version": "1.0.1", "dependencies": ["pyicons=1.*"]},
        {"name": "pretty", "version": "1.0.0", "dependencies": ["pyicons=1.*"]},
        {"name": "intl", "version": "5.0.0"},
        {"name": "intl", "version": "4.0.0"},
        {"name": "intl", "version": "3.2.0"},
        {"name": "intl", "version": "3.1.0"},
        {"name": "intl", "version": "3.0.0"},
        {"name": "intl-mod", "version": "1.0.0", "dependencies": ["intl=5.0.*"]},
        {"name": "intl-mod", "version": "1.0.1", "dependencies": ["intl=5.0.*"]},
        {"name": "libicons", "version": "2.1.0"},
        {"name": "libicons", "version": "2.0.1"},
        {"name": "libicons", "version": "2.0.0"},
        {"name": "libicons", "version": "1.2.1"},
        {"name": "libicons", "version": "1.2.0"},
        {"name": "libicons", "version": "1.0.0"},
    ]
    if missing_package:
        packages += [
            {"name": "dropdown", "version": "2.9.3", "dependencies": ["libnothere>1.0"]},
            {"name": "dropdown", "version": "2.9.2", "dependencies": ["libicons>10.0", "libnothere>1.0"]},
            {"name": "dropdown", "version": "2.9.1", "dependencies": ["libicons>10.0", "libnothere>1.0"]},
            {"name": "dropdown", "version": "2.9.0", "dependencies": ["libicons>10.0"]},
        ]

    return _create_problem_manual(
        packages=packages,
        specs=["menu", "pyicons=1.*", "intl=5.*", "intl-mod", "pretty>=1.0"],
    )


def create_pubgrub_missing() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_pubgrub_hard(True)


def create_conda_forge(
    specs: Sequence[str],
    glibc: str | None = "2.17.0",
    cuda: str | None = None,
) -> tuple[libmambapy.Solver, libmambapy.Pool]:
    pool = libmambapy.Pool()
    prefix_data = libmambapy.PrefixData(libmambapy.Path(str(tempfile.TemporaryDirectory())))
    virtual_pkgs = []
    if glibc is not None:
        virtual_pkgs.append(libmambapy.PackageInfo("__glibc", glibc, "", 0))
    if cuda is not None:
        virtual_pkgs.append(libmambapy.PackageInfo("__cuda", cuda, "", 0))
    prefix_data.add_packages(virtual_pkgs)
    repo = libmambapy.Repo(pool, prefix_data)
    repo.set_installed()
    mamba.utils.load_channels(pool, ["conda-forge"], [repo], prepend=False, platform="linux-64")
    solver_options = [(libmambapy.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
    solver = libmambapy.Solver(pool, solver_options)
    solver.add_jobs(specs, libmambapy.SOLVER_INSTALL)
    return solver, pool


def create_pytorch_cpu() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["python=2.7", "pytorch"], cuda=None)


def create_pytorch_cuda() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["python=2.7", "pytorch"], cuda="10.2.0")


def create_cudatoolkit() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(
        ["python=3.7", "cudatoolkit=11.1", "cudnn=8.0", "pytorch=1.8", "torchvision=0.9=*py37_cu111*"],
        cuda="11.1",
    )


def create_jpeg9b() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["python=3.7", "jpeg=9b"])


def create_r_base() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["r-base=3.5.* ", "pandas=0", "numpy<1.20.0", "matplotlib=2", "r-matchit=4.*"])


def create_scip() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["scip=8.*", "pyscipopt<4.0"])


def create_jupyterlab() -> tuple[libmambapy.Solver, libmambapy.Pool]:
    return create_conda_forge(["jupyterlab>3.0", "openssl=3.0.0"])
