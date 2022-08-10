#! /usr/bin/env bash

readonly tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}" || true' EXIT

cd "${tmp_dir}"
micromamba activate mamba-error
curl -sLJ 'https://github.com/mamba-org/mamba/archive/master.tar.gz' | tar -xz

cd mamba-master
micromamba install -n mamba-error --file mamba/environment-dev.yml
cmake -B build/ -D BUILD_SHARED=ON -D BUILD_LIBMAMBA=ON -D BUILD_LIBMAMBAPY=ON
cmake --build build/ --parallel
cmake --install build/ --prefix "${CONDA_PREFIX}"
python -m pip install --ignore-installed --no-deps --no-build-isolation libmambapy/ mamba/
