#! /usr/bin/env bash

cd "$(mktemp -d)"
curl -sLJ 'https://github.com/mamba-org/mamba/archive/master.tar.gz' | tar -xz -
micromamba install --file mamba-master/mamba/environment-dev.yml
cmake -B build/ -S mamba-master/ \
	-D BUILD_SHARED=ON -D BUILD_LIBMAMBA=ON -D BUILD_LIBMAMBAPY=ON
cmake --build build/ --parallel
cmake --install build/ --prefix "${CONDA_PREFIX}"
python -m pip install mamba-master/libmambapy/ mamba-master/mamba/
