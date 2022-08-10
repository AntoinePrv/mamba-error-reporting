# mamba-error-reporting
Proof of concept for improving mamba errors message about unsolvable environments.

## Installation
```bash
micromamba create -n mamba-error --file environment.yml
micromamba activate mamba-error
python -m pip install -e .
# Currently requires unreleased features of libmambapy
./install-mamba-latest.sh
```
