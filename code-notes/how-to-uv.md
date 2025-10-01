---
layout: default
title: How-to UV
parent: Code Notes
nav_order: 7
---

# How to UV

UV is a command-line tool for managing and switching between different virtual environments for python projects. It's an alterantive to poetry and anaconda. 

We can get `uv` from pip:

```bash
pip install uv
```

- `uv init [project-name]` - Initializes a new virtual environment at `./project-name` folder. This folder can be existing or non-existing.
- `uv add [ list[packages] ]` - Adds python packages to the current virtual environment (installs them).
- `uv remove [ list[packages] ]` - Removes python packages from the current virtual environment (uninstalls them).
- `uv run [name].py` - Runs a python script using the current virtual environment.
- `uv pip install -e .` - Installs the current project using `uv.lock` file.

## The difference between uv.lock and requirements.txt

You'll notice in `uv.lock` and `pyproject.toml` files that `uv` creates will have exact information on how to recreate the environment. This is different from `requirements.txt` which only lists the packages and their versions.

This can be compared to conda's `environment.yml` file which can figure out the dependencies and versions of packages to install. However, many times these conda environment files don't contain everything needed and we end up struggling to reproduce the exact environment to run the code.

