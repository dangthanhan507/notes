---
layout: default
title: Python Environments
parent: Code Notes
nav_order: 3
---

# Pyproject and UV

We set up python project management using `pyproject.toml` and `uv`.

Go into your project directory. Then run, `uv init`.

```bash
$ cd hello-world
$ uv init
```

uv will create the following:
```
├── .gitignore
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

We can run python scripts using `uv run [insert file]`.

uv creates virtual environment in uv.lock