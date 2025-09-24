---
layout: default
title: How-to PDB
parent: Code Notes
nav_order: 1
---

# How to PDB

## Using PDB for debugging in python

Insert the following lines of code where you want to start debugging:

```python
import pdb; pdb.set_trace()
```

This will pause the execution and drop you into the PDB interactive console. This is sketchier than just breakpoints in an IDE, but it works in any environment where you can run Python code.

## PDB Commands

Short list of useful PDB commands:

- `p`: Print the value of an expression. (`p x` prints the value of `x`)
- `continue`: Continue execution until the next breakpoint.
- `c`: short for `continue`.
- `s`: Step into the next line of code.
- `n`: Step to the next line in the current function.
- `q`: Quit the debugger and exit the program.

<b> NOTE: </b> I found that the python documentation for PDB is not very helpful. Look at GDB documentation for a more comprehensive list of commands.



