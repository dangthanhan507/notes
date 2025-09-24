---
layout: default
title: How-to Tmux
parent: Code Notes
nav_order: 2
---

# How to Tmux

## Using Tmux for terminal multiplexing

Create a new tmux session with a specific name:

```bash
tmux new -s [session-name]
```

Use `Ctrl+b` followed by `d` to detach from the session.

To reattach to a session:

```bash
tmux attach -d -t [session-name]
```

To list all tmux sessions:

```bash
tmux ls
```
To kill a specific tmux session:

```bash
tmux kill-session -t [session-name]
```

Extra things for session management:
```bash
ctrl+b c # Create a new window
ctrl+b , # Rename the current window
ctrl+b s # List all windows
```
