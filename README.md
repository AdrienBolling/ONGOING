# ONGOING

**Continuous Knowledge Modelling library**

ONGOING models the evolving knowledge of agents (e.g. technicians, workers) as n-dimensional grids. Each cell in the grid represents a region of an embedding space, and accumulates experience as the agent handles tasks ("tickets") in that region. Knowledge is derived from experience via a power-law transform inspired by learning curves: repeated exposure to similar tasks yields diminishing returns.

## Core concepts

- **Embedding space**: A continuous space (e.g. 2-D coordinates) representing the domain of possible tasks.
- **Knowledge grid**: A discretised version of that space. Each cell stores an experience count.
- **Gaussian propagation**: When an agent gains experience at a location, nearby cells also receive a fraction of that experience, controlled by `propagation_sigma`.
- **Knowledge vs. experience**: Raw experience counts are transformed into knowledge values using a power law (`experience^b`), where `b` is derived from the `learning_rate` parameter. This models diminishing returns from repeated exposure.
- **Transmission**: Knowledge can be transferred between agents. An agent absorbs a fraction (`transmission_factor`) of the experience difference from a more knowledgeable peer.
- **Decay**: Experience decays over time following a research-paper formula, modelling knowledge forgetting.

## Installation

```bash
pip install -e .
```

Requires Python 3.13+ with `numpy` and `scipy`.

For 2-D visualisation support:

```bash
pip install -e ".[viz]"
```

For development (tests):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from ongoing import KnowledgeGrid

# Create a 50x50 grid over a [0, 100] x [0, 100] embedding space
grid = KnowledgeGrid(
    shape=(50, 50),
    propagation_sigma=2.0,
    learning_rate=0.1,
    embedding_bounds=np.array([[0, 100], [0, 100]]),
)

# Simulate the agent handling tickets at various locations
grid.add_ticket_knowledge(np.array([25.0, 30.0]))
grid.add_ticket_knowledge(np.array([26.0, 31.0]))
grid.add_ticket_knowledge(np.array([80.0, 10.0]))

# Query knowledge at a specific location
knowledge = grid.get_knowledge(np.array([25.0, 30.0]))
experiences = grid.get_num_experiences(np.array([25.0, 30.0]))

print(f"Knowledge: {knowledge:.4f}, Experiences: {experiences:.4f}")

# Grid-wide statistics
print(f"Max knowledge: {grid.get_max_knowledge():.4f}")
print(f"Max experiences: {grid.get_max_experiences():.4f}")
```

### Knowledge transmission between agents

```python
senior = KnowledgeGrid(shape=(50, 50), learning_rate=0.1)
junior = KnowledgeGrid(shape=(50, 50), learning_rate=0.1, transmission_factor=0.3)

# Senior accumulates experience
for _ in range(20):
    senior.add_ticket_knowledge(np.array([50.0, 50.0]))

# Junior absorbs 30% of the experience gap from the senior
junior.transmit_from(senior)
```

### Saving and loading

```python
grid.save("my_agent.npz")
restored = KnowledgeGrid.load("my_agent.npz")
```

### Visualisation

```python
# 2-D grids: renders experience and knowledge heatmaps (requires matplotlib)
fig = grid.render()
fig.savefig("knowledge.png")

# Higher-dimensional grids: prints and returns a statistics table
stats = grid_3d.render()
```

## API reference

### `KnowledgeGrid(shape, propagation_sigma, transmission_factor, learning_rate, embedding_bounds, methods)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | `tuple[int, ...]` | *(required)* | Grid dimensions (e.g. `(50, 50)` for 2-D) |
| `propagation_sigma` | `float` | `1.0` | Std. dev. of the Gaussian used to spread experience to neighbouring cells |
| `transmission_factor` | `float` | `0.5` | Factor for knowledge transmission between agents |
| `learning_rate` | `float` | `0.1` | Controls the power-law exponent for the experience-to-knowledge transform |
| `embedding_bounds` | `np.ndarray` | `[[0,100]]` per dim | Shape `(n_dims, 2)` array of `[min, max]` bounds for each embedding dimension |
| `methods` | `list[str]` | `['propagation', 'transmission']` | Knowledge update methods to enable |

### Methods

| Method | Description |
|---|---|
| `add_ticket_knowledge(embedding)` | Record a new experience at the given embedding location. Propagates to neighbours via Gaussian blur. |
| `get_knowledge(embedding)` | Get the knowledge value (power-law transformed) at a location. |
| `get_num_experiences(embedding)` | Get the raw experience count at a location. |
| `get_max_knowledge()` / `get_max_experiences()` | Grid-wide maximums. |
| `embedding_to_coords(embedding)` | Convert continuous embedding to discrete grid coordinates. |
| `coords_to_embedding(coords)` | Convert grid coordinates back to continuous embedding. |
| `transmit_from(other)` | Absorb knowledge from another grid, scaled by `transmission_factor`. Only cells where the other grid has more experience are updated. |
| `decay_knowledge()` | Apply time-based decay to the experience grid. |
| `save(path)` | Persist the grid and all parameters to a `.npz` file. |
| `KnowledgeGrid.load(path)` | Class method to restore a grid from a `.npz` file. |
| `render()` | 2-D grids: matplotlib heatmap figure. Other dims: prints and returns a statistics dict. |

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```
