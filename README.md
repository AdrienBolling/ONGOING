# ONGOING

**Continuous Knowledge Modelling library**

ONGOING models the evolving knowledge of agents (e.g. technicians, workers) as n-dimensional grids. Each cell in the grid represents a region of an embedding space, and accumulates experience as the agent handles tasks ("tickets") in that region. Knowledge is derived from experience via a power-law transform inspired by learning curves: repeated exposure to similar tasks yields diminishing returns.

## Core concepts

- **Embedding space**: A continuous space (e.g. 2-D coordinates) representing the domain of possible tasks.
- **Knowledge grid**: A discretised version of that space. Each cell stores an experience count.
- **Gaussian propagation**: When an agent gains experience at a location, nearby cells also receive a fraction of that experience, controlled by `propagation_sigma`.
- **Knowledge vs. experience**: Raw experience counts are transformed into a dimensionless *knowledge factor* using a power law (`experience^b`), where `b = -log(LR)/log(2)` is Wright's learning exponent derived from `learning_rate` (the Wright LR, must be in `(0.5, 1.0)`). Users typically combine this factor with their own first-item production time `T_1` to recover a domain-specific quantity, e.g. `time_per_unit = T_1 / get_knowledge(...)`.
- **Transmission**: Knowledge can be transferred between agents. An agent absorbs a fraction (`transmission_factor`) of the experience difference from a more knowledgeable peer.
- **Decay**: Experience decays per call to `decay_knowledge()` following the LFCM forgetting formula from [Jaber et al. 2013](https://doi.org/10.1016/j.apm.2013.02.028) (Eqs. 4–6). The formula is written in `T_1`-normalised form, so two dimensionless hyperparameters control the dynamics: `forgetting_time = B/T_1` (time to total forgetting) and `decay_step = τ/T_1` (length of break applied per call).

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
    learning_rate=0.8,         # Wright's LR; lower = faster learning, must be in (0.5, 1.0)
    forgetting_time=2000.0,    # B / T_1
    decay_step=1.0,            # τ / T_1, length of break per decay_knowledge() call
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

### `KnowledgeGrid(shape, propagation_sigma, transmission_factor, learning_rate, forgetting_time, decay_step, embedding_bounds, methods)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | `tuple[int, ...]` | *(required)* | Grid dimensions (e.g. `(50, 50)` for 2-D) |
| `propagation_sigma` | `float` | `1.0` | Std. dev. of the Gaussian used to spread experience to neighbouring cells |
| `transmission_factor` | `float` | `0.5` | Factor for knowledge transmission between agents |
| `learning_rate` | `float` | `0.8` | Wright's LR; must be in `(0.5, 1.0)`. Lower = faster learning. The exponent `b = -log(LR)/log(2)` governs both the knowledge factor (`experience^b`) and the decay formula |
| `forgetting_time` | `float` | `2000.0` | `B / T_1` — time to total forgetting in units of first-item production time. Larger = slower forgetting |
| `decay_step` | `float` | `1.0` | `τ / T_1` — length of break applied by each `decay_knowledge()` call, in units of first-item production time |
| `embedding_bounds` | `np.ndarray` | `[[0,100]]` per dim | Shape `(n_dims, 2)` array of `[min, max]` bounds for each embedding dimension |
| `methods` | `list[str]` | `['propagation', 'transmission']` | Knowledge update methods to enable |

> **Note**: `T_1` (first-item production time) is intentionally kept user-side. The library works only with dimensionless ratios `B/T_1` and `τ/T_1`. To recover a domain-specific time-per-unit, combine the knowledge factor with your own `T_1`: `time_per_unit = T_1 / grid.get_knowledge(embedding)`.

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
