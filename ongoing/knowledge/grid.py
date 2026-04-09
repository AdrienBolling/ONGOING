from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.sparse.csgraph import shortest_path
from scipy import sparse


def add_gaussian_bump_scipy(arr, center, delta, sigma, mode='constant', truncate=3.0):
    """
    Works for n-D arrays.
    - mode: 'constant','reflect','nearest','mirror','wrap'
    - truncate: kernel radius (in sigmas)
    """
    imp = np.zeros_like(arr, dtype=float)
    imp[tuple(center)] = delta
    bump = gaussian_filter(imp, sigma=sigma, mode=mode, truncate=truncate)
    return arr.astype(float, copy=True) + bump


class KnowledgeGrid:

    """
    A grid-based representation of knowledge for an agent (typically a technician or worker).
    A grid is a n-D array (usually 2-D) where each cell contains information about the knowledge state of that location.
    """
    def __init__(
        self,
        shape: tuple[int, ...],
        propagation_sigma: float = 1.0,
        transmission_factor: float = 0.5,
        learning_rate: float = 0.1,
        embedding_bounds: np.ndarray | None = None,
        methods: list[str] | None = None,
    ):
        """
        Constructor for the KnowledgeGrid class.
        Args:
            shape (tuple[int, ...]): The shape of the grid (e.g., (length, width) for a 2D grid).
            propagation_sigma (float): Standard deviation for Gaussian propagation of knowledge.
            transmission_factor (float): Factor determining how much knowledge is transmitted between agents.
            learning_rate (float): Rate at which knowledge is learned or updated.
            embedding_bounds (np.ndarray): Bounds for the embeddings to map them to grid coordinates.
                A numpy array of shape (n_dims, 2) where each row is [min, max].
            methods (list[str]): List of methods to use for knowledge update
                (default: ['propagation', 'transmission']).
        """

        if embedding_bounds is None:
            embedding_bounds = np.array([[0, 100]] * len(shape))
        if methods is None:
            methods = ['propagation', 'transmission']

        self._shape = shape
        self._grid = np.zeros(shape, dtype=np.float64)
        self._propagation_sigma = propagation_sigma
        self._transmission_factor = transmission_factor
        self._learning_rate = learning_rate
        self._embedding_bounds = embedding_bounds
        self._methods = methods
        self._validate_methods()
        self._total_num_experiences = 0

        self.b = -np.log(self._learning_rate) / np.log(2)  # for decay formula

        # Physical cell spacing in embedding units (needed for geometric ops)
        span = self._embedding_bounds[:, 1] - self._embedding_bounds[:, 0]
        self._cell_spacing = span / (np.array(shape) - 1)

    def _validate_methods(self):
        valid_methods = {'propagation', 'transmission'}
        for method in self._methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}'. Valid methods are {valid_methods}.")

    # ------------------------------------------------------------------ #
    #  Coordinate mapping
    # ------------------------------------------------------------------ #

    def embedding_to_coords(self, embedding: np.ndarray) -> tuple[int, ...]:
        """
        Convert an embedding to grid coordinates.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            tuple[int, ...]: Discrete grid coordinates.
        """
        scaled = (embedding - self._embedding_bounds[:, 0]) / (
            self._embedding_bounds[:, 1] - self._embedding_bounds[:, 0]
        )
        grid_coords = np.clip(
            (scaled * (np.array(self._shape) - 1)).astype(int),
            0,
            np.array(self._shape) - 1,
        )
        return tuple(grid_coords)

    def coords_to_embedding(self, coords: tuple[int, ...]) -> np.ndarray:
        """
        Convert grid coordinates back to an approximate embedding.
        Args:
            coords (tuple[int, ...]): Discrete grid coordinates.
        Returns:
            np.ndarray: Continuous embedding (e.g., [x, y] coordinates).
        """
        scaled_coords = np.array(coords) / (np.array(self._shape) - 1)
        span = self._embedding_bounds[:, 1] - self._embedding_bounds[:, 0]
        return scaled_coords * span + self._embedding_bounds[:, 0]

    # ------------------------------------------------------------------ #
    #  Querying
    # ------------------------------------------------------------------ #

    def get_num_experiences(self, embedding: np.ndarray) -> float:
        """
        Get the raw experience count at a specific embedding location.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            float: Raw experience count at the specified embedding.
        """
        coords = self.embedding_to_coords(embedding)
        return self._grid[coords]

    def get_knowledge(self, embedding: np.ndarray) -> float:
        """
        Get the knowledge value at a specific embedding, derived from
        experience via a power-law transform.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            float: Knowledge value at the specified embedding.
        """
        coords = self.embedding_to_coords(embedding)
        return self._grid[coords] ** self.b

    def get_max_experiences(self) -> float:
        """
        Get the maximum experience count in the grid.
        """
        return np.max(self._grid)

    def get_max_knowledge(self) -> float:
        """
        Get the maximum knowledge value in the grid.
        """
        return np.max(self._grid) ** self.b

    # ------------------------------------------------------------------ #
    #  Knowledge updates
    # ------------------------------------------------------------------ #

    def add_ticket_knowledge(self, embedding: np.ndarray):
        """
        Add knowledge to the grid at a specific embedding.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        """
        coords = self.embedding_to_coords(embedding)
        self._total_num_experiences += 1
        self._grid = add_gaussian_bump_scipy(
            self._grid,
            center=coords,
            delta=1.0,
            sigma=self._propagation_sigma,
            mode='constant',
            truncate=3.0,
        )

    def transmit_from(self, other: KnowledgeGrid):
        """
        Absorb knowledge from another grid, scaled by this grid's
        transmission_factor.  Only cells where the other grid has more
        experience than this one are updated.

        Args:
            other (KnowledgeGrid): The grid to absorb knowledge from.
        """
        if self._shape != other._shape:
            raise ValueError(
                f"Grid shapes must match: {self._shape} vs {other._shape}"
            )
        diff = other._grid - self._grid
        gain = np.where(diff > 0, diff * self._transmission_factor, 0.0)
        self._grid = self._grid + gain

    def decay_knowledge(self):
        """
        Decay the knowledge in the grid over time using the research-paper
        formula.  Cells with zero or near-zero experience are left unchanged
        to avoid log(0) / division-by-zero.  Any cells that produce NaN from
        numerical edge cases in the formula are also left unchanged.
        """
        mask = self._grid > 1e-12
        g = self._grid[mask].copy()
        knowledge = g ** self.b

        denom = np.log((1 + self.b) / knowledge)
        # Avoid division by zero where denom is zero
        safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
        f = (self.b * (1 - self.b) * np.log(g)) / safe_denom

        base_s = (1 - self.b) + g ** (1 - self.b)
        # When base_s <= 0, the fractional exponent is undefined;
        # clamp to a small positive value so the power is computable.
        base_s = np.maximum(base_s, 1e-30)
        s = base_s ** (1 / (1 - self.b)) - g

        result = g ** (1 + f / self.b) * (g + s) ** (-f / self.b)

        # Replace any NaN/Inf with the original values (no decay for
        # cells where the formula is numerically unstable).
        bad = ~np.isfinite(result)
        result[bad] = g[bad]

        self._grid[mask] = result

    # ------------------------------------------------------------------ #
    #  Geometric / manifold features
    # ------------------------------------------------------------------ #

    def _knowledge_field(self) -> np.ndarray:
        """Return the knowledge scalar field (experience ** b), safe for
        zero cells."""
        return np.where(self._grid > 0, self._grid ** self.b, 0.0)

    def gradient_field(self) -> list[np.ndarray]:
        """
        Compute the gradient of the knowledge field in embedding-space
        units.  Returns a list of arrays, one per dimension, each with the
        same shape as the grid.

        Interpretation: at every cell, the gradient vector points in the
        direction of steepest knowledge increase.  Its magnitude tells you
        how sharply knowledge changes there.
        """
        k = self._knowledge_field()
        return np.gradient(k, *self._cell_spacing)

    def gradient_at(self, embedding: np.ndarray) -> np.ndarray:
        """
        Knowledge gradient vector at a single embedding location.

        Args:
            embedding: Continuous embedding coordinates.
        Returns:
            np.ndarray of shape (n_dims,) — the gradient vector in
            embedding-space units.
        """
        grad = self.gradient_field()
        coords = self.embedding_to_coords(embedding)
        return np.array([g[coords] for g in grad])

    def gradient_magnitude(self) -> np.ndarray:
        """
        Magnitude of the knowledge gradient at every cell.

        High values delineate the *knowledge frontier*: the boundary
        between regions the agent knows well and regions it does not.
        """
        grad = self.gradient_field()
        return np.sqrt(sum(g ** 2 for g in grad))

    def curvature(self) -> np.ndarray:
        """
        Scalar curvature of the knowledge field (discrete Laplacian).

        - Negative values → local peaks (specialisation centres).
        - Positive values → local valleys (knowledge gaps surrounded by
          more knowledgeable regions).
        - Near zero → flat or saddle regions.

        The Laplacian is computed with the correct cell spacing so values
        are comparable across grids with different resolutions.
        """
        k = self._knowledge_field()
        laplacian = np.zeros_like(k)
        for axis, dx in enumerate(self._cell_spacing):
            laplacian += np.gradient(np.gradient(k, dx, axis=axis), dx, axis=axis)
        return laplacian

    def curvature_at(self, embedding: np.ndarray) -> float:
        """Scalar curvature at a single embedding location."""
        coords = self.embedding_to_coords(embedding)
        return float(self.curvature()[coords])

    def knowledge_volume(self) -> float:
        """
        Total knowledge integrated over the embedding space (the
        hyper-volume under the knowledge surface).

        This is a single scalar summarising the agent's overall breadth
        of expertise, accounting for the physical size of each cell.
        """
        cell_vol = float(np.prod(self._cell_spacing))
        return float(np.sum(self._knowledge_field()) * cell_vol)

    def knowledge_entropy(self) -> float:
        """
        Shannon entropy of the normalised knowledge distribution.

        - High entropy → knowledge spread uniformly (generalist).
        - Low entropy  → knowledge concentrated in few cells (specialist).
        - Returns 0.0 when the grid is empty.
        """
        k = self._knowledge_field().ravel()
        total = k.sum()
        if total == 0:
            return 0.0
        p = k / total
        # Filter out zeros to avoid log(0)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def specialisation_index(self) -> float:
        """
        Ratio of the agent's entropy to maximum possible entropy
        (uniform distribution over all cells), inverted so that:

        - 1.0 → perfect specialist (all knowledge in one cell).
        - 0.0 → perfect generalist (uniform knowledge everywhere).
        """
        max_entropy = np.log(np.prod(self._shape))
        if max_entropy == 0:
            return 0.0
        return float(1.0 - self.knowledge_entropy() / max_entropy)

    def frontier(self, threshold: float | None = None) -> np.ndarray:
        """
        Boolean mask of *frontier cells*: locations where the gradient
        magnitude exceeds a threshold, marking the boundary between
        known and unknown territory.

        Args:
            threshold: Gradient-magnitude cutoff.  Defaults to the mean
                of all non-zero gradient magnitudes.
        Returns:
            Boolean array with the same shape as the grid.
        """
        mag = self.gradient_magnitude()
        if threshold is None:
            nonzero = mag[mag > 0]
            threshold = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0
        return mag > threshold

    def expertise_clusters(self, knowledge_threshold: float | None = None) -> tuple[np.ndarray, int]:
        """
        Connected components of cells whose knowledge exceeds a
        threshold.  Each connected component is a distinct *area of
        competence*.

        Args:
            knowledge_threshold: Minimum knowledge to count a cell as
                "competent".  Defaults to 10 % of peak knowledge.
        Returns:
            (labels, n_clusters) where labels is an integer array (same
            shape as the grid; 0 = below threshold) and n_clusters is
            the number of distinct competence regions.
        """
        k = self._knowledge_field()
        if knowledge_threshold is None:
            peak = np.max(k)
            if peak <= 0:
                return np.zeros_like(k, dtype=int), 0
            knowledge_threshold = 0.1 * peak
        binary = k >= knowledge_threshold
        if not np.any(binary):
            return np.zeros_like(k, dtype=int), 0
        labels, n = label(binary)
        return labels, int(n)

    def geodesic_distance(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Shortest-path distance between two embeddings on the knowledge
        manifold, where traversal cost through a cell is inversely
        proportional to its knowledge value.

        High-knowledge regions are cheap to cross; low-knowledge regions
        are expensive.  This measures how easily an agent can transfer
        expertise from one task domain to another given its current
        knowledge landscape.

        Uses Dijkstra on the grid adjacency graph (face-connected
        neighbours).

        Args:
            source: Embedding of the start location.
            target: Embedding of the destination.
        Returns:
            The geodesic cost (float).  np.inf if no path exists through
            cells with non-zero knowledge.
        """
        k = self._knowledge_field()
        # Cost = 1 / (knowledge + eps), scaled by physical step length
        eps = 1e-12
        cost = 1.0 / (k + eps)

        flat_size = int(np.prod(self._shape))

        # Build sparse adjacency: connect each cell to its face neighbours
        rows = []
        cols = []
        weights = []

        for axis in range(len(self._shape)):
            dx = self._cell_spacing[axis]
            # Indices of all cells and their +1-neighbour along this axis
            slices_src = [slice(None)] * len(self._shape)
            slices_dst = [slice(None)] * len(self._shape)
            slices_src[axis] = slice(0, self._shape[axis] - 1)
            slices_dst[axis] = slice(1, self._shape[axis])

            idx_src = np.arange(flat_size).reshape(self._shape)[tuple(slices_src)].ravel()
            idx_dst = np.arange(flat_size).reshape(self._shape)[tuple(slices_dst)].ravel()

            cost_flat = cost.ravel()
            edge_weight = 0.5 * (cost_flat[idx_src] + cost_flat[idx_dst]) * dx

            # Both directions
            rows.extend(idx_src)
            cols.extend(idx_dst)
            weights.extend(edge_weight)
            rows.extend(idx_dst)
            cols.extend(idx_src)
            weights.extend(edge_weight)

        graph = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(flat_size, flat_size)
        )

        src_coords = self.embedding_to_coords(source)
        tgt_coords = self.embedding_to_coords(target)
        src_flat = int(np.ravel_multi_index(src_coords, self._shape))
        tgt_flat = int(np.ravel_multi_index(tgt_coords, self._shape))

        dist = shortest_path(graph, directed=False, indices=src_flat)
        return float(dist[tgt_flat])

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path):
        """
        Save the grid and its parameters to a .npz file.
        Args:
            path: File path (will get .npz extension if not present).
        """
        np.savez(
            path,
            grid=self._grid,
            shape=np.array(self._shape),
            propagation_sigma=np.array(self._propagation_sigma),
            transmission_factor=np.array(self._transmission_factor),
            learning_rate=np.array(self._learning_rate),
            embedding_bounds=self._embedding_bounds,
            methods=np.array(self._methods),
            total_num_experiences=np.array(self._total_num_experiences),
        )

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGrid:
        """
        Load a KnowledgeGrid from a .npz file.
        Args:
            path: Path to the .npz file.
        Returns:
            KnowledgeGrid: Restored grid instance.
        """
        data = np.load(path, allow_pickle=False)
        shape = tuple(data['shape'])
        grid = cls(
            shape=shape,
            propagation_sigma=float(data['propagation_sigma']),
            transmission_factor=float(data['transmission_factor']),
            learning_rate=float(data['learning_rate']),
            embedding_bounds=data['embedding_bounds'],
            methods=list(data['methods']),
        )
        grid._grid = data['grid']
        grid._total_num_experiences = int(data['total_num_experiences'])
        return grid

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def render(self):
        """
        Visualise the knowledge grid.

        - For 2-D grids: renders a heatmap using matplotlib.
        - For other dimensionalities: prints a statistics table.

        Returns:
            matplotlib Figure for 2-D grids, or a dict of statistics otherwise.
        """
        if len(self._shape) == 2:
            return self._render_2d()
        return self._render_stats()

    def _render_2d(self):
        """Render a 2-D heatmap of the experience grid."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for 2-D rendering. "
                "Install it with:  pip install ongoing[viz]"
            )

        knowledge_grid = np.where(
            self._grid > 0, self._grid ** self.b, 0.0
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Experience heatmap
        bounds = self._embedding_bounds
        extent = [bounds[1, 0], bounds[1, 1], bounds[0, 0], bounds[0, 1]]
        im0 = axes[0].imshow(
            self._grid, origin='lower', aspect='auto', extent=extent,
        )
        axes[0].set_title('Experience')
        axes[0].set_xlabel(f'Dim 1 [{bounds[1, 0]}, {bounds[1, 1]}]')
        axes[0].set_ylabel(f'Dim 0 [{bounds[0, 0]}, {bounds[0, 1]}]')
        fig.colorbar(im0, ax=axes[0])

        # Knowledge heatmap
        im1 = axes[1].imshow(
            knowledge_grid, origin='lower', aspect='auto', extent=extent,
        )
        axes[1].set_title('Knowledge')
        axes[1].set_xlabel(f'Dim 1 [{bounds[1, 0]}, {bounds[1, 1]}]')
        axes[1].set_ylabel(f'Dim 0 [{bounds[0, 0]}, {bounds[0, 1]}]')
        fig.colorbar(im1, ax=axes[1])

        fig.suptitle('Knowledge Grid')
        fig.tight_layout()
        return fig

    def _render_stats(self) -> dict:
        """Print and return a statistics summary for non-2-D grids."""
        knowledge_grid = np.where(
            self._grid > 0, self._grid ** self.b, 0.0
        )
        nonzero = self._grid[self._grid > 0]

        stats = {
            'dimensions': len(self._shape),
            'shape': self._shape,
            'total_cells': int(np.prod(self._shape)),
            'total_experiences_added': self._total_num_experiences,
            'non_zero_cells': int(np.count_nonzero(self._grid)),
            'coverage (%)': float(np.count_nonzero(self._grid) / np.prod(self._shape) * 100),
            'experience_max': float(np.max(self._grid)),
            'experience_mean': float(np.mean(self._grid)),
            'experience_std': float(np.std(self._grid)),
            'experience_median': float(np.median(self._grid)),
            'knowledge_max': float(np.max(knowledge_grid)),
            'knowledge_mean': float(np.mean(knowledge_grid)),
            'knowledge_std': float(np.std(knowledge_grid)),
        }
        if len(nonzero) > 0:
            stats['experience_mean_nonzero'] = float(np.mean(nonzero))
            stats['experience_p25'] = float(np.percentile(nonzero, 25))
            stats['experience_p75'] = float(np.percentile(nonzero, 75))

        # Print as a formatted table
        max_key = max(len(k) for k in stats)
        print(f"\n{'─' * (max_key + 20)}")
        print(f" Knowledge Grid Statistics ({len(self._shape)}-D)")
        print(f"{'─' * (max_key + 20)}")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f" {key:<{max_key}}  {val:>12.4f}")
            else:
                print(f" {key:<{max_key}}  {str(val):>12}")
        print(f"{'─' * (max_key + 20)}\n")

        return stats
