from functools import partial
from typing import Tuple, List

from flax import struct
import jax
import jax.numpy as jnp

from jax.scipy.signal import convolve

import plotly.graph_objects as go
import numpy as np

@struct.dataclass
class Technician:
    id: int
    name: str
    learning_rate: float


VALID_PROPAGATIONS = ["direct", "gaussian"]


@partial(jax.jit, static_argnames=['grid_shape'])
def _blank_grid(grid_shape: Tuple[int]):
    """
    Initialize the knowledge grid
    """
    return jnp.zeros(grid_shape, dtype=jnp.float32)


@partial(jax.jit, static_argnames=['grid_cell_volume'])
def compute_hypervolume(knowledge_grid, grid_cell_volume):
    """
    Get the hypervolume of the knowledge grid
    """
    return jnp.sum(knowledge_grid) * grid_cell_volume


def gaussian_kernel_1d(size, sigma: float) -> jnp.ndarray:
    """Creates a 1D Gaussian kernel with the given size and sigma."""
    low = -size // 2 + 1
    high = size // 2 + 1
    x = jnp.arange(low, high)
    gauss_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d /= gauss_1d.sum()  # Normalize
    return gauss_1d


def gaussian_kernel(sizes, sigmas) -> jnp.ndarray:
    """Creates an N-dimensional Gaussian kernel where each dimension has its own size and sigma."""

    # Create a 1D kernel for each dimension
    kernels_1d = [gaussian_kernel_1d(size, sigma) for size, sigma in zip(sizes, sigmas)]

    # Start by using the first kernel
    kernel_nd = kernels_1d[0]

    # Iterate through the rest of the kernels and expand the dimensions using tensor products
    for kernel_1d in kernels_1d[1:]:
        kernel_nd = jnp.tensordot(kernel_nd, kernel_1d, axes=0)

    # Normalize the N-dimensional kernel
    kernel_nd /= jnp.sum(kernel_nd)

    return kernel_nd


@partial(jax.jit, static_argnames=['learning_rate'])
def _compute_experience_without_supervisor(knowledge_grid, coords: jnp.ndarray, learning_rate: float):
    """
    Compute the experience increase for a given ticket embedding
    :param ticket_embedding: jnp.ndarray - ticket embedding
    :return: float - new experience
    """
    previous_experience = get_experience(knowledge_grid, coords)
    return (jnp.log(1 + jnp.exp(previous_experience)) - previous_experience) * learning_rate + previous_experience


@partial(jax.jit, static_argnames=['threshold'])
def _propagate_increase(grid: jnp.ndarray, coord, incr, threshold, gaussian):
    """
    Increase the value at specific coordinates by the given increment and propagate it in a bell-curve manner.
    """
    delta_grid = jnp.zeros_like(grid)
    delta_grid = delta_grid.at[coord].add(incr)
    # Apply Gaussian filter using convolution
    propagated_grid = convolve(delta_grid, gaussian, mode='same')
    # Apply the threshold
    scaling_factor = incr / propagated_grid[coord]
    propagated_grid *= scaling_factor
    propagated_grid = jnp.where(propagated_grid < threshold, 0, propagated_grid)
    # Add the propagated values to the original grid
    updated_grid = grid + propagated_grid
    return updated_grid, scaling_factor, incr


@jax.jit
def get_experience(knowledge_grid, coord: jnp.ndarray):
    """
    Get the experience for a given ticket embedding.
    """
    return knowledge_grid[coord]


class KnowledgeGrid:
    """
    KnowledgeGrid class to represent the knowledge grid of a technician according to embedded tickets
    """

    def __init__(
            self,
            size: Tuple[int],
            technician: Technician,
            propagation_sigma: float = 1.0,
            transmission_factor: float = 0.5,
            propagation_threshold: float = 0.01,
            feature_max: Tuple[float] = 1.0,
            feature_min: Tuple[float] = 0.0,
            propagations: List[str] = None,
    ):
        """
        Constructor for KnowledgeGrid class

        Args:
            size: Tuple[int] - size of the knowledge grid, one size per dimension of the ticket embedding
            technician: Technician - technician who owns the knowledge grid
            propagation_sigma: float - sigma value for propagation
            transmission_factor: float - transmission factor for propagation
            propagation_threshold: float - propagation threshold
            feature_max: Tuple[float] - maximum value for each feature
            feature_min: Tuple[float] - minimum value for each feature
            propagations: List[str] - list of propagation methods to be used (default: ["direct", "gaussian"])
        """
        if propagations is None:
            propagations = ["direct", "gaussian"]

        self._size = size
        self._technician = technician
        self._propagation_sigma = propagation_sigma
        self._transmission_factor = transmission_factor
        self._propagation_threshold = propagation_threshold
        self._feature_max = feature_max
        self._feature_min = feature_min
        self._propagations = propagations

        # Check that all propagations are valid
        for propagation in propagations:
            if propagation not in VALID_PROPAGATIONS:
                raise ValueError(f"Invalid propagation method: {propagation}")

        self._grid = _blank_grid(self._size)
        self.shape = self._grid.shape
        
        self._feature_max = jnp.array(self._feature_max)
        self._feature_min = jnp.array(self._feature_min)

        # Replicate the sigma value for each dimension
        sigmas = jnp.array((self._propagation_sigma,) * len(self._size))
        sizes = jnp.array(self._size)
        scaled_sigmas = sigmas * sizes / (self._feature_max - self._feature_min)
        self.kernel_sizes = (2 * jnp.ceil(3 * scaled_sigmas) + 1).astype(int)

        self.gaussian = gaussian_kernel(self.kernel_sizes, scaled_sigmas)

        self._grid_cell_volume = jnp.prod(1 / (self._feature_max - self._feature_min)).item()

    def reset(self):
        """
        Reset the knowledge grid
        """
        self._grid = _blank_grid(self._size)

    def get_technician(self) -> Technician:
        """
        Get the technician who owns the knowledge grid
        :return: Technician - technician
        """
        return self._technician

    def get_grid(self) -> jnp.ndarray:
        """
        Get the knowledge grid
        :return: jnp.ndarray - knowledge
        """
        return self._grid

    def get_knowledge_iqr(self):
        """
        Get the interquartile range of the knowledge grid
        :return: - interquartile range of the knowledge grid
        """
        return jnp.percentile(self._grid, 75) - jnp.percentile(self._grid, 25)

    def get_hypervolume(self):
        """
        Get the hypervolume of the knowledge grid
        :return: - hypervolume of the knowledge grid
        """
        return compute_hypervolume(self._grid, self._grid_cell_volume)

    def coords_to_embedding(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Convert grid coordinates to ticket embedding
        :param coords: jnp.ndarray - grid coordinates
        :return: jnp.ndarray - ticket embedding
        """
        coords = jnp.array(coords)
        coords_normalized = (coords + 1) / jnp.array(self._size)  # Add 1 to reverse the earlier subtraction
        embedding = coords_normalized * (self._feature_max - self._feature_min) + jnp.array(
            self._feature_min)
        return embedding

    def embedding_to_coords(self, embedding: jnp.ndarray):
        """
        Convert ticket embedding to grid coordinates
        :param embedding: jnp.ndarray - ticket embedding
        :return: jnp.ndarray - grid coordinates (discrete)
        """
        coords_normalized = (embedding - self._feature_min) / (
                self._feature_max - self._feature_min)
        coords = (coords_normalized * jnp.array(self._size) - 1).astype(int)
        return tuple(coords)

    def get_knowledge(self, ticket_embedding: jnp.ndarray) -> float:
        """
        Get the knowledge for a given ticket embedding
        :param ticket_embedding: jnp.ndarray - ticket embedding
        :return: float - knowledge
        """
        coords = self.embedding_to_coords(ticket_embedding)
        return get_experience(self._grid, coords)

    def add_ticket_knowledge(self, ticket_embedding: jnp.ndarray):
        """
        Add ticket knowledge to the knowledge grid
        :param ticket_embedding: jnp.ndarray - ticket embedding
        :param supervisor: Technician - supervisor technician
        """
        coords = self.embedding_to_coords(ticket_embedding)
        new_experience = _compute_experience_without_supervisor(self._grid, coords,
                                                                self._technician.learning_rate)

        # Propagate the increase in knowledge
        incr = new_experience - get_experience(self._grid, coords)
        self._grid, scaling_factor, incr = _propagate_increase(self._grid, coords, incr,
                                         self._propagation_threshold, self.gaussian)

    def get_max_knowledge(self):
        """
        Get the maximum knowledge in the knowledge grid
        :return: float - maximum knowledge
        """
        return jnp.max(self._grid)

    def render(self, dim1: int, dim2: int, reduction='slice', slice_index=0, streamlit=False, max_knowledge=None):
        """
        Render a 3D plot of the grid using two chosen dimensions (dim1, dim2) and reducing the others.

        Args:
        - dim1 (int): The first dimension to plot on the X axis.
        - dim2 (int): The second dimension to plot on the Y axis.
        - reduction (str): How to handle the remaining dimensions ('mean', 'sum', or 'slice').
        - slice_index (int): If using 'slice' reduction, the index to slice the remaining dimensions at.

        Returns:
        - Plotly 3D plot figure.
        """

        if max_knowledge is None:
            max_knowledge = self.get_max_knowledge()
        if type(max_knowledge) is str and max_knowledge == 'percentage':
            max_knowledge = self.get_max_knowledge()

        # Step 1: Handle the reduction of dimensions other than dim1 and dim2
        grid = self._grid  # Assuming self._grid holds the knowledge grid

        other_dims = [i for i in range(grid.ndim) if i not in (dim1, dim2)]

        # Step 2: Reduce the other dimensions
        if reduction == 'mean':
            # Collapse the other dimensions by taking the mean along them
            for dim in other_dims:
                grid = jnp.mean(grid, axis=dim, keepdims=False)
        elif reduction == 'sum':
            # Collapse the other dimensions by summing along them
            for dim in other_dims:
                grid = jnp.sum(grid, axis=dim, keepdims=False)
        elif reduction == 'slice':
            # Slice the other dimensions at the specified index
            for dim in other_dims:
                grid = jnp.take(grid, slice_index, axis=dim)

        # Step 3: Prepare the X and Y axes using the specified dimensions
        x = jnp.arange(grid.shape[dim1])
        y = jnp.arange(grid.shape[dim2])

        # Step 4: Create a meshgrid for the plot
        X, Y = jnp.meshgrid(x, y)

        # Step 5: Z-values correspond to the grid values along dim1 and dim2
        Z = grid[:, :]  # Adjust this slicing based on the grid's shape after reduction
        
        if type(max_knowledge) is str and max_knowledge == 'percentage':
            Z = Z / jnp.max(Z) * 100

        # Step 6: Generate the 3D plot using Plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

        # Step 7: Customize the plot layout
        fig.update_layout(
            title="Knowledge Grid Representation",
            scene=dict(
                xaxis_title=f"Dimension {dim1}",
                yaxis_title=f"Dimension {dim2}",
                zaxis_title=("Knowledge" if type(max_knowledge) is not str else "Knowledge repartition of the technician(%)"),
                zaxis_range=[0, max_knowledge],
            ),
            autosize=True,
        )
        if streamlit:
            import streamlit as st
            st.plotly_chart(fig)
        else:
            return fig