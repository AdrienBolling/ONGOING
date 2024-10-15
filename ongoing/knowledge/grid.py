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


@partial(jax.jit, static_argnames=['grid_size'])
def _blank_grid(grid_shape: Tuple[int]):
    """
    Initialize the experience grid
    """
    return jnp.zeros(grid_shape, dtype=jnp.float32)


@partial(jax.jit, static_argnames=['grid_cell_volume'])
def compute_hypervolume(experience_grid, grid_cell_volume):
    """
    Get the hypervolume of the experience grid
    """
    return jnp.sum(experience_grid) * grid_cell_volume


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
def _compute_experience_without_supervisor(experience_grid, ticket_embedding: jnp.ndarray, learning_rate: float):
    """
    Compute the experience increase for a given ticket embedding
    :param ticket_embedding: jnp.ndarray - ticket embedding
    :return: float - new experience
    """
    previous_experience = get_experience(experience_grid, ticket_embedding)
    return previous_experience + jnp.log1p(jnp.exp(-previous_experience)) * learning_rate


@partial(jax.jit, static_argnames=['threshold'])
def _propagate_increase(grid: jnp.ndarray, coord, incr, threshold, gaussian) -> jnp.ndarray:
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
    return updated_grid


@jax.jit
def get_experience(experience_grid, ticket_embedding: jnp.ndarray):
    """
    Get the experience for a given ticket embedding.
    """
    coord = tuple(ticket_embedding.astype(int))
    return experience_grid[coord]


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
            propagation_threshold: float = 0.1,
            ticket_embedding_shape: Tuple[int] = (10,),
            feature_max: Tuple[float] = 1.0,
            feature_min: Tuple[float] = 0.0,
            propagations: List[str] = None,
    ):
        """
        Constructor for KnowledgeGrid class

        Args:
            size: Tuple[int] - size of the knowledge grid
            technician: Technician - technician who owns the knowledge grid
            propagation_sigma: float - sigma value for propagation
            transmission_factor: float - transmission factor for propagation
            propagation_threshold: float - propagation threshold
            ticket_embedding_shape: Tuple[int] - shape of the ticket embedding
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
        self._ticket_embedding_shape = ticket_embedding_shape
        self._feature_max = feature_max
        self._feature_min = feature_min
        self._propagations = propagations

        # Check if feature_max and feature_min are tuples of same shape as ticket_embedding_shape, or single values
        if isinstance(feature_max, float):
            self._feature_max = (feature_max,) * len(ticket_embedding_shape)

        if isinstance(feature_min, float):
            self._feature_min = (feature_min,) * len(ticket_embedding_shape)

        if len(self._feature_max) != len(ticket_embedding_shape) or len(self._feature_min) != len(
                ticket_embedding_shape):
            raise ValueError("feature_max and feature_min should have the same shape as ticket_embedding_shape, "
                             "or be single values")

        if isinstance(size, int):
            self._size = (size,) * len(ticket_embedding_shape)

        if len(self._size) != len(ticket_embedding_shape):
            raise ValueError("size should have the same shape as ticket_embedding_shape, or be a single value")

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

        self._grid_cell_volume = jnp.prod(1 / (self._feature_max - self._feature_min))

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
        Get the interquartile range of the experience grid
        :return: - interquartile range of the experience grid
        """
        return jnp.percentile(self._grid, 75) - jnp.percentile(self._grid, 25)

    def get_hypervolume(self):
        """
        Get the hypervolume of the experience grid
        :return: - hypervolume of the experience grid
        """
        return compute_hypervolume(self._grid, self.grid_cell_volume)

    def coords_to_embedding(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Convert grid coordinates to ticket embedding
        :param coords: jnp.ndarray - grid coordinates
        :return: jnp.ndarray - ticket embedding
        """
        coords_normalized = (coords + 1) / jnp.array(self._size)  # Add 1 to reverse the earlier subtraction
        embedding = coords_normalized * (self._feature_max - self._feature_min) + jnp.array(
            self._feature_min)
        return embedding

    def embedding_to_coords(self, embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Convert ticket embedding to grid coordinates
        :param embedding: jnp.ndarray - ticket embedding
        :return: jnp.ndarray - grid coordinates (discrete)
        """
        coords_normalized = (embedding - self._feature_min) / (
                self._feature_max - self._feature_min)
        coords = (coords_normalized * jnp.array(self._size) - 1).astype(int)
        return coords

    def get_knowledge(self, ticket_embedding: jnp.ndarray) -> float:
        """
        Get the knowledge for a given ticket embedding
        :param ticket_embedding: jnp.ndarray - ticket embedding
        :return: float - knowledge
        """
        return get_experience(self._grid, ticket_embedding)

    def add_ticket_knowledge(self, ticket_embedding: jnp.ndarray):
        """
        Add ticket knowledge to the knowledge grid
        :param ticket_embedding: jnp.ndarray - ticket embedding
        :param supervisor: Technician - supervisor technician
        """
        new_experience = _compute_experience_without_supervisor(self._grid, ticket_embedding,
                                                                self._technician.learning_rate)

        self._grid = _propagate_increase(self._grid, tuple(ticket_embedding.astype(int)), new_experience,
                                         self._propagation_threshold, self.gaussian)


    def render(self, mode='2d', style='surface', dim1: int = 0, dim2: int = 1):
        """
        Renders the experience grid
        :param mode: str - mode of rendering
        :return:
        """
        if mode == '2d':
            if style == 'surface':
                self._render2d_surface(dim1, dim2)
            elif style == 'scatter':
                self._render2d(dim1, dim2)
            else:
                raise ValueError(f"Unknown rendering style: {style}")
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")

    import plotly.graph_objects as go
    import numpy as np

    def _render2d(self, dim1: int, dim2: int):
        """
        Render a 2D slice of the experience grid along the chosen dimensions (dim1, dim2) in 3D space.

        :param dim1: The first dimension to visualize.
        :param dim2: The second dimension to visualize.
        """
        # Extract the grid from the ExperienceGrid object
        grid = self._grid

        # Initialize x, y, and z lists to store the coordinates and experience values
        x, y, z = [], [], []

        # Calculate the step size for each coordinate based on grid size and value range
        x_step = (self._feature_max[dim1] - self._feature_min[dim1]) / self._size[dim1]
        y_step = (self._feature_max[dim2] - self._feature_min[dim2]) / self._size[dim2]

        # Iterate over the selected dimensions of the grid
        for i in range(grid.shape[dim1]):
            for j in range(grid.shape[dim2]):
                # Map grid indices to actual feature values for the selected dimensions
                x_coord = self._feature_min[dim1] + i * x_step
                y_coord = self._feature_min[dim2] + j * y_step

                # Append coordinates and experience values
                x.append(x_coord)
                y.append(y_coord)
                z.append(float(grid[i, j]))  # Assuming the grid stores experience directly at these indices

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                           mode='markers',
                                           marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8))])

        # Update plot layout
        fig.update_layout(title=f'2D Slice of Experience Grid (Dimensions {dim1}, {dim2}) in 3D Space',
                          scene=dict(
                              xaxis_title=f'Coord {dim1}',
                              yaxis_title=f'Coord {dim2}',
                              zaxis_title='Experience Value'),
                          margin=dict(l=0, r=0, b=0, t=0))

        # Show the plot
        fig.show()

    def _render2d_surface(self, dim1: int, dim2: int):
        """
        Render a 2D surface slice of the experience grid along the chosen dimensions (dim1, dim2) in 3D space.

        :param dim1: The first dimension to visualize.
        :param dim2: The second dimension to visualize.
        """
        # Extract the grid from the ExperienceGrid object
        grid = self._grid

        # Initialize x and y lists to store the coordinates
        x, y = [], []

        # Calculate the step size for each coordinate based on grid size and value range
        x_step = (self._feature_max[dim1] - self._feature_min[dim1]) / self._size[dim1]
        y_step = (self._feature_max[dim2] - self._feature_min[dim2]) / self._size[dim2]

        # Iterate over the selected dimensions of the grid to create the x and y coordinates
        for i in range(grid.shape[dim1]):
            x_coord = self._feature_min[dim1] + i * x_step
            x.append(x_coord)

        for j in range(grid.shape[dim2]):
            y_coord = self._feature_min[dim2] + j * y_step
            y.append(y_coord)

        # Convert the lists to a meshgrid for the surface plot
        x, y = np.meshgrid(x, y)

        # Create a surface plot
        fig = go.Figure(data=[go.Surface(z=grid, x=x, y=y, colorscale='Viridis')])

        # Update plot layout
        fig.update_layout(title=f'2D Surface Slice of Experience Grid (Dimensions {dim1}, {dim2})',
                          scene=dict(
                              xaxis_title=f'Coord {dim1}',
                              yaxis_title=f'Coord {dim2}',
                              zaxis_title='Experience Value'),
                          margin=dict(l=0, r=0, b=0, t=0))

        # Show the plot
        fig.show()
