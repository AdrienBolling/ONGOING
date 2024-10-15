from __future__ import annotations

from functools import partial
from typing import Tuple, List

from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Technician:
    id: int
    name: str
    learning_rate: float


VALID_PROPAGATIONS = ["direct", "gaussian"]


@partial(jax.jit, static_argnames=['grid_size', 'ticket_embedding_shape'])
def _blank_grid(grid_size: int, ticket_embedding_shape: int):
    """
    Initialize the experience grid
    :return:
    """
    shape = tuple([grid_size for _ in range(ticket_embedding_shape)])
    return jnp.zeros(shape, dtype=jnp.float32)


class KnowledgeGrid:
    """
    KnowledgeGrid class to represent the knowledge grid of a technician according to embedded tickets
    """

    def __init__(
            self,
            size: int | Tuple[int],
            technician: Technician,
            propagation_sigma: float = 1.0,
            transmission_factor: float = 0.5,
            propagation_threshold: float = 0.1,
            ticket_embedding_shape: Tuple[int] = (10,),
            feature_max: float | Tuple[float] = 1.0,
            feature_min: float | Tuple[float] = 0.0,
            propagations: List[str] = None,
    ):
        """
        Constructor for KnowledgeGrid class

        Args:
            size: int | Tuple[int] - size of the knowledge grid
            technician: Technician - technician who owns the knowledge grid
            propagation_sigma: float - sigma value for propagation
            transmission_factor: float - transmission factor for propagation
            propagation_threshold: float - propagation threshold
            ticket_embedding_shape: Tuple[int] - shape of the ticket embedding
            feature_max: float | Tuple[float] - maximum value for each feature
            feature_min: float | Tuple[float] - minimum value for each feature
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

        self._grid = self._initialize_knowledge_grid()
