from typing import Tuple, List
import numpy as np
import numba as nb
from scipy.ndimage import gaussian_filter

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
    A grid-based represntation of knowledge for an agent. (typically a technician or worker).
    A grid is a n-D array (usually 2-D) where each cell contains information about the knowledge state of that location.
    """
    def __init__(
        self,
        shape: Tuple[int],
        propagation_sigma: float = 1.0,
        transmission_factor: float = 0.5,
        learning_rate: float = 0.1,
        embedding_bounds: np.ndarray = np.array([[0, 100], [0, 100]]),  # Assuming 2D embeddings with x and y bounds
        methods: List[str] = ['propagation', 'transmission'],
    ):
        """
        Constructor for the KnowledgeGrid class.
        Args:
            shape (Tuple[int]): The shape of the grid (e.g., (length, width) for a 2D grid).
            propagation_sigma (float): Standard deviation for Gaussian propagation of knowledge.
            transmission_factor (float): Factor determining how much knowledge is transmitted between agents.
            learning_rate (float): Rate at which knowledge is learned or updated.
            embedding_bounds (np.ndarray): Bounds for the embeddings to map them to grid coordinates. (A numpy array of shape (z, 2) where z is the number of dimensions of the input embedding space)
            methods (List[str]): List of methods to use for knowledge update
        """
        
        self._shape = shape
        self._grid = np.zeros(shape, dtype=np.float32)
        self._propagation_sigma = propagation_sigma
        self._transmission_factor = transmission_factor
        self._learning_rate = learning_rate
        self._embedding_bounds = embedding_bounds
        self._methods = methods
        self._validate_methods()
        
        self.b = -np.log(self._learning_rate)/np.log(2)  # for decay formula
        
    def _validate_methods(self):
        valid_methods = {'propagation', 'transmission'}
        for method in self._methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}'. Valid methods are {valid_methods}.")
            
    def embedding_to_coords(self, embedding: np.ndarray) -> Tuple[int]:
        """
        Convert an embedding to grid coordinates.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            Tuple[int]: Discrete grid coordinates.
        """
        # Scale the embedding to the grid size
        scaled_embedding = (embedding - self._embedding_bounds[:, 0]) / (self._embedding_bounds[:, 1] - self._embedding_bounds[:, 0])
        grid_coords = np.clip((scaled_embedding * (np.array(self._shape) - 1)).astype(int), 0, np.array(self._shape) - 1)
        return tuple(grid_coords)
    
    def coords_to_embedding(self, coords: Tuple[int]) -> np.ndarray:
        """
        Convert grid coordinates back to an approximate embedding.
        Args:
            coords (Tuple[int]): Discrete grid coordinates.
        Returns:
            np.ndarray: Continuous embedding (e.g., [x, y] coordinates).
        """
        scaled_coords = np.array(coords) / (np.array(self._shape) - 1)
        embedding = scaled_coords * (self._embedding_bounds[:, 1] - self._embedding_bounds[:, 0]) + self._embedding_bounds[:, 0]
        return embedding
    
    def get_num_experiences(self, embedding: np.ndarray) -> float:
        """
        Get the knowledge value at a specific embedding.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            float: Knowledge value at the specified embedding.
        """
        coords = self.embedding_to_coords(embedding)
        return self._grid[coords]
    
    def get_knowledge(self, embedding: np.ndarray) -> float:
        """
        Get the knowledge value at a specific embedding.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
        Returns:
            float: Knowledge value at the specified embedding.
        """
        coords = self.embedding_to_coords(embedding)
        return self._grid[coords]**self.b
    
    def get_max_knowledge(self) -> float:
        """
        Get the maximum knowledge value in the grid.
        Returns:
            float: Maximum knowledge value in the grid.
        """
        return np.max(self._grid)
    
    def add_ticket_knowledge(self, embedding: np.ndarray):
        """
        Add knowledge to the grid at a specific embedding.
        Args:
            embedding (np.ndarray): Continuous embedding (e.g., [x, y] coordinates).
            knowledge_value (float): Knowledge value to add.
        """
        coords = self.embedding_to_coords(embedding)
        self._grid = add_gaussian_bump_scipy(
            self._grid,
            center=coords,
            delta=1.0,
            sigma=self._propagation_sigma,
            mode='constant',
            truncate=3.0
        )
        
    def decay_knowledge(self):
        """
        Decay the knowledge in the grid over time.
        """
        f = (self.b*(1-self.b)*np.log(self._grid))/(np.log((1+self.B)/(self._grid**self.b)))
        s = ((1 - self.b) + self._grid**(1-self.b))**(1/1-self.b) - self._grid
        self._grid = (self._grid)**(1+f/self.b) * (self._grid + s)**(-f/self.b)