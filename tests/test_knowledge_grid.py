from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ongoing.knowledge.grid import KnowledgeGrid


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def grid_2d():
    return KnowledgeGrid(
        shape=(20, 20),
        propagation_sigma=1.0,
        learning_rate=0.1,
        embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
    )


@pytest.fixture
def grid_3d():
    return KnowledgeGrid(
        shape=(10, 10, 10),
        propagation_sigma=1.0,
        learning_rate=0.1,
        embedding_bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    )


# ------------------------------------------------------------------ #
#  Construction
# ------------------------------------------------------------------ #

class TestConstruction:
    def test_default_bounds_match_shape_dims(self):
        grid = KnowledgeGrid(shape=(5, 5, 5))
        assert grid._embedding_bounds.shape == (3, 2)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid method"):
            KnowledgeGrid(shape=(5, 5), methods=['bogus'])

    def test_grid_starts_at_zero(self, grid_2d):
        assert np.all(grid_2d._grid == 0)
        assert grid_2d._total_num_experiences == 0


# ------------------------------------------------------------------ #
#  Coordinate round-trip
# ------------------------------------------------------------------ #

class TestCoordinateMapping:
    def test_round_trip_corners(self, grid_2d):
        """Embedding -> coords -> embedding should be close to original at corners."""
        for emb in [np.array([0.0, 0.0]), np.array([10.0, 10.0])]:
            coords = grid_2d.embedding_to_coords(emb)
            recovered = grid_2d.coords_to_embedding(coords)
            np.testing.assert_allclose(recovered, emb, atol=1.0)

    def test_round_trip_center(self, grid_2d):
        emb = np.array([5.0, 5.0])
        coords = grid_2d.embedding_to_coords(emb)
        recovered = grid_2d.coords_to_embedding(coords)
        np.testing.assert_allclose(recovered, emb, atol=1.0)

    def test_coords_clipped_to_bounds(self, grid_2d):
        """Out-of-bounds embeddings should be clipped to grid edges."""
        coords = grid_2d.embedding_to_coords(np.array([-5.0, 200.0]))
        assert coords[0] == 0
        assert coords[1] == grid_2d._shape[1] - 1


# ------------------------------------------------------------------ #
#  Adding knowledge
# ------------------------------------------------------------------ #

class TestAddKnowledge:
    def test_single_ticket_increases_experience(self, grid_2d):
        emb = np.array([5.0, 5.0])
        grid_2d.add_ticket_knowledge(emb)
        assert grid_2d.get_num_experiences(emb) > 0
        assert grid_2d._total_num_experiences == 1

    def test_propagation_affects_neighbours(self, grid_2d):
        emb = np.array([5.0, 5.0])
        grid_2d.add_ticket_knowledge(emb)
        neighbour = np.array([5.5, 5.0])
        assert grid_2d.get_num_experiences(neighbour) > 0

    def test_repeated_tickets_accumulate(self, grid_2d):
        emb = np.array([5.0, 5.0])
        grid_2d.add_ticket_knowledge(emb)
        first = grid_2d.get_num_experiences(emb)
        grid_2d.add_ticket_knowledge(emb)
        second = grid_2d.get_num_experiences(emb)
        assert second > first

    def test_knowledge_less_than_experience(self, grid_2d):
        """With learning_rate < 1, knowledge exponent b > 0 and
        knowledge = exp^b should differ from raw experience."""
        emb = np.array([5.0, 5.0])
        for _ in range(5):
            grid_2d.add_ticket_knowledge(emb)
        exp = grid_2d.get_num_experiences(emb)
        know = grid_2d.get_knowledge(emb)
        # They should not be equal (unless b == 1)
        assert exp != pytest.approx(know)


# ------------------------------------------------------------------ #
#  Max queries
# ------------------------------------------------------------------ #

class TestMaxQueries:
    def test_max_on_empty_grid(self, grid_2d):
        assert grid_2d.get_max_experiences() == 0.0
        assert grid_2d.get_max_knowledge() == 0.0

    def test_max_after_tickets(self, grid_2d):
        grid_2d.add_ticket_knowledge(np.array([5.0, 5.0]))
        assert grid_2d.get_max_experiences() > 0
        assert grid_2d.get_max_knowledge() > 0


# ------------------------------------------------------------------ #
#  Transmission
# ------------------------------------------------------------------ #

class TestTransmission:
    def test_transmit_from_increases_knowledge(self, grid_2d):
        other = KnowledgeGrid(
            shape=(20, 20),
            propagation_sigma=1.0,
            learning_rate=0.1,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        emb = np.array([5.0, 5.0])
        other.add_ticket_knowledge(emb)
        other.add_ticket_knowledge(emb)

        before = grid_2d.get_num_experiences(emb)
        grid_2d.transmit_from(other)
        after = grid_2d.get_num_experiences(emb)
        assert after > before

    def test_transmit_respects_factor(self):
        a = KnowledgeGrid(shape=(10, 10), transmission_factor=0.5)
        b = KnowledgeGrid(shape=(10, 10), transmission_factor=0.5)
        emb = np.array([50.0, 50.0])
        b.add_ticket_knowledge(emb)
        donor_val = b.get_num_experiences(emb)

        a.transmit_from(b)
        received = a.get_num_experiences(emb)
        assert received == pytest.approx(donor_val * 0.5, rel=1e-6)

    def test_transmit_no_negative_transfer(self):
        """If receiver already knows more, no transfer should occur."""
        a = KnowledgeGrid(shape=(10, 10))
        b = KnowledgeGrid(shape=(10, 10))
        emb = np.array([50.0, 50.0])
        a.add_ticket_knowledge(emb)
        a.add_ticket_knowledge(emb)
        b.add_ticket_knowledge(emb)

        before = a._grid.copy()
        a.transmit_from(b)
        np.testing.assert_array_equal(a._grid, before)

    def test_transmit_shape_mismatch_raises(self):
        a = KnowledgeGrid(shape=(10, 10))
        b = KnowledgeGrid(shape=(5, 5))
        with pytest.raises(ValueError, match="Grid shapes must match"):
            a.transmit_from(b)


# ------------------------------------------------------------------ #
#  Decay
# ------------------------------------------------------------------ #

class TestDecay:
    def test_decay_on_empty_grid(self, grid_2d):
        """Should not crash on an empty grid."""
        grid_2d.decay_knowledge()
        assert np.all(grid_2d._grid == 0)

    def test_decay_reduces_experience(self, grid_2d):
        emb = np.array([5.0, 5.0])
        for _ in range(10):
            grid_2d.add_ticket_knowledge(emb)
        before = grid_2d.get_max_experiences()
        grid_2d.decay_knowledge()
        after = grid_2d.get_max_experiences()
        assert after < before


# ------------------------------------------------------------------ #
#  Serialization
# ------------------------------------------------------------------ #

class TestSerialization:
    def test_save_and_load_roundtrip(self, grid_2d):
        grid_2d.add_ticket_knowledge(np.array([3.0, 7.0]))
        grid_2d.add_ticket_knowledge(np.array([5.0, 5.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'grid.npz'
            grid_2d.save(path)
            loaded = KnowledgeGrid.load(path)

        np.testing.assert_array_equal(loaded._grid, grid_2d._grid)
        assert loaded._shape == grid_2d._shape
        assert loaded._propagation_sigma == grid_2d._propagation_sigma
        assert loaded._transmission_factor == grid_2d._transmission_factor
        assert loaded._learning_rate == grid_2d._learning_rate
        assert loaded._total_num_experiences == grid_2d._total_num_experiences
        np.testing.assert_array_equal(loaded._embedding_bounds, grid_2d._embedding_bounds)


# ------------------------------------------------------------------ #
#  Rendering
# ------------------------------------------------------------------ #

class TestRender:
    def test_render_stats_for_3d(self, grid_3d):
        grid_3d.add_ticket_knowledge(np.array([0.5, 0.5, 0.5]))
        stats = grid_3d.render()
        assert isinstance(stats, dict)
        assert stats['dimensions'] == 3
        assert stats['non_zero_cells'] > 0

    def test_render_2d_returns_figure(self, grid_2d):
        pytest.importorskip('matplotlib')
        grid_2d.add_ticket_knowledge(np.array([5.0, 5.0]))
        fig = grid_2d.render()
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)


# ------------------------------------------------------------------ #
#  Geometric / manifold features
# ------------------------------------------------------------------ #

@pytest.fixture
def populated_2d():
    """A 2-D grid with a clear knowledge peak for geometric tests."""
    grid = KnowledgeGrid(
        shape=(30, 30),
        propagation_sigma=2.0,
        learning_rate=0.1,
        embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
    )
    # Concentrated cluster
    for _ in range(15):
        grid.add_ticket_knowledge(np.array([5.0, 5.0]))
    # A second, weaker cluster
    for _ in range(5):
        grid.add_ticket_knowledge(np.array([2.0, 8.0]))
    return grid


class TestGradient:
    def test_gradient_field_shape(self, populated_2d):
        grad = populated_2d.gradient_field()
        assert len(grad) == 2
        for g in grad:
            assert g.shape == populated_2d._shape

    def test_gradient_zero_on_empty_grid(self, grid_2d):
        grad = grid_2d.gradient_field()
        for g in grad:
            np.testing.assert_array_equal(g, 0.0)

    def test_gradient_at_returns_vector(self, populated_2d):
        vec = populated_2d.gradient_at(np.array([5.0, 5.0]))
        assert vec.shape == (2,)

    def test_gradient_near_zero_at_peak(self, populated_2d):
        """The gradient at the peak of a Gaussian bump should be near zero."""
        vec = populated_2d.gradient_at(np.array([5.0, 5.0]))
        mag = np.linalg.norm(vec)
        # Compare to the max gradient magnitude anywhere in the grid
        max_mag = np.max(populated_2d.gradient_magnitude())
        assert mag < 0.3 * max_mag  # peak is relatively flat

    def test_gradient_magnitude_shape(self, populated_2d):
        mag = populated_2d.gradient_magnitude()
        assert mag.shape == populated_2d._shape
        assert np.all(mag >= 0)


class TestCurvature:
    def test_curvature_shape(self, populated_2d):
        curv = populated_2d.curvature()
        assert curv.shape == populated_2d._shape

    def test_negative_curvature_at_peak(self, populated_2d):
        """A knowledge peak should have negative Laplacian (concave down)."""
        curv_val = populated_2d.curvature_at(np.array([5.0, 5.0]))
        assert curv_val < 0

    def test_curvature_at_returns_scalar(self, populated_2d):
        val = populated_2d.curvature_at(np.array([5.0, 5.0]))
        assert isinstance(val, float)


class TestKnowledgeVolume:
    def test_volume_zero_on_empty(self, grid_2d):
        assert grid_2d.knowledge_volume() == 0.0

    def test_volume_positive_after_tickets(self, populated_2d):
        assert populated_2d.knowledge_volume() > 0

    def test_volume_increases_with_tickets(self, grid_2d):
        grid_2d.add_ticket_knowledge(np.array([5.0, 5.0]))
        vol1 = grid_2d.knowledge_volume()
        grid_2d.add_ticket_knowledge(np.array([8.0, 2.0]))
        vol2 = grid_2d.knowledge_volume()
        assert vol2 > vol1


class TestEntropy:
    def test_entropy_zero_on_empty(self, grid_2d):
        assert grid_2d.knowledge_entropy() == 0.0

    def test_entropy_positive_after_tickets(self, populated_2d):
        assert populated_2d.knowledge_entropy() > 0

    def test_specialist_has_higher_specialisation_index(self):
        """A grid with one concentrated cluster should be more specialised
        than one with knowledge spread across two distant locations."""
        specialist = KnowledgeGrid(
            shape=(30, 30), propagation_sigma=1.0,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        generalist = KnowledgeGrid(
            shape=(30, 30), propagation_sigma=1.0,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        for _ in range(20):
            specialist.add_ticket_knowledge(np.array([5.0, 5.0]))

        for _ in range(10):
            generalist.add_ticket_knowledge(np.array([2.0, 2.0]))
            generalist.add_ticket_knowledge(np.array([8.0, 8.0]))

        assert specialist.specialisation_index() > generalist.specialisation_index()

    def test_specialisation_between_zero_and_one(self, populated_2d):
        si = populated_2d.specialisation_index()
        assert 0.0 <= si <= 1.0


class TestFrontier:
    def test_frontier_shape(self, populated_2d):
        mask = populated_2d.frontier()
        assert mask.shape == populated_2d._shape
        assert mask.dtype == bool

    def test_frontier_empty_on_empty_grid(self, grid_2d):
        mask = grid_2d.frontier()
        assert not np.any(mask)

    def test_frontier_exists_around_peak(self, populated_2d):
        mask = populated_2d.frontier()
        assert np.any(mask)
        # The peak itself should NOT be on the frontier (gradient ~ 0 there)
        peak_coords = populated_2d.embedding_to_coords(np.array([5.0, 5.0]))
        assert not mask[peak_coords]


class TestExpertiseClusters:
    def test_two_clusters_detected(self):
        """Two well-separated peaks should yield two clusters."""
        grid = KnowledgeGrid(
            shape=(50, 50), propagation_sigma=1.0,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        for _ in range(10):
            grid.add_ticket_knowledge(np.array([1.0, 1.0]))
        for _ in range(10):
            grid.add_ticket_knowledge(np.array([9.0, 9.0]))
        _, n = grid.expertise_clusters()
        assert n >= 2

    def test_single_cluster_for_single_peak(self):
        grid = KnowledgeGrid(
            shape=(30, 30), propagation_sigma=1.0,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        for _ in range(10):
            grid.add_ticket_knowledge(np.array([5.0, 5.0]))
        _, n = grid.expertise_clusters()
        assert n == 1

    def test_zero_clusters_on_empty(self, grid_2d):
        _, n = grid_2d.expertise_clusters()
        assert n == 0

    def test_labels_shape(self, populated_2d):
        labels, _ = populated_2d.expertise_clusters()
        assert labels.shape == populated_2d._shape


class TestGeodesicDistance:
    def test_distance_to_self_is_zero(self, populated_2d):
        emb = np.array([5.0, 5.0])
        d = populated_2d.geodesic_distance(emb, emb)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self, populated_2d):
        a = np.array([5.0, 5.0])
        b = np.array([2.0, 8.0])
        assert populated_2d.geodesic_distance(a, b) == pytest.approx(
            populated_2d.geodesic_distance(b, a), rel=1e-6
        )

    def test_through_knowledge_cheaper(self):
        """Path through a high-knowledge corridor should be shorter than
        going through unknown territory."""
        grid = KnowledgeGrid(
            shape=(20, 20), propagation_sigma=1.5,
            embedding_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
        # Build a corridor from (1,5) to (9,5)
        for x in np.linspace(1, 9, 20):
            grid.add_ticket_knowledge(np.array([x, 5.0]))
            grid.add_ticket_knowledge(np.array([x, 5.0]))

        near_corridor = grid.geodesic_distance(
            np.array([1.0, 5.0]), np.array([9.0, 5.0])
        )
        off_corridor = grid.geodesic_distance(
            np.array([1.0, 1.0]), np.array([9.0, 1.0])
        )
        assert near_corridor < off_corridor
