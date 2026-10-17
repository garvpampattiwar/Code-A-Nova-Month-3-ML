"""
Automated unit and integration tests for MovieRecommender in CineMatch.
Tests core recommendation functionality, relevance, edge cases, error handling,
filtering, self-exclusion, and deduplication.
"""

import pytest
import numpy as np
from src.recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender():
    """Shared fitted recommender instance for efficient test execution."""
    rec = MovieRecommender()
    rec.load()
    return rec


class TestMovieRecommenderCore:
    """Test standard recommendation behaviors."""

    def test_recommender_is_fitted(self, recommender):
        assert recommender.df is not None
        assert recommender.tfidf_matrix is not None
        assert recommender.tfidf_matrix.shape[0] > 9000

    def test_recommendation_output_schema(self, recommender):
        res = recommender.get_recommendations("Toy Story (1995)", top_n=5)
        assert res["status"] == "success"
        assert res["target_movie"] is not None
        assert isinstance(res["recommendations"], list)
        assert len(res["recommendations"]) == 5

        # Check fields of recommendation items
        first_rec = res["recommendations"][0]
        expected_keys = [
            "movieId", "title", "clean_title", "year", "genres",
            "rating_count", "rating_mean", "similarity_score",
            "match_percentage", "explanation"
        ]
        for key in expected_keys:
            assert key in first_rec, f"Missing key '{key}' in recommendation"

    def test_self_exclusion(self, recommender):
        """Selected movie must NEVER appear in its own recommendations."""
        target_title = "Toy Story (1995)"
        res = recommender.get_recommendations(target_title, top_n=10)
        target_id = res["target_movie"]["movieId"]
        rec_ids = [r["movieId"] for r in res["recommendations"]]
        assert target_id not in rec_ids, "Target movie was erroneously included in its recommendations"

    def test_no_duplicate_recommendations(self, recommender):
        """Ensure all returned recommended movies are strictly unique."""
        res = recommender.get_recommendations("Inception (2010)", top_n=15)
        rec_ids = [r["movieId"] for r in res["recommendations"]]
        assert len(rec_ids) == len(set(rec_ids)), "Duplicate movie IDs found in recommendations"

    def test_custom_top_n_count(self, recommender):
        """Ensure top_n parameter is strictly respected."""
        for n in [1, 3, 7, 12]:
            res = recommender.get_recommendations("Forrest Gump (1994)", top_n=n)
            assert len(res["recommendations"]) == n


class TestRecommenderEdgeCases:
    """Test unusual, missing, or invalid queries."""

    def test_unknown_movie_title(self, recommender):
        """Unknown titles should return graceful not_found status with fallback recommendations."""
        res = recommender.get_recommendations("Nonexistent Movie XYZ 999", top_n=5)
        assert res["status"] == "not_found"
        assert res["target_movie"] is None
        # Should provide fallback popular recommendations rather than crashing
        assert len(res["recommendations"]) == 5

    def test_empty_string_input(self, recommender):
        """Empty input should not raise unhandled exceptions."""
        res = recommender.get_recommendations("", top_n=5)
        assert res["status"] == "not_found"
        assert len(res["recommendations"]) == 5

    def test_whitespace_input(self, recommender):
        res = recommender.get_recommendations("    ", top_n=5)
        assert res["status"] == "not_found"

    def test_case_insensitivity(self, recommender):
        """Search should be case-insensitive."""
        res_lower = recommender.get_recommendations("toy story", top_n=5)
        res_upper = recommender.get_recommendations("TOY STORY", top_n=5)
        assert res_lower["status"] == "success"
        assert res_upper["status"] == "success"
        assert res_lower["target_movie"]["movieId"] == res_upper["target_movie"]["movieId"]

    def test_clean_title_without_year(self, recommender):
        """Searching without release year should match the movie."""
        res = recommender.get_recommendations("The Matrix", top_n=5)
        assert res["status"] == "success"
        assert "Matrix" in res["target_movie"]["title"]

    def test_query_by_movie_id(self, recommender):
        """Numeric ID lookup should work directly."""
        res = recommender.get_recommendations(1, top_n=5)  # Toy Story is movieId 1
        assert res["status"] == "success"
        assert res["target_movie"]["movieId"] == 1


class TestFilteringAndHybrid:
    """Test genre filters, year filters, and hybrid options."""

    def test_genre_filter(self, recommender):
        """Recommendations with genre filter must only contain that genre."""
        res = recommender.get_recommendations("Toy Story (1995)", top_n=8, genre_filter="Animation")
        assert res["status"] == "success"
        for rec in res["recommendations"]:
            assert "Animation" in rec["genres_list"]

    def test_year_range_filter(self, recommender):
        """Filtered recommendations must satisfy year range."""
        res = recommender.get_recommendations(
            "Star Wars: Episode IV - A New Hope (1977)",
            top_n=5,
            year_min=1980,
            year_max=2000,
        )
        assert res["status"] == "success"
        for rec in res["recommendations"]:
            if rec["year"] is not None:
                assert 1980 <= rec["year"] <= 2000

    def test_pure_content_vs_hybrid_weight(self, recommender):
        """Changing hybrid weight should run without error."""
        res_content = recommender.get_recommendations("Toy Story (1995)", top_n=5, hybrid_weight=1.0)
        res_hybrid = recommender.get_recommendations("Toy Story (1995)", top_n=5, hybrid_weight=0.5)
        assert res_content["status"] == "success"
        assert res_hybrid["status"] == "success"


class TestRelevanceAndExplaining:
    """Validate semantic and franchise relevance on classic test cases."""

    def test_toy_story_relevance(self, recommender):
        """Toy Story should recommend animated or franchise counterparts."""
        res = recommender.get_recommendations("Toy Story", top_n=5)
        titles = [r["title"] for r in res["recommendations"]]
        # Toy Story 3 or Antz or Monsters Inc or Bug's Life should appear
        franchise_or_pixar = any(
            any(w in t for w in ["Toy Story", "Antz", "Monsters", "Bug's Life", "Shrek"])
            for t in titles
        )
        assert franchise_or_pixar, f"Expected animated/pixar recommendations, got: {titles}"

    def test_godfather_relevance(self, recommender):
        """The Godfather should recommend Godfather Part II or crime dramas."""
        res = recommender.get_recommendations("Godfather, The", top_n=5)
        titles = [r["title"] for r in res["recommendations"]]
        godfather_2 = any("Godfather: Part II" in t for t in titles)
        assert godfather_2, f"Godfather should recommend Godfather Part II, got: {titles}"

    def test_explanation_field(self, recommender):
        """Every recommendation should include an explanatory string."""
        res = recommender.get_recommendations("Inception (2010)", top_n=5)
        for rec in res["recommendations"]:
            assert "explanation" in rec
            assert len(rec["explanation"]) > 5

    def test_search_autocomplete(self, recommender):
        """Live search should return relevant suggestions."""
        results = recommender.search_movies("Matrix", limit=5)
        assert len(results) > 0
        assert any("Matrix" in r["title"] for r in results)
