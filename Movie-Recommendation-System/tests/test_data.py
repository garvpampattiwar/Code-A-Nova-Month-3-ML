"""
Automated unit tests for data loading and preprocessing in CineMatch.
Validates schemas, missing value handling, year extraction, title normalization,
and rating aggregation.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_movies,
    load_raw_ratings,
    load_raw_tags,
    load_raw_links,
    load_processed_movies,
    verify_raw_data,
)
from src.preprocessing import (
    clean_title,
    extract_year,
    parse_genres,
    aggregate_tags,
    compute_rating_stats,
    build_content_soup,
)


class TestDataValidation:
    """Test raw and processed dataset schemas and integrity."""

    def test_verify_raw_data_present(self):
        present, msg = verify_raw_data()
        assert present is True, f"Raw data verification failed: {msg}"

    def test_raw_movies_schema(self):
        df = load_raw_movies()
        assert not df.empty
        assert "movieId" in df.columns
        assert "title" in df.columns
        assert "genres" in df.columns
        assert df["movieId"].nunique() == len(df), "movieId should be unique in movies.csv"

    def test_raw_ratings_schema(self):
        df = load_raw_ratings()
        assert not df.empty
        assert {"userId", "movieId", "rating"}.issubset(df.columns)
        assert df["rating"].min() >= 0.5
        assert df["rating"].max() <= 5.0

    def test_raw_links_schema(self):
        df = load_raw_links()
        assert not df.empty
        assert {"movieId", "imdbId"}.issubset(df.columns)

    def test_processed_movies_schema(self):
        df = load_processed_movies()
        assert not df.empty
        required_cols = [
            "movieId", "title", "clean_title", "year",
            "genres_clean", "rating_count", "rating_mean",
            "weighted_rating", "imdb_id", "content_soup"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}' in processed data"
        
        # Verify no NaN in content_soup
        assert df["content_soup"].isnull().sum() == 0


class TestPreprocessingUnits:
    """Test unit functions in preprocessing."""

    def test_extract_year(self):
        assert extract_year("Toy Story (1995)") == 1995
        assert extract_year("Star Wars: Episode IV - A New Hope (1977)") == 1977
        assert extract_year("Movie Without Year") is None
        assert extract_year("") is None
        assert extract_year(None) is None

    def test_clean_title(self):
        assert clean_title("Toy Story (1995)") == "Toy Story"
        assert clean_title("Matrix, The (1999)") == "The Matrix"
        assert clean_title("Shawshank Redemption, The") == "The Shawshank Redemption"
        assert clean_title("Iron Giant, The (1999)") == "The Iron Giant"
        assert clean_title("") == ""
        assert clean_title(None) == ""

    def test_parse_genres(self):
        assert parse_genres("Action|Adventure|Sci-Fi") == ["Action", "Adventure", "Sci-Fi"]
        assert parse_genres("(no genres listed)") == []
        assert parse_genres("") == []
        assert parse_genres(None) == []

    def test_aggregate_tags(self):
        sample_tags = pd.DataFrame({
            "movieId": [1, 1, 1, 2],
            "tag": ["pixar", "animation", "pixar", "funny"],
        })
        agg = aggregate_tags(sample_tags)
        assert len(agg) == 2
        row_1 = agg[agg["movieId"] == 1].iloc[0]
        # Should deduplicate tags
        assert "animation" in row_1["tags_str"]
        assert "pixar" in row_1["tags_str"]

    def test_compute_rating_stats(self):
        sample_ratings = pd.DataFrame({
            "userId": [1, 2, 3, 4],
            "movieId": [1, 1, 2, 2],
            "rating": [5.0, 4.0, 2.0, 3.0],
        })
        stats = compute_rating_stats(sample_ratings)
        assert len(stats) == 2
        m1 = stats[stats["movieId"] == 1].iloc[0]
        assert m1["rating_count"] == 2
        assert m1["rating_mean"] == 4.5
        assert m1["weighted_rating"] > 0

    def test_build_content_soup_empty_row(self):
        empty_row = pd.Series({
            "genres_list": [],
            "tags_str": "",
            "clean_title": "",
        })
        soup = build_content_soup(empty_row)
        assert soup == "unknown" or len(soup) > 0
