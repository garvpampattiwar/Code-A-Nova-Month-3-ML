"""
Preprocessing pipeline for CineMatch Movie Recommendation System.
Performs data cleaning, feature engineering, tag aggregation, rating aggregation,
and content soup construction for TF-IDF vectorization.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.data_loader import (
    load_raw_movies,
    load_raw_ratings,
    load_raw_tags,
    load_raw_links,
    get_data_paths,
)


def extract_year(title: str) -> Optional[int]:
    """
    Extract 4-digit release year from movie title.
    Example: 'Toy Story (1995)' -> 1995
    """
    if not isinstance(title, str):
        return None
    match = re.search(r'\((\d{4})\)\s*$', title.strip())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def clean_title(title: str) -> str:
    """
    Clean movie title:
    - Removes trailing year in parentheses: 'Toy Story (1995)' -> 'Toy Story'
    - Normalizes trailing articles: 'Matrix, The' -> 'The Matrix'
    - Cleans extra whitespace
    """
    if not isinstance(title, str):
        return ""
    
    # Strip trailing year like (1995) or (1995-1996)
    cleaned = re.sub(r'\s*\(\d{4}(?:[–-]\d{4})?\)\s*$', '', title).strip()
    
    # Handle 'Title, The' or 'Title, A' or 'Title, An'
    articles = [", The", ", A", ", An"]
    for art in articles:
        if cleaned.endswith(art):
            prefix = art.replace(", ", "").strip()
            core = cleaned[:-len(art)].strip()
            cleaned = f"{prefix} {core}"
            break

    return cleaned


def parse_genres(genre_str: str) -> List[str]:
    """
    Parse pipe-separated genres string into clean list of genres.
    Replaces '(no genres listed)' with empty list.
    """
    if not isinstance(genre_str, str) or genre_str.strip() == "(no genres listed)":
        return []
    return [g.strip() for g in genre_str.split("|") if g.strip()]


def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group tags by movieId and aggregate into deduplicated space-separated string.
    """
    if tags_df.empty or "movieId" not in tags_df.columns or "tag" not in tags_df.columns:
        return pd.DataFrame(columns=["movieId", "tags_str"])

    clean_tags = tags_df.copy()
    clean_tags["tag"] = clean_tags["tag"].astype(str).str.strip().str.lower()
    
    # Remove empty or non-alphanumeric tags
    clean_tags = clean_tags[clean_tags["tag"].str.len() > 1]
    
    # Group and deduplicate tags per movie
    agg = clean_tags.groupby("movieId")["tag"].apply(
        lambda s: " ".join(sorted(set(s)))
    ).reset_index()
    agg.rename(columns={"tag": "tags_str"}, inplace=True)
    return agg


def compute_rating_stats(ratings_df: pd.DataFrame, min_votes_quantile: float = 0.50) -> pd.DataFrame:
    """
    Compute rating statistics per movie:
    - rating_count (vote count)
    - rating_mean (average rating)
    - weighted_rating (IMDb Bayesian weighted formula)
    """
    if ratings_df.empty or "movieId" not in ratings_df.columns:
        return pd.DataFrame(columns=["movieId", "rating_count", "rating_mean", "weighted_rating"])

    stats = ratings_df.groupby("movieId")["rating"].agg(
        rating_count="count",
        rating_mean="mean"
    ).reset_index()

    # Bayesian weighted rating formula:
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    C = ratings_df["rating"].mean()
    m = stats["rating_count"].quantile(min_votes_quantile)
    if m < 1:
        m = 1

    stats["weighted_rating"] = stats.apply(
        lambda row: (row["rating_count"] / (row["rating_count"] + m)) * row["rating_mean"] +
                    (m / (row["rating_count"] + m)) * C,
        axis=1
    )
    stats["rating_mean"] = stats["rating_mean"].round(2)
    stats["weighted_rating"] = stats["weighted_rating"].round(2)
    return stats


def format_imdb_links(links_df: pd.DataFrame) -> pd.DataFrame:
    """Format IMDb 7-digit ID and full URL."""
    if links_df.empty or "movieId" not in links_df.columns:
        return pd.DataFrame(columns=["movieId", "imdb_id", "imdb_url"])
    
    links = links_df.copy()
    links["imdbId"] = links["imdbId"].fillna(0).astype(int)
    links["imdb_id"] = links["imdbId"].apply(lambda x: f"tt{x:07d}" if x > 0 else "")
    links["imdb_url"] = links["imdb_id"].apply(
        lambda x: f"https://www.imdb.com/title/{x}/" if x else ""
    )
    return links[["movieId", "imdb_id", "imdb_url"]]


def build_content_soup(row: pd.Series) -> str:
    """
    Construct text soup combining genres (weighted by repetition), tags, and title keywords.
    Genres are repeated 2x to ensure strong categorical affinity in TF-IDF.
    """
    parts = []
    
    # Genres repeated 2 times for feature weighting
    genres = row.get("genres_list", [])
    if isinstance(genres, list) and genres:
        # e.g. "Action Adventure Sci-Fi"
        g_str = " ".join([re.sub(r'[^a-zA-Z0-9]', '', g) for g in genres])
        parts.append(g_str)
        parts.append(g_str)
    
    # User tags
    tags = row.get("tags_str", "")
    if isinstance(tags, str) and tags.strip():
        parts.append(tags.strip())

    # Clean title keywords
    clean_t = row.get("clean_title", "")
    if isinstance(clean_t, str) and clean_t.strip():
        # Strip common stop characters
        t_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_t).lower()
        parts.append(t_tokens.strip())

    soup = " ".join(parts).strip()
    return soup if soup else "unknown"


def run_preprocessing(save_output: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Loads raw movies, ratings, tags, links.
    2. Cleans titles and extracts years.
    3. Normalizes genres.
    4. Aggregates tags and ratings.
    5. Builds content soup for ML recommendation.
    6. Saves to data/processed/movies_processed.csv.
    """
    print("[INFO] Starting MovieLens preprocessing pipeline...")
    movies_df = load_raw_movies()
    ratings_df = load_raw_ratings()
    tags_df = load_raw_tags()
    links_df = load_raw_links()

    print(f"  Raw Movies: {len(movies_df):,} records")
    print(f"  Raw Ratings: {len(ratings_df):,} records")
    print(f"  Raw Tags: {len(tags_df):,} records")

    # 1. Title and Year Processing
    movies_df["year"] = movies_df["title"].apply(extract_year)
    movies_df["clean_title"] = movies_df["title"].apply(clean_title)

    # 2. Genres
    movies_df["genres_list"] = movies_df["genres"].apply(parse_genres)
    movies_df["genres_clean"] = movies_df["genres_list"].apply(
        lambda gl: ", ".join(gl) if gl else "Uncategorized"
    )

    # 3. Aggregated Tags
    tags_agg = aggregate_tags(tags_df)
    merged = movies_df.merge(tags_agg, on="movieId", how="left")
    merged["tags_str"] = merged["tags_str"].fillna("")

    # 4. Rating Statistics
    rating_stats = compute_rating_stats(ratings_df)
    merged = merged.merge(rating_stats, on="movieId", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["rating_mean"] = merged["rating_mean"].fillna(0.0)
    merged["weighted_rating"] = merged["weighted_rating"].fillna(0.0)

    # 5. IMDb Links
    links_clean = format_imdb_links(links_df)
    merged = merged.merge(links_clean, on="movieId", how="left")
    merged["imdb_id"] = merged["imdb_id"].fillna("")
    merged["imdb_url"] = merged["imdb_url"].fillna("")

    # 6. Build Content Soup
    merged["content_soup"] = merged.apply(build_content_soup, axis=1)

    # Drop duplicate movieIds if any
    merged = merged.drop_duplicates(subset=["movieId"]).reset_index(drop=True)

    if save_output:
        paths = get_data_paths()
        os.makedirs(os.path.dirname(paths["processed_movies"]), exist_ok=True)
        # Store genres_list as pipe-separated string for CSV portability
        csv_export = merged.copy()
        csv_export["genres_list"] = csv_export["genres_list"].apply(lambda gl: "|".join(gl))
        csv_export.to_csv(paths["processed_movies"], index=False)
        print(f"[SUCCESS] Saved preprocessed dataset to: {paths['processed_movies']}")
        print(f"  Total processed movies: {len(merged):,}")

    return merged


if __name__ == "__main__":
    run_preprocessing(save_output=True)
