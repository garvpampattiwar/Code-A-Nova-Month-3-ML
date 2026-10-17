"""
Core recommendation engine for CineMatch.
Provides content-based and hybrid movie recommendations using TF-IDF,
cosine similarity, on-demand sparse vector projection, explainability,
and flexible filtering.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data_loader import load_processed_movies, get_data_paths


class MovieRecommender:
    """
    Production-ready Movie Recommender implementing Content-Based Filtering
    and Hybrid Quality Re-ranking on MovieLens dataset.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.title_to_idx: Dict[str, int] = {}
        self.clean_title_to_idx: Dict[str, int] = {}
        self.id_to_idx: Dict[int, int] = {}
        self.all_genres: List[str] = []

    def fit(self, df: Optional[pd.DataFrame] = None) -> "MovieRecommender":
        """
        Fit the recommendation model:
        1. Prepares internal indices for exact, clean, and ID lookups.
        2. Fits TF-IDF vectorizer on content soup (genres + tags + title tokens).
        3. Computes and holds sparse TF-IDF matrix.
        """
        if df is not None:
            self.df = df.copy().reset_index(drop=True)
        else:
            self.df = load_processed_movies(self.data_path).reset_index(drop=True)

        # Ensure correct column types
        self.df["movieId"] = self.df["movieId"].astype(int)
        self.df["title"] = self.df["title"].astype(str)
        self.df["clean_title"] = self.df["clean_title"].fillna("").astype(str)
        self.df["content_soup"] = self.df["content_soup"].fillna("").astype(str)

        # Parse genres_list if stored as string
        if "genres_list" in self.df.columns and isinstance(self.df["genres_list"].iloc[0], str):
            self.df["genres_list_parsed"] = self.df["genres_list"].apply(
                lambda s: [g.strip() for g in s.split("|") if g.strip()] if pd.notna(s) and s else []
            )
        elif "genres_list" in self.df.columns:
            self.df["genres_list_parsed"] = self.df["genres_list"]
        else:
            self.df["genres_list_parsed"] = self.df["genres"].apply(
                lambda s: [g.strip() for g in s.split("|") if g.strip()] if pd.notna(s) and s != "(no genres listed)" else []
            )

        # Build comprehensive genres set
        unique_genres = set()
        for glist in self.df["genres_list_parsed"]:
            unique_genres.update(glist)
        self.all_genres = sorted([g for g in unique_genres if g and g != "(no genres listed)"])

        # Create quick lookup mappings
        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(self.df["movieId"])}
        self.title_to_idx = {t.lower().strip(): idx for idx, t in enumerate(self.df["title"])}
        self.clean_title_to_idx = {
            ct.lower().strip(): idx for idx, ct in enumerate(self.df["clean_title"]) if ct
        }

        # Fit TF-IDF Vectorizer (sparse representation)
        print("[INFO] Vectorizing content soup using TF-IDF (1-2 grams)...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=25000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["content_soup"])
        print(f"[INFO] TF-IDF Matrix created: {self.tfidf_matrix.shape[0]} items x {self.tfidf_matrix.shape[1]} features.")
        return self

    def find_movie_index(self, query: Union[str, int]) -> Optional[int]:
        """
        Resolve movie index by:
        1. Direct movieId integer match
        2. Exact full title match (case-insensitive)
        3. Clean title match (e.g. without year)
        4. Substring / partial match ranking by popularity
        """
        if self.df is None or self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # 1. Match by movieId
        if isinstance(query, (int, np.integer)):
            return self.id_to_idx.get(int(query))

        if not isinstance(query, str) or not query.strip():
            return None

        q_clean = query.strip().lower()

        # Try parsing as integer ID in string
        if q_clean.isdigit():
            mid = int(q_clean)
            if mid in self.id_to_idx:
                return self.id_to_idx[mid]

        # 2. Match exact full title
        if q_clean in self.title_to_idx:
            return self.title_to_idx[q_clean]

        # 3. Match clean title
        if q_clean in self.clean_title_to_idx:
            return self.clean_title_to_idx[q_clean]

        # 4. Partial substring search - select candidate with highest rating_count
        matches = self.df[
            self.df["title"].str.lower().str.contains(re.escape(q_clean), regex=True, na=False) |
            self.df["clean_title"].str.lower().str.contains(re.escape(q_clean), regex=True, na=False)
        ]
        if not matches.empty:
            # Sort by rating_count descending to get the most prominent version
            best_match = matches.sort_values(by="rating_count", ascending=False).iloc[0]
            return self.id_to_idx[best_match["movieId"]]

        return None

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Live search method for UI autocomplete. Returns list of candidate movie dicts.
        """
        if self.df is None:
            raise RuntimeError("Model is not fitted.")
        if not query or not query.strip():
            # Return top popular movies when query is empty
            return self.get_top_popular(top_n=limit)

        q = query.strip().lower()
        # Case-insensitive substring match
        matched = self.df[
            self.df["title"].str.lower().str.contains(re.escape(q), regex=True, na=False) |
            self.df["clean_title"].str.lower().str.contains(re.escape(q), regex=True, na=False)
        ]

        if matched.empty:
            return []

        # Rank matches: exact clean title first, then starts with, then by rating count
        def rank_score(row):
            t_low = row["title"].lower()
            ct_low = row["clean_title"].lower()
            if ct_low == q or t_low == q:
                return 100000 + row["rating_count"]
            if ct_low.startswith(q) or t_low.startswith(q):
                return 50000 + row["rating_count"]
            return row["rating_count"]

        ranked_indices = matched.apply(rank_score, axis=1).sort_values(ascending=False).index[:limit]
        return self._format_movie_records(self.df.loc[ranked_indices])

    def get_recommendations(
        self,
        movie_title_or_id: Union[str, int],
        top_n: int = 10,
        genre_filter: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_rating: Optional[float] = None,
        hybrid_weight: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Generate top N movie recommendations.

        Parameters:
        - movie_title_or_id: Title or movieId of the target movie.
        - top_n: Number of recommendations to return (default: 10).
        - genre_filter: Optional genre to filter recommendations (e.g. 'Action').
        - year_min: Minimum release year filter.
        - year_max: Maximum release year filter.
        - min_rating: Minimum average rating filter.
        - hybrid_weight: Content similarity weight in [0, 1].
                         1.0 = Pure content similarity.
                         0.7-0.9 = Hybrid with quality/popularity re-ranking.

        Returns:
        Dict containing:
        - target_movie: target movie metadata dictionary
        - recommendations: list of recommended movie dictionaries with similarity & explanations
        - status: 'success' or error message
        """
        if self.df is None or self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        idx = self.find_movie_index(movie_title_or_id)
        if idx is None:
            return {
                "status": "not_found",
                "message": f"Movie '{movie_title_or_id}' could not be found in the CineMatch database.",
                "target_movie": None,
                "recommendations": self.get_top_popular(top_n=top_n),
            }

        target_row = self.df.iloc[idx]
        target_genres = set(target_row["genres_list_parsed"])
        target_tags = set(
            [t.strip() for t in str(target_row.get("tags_str", "")).split() if len(t.strip()) > 2]
        )

        # On-demand sparse cosine similarity computation (linear_kernel)
        target_vec = self.tfidf_matrix[idx : idx + 1]
        cosine_sim = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Quality normalization for hybrid re-ranking
        # Normalized weighted rating between 0 and 1
        max_wr = self.df["weighted_rating"].max()
        min_wr = self.df["weighted_rating"].min()
        wr_range = (max_wr - min_wr) if (max_wr - min_wr) > 0 else 1.0
        normalized_quality = (self.df["weighted_rating"] - min_wr) / wr_range

        # Clip hybrid weight between 0.0 and 1.0
        alpha = max(0.0, min(1.0, float(hybrid_weight)))
        final_scores = (alpha * cosine_sim) + ((1.0 - alpha) * normalized_quality.values)

        # Candidates excluding target movie itself
        candidate_indices = np.argsort(final_scores)[::-1]

        recs = []
        seen_ids = {int(target_row["movieId"])}

        for cand_idx in candidate_indices:
            if cand_idx == idx:
                continue

            row = self.df.iloc[cand_idx]
            mid = int(row["movieId"])
            if mid in seen_ids:
                continue

            # Filtering
            if genre_filter and genre_filter != "All":
                if genre_filter not in row["genres_list_parsed"]:
                    continue

            if year_min is not None and pd.notna(row["year"]) and row["year"] < year_min:
                continue

            if year_max is not None and pd.notna(row["year"]) and row["year"] > year_max:
                continue

            if min_rating is not None and row["rating_mean"] < min_rating:
                continue

            # Calculate explainability metrics
            cand_genres = set(row["genres_list_parsed"])
            cand_tags = set(
                [t.strip() for t in str(row.get("tags_str", "")).split() if len(t.strip()) > 2]
            )
            shared_genres = sorted(list(target_genres.intersection(cand_genres)))
            shared_tags = sorted(list(target_tags.intersection(cand_tags)))

            sim_score = float(cosine_sim[cand_idx])
            hybrid_score = float(final_scores[cand_idx])

            rec_item = self._format_single_movie(row)
            rec_item["similarity_score"] = round(sim_score, 4)
            rec_item["match_percentage"] = int(round(sim_score * 100))
            rec_item["hybrid_score"] = round(hybrid_score, 4)
            rec_item["shared_genres"] = shared_genres
            rec_item["shared_tags"] = shared_tags
            rec_item["explanation"] = self._generate_explanation(
                target_row["clean_title"],
                row["clean_title"],
                shared_genres,
                shared_tags,
                rec_item["match_percentage"],
            )

            recs.append(rec_item)
            seen_ids.add(mid)

            if len(recs) >= top_n:
                break

        target_info = self._format_single_movie(target_row)
        return {
            "status": "success",
            "message": f"Found {len(recs)} recommendations for '{target_row['clean_title']}'.",
            "target_movie": target_info,
            "recommendations": recs,
        }

    def _generate_explanation(
        self,
        target_title: str,
        rec_title: str,
        shared_genres: List[str],
        shared_tags: List[str],
        match_pct: int,
    ) -> str:
        """Create human-readable explanation for why the movie is recommended."""
        reasons = []
        if shared_genres:
            genres_str = ", ".join(shared_genres)
            reasons.append(f"Shares genre: {genres_str}")
        if shared_tags:
            tags_str = ", ".join(shared_tags[:3])
            reasons.append(f"Matching themes: {tags_str}")
        if not reasons:
            reasons.append("High thematic & metadata correlation")
        
        return f"{match_pct}% Match — " + " • ".join(reasons)

    def get_top_popular(self, top_n: int = 10, genre_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return top movies ranked by rating count (popularity)."""
        if self.df is None:
            raise RuntimeError("Model is not fitted.")
        subset = self.df
        if genre_filter and genre_filter != "All":
            subset = subset[subset["genres_list_parsed"].apply(lambda glist: genre_filter in glist)]
        
        sorted_df = subset.sort_values(by="rating_count", ascending=False).head(top_n)
        return self._format_movie_records(sorted_df)

    def get_top_rated(
        self, top_n: int = 10, genre_filter: Optional[str] = None, min_votes: int = 25
    ) -> List[Dict[str, Any]]:
        """Return top movies ranked by Bayesian weighted rating with minimum vote threshold."""
        if self.df is None:
            raise RuntimeError("Model is not fitted.")
        subset = self.df[self.df["rating_count"] >= min_votes]
        if genre_filter and genre_filter != "All":
            subset = subset[subset["genres_list_parsed"].apply(lambda glist: genre_filter in glist)]
        
        sorted_df = subset.sort_values(by="weighted_rating", ascending=False).head(top_n)
        return self._format_movie_records(sorted_df)

    def get_movies_by_genre(
        self, genre: str, top_n: int = 12, sort_by: str = "weighted_rating"
    ) -> List[Dict[str, Any]]:
        """Browse movies by genre with configurable sorting."""
        if self.df is None:
            raise RuntimeError("Model is not fitted.")
        subset = self.df[self.df["genres_list_parsed"].apply(lambda glist: genre in glist)]
        if sort_by in subset.columns:
            subset = subset.sort_values(by=sort_by, ascending=False)
        return self._format_movie_records(subset.head(top_n))

    def _format_single_movie(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a dataframe row to a structured dictionary."""
        return {
            "movieId": int(row["movieId"]),
            "title": str(row["title"]),
            "clean_title": str(row["clean_title"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "genres": str(row.get("genres_clean", row.get("genres", ""))),
            "genres_list": row.get("genres_list_parsed", []),
            "rating_count": int(row["rating_count"]),
            "rating_mean": float(row["rating_mean"]),
            "weighted_rating": float(row["weighted_rating"]),
            "imdb_id": str(row.get("imdb_id", "")),
            "imdb_url": str(row.get("imdb_url", "")),
            "tags": str(row.get("tags_str", "")),
        }

    def _format_movie_records(self, df_subset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format dataframe rows into list of movie dicts."""
        return [self._format_single_movie(row) for _, row in df_subset.iterrows()]

    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Persist model artifacts (vectorizer and sparse TF-IDF matrix).
        """
        paths = get_data_paths()
        target_dir = model_dir or paths["models_dir"]
        os.makedirs(target_dir, exist_ok=True)

        vec_path = os.path.join(target_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(target_dir, "tfidf_matrix.joblib")
        
        joblib.dump(self.tfidf_vectorizer, vec_path, compress=3)
        joblib.dump(self.tfidf_matrix, matrix_path, compress=3)
        print(f"[SUCCESS] Recommender artifacts persisted to: {target_dir}")
        return target_dir

    def load(self, model_dir: Optional[str] = None, data_path: Optional[str] = None) -> "MovieRecommender":
        """
        Load persisted model artifacts from disk for faster startup.
        """
        paths = get_data_paths()
        target_dir = model_dir or paths["models_dir"]
        vec_path = os.path.join(target_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(target_dir, "tfidf_matrix.joblib")

        if not os.path.exists(vec_path) or not os.path.exists(matrix_path):
            print(f"[INFO] Artifacts not found in {target_dir}. Fitting from processed data...")
            return self.fit()

        print(f"[INFO] Loading artifacts from {target_dir}...")
        self.df = load_processed_movies(data_path or self.data_path).reset_index(drop=True)
        self.df["movieId"] = self.df["movieId"].astype(int)
        self.df["title"] = self.df["title"].astype(str)
        self.df["clean_title"] = self.df["clean_title"].fillna("").astype(str)
        
        # Parse genres_list
        if "genres_list" in self.df.columns and isinstance(self.df["genres_list"].iloc[0], str):
            self.df["genres_list_parsed"] = self.df["genres_list"].apply(
                lambda s: [g.strip() for g in s.split("|") if g.strip()] if pd.notna(s) and s else []
            )
        elif "genres_list" in self.df.columns:
            self.df["genres_list_parsed"] = self.df["genres_list"]
        else:
            self.df["genres_list_parsed"] = self.df["genres"].apply(
                lambda s: [g.strip() for g in s.split("|") if g.strip()] if pd.notna(s) and s != "(no genres listed)" else []
            )

        unique_genres = set()
        for glist in self.df["genres_list_parsed"]:
            unique_genres.update(glist)
        self.all_genres = sorted([g for g in unique_genres if g and g != "(no genres listed)"])

        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(self.df["movieId"])}
        self.title_to_idx = {t.lower().strip(): idx for idx, t in enumerate(self.df["title"])}
        self.clean_title_to_idx = {
            ct.lower().strip(): idx for idx, ct in enumerate(self.df["clean_title"]) if ct
        }

        self.tfidf_vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = joblib.load(matrix_path)
        print("[SUCCESS] Loaded MovieRecommender artifacts successfully.")
        return self


def evaluate_recommender(
    recommender: MovieRecommender, sample_size: int = 150, top_k: int = 10
) -> Dict[str, Any]:
    """
    Offline evaluation of recommendation quality:
    - Genre Precision@K: Mean proportion of recommendations sharing at least one genre with query
    - Intra-List Diversity (ILD): Mean pairwise distance among recommended items in feature space
    - Catalog Coverage: Percentage of unique movies recommended across queries
    """
    if recommender.df is None:
        raise RuntimeError("Recommender is not fitted.")

    np.random.seed(42)
    sample_indices = np.random.choice(len(recommender.df), size=min(sample_size, len(recommender.df)), replace=False)
    
    genre_precisions = []
    recommended_movie_ids = set()
    intra_list_diversities = []

    for idx in sample_indices:
        query_row = recommender.df.iloc[idx]
        query_genres = set(query_row["genres_list_parsed"])
        if not query_genres:
            continue

        res = recommender.get_recommendations(int(query_row["movieId"]), top_n=top_k)
        recs = res["recommendations"]
        if not recs:
            continue

        # Genre Precision@K
        matches = [1 if set(r["genres_list"]).intersection(query_genres) else 0 for r in recs]
        genre_precisions.append(np.mean(matches))

        # Recommendations tracking
        for r in recs:
            recommended_movie_ids.add(r["movieId"])

        # Intra-list Diversity using cosine distance
        rec_indices = [recommender.id_to_idx[r["movieId"]] for r in recs if r["movieId"] in recommender.id_to_idx]
        if len(rec_indices) > 1:
            rec_submatrix = recommender.tfidf_matrix[rec_indices]
            sim_pairs = linear_kernel(rec_submatrix, rec_submatrix)
            # Distance = 1 - similarity
            dist_pairs = 1.0 - sim_pairs
            # Average upper triangle
            n_items = len(rec_indices)
            upper_tri_indices = np.triu_indices(n_items, k=1)
            ild = float(np.mean(dist_pairs[upper_tri_indices]))
            intra_list_diversities.append(ild)

    coverage_pct = round((len(recommended_movie_ids) / len(recommender.df)) * 100, 2)
    mean_genre_prec = round(float(np.mean(genre_precisions)), 4) if genre_precisions else 0.0
    mean_ild = round(float(np.mean(intra_list_diversities)), 4) if intra_list_diversities else 0.0

    return {
        "sample_size": len(genre_precisions),
        "top_k": top_k,
        "genre_precision_at_k": mean_genre_prec,
        "genre_precision_percentage": round(mean_genre_prec * 100, 2),
        "intra_list_diversity": mean_ild,
        "catalog_coverage_percentage": coverage_pct,
        "unique_movies_recommended": len(recommended_movie_ids),
        "total_catalog_size": len(recommender.df),
    }
