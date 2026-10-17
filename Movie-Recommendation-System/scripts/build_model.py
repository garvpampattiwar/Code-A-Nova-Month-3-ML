"""
Model building and evaluation script for CineMatch Movie Recommendation System.
Coordinates:
1. Dataset verification / download
2. Preprocessing pipeline execution
3. Recommender fitting and artifact persistence
4. Offline evaluation metrics calculation and report generation
"""

import os
import sys
import json
import time

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from scripts.download_data import download_movielens_small
from src.preprocessing import run_preprocessing
from src.recommender import MovieRecommender, evaluate_recommender
from src.data_loader import get_data_paths


def main():
    print("=" * 65)
    print("      CINEMATCH — MODEL BUILDING & OFFLINE EVALUATION       ")
    print("=" * 65)

    start_time = time.time()

    # Step 1: Ensure raw dataset exists
    print("\n[STEP 1/4] Checking MovieLens dataset...")
    success = download_movielens_small()
    if not success:
        print("[ERROR] Could not obtain dataset. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Run preprocessing pipeline
    print("\n[STEP 2/4] Running data cleaning & feature engineering...")
    processed_df = run_preprocessing(save_output=True)

    # Step 3: Fit recommender model & persist artifacts
    print("\n[STEP 3/4] Fitting TF-IDF Sparse Recommender...")
    recommender = MovieRecommender()
    recommender.fit(processed_df)

    paths = get_data_paths()
    models_dir = paths["models_dir"]
    recommender.save(models_dir)

    # Step 4: Run Offline Evaluation
    print("\n[STEP 4/4] Conducting Offline Evaluation...")
    eval_metrics = evaluate_recommender(recommender, sample_size=200, top_k=10)
    
    print("\n" + "-" * 50)
    print("           OFFLINE EVALUATION RESULTS            ")
    print("-" * 50)
    print(f"  Test Sample Queries       : {eval_metrics['sample_size']}")
    print(f"  Recommendations per Query : Top {eval_metrics['top_k']}")
    print(f"  Genre Precision@10        : {eval_metrics['genre_precision_percentage']}%")
    print(f"  Intra-List Diversity (ILD): {eval_metrics['intra_list_diversity']} (scale 0-1)")
    print(f"  Catalog Coverage          : {eval_metrics['catalog_coverage_percentage']}% ({eval_metrics['unique_movies_recommended']}/{eval_metrics['total_catalog_size']} movies)")
    print("-" * 50)

    # Save evaluation report to reports/
    reports_dir = os.path.join(WORKSPACE_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "evaluation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"[SUCCESS] Offline evaluation report saved to: {report_path}")

    # Test sample recommendation query
    print("\n[VERIFICATION] Testing sample recommendation for 'Toy Story':")
    sample_res = recommender.get_recommendations("Toy Story", top_n=5)
    print(f"Target: {sample_res['target_movie']['title']} ({sample_res['target_movie']['genres']})")
    for i, r in enumerate(sample_res["recommendations"], 1):
        print(f"  {i}. {r['title']} [{r['genres']}] — Match: {r['match_percentage']}% | {r['explanation']}")

    total_time = round(time.time() - start_time, 2)
    print(f"\n[COMPLETE] Model build and verification finished in {total_time}s.")


if __name__ == "__main__":
    main()
