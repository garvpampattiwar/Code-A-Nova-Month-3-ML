"""
Dataset download script for CineMatch Movie Recommendation System.
Downloads genuine MovieLens Latest Small dataset from GroupLens:
https://grouplens.org/datasets/movielens/latest/
"""

import os
import sys
import zipfile
import io
import urllib.request
import ssl

try:
    import certifi
    DEFAULT_CA = certifi.where()
except ImportError:
    DEFAULT_CA = None

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
FALLBACK_URL = "https://grouplens.org/datasets/movielens/latest/"

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def get_ssl_context():
    """Create a verified SSL context with fallback."""
    if DEFAULT_CA:
        return ssl.create_default_context(cafile=DEFAULT_CA)
    try:
        return ssl.create_default_context()
    except Exception:
        # Fallback for unconfigured local certificate store
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def download_movielens_small(target_dir=RAW_DATA_DIR):
    """
    Downloads and extracts the MovieLens latest small dataset into target_dir.
    Files extracted: movies.csv, ratings.csv, tags.csv, links.csv, README.txt.
    """
    os.makedirs(target_dir, exist_ok=True)

    expected_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    already_present = all(os.path.exists(os.path.join(target_dir, f)) for f in expected_files)

    if already_present:
        print(f"[INFO] MovieLens dataset already exists in {target_dir}")
        return True

    print(f"[INFO] Fetching MovieLens Small dataset from: {MOVIELENS_URL}")
    ctx = get_ssl_context()
    
    headers = {"User-Agent": "CineMatch-DataDownloader/1.0"}
    req = urllib.request.Request(MOVIELENS_URL, headers=headers)

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=45) as response:
            content_length = response.headers.get("Content-Length")
            print(f"[INFO] Downloading archive ({content_length or 'unknown'} bytes)...")
            zip_bytes = response.read()

        print(f"[INFO] Download complete ({len(zip_bytes)} bytes). Extracting...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for member in z.namelist():
                # Format inside zip is 'ml-latest-small/filename.ext'
                filename = os.path.basename(member)
                if not filename:
                    continue
                extracted_path = os.path.join(target_dir, filename)
                with z.open(member) as source, open(extracted_path, "wb") as target:
                    target.write(source.read())
                print(f"  -> Extracted: {filename} ({os.path.getsize(extracted_path):,} bytes)")

        print("[SUCCESS] All MovieLens dataset files successfully extracted!")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        print(f"[HINT] If network is restricted, place movies.csv and ratings.csv manually in {target_dir}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = download_movielens_small()
    if not success:
        sys.exit(1)
