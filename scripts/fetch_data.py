"""
Auto-fetch script for the 20 Newsgroups dataset.
Downloads and extracts the dataset if it is missing locally.
"""
import os
import zipfile
import tarfile
import urllib.request

RAW_DIR = os.path.join("data", "raw")
DATASET_DIR = os.path.join(RAW_DIR, "20_newsgroups")
ZIP_SOURCE = os.path.abspath(os.path.join("..", "twenty+newsgroups.zip"))
TAR_FILE = os.path.join(RAW_DIR, "20_newsgroups.tar.gz")
DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"


def fetch_data():
    """Check if dataset exists, if not extract or download it."""
    if os.path.isdir(DATASET_DIR):
        count = sum(len(files) for _, _, files in os.walk(DATASET_DIR))
        print(f"Dataset already exists at {DATASET_DIR} ({count} files). Skipping.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    # Option 1: Extract from local zip
    if os.path.exists(ZIP_SOURCE):
        print(f"Found local zip: {ZIP_SOURCE}")
        print("Extracting zip...")
        with zipfile.ZipFile(ZIP_SOURCE, "r") as z:
            z.extractall(RAW_DIR)
        print("Extracting tar.gz...")
        with tarfile.open(TAR_FILE, "r:gz") as t:
            t.extractall(RAW_DIR)
        print("Done!")
        return

    # Option 2: Download from UCI repository
    if not os.path.exists(TAR_FILE):
        print(f"Downloading dataset from {DOWNLOAD_URL}...")
        urllib.request.urlretrieve(DOWNLOAD_URL, TAR_FILE)
        print("Download complete.")

    print("Extracting tar.gz...")
    with tarfile.open(TAR_FILE, "r:gz") as t:
        t.extractall(RAW_DIR)
    print("Done!")


if __name__ == "__main__":
    fetch_data()
