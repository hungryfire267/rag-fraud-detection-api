from dotenv import load_dotenv
import kagglehub
from pathlib import Path
import os
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

print("Username:", os.getenv("KAGGLE_USERNAME"))
print("Key found:", os.getenv("KAGGLE_KEY") is not None)
print("Key length:", len(os.getenv("KAGGLE_KEY") or ""))


def download_competition_data(competition: str, dest_dir: str = "data/raw") -> Path:
    cache_path = kagglehub.competition_download(competition)

    dest = BASE_DIR / dest_dir
    dest.mkdir(parents=True, exist_ok=True)

    for f in os.listdir(cache_path):
        shutil.copy(os.path.join(cache_path, f), dest)

    print(f"Files copied to: {dest}")
    return dest


def remove_irrelevant_files(dest: Path) -> Path:
    irrelevant_files_list = [
        "sample_submission.csv",
        "test_transaction.csv",
        "test_identity.csv",
    ]
    for file in irrelevant_files_list:
        file_path = dest / file
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted: {file_path}")
            else:
                print(f"Error: {file_path} does not exist.")
        except PermissionError:
            print(f"Permission denied: {file_path} is open or restricted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return dest


def rename_files(dest: Path) -> Path:
    rename_file_dict = {
        "train_transaction.csv": "transaction.csv",
        "train_identity.csv": "identity.csv",
    }
    for original, renamed in rename_file_dict.items():
        original_path = dest / original
        renamed_path = dest / renamed
        if original_path.exists():
            original_path.rename(renamed_path)
            print(f"Renamed {original} -> {renamed}")
        else:
            print(f"Skipped: {original_path} does not exist.")
    return dest


if __name__ == "__main__":
    dest = download_competition_data('ieee-fraud-detection')
    dest = remove_irrelevant_files(dest)
    dest = rename_files(dest)