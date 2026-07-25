from dotenv import load_dotenv
import kagglehub
from pathlib import Path
import os
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

def download_competition_data(competition: str, dest_dir: str="data/raw"): 
    cache_path = kagglehub.competition_download(competition)
    
    dest = os.path.join(BASE_DIR, dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    for f in os.listdir(cache_path): 
        shutil.copy(os.path.join(cache_path, f), dest)

    print(f"Files copied to: {dest}")
    return dest 

def remove_irrelevant_files(dest: str): 
    irrelevant_files_list = [
        "sample_submission.csv", 
        "test_transaction.csv", 
        "test_identity.csv"
    ]
    for file in irrelevant_files_list: 
        file_path = os.path.join(dest, file)
        try: 
            if os.path.exists(file_path): 
                os.remove(file_path)
                print(f"Successfully deleted: {file_path}")
            else: 
                print(f"Error: {file_path} does not exist.")
        except PermissionError:
            print(f"Permission denied: {file_path} is open or restricted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return dest
    
def rename_files(dest: str): 
    rename_file_dict = { 
        "train_transaction.csv": "transaction.csv",
        "train_identity.csv": "identity.csv"
    }
    for original, renamed in rename_file_dict.items(): 
        os.rename(original, renamed)
    return dest
        
    
if __name__ == "__main__": 
    dest = download_competition_data()
    dest = remove_irrelevant_files(dest)
    dest = rename_files(dest)
    