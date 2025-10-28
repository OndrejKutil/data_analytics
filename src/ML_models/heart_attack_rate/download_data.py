import kagglehub
import os
import glob

def download_data() -> str:

    path = kagglehub.dataset_download("fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid")
    
    # Look for CSV files in the downloaded directory
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    
    if len(csv_files) > 1:
        print(f"Multiple CSV files found: {[os.path.basename(f) for f in csv_files]}")
        print(f"Using the first one: {os.path.basename(csv_files[0])}")
    
    return csv_files[0]