import kagglehub
import os
import glob
import pandas as pd
import re

# ===============================================
# Download data
# ===============================================

def download_data() -> str:

    # download the dataset from Kaggle
    path = kagglehub.dataset_download('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')
    
    # returns a list of paths matching the given pattern
    # in this case, it will return all CSV files in the dataset directory
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    
    # if no CSV files are found, raise an error
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {path}')
    
    # if multiple CSV files are found, print a warning and use the first one
    if len(csv_files) > 1:
        print(f'Multiple CSV files found: {[os.path.basename(f) for f in csv_files]}')
        print(f'Using the first one: {os.path.basename(csv_files[0])}')
    
    return csv_files[0]

# ===============================================
# Load data
# ===============================================

df = pd.read_csv(download_data())

def clean_text(text: str) -> str:
    '''Clean the text data by removing special characters and extra spaces.'''
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces with a single space
    text = re.sub(r'<.*?>', '', text)           # Remove HTML tags
    return text.strip()

df['review'] = df['review'].apply(clean_text)

# Map sentiment labels to integers
df['label'] = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})

def print_overview(df: pd.DataFrame):
    """Print an overview of the dataset."""
    print('=' * 100)
    print('Dataset Overview')
    print(df.head())
    print('=' * 100)
    print(f'Dataset shape: {df.shape}')
    print(f'Number of unique reviews: {df["review"].nunique()}')
    print(f'Columns: {df.columns.tolist()}')
    print('=' * 100)
    print()

print_overview(df)

# ===============================================
# Prepare data
# ===============================================

x = df['review']
y = df['label']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# Stratify the split to maintain the distribution of labels
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures reproducibility of the split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)