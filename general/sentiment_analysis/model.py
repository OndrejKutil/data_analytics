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

# ===============================================
# Tokenization
# ===============================================

from transformers import BertTokenizer

# Load the BERT tokenizer
# This will download the tokenizer files if not already cached
# If you have a custom path for the tokenizer, you can specify it here
# For example, if you have downloaded the tokenizer files to a specific path:
# tokenizer = BertTokenizer.from_pretrained('path/to/your/tokenizer')
#
# Otherwise, you can use the default BERT tokenizer

from dotenv import load_dotenv
import os
load_dotenv()

model_path = os.getenv('model_path')

tokenizer = BertTokenizer.from_pretrained(model_path)

# Tokenize the text data
x_train_encoded = tokenizer(list(x_train), padding=True, truncation=True, max_length=256)
x_test_encoded = tokenizer(list(x_test), padding=True, truncation=True, max_length=256)

# ===============================================
# Create Dataset class
# ===============================================

import torch

# Create a custom Dataset class for PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Override the __getitem__ method to return the encoded inputs and labels
    # This method is called when accessing an item from the dataset
    # idx is the index of the item to be retrieved
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    # Override the __len__ method to return the length of the dataset
    def __len__(self):
        return len(self.labels)

# Create datasets for training and testing
train_dataset = SentimentDataset(x_train_encoded, y_train.tolist())
test_dataset = SentimentDataset(x_test_encoded, y_test.tolist())

# ===============================================
# Model training
# ===============================================

from transformers import BertForSequenceClassification

#? If you are running this code for the first time, you need to download the model.
#? Uncomment the line below to download the model.
#
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
#
# On first download, the model will be downloaded to the cache directory.
# For higher download speed, install hf_xet package:
# pip install hf_xet
#
# C://Users/<YourUsername>/.cache/huggingface/hub/
# models--bert-base-uncased
#   └── snapshots
#       └── <hash>/
#           ├── config.json
#           ├── pytorch_model.bin
#           ├── tokenizer_config.json
#           ├── vocab.txt
#
# Then it can be reused locally without downloading again.
# Simply  copy the model path to the model name.
# 
# So instead of this:
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
#
# You can use:
# path = 'C://Users/<YourUsername>/.cache/huggingface/hub/models--bert-base-uncased/snapshots/<hash>'
# model = BertForSequenceClassification.from_pretrained(path, num_labels=3)
#
# Or you can copy the model folder to your custom path and set an environment variable for example

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

# ===============================================
# Training the model
# ===============================================

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}



training_args = TrainingArguments(
    output_dir="./general/sentiment_analysis/results",      # Where to save model checkpoints
    eval_strategy="epoch",                                  # Evaluate at end of each epoch
    save_strategy="epoch",                                  # Save model at end of each epoch
    learning_rate=2e-5,                                     # Learning rate for the optimizer
    per_device_train_batch_size=16,                         # Batch size for training
    per_device_eval_batch_size=16,                          # Batch size for evaluation
    num_train_epochs=3,                                     # Number of training epochs
    weight_decay=0.01,                                      # Weight decay for optimizer
    load_best_model_at_end=True,                            # Load best model at end of training
    logging_dir="./logs",                                   # Directory for storing logs
    logging_steps=5,                                        # Log every 5 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./general/sentiment_analysis/model")
tokenizer.save_pretrained("./general/sentiment_analysis/model")


