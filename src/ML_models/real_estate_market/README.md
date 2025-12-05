# Real Estate Price Prediction

Minimal overview of the scraping + preprocessing + modeling pipeline for Prague listings.

## Structure

```text
src/ML_models/real_estate_market/
├─ src/
│  ├─ scraper/
│  │  └─ scraper.py            # Async link + detail scraping, batching, saving
│  ├─ preprocessing/
│  │  ├─ preprocessing.py      # Feature extraction and consolidation
│  │  └─ preprocessed_listings.csv (generated)
│  └─ model/
│     └─ model.ipynb           # XGBoost training, tuning, evaluation
└─ docs/
    ├─ concurrent_saving.md     # Concurrency, memory usage, incremental appends
    ├─ feature_extraction.md    # What features we extract and why
    └─ model_choice_and_memory.md # Model rationale and memory considerations
```

## Quick Links

- Concurrent saving and memory: [Saving and memory docs](./docs/concurrent_saving.md)
- Features and rationale: [feature extraction docs](./docs/feature_extraction.md)
- Model choice: [Model choice](./docs/model_choice.md)

## Quick Start

1. Scrape (adjust `PAGES_TO_SCRAPE`, `BATCH_SIZE`, `SAVE_FILES` inside `scraper.py`):

```bash
# Windows PowerShell example
python -m src.ML_models.real_estate_market.src.scraper.scraper
```

2. Preprocess features:

```bash
python -m src.ML_models.real_estate_market.src.preprocessing.preprocessing
```

3. Train/evaluate model in notebook: open `src/ML_models/real_estate_market/src/model/model.ipynb`.

Notes:

- For long runs, prefer incremental appends (see docs) to keep memory bounded.
- Reduce `BATCH_SIZE` if you encounter high memory usage.
