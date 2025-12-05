# Model Choice

This document explains the model used for price prediction and how to manage memory during scraping and modeling.

## Model Overview

- Notebook: `src/model/model.ipynb`
- Baseline and tuned models: `XGBRegressor` (XGBoost), with `GridSearchCV` for hyperparameter tuning.
- Why XGBoost for tabular real estate data:
  - Handles non‑linear interactions and complex feature mixes (categorical + numeric)
  - Robust to monotonicity and wide ranges in area/price
  - Works well with moderate missingness (after basic preprocessing)
  - Strong out‑of‑the‑box performance on tabular problems
  - Offers feature importance for interpretability (global insights)

## Targets and Metrics

- Target: Total price (CZK). Optionally, price per m² can be modeled as an alternate target.
- Typical metrics: RMSE for error in price space, R² for explained variance. Residual diagnostics used for sanity checks.

## Features Used

See `docs/feature_extraction.md` for a detailed list. In short: location, areas, structural attributes, amenities, energy/utilities, transport/surroundings, and text‑derived flags.

## Training Pipeline Notes

- Encoding: Convert categorical strings to encodings appropriate for tree models (one‑hot or label encoding).
- Missing values: Impute sensibly (median/mode) to keep samples.
- Regularization: Use tuned tree depth, learning rate, min child weight, subsample/colsample to reduce overfitting.
- Validation: Cross‑validation via `GridSearchCV` ensures robust estimates across folds.
