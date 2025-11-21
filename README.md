# Real Estate Price Prediction (Divar Dataset)

This project builds a full **price prediction pipeline for real-estate listings** using data from the Divar platform.  
The main focus is on:

- Cleaning and enriching raw listing data
- Engineering meaningful features from text and structured fields
- Training and evaluating regression models to predict **property price (log-transformed)**

## Project Overview

The pipeline consists of two main stages:

1. **Data Cleaning & Feature Engineering**  
   Several modular notebooks process different groups of columns from the raw `Divar.csv` file and export cleaned CSV files (e.g. `cleaned_property_features.csv`, `total_price.csv`, `total_price_log.csv`).

2. **Modeling & Evaluation**  
   The final merged dataset is used in `RegressionModel.ipynb` to train regression models (Random Forest, Linear Regression, Decision Tree) on a **log-transformed price-per-unit** target.

---

## 1. Feature Engineering & Data Preparation

### 1.1 Financial Columns (Rent / Price / Credit)  
Notebook: `clean_columns_9_to_22.ipynb`

- Works on columns 9–22 (and some related columns 51–57) of `Divar.csv`
- Cleans and analyzes:
  - `rent_value`
  - `price_value`
  - `credit_value`
- Builds and validates a consistent **total_price** field from different combinations of rent, price, and credit.
- Exports cleaned price-related data (later used to create `total_price.csv` and `total_price_log.csv`).

### 1.2 Text-Informed Structural Features  
Notebook: `clean_columns_25_to_42.ipynb`

- Focuses on columns 25–42 plus basic ID/title/description columns.
- Uses a combination of:
  - Pattern-based rules (`regex`)
  - **LLM/OpenAI API calls** on `title` and `description`

- Fills missing values where possible and clips invalid values (e.g., very large room counts).

---

### 1.3 Location Features  
Notebook: `General_location_columns.ipynb`

- Extracts location-related columns from `Divar.csv`:
  - `cat3_slug` (category)
  - `city_slug`
  - `neighborhood_slug`
  - 
- Fixes missing neighborhood values using city-level modes.
- Prepares a compact location dataset (`location.csv`) that can be merged with other feature sets.

### 1.4 Land & Building Size / Improvements  
Notebook: `Land_improvements_columns.ipynb`

- Works on land and building size columns:
  - `land_size`
  - `building_size`
- Uses AI-generated helpers (`land_size_ai`, `building_size_ai`) to:
  - Fill missing values
  - Enforce logical relationships between land and building size (especially for villa-type properties)
- Handles special cases:
  - When only one of land/building sizes is known
  - When AI predictions are available for both
- Produces consistent land/improvement features for modeling.

### 1.5 Final Merged Features

All of the above processed pieces are merged into:

- `cleaned_property_features.csv` – main feature matrix
- `total_price.csv` / `total_price_log.csv` – cleaned target-related columns

These are the inputs to the modeling notebook.

## 2. Modeling: RegressionModel.ipynb

Notebook: `RegressionModel.ipynb`

### 2.1 Data Assembly

- Loads:
  - `cleaned_property_features.csv`
  - `total_price.csv`
  - `total_price_log.csv`
- Merges them into a single `DataFrame`.
- Fills missing `total_price_cleaned` values **group-wise** using:
  - `region_area_mapped`
  - `total_excl`
  - `Land`
  - `Land_per_unit`
- Scales `total_price` by `1e6` for better numeric stability.

### 2.2 Target Definition

The main target for modeling is:

df['log_cl_price'] = (np.log(df["total_price_cleaned"]) / np.log(1e6)) / df['Land']
