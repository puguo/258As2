# CSE 258 Assignment 2: Restaurant Recommendation System

Top-K restaurant recommendation system using similarity-based collaborative filtering.

## Quick Start

1. **Get the data**: Place `google_restaraunt.json` in the root directory
2. **Run preprocessing**: Open and run [`notebooks/preprocess_data.ipynb`](notebooks/preprocess_data.ipynb)
3. **Build model**: Use the generated preprocessed data for your recommendation model

## Project Structure

```
.
├── notebooks/
│   └── preprocess_data.ipynb    # Main preprocessing pipeline
├── analysis/                     # Dataset analysis (optional)
├── docs/                         # Assignment documentation
└── README.md
```

## Preprocessing Pipeline

The [`preprocess_data.ipynb`](notebooks/preprocess_data.ipynb) notebook performs:

1. **Data Loading & Cleaning**: Load 1.5M reviews, remove duplicates
2. **Filtering**: Keep users with ≥3 reviews, businesses with ≥5 reviews
3. **Train/Val/Test Split**: Per-user stratified split (70/15/15)
4. **Sparse Matrices**: Create CSR matrices for efficient computation
5. **Statistics**: Compute business stats for cold-start handling

### Output Files

The notebook generates:
- `preprocessed_data.pkl` - All data structures (matrices, mappings, stats)
- `train_data.csv`, `val_data.csv`, `test_data.csv` - CSV versions
- `business_stats.csv` - Business statistics
- `id_mappings.json` - User/business ID mappings

### Preprocessing Results

- **Final Dataset**: 513,868 reviews
- **Users**: 98,975
- **Businesses**: 28,274
- **Sparsity**: 99.98%
- **Train**: 342,665 reviews (66.7%)
- **Val**: 24,950 reviews (4.9%)
- **Test**: 146,253 reviews (28.5%)

## Usage

```python
import pickle

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access data structures
train_matrix = data['train_matrix']  # Sparse CSR matrix
user_to_idx = data['user_to_idx']
business_to_idx = data['business_to_idx']
```

## Requirements

```bash
pip install pandas numpy scikit-learn scipy
```

## Course Information

**CSE 258** - Web Mining and Recommender Systems
UC San Diego, 2025
