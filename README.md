# CSE 258 Assignment 2: Restaurant Recommendation System

Top-K restaurant recommendation system using similarity-based collaborative filtering on Google restaurant review data.

## Project Overview

This project builds a recommendation system to suggest the top-K restaurants to users based on similarity metrics. We use collaborative filtering techniques on a large-scale restaurant review dataset.

## Dataset

- **Source**: Google Restaurant Reviews Dataset
- **Original Size**: 1,487,747 reviews
- **After Preprocessing**: 513,868 reviews
- **Users**: 98,975
- **Restaurants**: 28,274
- **Sparsity**: 99.98%

### Data Statistics (After Preprocessing)

- **Training Set**: 342,665 reviews (66.7%)
- **Validation Set**: 24,950 reviews (4.9%)
- **Test Set**: 146,253 reviews (28.5%)
- **Average Reviews per User**: 3.46
- **Average Reviews per Restaurant**: 12.12

### Rating Distribution (Training Set)
- 5 stars: 64.3%
- 4 stars: 24.7%
- 3 stars: 7.5%
- 2 stars: 2.2%
- 1 star: 1.4%

## Setup Instructions

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Getting the Data

1. Obtain the `google_restaraunt.json` dataset
2. Place it in the root directory of this project

### Running the Preprocessing

**Option 1: Using Jupyter Notebooks (Recommended for Course Submission)**
```bash
# Start Jupyter
jupyter notebook

# Then open and run:
# - analyze_dataset.ipynb
# - preprocess_data.ipynb
```

**Option 2: Using Python Scripts**
```bash
# Analyze the dataset distribution
python analyze_dataset.py

# Preprocess the data (takes ~30 minutes)
python preprocess_data.py
```

The preprocessing script will generate:
- `preprocessed_data.pkl` - Main preprocessed dataset with sparse matrices
- `train_data.csv` - Training set
- `val_data.csv` - Validation set
- `test_data.csv` - Test set
- `business_stats.csv` - Restaurant statistics for cold-start handling
- `id_mappings.json` - User and business ID mappings

## Project Structure

```
.
├── README.md
├── analyze_dataset.ipynb        # Dataset analysis notebook (MAIN)
├── preprocess_data.ipynb        # Data preprocessing notebook (MAIN)
├── user_business_distribution.png  # Distribution analysis visualization
└── dataset_analysis.png         # Initial dataset analysis
```

## Data Preprocessing Pipeline

The preprocessing script performs the following steps:

1. **Load Data**: Read JSON Lines format dataset
2. **Clean Data**: Remove duplicates and handle missing values
3. **Filter Sparse Data**:
   - Keep users with ≥3 reviews
   - Keep restaurants with ≥5 reviews
   - Iterative filtering until convergence
4. **Create ID Mappings**: Map user/business IDs to integer indices
5. **Train/Val/Test Split**: Per-user stratified splitting (70/15/15)
6. **Create Sparse Matrices**: Build CSR matrices for efficient computation
7. **Compute Statistics**: Calculate restaurant statistics for cold-start scenarios

## Recommendation Approach

**Similarity-Based Collaborative Filtering**

Given the data characteristics (restaurants have more reviews than users on average), an item-based collaborative filtering approach is recommended:

1. Compute restaurant-restaurant similarity matrix using cosine similarity
2. For each user, identify restaurants they've rated highly
3. Find similar restaurants
4. Recommend top-K unrated similar restaurants

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
- **MAP@K**: Mean Average Precision

## Usage Example

```python
import pickle

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access the data
train_matrix = data['train_matrix']  # Sparse user-item matrix
test_matrix = data['test_matrix']
user_to_idx = data['user_to_idx']
business_to_idx = data['business_to_idx']
business_stats = data['business_stats']
```

## Team Members

- [Your Name]
- [Teammate Names]

## Course Information

- **Course**: CSE 258 - Web Mining and Recommender Systems
- **Institution**: UC San Diego
- **Year**: 2025
