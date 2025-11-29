import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle

print("="*60)
print("DATA PREPROCESSING FOR TOP-K RESTAURANT RECOMMENDATION")
print("="*60)

# Load the data
print("\n[1/7] Loading dataset...")
data = []
with open('google_restaraunt.json', 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
print(f"   Total reviews: {len(df):,}")
print(f"   Unique users: {df['user_id'].nunique():,}")
print(f"   Unique businesses: {df['business_id'].nunique():,}")

# Clean data
print("\n[2/7] Cleaning data...")
# Remove any duplicates
df = df.drop_duplicates(subset=['user_id', 'business_id'])
print(f"   After removing duplicates: {len(df):,} reviews")

# Check for missing values
missing = df[['user_id', 'business_id', 'rating']].isnull().sum()
if missing.sum() > 0:
    print(f"   Found missing values: {missing[missing > 0]}")
    df = df.dropna(subset=['user_id', 'business_id', 'rating'])
    print(f"   After removing missing: {len(df):,} reviews")
else:
    print("   No missing values found")

# Filter sparse data
print("\n[3/7] Filtering sparse users and businesses...")
min_user_reviews = 3  # Users must have at least 3 reviews
min_business_reviews = 5  # Businesses must have at least 5 reviews

# Iteratively filter (because filtering businesses affects users and vice versa)
prev_size = 0
iteration = 0
while len(df) != prev_size and iteration < 10:
    prev_size = len(df)
    iteration += 1

    user_counts = df['user_id'].value_counts()
    business_counts = df['business_id'].value_counts()

    valid_users = user_counts[user_counts >= min_user_reviews].index
    valid_businesses = business_counts[business_counts >= min_business_reviews].index

    df = df[df['user_id'].isin(valid_users) & df['business_id'].isin(valid_businesses)]

    if iteration > 1:
        print(f"   Iteration {iteration}: {len(df):,} reviews")

print(f"\n   Final dataset after filtering:")
print(f"   Reviews: {len(df):,}")
print(f"   Users: {df['user_id'].nunique():,}")
print(f"   Businesses: {df['business_id'].nunique():,}")
print(f"   Sparsity: {(1 - len(df)/(df['user_id'].nunique() * df['business_id'].nunique()))*100:.4f}%")

# Create user and business mappings
print("\n[4/7] Creating ID mappings...")
unique_users = df['user_id'].unique()
unique_businesses = df['business_id'].unique()

user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

business_to_idx = {bid: idx for idx, bid in enumerate(unique_businesses)}
idx_to_business = {idx: bid for bid, idx in business_to_idx.items()}

# Add mapped indices to dataframe
df['user_idx'] = df['user_id'].map(user_to_idx)
df['business_idx'] = df['business_id'].map(business_to_idx)

print(f"   Created mappings for {len(user_to_idx):,} users and {len(business_to_idx):,} businesses")

# Split data into train/validation/test sets
print("\n[5/7] Splitting into train/validation/test sets...")
# Use stratified split to ensure each user appears in training set
# For recommendation systems, we typically do per-user splits

train_data = []
val_data = []
test_data = []

for user_id in df['user_id'].unique():
    user_reviews = df[df['user_id'] == user_id]
    n_reviews = len(user_reviews)

    if n_reviews >= 5:
        # For users with many reviews: 70% train, 15% val, 15% test
        train_size = int(0.7 * n_reviews)
        val_size = int(0.15 * n_reviews)

        # Shuffle user's reviews
        user_reviews = user_reviews.sample(frac=1, random_state=42)

        train_data.append(user_reviews.iloc[:train_size])
        val_data.append(user_reviews.iloc[train_size:train_size+val_size])
        test_data.append(user_reviews.iloc[train_size+val_size:])
    else:
        # For users with few reviews: put most in train, 1 in test
        user_reviews = user_reviews.sample(frac=1, random_state=42)
        train_data.append(user_reviews.iloc[:-1])
        test_data.append(user_reviews.iloc[-1:])

train_df = pd.concat(train_data, ignore_index=True)
val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
test_df = pd.concat(test_data, ignore_index=True)

print(f"   Train set: {len(train_df):,} reviews ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation set: {len(val_df):,} reviews ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test set: {len(test_df):,} reviews ({len(test_df)/len(df)*100:.1f}%)")

# Create user-item interaction matrices
print("\n[6/7] Creating interaction matrices...")
from scipy.sparse import csr_matrix

def create_interaction_matrix(df_subset, n_users, n_businesses):
    """Create sparse user-item interaction matrix"""
    rows = df_subset['user_idx'].values
    cols = df_subset['business_idx'].values
    ratings = df_subset['rating'].values

    matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_businesses))
    return matrix

n_users = len(user_to_idx)
n_businesses = len(business_to_idx)

train_matrix = create_interaction_matrix(train_df, n_users, n_businesses)
val_matrix = create_interaction_matrix(val_df, n_users, n_businesses) if len(val_df) > 0 else None
test_matrix = create_interaction_matrix(test_df, n_users, n_businesses)

print(f"   Train matrix shape: {train_matrix.shape}")
print(f"   Train matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.4f}%")
if val_matrix is not None:
    print(f"   Val matrix shape: {val_matrix.shape}")
print(f"   Test matrix shape: {test_matrix.shape}")

# Compute business statistics for cold-start handling
print("\n[7/7] Computing business statistics...")
business_stats = train_df.groupby('business_id').agg({
    'rating': ['mean', 'count', 'std']
}).reset_index()
business_stats.columns = ['business_id', 'avg_rating', 'num_ratings', 'std_rating']
business_stats['std_rating'] = business_stats['std_rating'].fillna(0)

# Add business index
business_stats['business_idx'] = business_stats['business_id'].map(business_to_idx)

print(f"   Computed statistics for {len(business_stats):,} businesses")
print(f"   Average rating across all businesses: {business_stats['avg_rating'].mean():.3f}")

# Save preprocessed data
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

# Save as pickle for fast loading
save_data = {
    'train_df': train_df,
    'val_df': val_df,
    'test_df': test_df,
    'train_matrix': train_matrix,
    'val_matrix': val_matrix,
    'test_matrix': test_matrix,
    'user_to_idx': user_to_idx,
    'idx_to_user': idx_to_user,
    'business_to_idx': business_to_idx,
    'idx_to_business': idx_to_business,
    'business_stats': business_stats,
    'n_users': n_users,
    'n_businesses': n_businesses,
}

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print("✓ Saved preprocessed_data.pkl")

# Also save CSV versions for easy inspection
train_df.to_csv('train_data.csv', index=False)
if len(val_df) > 0:
    val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
business_stats.to_csv('business_stats.csv', index=False)
print("✓ Saved CSV files (train_data.csv, val_data.csv, test_data.csv, business_stats.csv)")

# Save mappings as JSON for readability
mappings = {
    'user_to_idx': user_to_idx,
    'business_to_idx': business_to_idx,
}
with open('id_mappings.json', 'w') as f:
    json.dump(mappings, f, indent=2)
print("✓ Saved id_mappings.json")

# Print summary statistics
print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Total users: {n_users:,}")
print(f"Total businesses: {n_businesses:,}")
print(f"Total interactions: {len(df):,}")
print(f"")
print(f"Train interactions: {len(train_df):,}")
print(f"Val interactions: {len(val_df):,}")
print(f"Test interactions: {len(test_df):,}")
print(f"")
print(f"Avg reviews per user: {len(train_df)/n_users:.2f}")
print(f"Avg reviews per business: {len(train_df)/n_businesses:.2f}")
print(f"")
print(f"Rating distribution (train):")
for rating in sorted(train_df['rating'].unique()):
    count = (train_df['rating'] == rating).sum()
    pct = count / len(train_df) * 100
    print(f"  {rating} stars: {count:,} ({pct:.1f}%)")

print("\n✓ Preprocessing complete!")
print("\nTo load the preprocessed data:")
print("  import pickle")
print("  with open('preprocessed_data.pkl', 'rb') as f:")
print("      data = pickle.load(f)")
