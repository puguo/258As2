import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Load the data
print("Loading dataset...")
data = []
with open('google_restaraunt.json', 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
print(f"Total reviews: {len(df)}")
print(f"Unique businesses: {df['business_id'].nunique()}")
print(f"Unique users: {df['user_id'].nunique()}")

# Calculate review counts
user_review_counts = df['user_id'].value_counts()
business_review_counts = df['business_id'].value_counts()

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('User & Business Review Distribution Analysis', fontsize=16, fontweight='bold')

# ============ USER ANALYSIS ============

# 1. User Review Count Distribution (Histogram)
ax1 = axes[0, 0]
ax1.hist(user_review_counts.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Reviews per User')
ax1.set_ylabel('Number of Users')
ax1.set_title('Distribution of Reviews per User')
ax1.set_yscale('log')
ax1.axvline(user_review_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_review_counts.mean():.1f}')
ax1.axvline(user_review_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {user_review_counts.median():.0f}')
ax1.legend()

# 2. User Review Count - Cumulative Distribution
ax2 = axes[0, 1]
user_counts_sorted = np.sort(user_review_counts.values)[::-1]
cumulative_pct = np.cumsum(user_counts_sorted) / user_counts_sorted.sum() * 100
user_pct = np.arange(1, len(user_counts_sorted) + 1) / len(user_counts_sorted) * 100
ax2.plot(user_pct, cumulative_pct, color='steelblue', linewidth=2)
ax2.set_xlabel('% of Users (sorted by activity)')
ax2.set_ylabel('% of Total Reviews')
ax2.set_title('Cumulative Review Contribution by Users')
ax2.axhline(50, color='gray', linestyle=':', alpha=0.7)
ax2.axhline(80, color='gray', linestyle=':', alpha=0.7)
# Find what % of users contribute 50% and 80% of reviews
idx_50 = np.searchsorted(cumulative_pct, 50)
idx_80 = np.searchsorted(cumulative_pct, 80)
pct_users_50 = user_pct[idx_50] if idx_50 < len(user_pct) else 100
pct_users_80 = user_pct[idx_80] if idx_80 < len(user_pct) else 100
ax2.axvline(pct_users_50, color='red', linestyle='--', alpha=0.7, label=f'{pct_users_50:.1f}% users → 50% reviews')
ax2.axvline(pct_users_80, color='orange', linestyle='--', alpha=0.7, label=f'{pct_users_80:.1f}% users → 80% reviews')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. User Review Count Breakdown (Pie/Bar)
ax3 = axes[0, 2]
user_bins = [0, 1, 2, 4, 9, 19, float('inf')]
user_labels = ['1', '2', '3-4', '5-9', '10-19', '20+']
user_binned = pd.cut(user_review_counts, bins=user_bins, labels=user_labels)
user_bin_counts = user_binned.value_counts().reindex(user_labels)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(user_labels)))
bars = ax3.bar(user_labels, user_bin_counts.values, color=colors, edgecolor='black')
ax3.set_xlabel('Number of Reviews')
ax3.set_ylabel('Number of Users')
ax3.set_title('Users by Review Count Category')
for bar, count in zip(bars, user_bin_counts.values):
    pct = count / len(user_review_counts) * 100
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============ BUSINESS ANALYSIS ============

# 4. Business Review Count Distribution (Histogram)
ax4 = axes[1, 0]
ax4.hist(business_review_counts.values, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Number of Reviews per Business')
ax4.set_ylabel('Number of Businesses')
ax4.set_title('Distribution of Reviews per Business')
ax4.set_yscale('log')
ax4.axvline(business_review_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {business_review_counts.mean():.1f}')
ax4.axvline(business_review_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {business_review_counts.median():.0f}')
ax4.legend()

# 5. Business Review Count - Cumulative Distribution
ax5 = axes[1, 1]
biz_counts_sorted = np.sort(business_review_counts.values)[::-1]
cumulative_pct_biz = np.cumsum(biz_counts_sorted) / biz_counts_sorted.sum() * 100
biz_pct = np.arange(1, len(biz_counts_sorted) + 1) / len(biz_counts_sorted) * 100
ax5.plot(biz_pct, cumulative_pct_biz, color='coral', linewidth=2)
ax5.set_xlabel('% of Businesses (sorted by popularity)')
ax5.set_ylabel('% of Total Reviews')
ax5.set_title('Cumulative Review Distribution by Business')
ax5.axhline(50, color='gray', linestyle=':', alpha=0.7)
ax5.axhline(80, color='gray', linestyle=':', alpha=0.7)
idx_50_biz = np.searchsorted(cumulative_pct_biz, 50)
idx_80_biz = np.searchsorted(cumulative_pct_biz, 80)
pct_biz_50 = biz_pct[idx_50_biz] if idx_50_biz < len(biz_pct) else 100
pct_biz_80 = biz_pct[idx_80_biz] if idx_80_biz < len(biz_pct) else 100
ax5.axvline(pct_biz_50, color='red', linestyle='--', alpha=0.7, label=f'{pct_biz_50:.1f}% businesses → 50% reviews')
ax5.axvline(pct_biz_80, color='orange', linestyle='--', alpha=0.7, label=f'{pct_biz_80:.1f}% businesses → 80% reviews')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Business Review Count Breakdown (Bar)
ax6 = axes[1, 2]
biz_bins = [0, 4, 9, 19, 49, 99, float('inf')]
biz_labels = ['1-4', '5-9', '10-19', '20-49', '50-99', '100+']
biz_binned = pd.cut(business_review_counts, bins=biz_bins, labels=biz_labels)
biz_bin_counts = biz_binned.value_counts().reindex(biz_labels)
colors_biz = plt.cm.Oranges(np.linspace(0.3, 0.9, len(biz_labels)))
bars = ax6.bar(biz_labels, biz_bin_counts.values, color=colors_biz, edgecolor='black')
ax6.set_xlabel('Number of Reviews')
ax6.set_ylabel('Number of Businesses')
ax6.set_title('Businesses by Review Count Category')
for bar, count in zip(bars, biz_bin_counts.values):
    pct = count / len(business_review_counts) * 100
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('user_business_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'user_business_distribution.png'")

# Print detailed statistics
print("\n" + "="*60)
print("USER REVIEW STATISTICS")
print("="*60)
print(f"Total unique users: {len(user_review_counts):,}")
print(f"\nReviews per user:")
print(f"  Mean: {user_review_counts.mean():.2f}")
print(f"  Median: {user_review_counts.median():.0f}")
print(f"  Std: {user_review_counts.std():.2f}")
print(f"  Min: {user_review_counts.min()}")
print(f"  Max: {user_review_counts.max()}")
print(f"\nUser breakdown:")
print(f"  Users with exactly 1 review: {(user_review_counts == 1).sum():,} ({(user_review_counts == 1).sum()/len(user_review_counts)*100:.1f}%)")
print(f"  Users with 2 reviews: {(user_review_counts == 2).sum():,} ({(user_review_counts == 2).sum()/len(user_review_counts)*100:.1f}%)")
print(f"  Users with 3-4 reviews: {((user_review_counts >= 3) & (user_review_counts <= 4)).sum():,} ({((user_review_counts >= 3) & (user_review_counts <= 4)).sum()/len(user_review_counts)*100:.1f}%)")
print(f"  Users with 5-9 reviews: {((user_review_counts >= 5) & (user_review_counts <= 9)).sum():,} ({((user_review_counts >= 5) & (user_review_counts <= 9)).sum()/len(user_review_counts)*100:.1f}%)")
print(f"  Users with 10+ reviews: {(user_review_counts >= 10).sum():,} ({(user_review_counts >= 10).sum()/len(user_review_counts)*100:.1f}%)")

print("\n" + "="*60)
print("BUSINESS REVIEW STATISTICS")
print("="*60)
print(f"Total unique businesses: {len(business_review_counts):,}")
print(f"\nReviews per business:")
print(f"  Mean: {business_review_counts.mean():.2f}")
print(f"  Median: {business_review_counts.median():.0f}")
print(f"  Std: {business_review_counts.std():.2f}")
print(f"  Min: {business_review_counts.min()}")
print(f"  Max: {business_review_counts.max()}")
print(f"\nBusiness breakdown:")
print(f"  Businesses with 1-4 reviews: {(business_review_counts <= 4).sum():,} ({(business_review_counts <= 4).sum()/len(business_review_counts)*100:.1f}%)")
print(f"  Businesses with 5-9 reviews: {((business_review_counts >= 5) & (business_review_counts <= 9)).sum():,} ({((business_review_counts >= 5) & (business_review_counts <= 9)).sum()/len(business_review_counts)*100:.1f}%)")
print(f"  Businesses with 10-19 reviews: {((business_review_counts >= 10) & (business_review_counts <= 19)).sum():,} ({((business_review_counts >= 10) & (business_review_counts <= 19)).sum()/len(business_review_counts)*100:.1f}%)")
print(f"  Businesses with 20-49 reviews: {((business_review_counts >= 20) & (business_review_counts <= 49)).sum():,} ({((business_review_counts >= 20) & (business_review_counts <= 49)).sum()/len(business_review_counts)*100:.1f}%)")
print(f"  Businesses with 50-99 reviews: {((business_review_counts >= 50) & (business_review_counts <= 99)).sum():,} ({((business_review_counts >= 50) & (business_review_counts <= 99)).sum()/len(business_review_counts)*100:.1f}%)")
print(f"  Businesses with 100+ reviews: {(business_review_counts >= 100).sum():,} ({(business_review_counts >= 100).sum()/len(business_review_counts)*100:.1f}%)")

print("\n" + "="*60)
print("SPARSITY ANALYSIS (for recommendation systems)")
print("="*60)
total_possible = len(user_review_counts) * len(business_review_counts)
actual_reviews = len(df)
sparsity = (1 - actual_reviews / total_possible) * 100
print(f"User-Business matrix size: {len(user_review_counts):,} x {len(business_review_counts):,} = {total_possible:,}")
print(f"Actual reviews: {actual_reviews:,}")
print(f"Sparsity: {sparsity:.6f}%")
