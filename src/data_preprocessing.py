"""
Data preprocessing pipeline for Robust Recommendation System.

Steps:
1. Load raw Amazon Electronics reviews
2. Filter high-variance items (Step A)
3. Pull interactions for selected items (Step B)
4. Enforce minimum activity constraints (Step C)
5. Save filtered dataset
"""
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import orjson

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path, max_lines: int = None) -> List[Dict]:
    """Load raw JSON lines data efficiently using orjson."""
    logger.info(f"Loading data from {filepath}")
    data = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading reviews")):
            if max_lines and i >= max_lines:
                break
            try:
                record = orjson.loads(line)
                # Only keep records with required fields
                if all(k in record for k in ['reviewerID', 'asin', 'overall', 'reviewText']):
                    data.append({
                        'user_id': record['reviewerID'],
                        'item_id': record['asin'],
                        'rating': float(record['overall']),
                        'review_text': record.get('reviewText', ''),
                        'summary': record.get('summary', ''),
                        'verified': record.get('verified', False),
                        'timestamp': record.get('unixReviewTime', 0)
                    })
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    logger.info(f"Loaded {len(data):,} valid reviews")
    return data


def compute_item_statistics(data: List[Dict]) -> pd.DataFrame:
    """Compute rating statistics per item."""
    logger.info("Computing item statistics...")
    
    item_ratings = defaultdict(list)
    item_reviews = defaultdict(list)
    
    for record in tqdm(data, desc="Aggregating by item"):
        item_ratings[record['item_id']].append(record['rating'])
        if record['review_text']:
            item_reviews[record['item_id']].append(len(record['review_text']))
    
    stats = []
    for item_id, ratings in item_ratings.items():
        ratings_arr = np.array(ratings)
        stats.append({
            'item_id': item_id,
            'count': len(ratings),
            'mean_rating': np.mean(ratings_arr),
            'std_rating': np.std(ratings_arr),
            'min_rating': np.min(ratings_arr),
            'max_rating': np.max(ratings_arr),
            'avg_review_len': np.mean(item_reviews[item_id]) if item_reviews[item_id] else 0
        })
    
    df = pd.DataFrame(stats)
    logger.info(f"Computed statistics for {len(df):,} items")
    return df


def select_high_variance_items(
    item_stats: pd.DataFrame,
    min_ratings: int = 50,
    top_k: int = 1500
) -> List[str]:
    """
    Select items that are:
    1. Popular enough (count >= min_ratings)
    2. High-variance in ratings
    3. Have enough text (reviews present)
    
    Uses count * std as selection score (balances volume + dissonance potential).
    """
    logger.info(f"Selecting high-variance items (min_ratings={min_ratings}, top_k={top_k})")
    
    # Filter by minimum rating count
    filtered = item_stats[item_stats['count'] >= min_ratings].copy()
    logger.info(f"Items with >= {min_ratings} ratings: {len(filtered):,}")
    
    # Filter by having reviews
    filtered = filtered[filtered['avg_review_len'] > 50]
    logger.info(f"Items with substantial reviews: {len(filtered):,}")
    
    # Compute selection score: count * std
    filtered['selection_score'] = filtered['count'] * filtered['std_rating']
    
    # Select top K by score
    selected = filtered.nlargest(top_k, 'selection_score')
    
    logger.info(f"Selected {len(selected):,} high-variance items")
    logger.info(f"  Mean rating std: {selected['std_rating'].mean():.3f}")
    logger.info(f"  Mean count: {selected['count'].mean():.1f}")
    
    return selected['item_id'].tolist()


def filter_interactions(
    data: List[Dict],
    selected_items: List[str]
) -> List[Dict]:
    """Pull all interactions for selected items."""
    logger.info(f"Filtering interactions for {len(selected_items):,} items")
    
    selected_set = set(selected_items)
    filtered = [r for r in tqdm(data, desc="Filtering") if r['item_id'] in selected_set]
    
    logger.info(f"Filtered to {len(filtered):,} interactions")
    return filtered


def enforce_minimum_activity(
    data: List[Dict],
    min_user_interactions: int = 5,
    min_item_interactions: int = 20,
    max_iterations: int = 5
) -> List[Dict]:
    """
    Iteratively filter to enforce minimum activity constraints.
    Removes users/items with too few interactions.
    """
    logger.info(f"Enforcing minimum activity (users>={min_user_interactions}, items>={min_item_interactions})")
    
    current_data = data
    
    for iteration in range(max_iterations):
        initial_size = len(current_data)
        
        # Count user interactions
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for r in current_data:
            user_counts[r['user_id']] += 1
            item_counts[r['item_id']] += 1
        
        # Filter users
        valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
        current_data = [r for r in current_data if r['user_id'] in valid_users]
        
        # Recount items
        item_counts = defaultdict(int)
        for r in current_data:
            item_counts[r['item_id']] += 1
        
        # Filter items
        valid_items = {i for i, c in item_counts.items() if c >= min_item_interactions}
        current_data = [r for r in current_data if r['item_id'] in valid_items]
        
        final_size = len(current_data)
        removed = initial_size - final_size
        
        logger.info(f"  Iteration {iteration + 1}: {initial_size:,} -> {final_size:,} ({removed:,} removed)")
        
        if removed == 0:
            break
    
    # Final statistics
    unique_users = len(set(r['user_id'] for r in current_data))
    unique_items = len(set(r['item_id'] for r in current_data))
    
    logger.info(f"Final dataset: {len(current_data):,} interactions")
    logger.info(f"  Unique users: {unique_users:,}")
    logger.info(f"  Unique items: {unique_items:,}")
    logger.info(f"  Density: {len(current_data) / (unique_users * unique_items) * 100:.4f}%")
    
    return current_data


def create_id_mappings(data: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create integer ID mappings for users and items."""
    unique_users = sorted(set(r['user_id'] for r in data))
    unique_items = sorted(set(r['item_id'] for r in data))
    
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {i: idx for idx, i in enumerate(unique_items)}
    
    return user2idx, item2idx


def save_processed_data(
    data: List[Dict],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    output_dir: Path
):
    """Save processed data in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)
    
    # Save full data with reviews
    df.to_parquet(output_dir / 'interactions_full.parquet', index=False)
    logger.info(f"Saved full interactions to {output_dir / 'interactions_full.parquet'}")
    
    # Save CF-ready format (no text, just IDs and ratings)
    cf_df = df[['user_idx', 'item_idx', 'rating', 'timestamp']].copy()
    cf_df.to_csv(output_dir / 'interactions_cf.csv', index=False)
    logger.info(f"Saved CF interactions to {output_dir / 'interactions_cf.csv'}")
    
    # Save ID mappings
    mappings = {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': {v: k for k, v in user2idx.items()},
        'idx2item': {v: k for k, v in item2idx.items()}
    }
    with open(output_dir / 'id_mappings.json', 'w') as f:
        json.dump(mappings, f)
    logger.info(f"Saved ID mappings to {output_dir / 'id_mappings.json'}")
    
    # Save statistics
    stats = {
        'num_users': len(user2idx),
        'num_items': len(item2idx),
        'num_interactions': len(data),
        'density': len(data) / (len(user2idx) * len(item2idx)),
        'rating_distribution': df['rating'].value_counts().to_dict()
    }
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {output_dir / 'dataset_stats.json'}")
    
    return df


def main():
    """Run the full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # File paths
    raw_data_path = config.DATA_DIR / 'Electronics_5.json'
    output_dir = config.OUTPUT_DIR / 'processed'
    
    # Step 1: Load raw data
    data = load_raw_data(raw_data_path)
    
    # Step 2: Compute item statistics
    item_stats = compute_item_statistics(data)
    item_stats.to_csv(output_dir.parent / 'item_statistics.csv', index=False)
    
    # Step 3: Select high-variance items (Step A)
    selected_items = select_high_variance_items(
        item_stats,
        min_ratings=config.MIN_ITEM_RATINGS_FOR_VARIANCE,
        top_k=config.TOP_VARIANCE_ITEMS
    )
    
    # Step 4: Filter interactions for selected items (Step B)
    filtered_data = filter_interactions(data, selected_items)
    
    # Step 5: Enforce minimum activity (Step C)
    clean_data = enforce_minimum_activity(
        filtered_data,
        min_user_interactions=config.MIN_USER_INTERACTIONS,
        min_item_interactions=config.MIN_ITEM_INTERACTIONS
    )
    
    # Step 6: Create ID mappings and save
    user2idx, item2idx = create_id_mappings(clean_data)
    df = save_processed_data(clean_data, user2idx, item2idx, output_dir)
    
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    
    return df


if __name__ == '__main__':
    main()
