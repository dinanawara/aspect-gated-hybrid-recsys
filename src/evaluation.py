"""
Evaluation metrics for recommender systems.

Implements:
- Hit Rate @ K (HR@K)
- Normalized Discounted Cumulative Gain @ K (NDCG@K)
- Mean Reciprocal Rank (MRR)
- Precision @ K
- Recall @ K
"""
import logging
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def hit_rate_at_k(recommendations: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Hit Rate @ K: 1 if any ground truth item is in top-k, else 0.
    """
    top_k = recommendations[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def ndcg_at_k(recommendations: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    """
    top_k = recommendations[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i + 2 because i is 0-indexed
    
    # Ideal DCG (all relevant items at top)
    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommendations: List[int], ground_truth: Set[int]) -> float:
    """
    Mean Reciprocal Rank: 1/position of first relevant item.
    """
    for i, item in enumerate(recommendations):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(recommendations: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Precision @ K: fraction of top-k that are relevant.
    """
    top_k = recommendations[:k]
    relevant = sum(1 for item in top_k if item in ground_truth)
    return relevant / k


def recall_at_k(recommendations: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Recall @ K: fraction of relevant items that appear in top-k.
    """
    if not ground_truth:
        return 0.0
    top_k = recommendations[:k]
    relevant = sum(1 for item in top_k if item in ground_truth)
    return relevant / len(ground_truth)


def evaluate_model(
    model,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    device: str = 'cpu',
    sample_users: int = None
) -> Dict[str, float]:
    """
    Evaluate a recommendation model on test data.
    
    Args:
        model: Trained BPR-MF model
        test_data: Test interactions
        train_data: Training interactions (to exclude from recommendations)
        k_values: List of K values for metrics
        device: Device for model inference
        sample_users: If set, sample this many users for faster evaluation
    
    Returns:
        Dictionary of metric name -> value
    """
    model.eval()
    
    # Build user test sets
    test_user_items = test_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    users = list(test_user_items.keys())
    if sample_users and sample_users < len(users):
        users = np.random.choice(users, sample_users, replace=False).tolist()
    
    # Initialize metric accumulators
    metrics = {f'{m}@{k}': [] for m in ['hr', 'ndcg', 'precision', 'recall'] for k in k_values}
    metrics['mrr'] = []
    
    max_k = max(k_values)
    
    # Precompute all item scores for efficiency
    with torch.no_grad():
        all_items = torch.arange(model.num_items, device=device)
        item_embeddings = model.item_embeddings(all_items)
        item_biases = model.item_biases(all_items).squeeze(-1)
    
    for user_idx in tqdm(users, desc="Evaluating"):
        ground_truth = test_user_items.get(user_idx, set())
        if not ground_truth:
            continue
        
        train_items = train_user_items.get(user_idx, set())
        
        # Get predictions
        with torch.no_grad():
            user = torch.tensor([user_idx], device=device)
            user_emb = model.user_embeddings(user)
            
            scores = (user_emb @ item_embeddings.T).squeeze() + item_biases
            scores = scores.cpu().numpy()
        
        # Mask training items
        for item in train_items:
            if item < len(scores):
                scores[item] = float('-inf')
        
        # Get top-k recommendations
        recommendations = np.argsort(scores)[::-1][:max_k].tolist()
        
        # Compute metrics
        for k in k_values:
            metrics[f'hr@{k}'].append(hit_rate_at_k(recommendations, ground_truth, k))
            metrics[f'ndcg@{k}'].append(ndcg_at_k(recommendations, ground_truth, k))
            metrics[f'precision@{k}'].append(precision_at_k(recommendations, ground_truth, k))
            metrics[f'recall@{k}'].append(recall_at_k(recommendations, ground_truth, k))
        
        metrics['mrr'].append(mrr(recommendations, ground_truth))
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items() if v}


def evaluate_agent_accuracy(
    df: pd.DataFrame,
    manual_labels: pd.DataFrame,
    label_col: str = 'llm_label'
) -> Dict[str, float]:
    """
    Evaluate agent judgment accuracy against manual labels.
    
    Args:
        df: DataFrame with LLM predictions
        manual_labels: DataFrame with manual labels
        label_col: Column name for predicted labels
    
    Returns:
        Precision, Recall, F1 for dissonance detection
    """
    # Merge predictions and labels
    merged = df.merge(
        manual_labels[['user_id', 'item_id', 'manual_label']],
        on=['user_id', 'item_id'],
        how='inner'
    )
    
    if len(merged) == 0:
        logger.warning("No matching labels found!")
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    # Binary: dissonant vs consistent
    pred_dissonant = merged[label_col].isin(['strongly_dissonant', 'logistics_complaint', 'incentivized'])
    true_dissonant = merged['manual_label'].isin(['dissonant', 'strongly_dissonant', 'logistics_complaint', 'incentivized'])
    
    tp = (pred_dissonant & true_dissonant).sum()
    fp = (pred_dissonant & ~true_dissonant).sum()
    fn = (~pred_dissonant & true_dissonant).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Agent Accuracy Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    
    # Per-class breakdown
    logger.info(f"\nPer-class breakdown:")
    for label in merged[label_col].unique():
        count = (merged[label_col] == label).sum()
        logger.info(f"  {label}: {count}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': len(merged)
    }


def create_manual_labeling_sample(
    df: pd.DataFrame,
    n_samples: int = 100,
    stratify_by: str = 'mismatch_type'
) -> pd.DataFrame:
    """
    Create a sample for manual labeling.
    
    Exports a stratified sample to a CSV for manual annotation.
    """
    if stratify_by and stratify_by in df.columns:
        # Stratified sampling
        samples = []
        per_stratum = n_samples // df[stratify_by].nunique()
        
        for stratum, group in df.groupby(stratify_by):
            n = min(len(group), per_stratum)
            samples.append(group.sample(n, random_state=config.RANDOM_SEED))
        
        sample_df = pd.concat(samples, ignore_index=True)
        
        # Fill remaining
        remaining = n_samples - len(sample_df)
        if remaining > 0:
            unsampled = df[~df.index.isin(sample_df.index)]
            extra = unsampled.sample(min(remaining, len(unsampled)), random_state=config.RANDOM_SEED)
            sample_df = pd.concat([sample_df, extra], ignore_index=True)
    else:
        sample_df = df.sample(min(n_samples, len(df)), random_state=config.RANDOM_SEED)
    
    # Select columns for labeling
    output_cols = [
        'user_id', 'item_id', 'rating', 'review_text', 'summary',
        'sentiment_compound', 'mismatch_type', 'priority_score'
    ]
    available_cols = [c for c in output_cols if c in sample_df.columns]
    
    sample_df = sample_df[available_cols].copy()
    sample_df['manual_label'] = ''  # Empty column for manual annotation
    sample_df['notes'] = ''
    
    # Save for annotation
    output_path = config.OUTPUT_DIR / 'manual_labeling_sample.csv'
    sample_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(sample_df)} samples to {output_path}")
    logger.info("Please annotate the 'manual_label' column with:")
    logger.info("  - consistent")
    logger.info("  - mild_inconsistent")
    logger.info("  - strongly_dissonant")
    logger.info("  - logistics_complaint")
    logger.info("  - incentivized")
    
    return sample_df


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table of results."""
    models = list(results.keys())
    metrics = list(list(results.values())[0].keys())
    
    # Header
    header = "Model".ljust(20) + "".join(m.ljust(12) for m in metrics)
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Rows
    for model in models:
        row = model.ljust(20)
        for metric in metrics:
            val = results[model].get(metric, 0)
            row += f"{val:.4f}".ljust(12)
        print(row)
    
    print("=" * len(header))
    
    # Improvement row if multiple models
    if len(models) > 1:
        print("\nImprovement over baseline:")
        baseline = models[0]
        for model in models[1:]:
            for metric in metrics:
                baseline_val = results[baseline].get(metric, 0)
                model_val = results[model].get(metric, 0)
                if baseline_val > 0:
                    pct = (model_val - baseline_val) / baseline_val * 100
                    print(f"  {metric}: {pct:+.2f}%")


if __name__ == '__main__':
    # Example: Create labeling sample
    prefilter_path = config.OUTPUT_DIR / 'prefilter_results.parquet'
    if prefilter_path.exists():
        df = pd.read_parquet(prefilter_path)
        create_manual_labeling_sample(df, n_samples=100)
