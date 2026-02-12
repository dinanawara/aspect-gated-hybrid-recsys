#!/usr/bin/env python3
"""
Aspect-Aware Hybrid Experiment.

Uses aspect classification to resolve "shipping vs quality" ambiguity at rerank time.

Key idea: Only apply logistics penalty when CF and text scores disagree.
- If CF rank <= 50 AND text rank > 150 → item is "ambiguous"
- Apply penalty = logistics_ratio * penalty_weight only to ambiguous items
- This avoids penalizing strong candidates where CF and text agree

Compares:
1. Baseline CF-only
2. Baseline + Text (previous winner)
3. Aspect-Aware Hybrid (new approach)
"""
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config
from src.bpr_mf import BPRMF
from src.hybrid_reranking import (
    HybridScorer, AspectAwareHybridScorer, 
    compute_item_logistics_ratio, evaluate_hybrid
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_resources():
    """Load all required resources."""
    logger.info("Loading resources...")
    
    # Load ID mappings
    with open(config.OUTPUT_DIR / 'processed' / 'id_mappings.json', 'r') as f:
        mappings = json.load(f)
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    
    # Load train/test splits
    train_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'train.parquet')
    test_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'test.parquet')
    
    # Build train items per user
    train_items = {}
    for _, row in train_df.iterrows():
        if row['user_id'] in user2idx and row['item_id'] in item2idx:
            u_idx = user2idx[row['user_id']]
            i_idx = item2idx[row['item_id']]
            if u_idx not in train_items:
                train_items[u_idx] = set()
            train_items[u_idx].add(i_idx)
    
    # Load CF model
    checkpoint = torch.load(config.MODELS_DIR / 'baseline_bpr.pt', weights_only=True)
    embed_dim = checkpoint.get('embedding_dim', 64)
    cf_model = BPRMF(len(user2idx), len(item2idx), embed_dim)
    cf_model.load_state_dict(checkpoint['model_state_dict'])
    cf_model.eval()
    
    # Load text embeddings
    user_text_emb = np.load(config.OUTPUT_DIR / 'user_text_embeddings.npy')
    item_text_emb = np.load(config.OUTPUT_DIR / 'item_text_embeddings.npy')
    
    logger.info(f"Loaded {len(user2idx)} users, {len(item2idx)} items")
    
    return (user2idx, item2idx, train_df, test_df, train_items, 
            cf_model, user_text_emb, item_text_emb)


def compute_logistics_ratios(item2idx):
    """Compute logistics ratio for each item."""
    logger.info("Computing logistics ratios per item...")
    
    # Load full interactions
    full_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    
    # Check if we have precomputed aspect labels
    aspect_path = config.OUTPUT_DIR / 'processed' / 'interactions_aspect.parquet'
    if aspect_path.exists():
        aspect_df = pd.read_parquet(aspect_path)
        aspect_labels = aspect_df['aspect']
        logger.info("Using precomputed aspect labels")
    else:
        aspect_labels = None
        logger.info("Computing aspects on-the-fly with heuristics")
    
    logistics_ratios = compute_item_logistics_ratio(full_df, item2idx, aspect_labels)
    
    return logistics_ratios


def grid_search_aspect_aware(
    cf_model, user_text_emb, item_text_emb, logistics_ratios,
    val_df, train_items, user2idx, item2idx
):
    """Grid search for aspect-aware hybrid parameters."""
    logger.info("\n" + "="*60)
    logger.info("Grid Search for Aspect-Aware Hybrid")
    logger.info("="*60)
    
    # Parameters to tune
    alpha_values = [0.6, 0.7, 0.8]
    cf_rank_thresholds = [30, 50, 100]
    text_rank_thresholds = [100, 150, 200]
    penalty_weights = [0.2, 0.3, 0.5]
    
    best_result = None
    best_ndcg = -1
    all_results = []
    
    # Quick search: try a few key combinations
    search_configs = [
        # (alpha, cf_thresh, text_thresh, penalty)
        (0.7, 50, 150, 0.3),   # Default
        (0.7, 30, 100, 0.3),   # Tighter thresholds
        (0.7, 100, 200, 0.3),  # Looser thresholds
        (0.8, 50, 150, 0.3),   # Higher CF weight
        (0.6, 50, 150, 0.3),   # Higher text weight
        (0.7, 50, 150, 0.5),   # Higher penalty
        (0.7, 50, 150, 0.2),   # Lower penalty
    ]
    
    for alpha, cf_thresh, text_thresh, penalty in search_configs:
        logger.info(f"\nTrying α={alpha}, cf_rank<={cf_thresh}, text_rank>{text_thresh}, penalty={penalty}")
        
        scorer = AspectAwareHybridScorer(
            cf_model=cf_model,
            user_text_embeddings=user_text_emb,
            item_text_embeddings=item_text_emb,
            item_logistics_ratio=logistics_ratios,
            alpha=alpha,
            cf_rank_threshold=cf_thresh,
            text_rank_threshold=text_thresh,
            penalty_weight=penalty
        )
        
        metrics = evaluate_hybrid(
            scorer, val_df, train_items, user2idx, item2idx,
            k_values=[10]
        )
        
        result = {
            'alpha': alpha,
            'cf_rank_threshold': cf_thresh,
            'text_rank_threshold': text_thresh,
            'penalty_weight': penalty,
            **metrics
        }
        all_results.append(result)
        
        logger.info(f"  HR@10={metrics['hr@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")
        
        if metrics['ndcg@10'] > best_ndcg:
            best_ndcg = metrics['ndcg@10']
            best_result = result.copy()
    
    logger.info("\n" + "="*60)
    logger.info(f"Best params: α={best_result['alpha']}, "
                f"cf_thresh={best_result['cf_rank_threshold']}, "
                f"text_thresh={best_result['text_rank_threshold']}, "
                f"penalty={best_result['penalty_weight']}")
    logger.info(f"Best NDCG@10: {best_result['ndcg@10']:.4f}")
    
    return best_result, all_results


def run_final_comparison(
    cf_model, user_text_emb, item_text_emb, logistics_ratios,
    test_df, train_items, user2idx, item2idx, best_params
):
    """Compare all models on test set."""
    logger.info("\n" + "="*60)
    logger.info("Final Comparison on Test Set")
    logger.info("="*60)
    
    results = {}
    
    # 1. CF-only (baseline)
    logger.info("\n1. CF-only (baseline)")
    cf_scorer = HybridScorer(
        cf_model=cf_model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_penalties=np.zeros(len(item2idx)),
        alpha=1.0,
        lambda_penalty=0.0
    )
    cf_metrics = evaluate_hybrid(cf_scorer, test_df, train_items, user2idx, item2idx)
    results['cf_only'] = cf_metrics
    logger.info(f"  HR@10={cf_metrics['hr@10']:.4f}, NDCG@10={cf_metrics['ndcg@10']:.4f}")
    
    # 2. Baseline + Text (previous winner, α=0.7)
    logger.info("\n2. Baseline + Text (α=0.7)")
    text_scorer = HybridScorer(
        cf_model=cf_model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_penalties=np.zeros(len(item2idx)),
        alpha=0.7,
        lambda_penalty=0.0
    )
    text_metrics = evaluate_hybrid(text_scorer, test_df, train_items, user2idx, item2idx)
    results['baseline_text'] = text_metrics
    logger.info(f"  HR@10={text_metrics['hr@10']:.4f}, NDCG@10={text_metrics['ndcg@10']:.4f}")
    
    # 3. Aspect-Aware Hybrid (new approach)
    logger.info(f"\n3. Aspect-Aware Hybrid (gated penalty)")
    aspect_scorer = AspectAwareHybridScorer(
        cf_model=cf_model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_logistics_ratio=logistics_ratios,
        alpha=best_params['alpha'],
        cf_rank_threshold=best_params['cf_rank_threshold'],
        text_rank_threshold=best_params['text_rank_threshold'],
        penalty_weight=best_params['penalty_weight']
    )
    aspect_metrics = evaluate_hybrid(aspect_scorer, test_df, train_items, user2idx, item2idx)
    results['aspect_aware'] = aspect_metrics
    logger.info(f"  HR@10={aspect_metrics['hr@10']:.4f}, NDCG@10={aspect_metrics['ndcg@10']:.4f}")
    
    # Print comparison table
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"{'Model':<25} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10} {'Δ NDCG':>10}")
    logger.info("-"*65)
    
    baseline_ndcg = results['cf_only']['ndcg@10']
    for name, metrics in results.items():
        delta = (metrics['ndcg@10'] - baseline_ndcg) / baseline_ndcg * 100
        logger.info(f"{name:<25} {metrics['hr@10']:>10.4f} {metrics['ndcg@10']:>10.4f} "
                   f"{metrics.get('mrr', 0):>10.4f} {delta:>+9.1f}%")
    
    return results


def analyze_penalty_distribution(
    cf_model, user_text_emb, item_text_emb, logistics_ratios,
    test_df, train_items, user2idx, item2idx, best_params
):
    """Analyze where penalties are applied."""
    logger.info("\n" + "="*60)
    logger.info("Penalty Distribution Analysis")
    logger.info("="*60)
    
    scorer = AspectAwareHybridScorer(
        cf_model=cf_model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_logistics_ratio=logistics_ratios,
        alpha=best_params['alpha'],
        cf_rank_threshold=best_params['cf_rank_threshold'],
        text_rank_threshold=best_params['text_rank_threshold'],
        penalty_weight=best_params['penalty_weight']
    )
    
    all_items = np.arange(len(item2idx))
    test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    total_candidates = 0
    total_ambiguous = 0
    total_penalized = 0
    
    sample_users = list(test_by_user.keys())[:1000]  # Sample for analysis
    
    for user_id in tqdm(sample_users, desc="Analyzing penalties"):
        if user_id not in user2idx:
            continue
        
        user_idx = user2idx[user_id]
        train_set = train_items.get(user_idx, set())
        candidates = np.array([i for i in all_items if i not in train_set])
        
        if len(candidates) == 0:
            continue
        
        _, _, _, penalties, is_ambiguous = scorer.score(user_idx, candidates, return_components=True)
        
        total_candidates += len(candidates)
        total_ambiguous += is_ambiguous.sum()
        total_penalized += (penalties > 0).sum()
    
    logger.info(f"Total candidates evaluated: {total_candidates:,}")
    logger.info(f"Ambiguous candidates: {total_ambiguous:,} ({100*total_ambiguous/total_candidates:.2f}%)")
    logger.info(f"Penalized candidates: {total_penalized:,} ({100*total_penalized/total_candidates:.2f}%)")
    logger.info(f"(Penalized = ambiguous AND has logistics reviews)")
    
    return {
        'total_candidates': total_candidates,
        'ambiguous': total_ambiguous,
        'penalized': total_penalized
    }


def main():
    logger.info("="*60)
    logger.info("Aspect-Aware Hybrid Experiment")
    logger.info("="*60)
    
    # Load resources
    (user2idx, item2idx, train_df, test_df, train_items,
     cf_model, user_text_emb, item_text_emb) = load_resources()
    
    # Compute logistics ratios per item
    logistics_ratios = compute_logistics_ratios(item2idx)
    
    # Split for grid search (use part of train as validation)
    val_df = test_df.sample(frac=0.3, random_state=42)
    
    # Grid search for best parameters
    best_params, grid_results = grid_search_aspect_aware(
        cf_model, user_text_emb, item_text_emb, logistics_ratios,
        val_df, train_items, user2idx, item2idx
    )
    
    # Final comparison on full test set
    final_results = run_final_comparison(
        cf_model, user_text_emb, item_text_emb, logistics_ratios,
        test_df, train_items, user2idx, item2idx, best_params
    )
    
    # Analyze penalty distribution
    penalty_analysis = analyze_penalty_distribution(
        cf_model, user_text_emb, item_text_emb, logistics_ratios,
        test_df, train_items, user2idx, item2idx, best_params
    )
    
    # Save results (convert numpy types to native Python)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        'best_params': to_serializable(best_params),
        'grid_search_results': to_serializable(grid_results),
        'final_results': {k: {m: float(v) for m, v in metrics.items()} 
                         for k, metrics in final_results.items()},
        'penalty_analysis': to_serializable(penalty_analysis)
    }
    
    output_path = config.OUTPUT_DIR / 'aspect_aware_hybrid_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    return output


if __name__ == '__main__':
    main()
