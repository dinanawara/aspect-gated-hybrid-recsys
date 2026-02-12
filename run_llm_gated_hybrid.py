"""
LLM-Gated Hybrid Experiment.

Uses LLM to classify reviews as product/external/mixed,
then applies gated penalty only when:
1. LLM says "external" 
2. AND CF-Text disagree

This makes LLM the decision-maker for aspect classification.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from src.bpr_mf import BPRMF
from src.hybrid_reranking import (
    LLMGatedHybridScorer,
    AspectAwareHybridScorer,
    HybridScorer,
    compute_item_external_ratio_from_llm,
    compute_item_logistics_ratio
)
from src.llm_aspect_classifier import classify_all_reviews

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_resources() -> Tuple:
    """Load all required resources."""
    logger.info("Loading resources...")
    
    # Load mappings from JSON
    with open(config.OUTPUT_DIR / 'processed' / 'id_mappings.json', 'r') as f:
        mappings = json.load(f)
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    
    # Load data splits
    train_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'train.parquet')
    val_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'val.parquet')
    test_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'test.parquet')
    
    # Load full interactions for aspect classification
    interactions_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    
    # Build train items per user
    train_items = {}
    for _, row in train_df.iterrows():
        uid = user2idx.get(row['user_id'])
        iid = item2idx.get(row['item_id'])
        if uid is not None and iid is not None:
            if uid not in train_items:
                train_items[uid] = set()
            train_items[uid].add(iid)
    
    # Load CF model
    embed_dim = 64
    model = BPRMF(len(user2idx), len(item2idx), embed_dim)
    checkpoint = torch.load(config.MODELS_DIR / 'baseline_bpr.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load text embeddings
    user_text_emb = np.load(config.OUTPUT_DIR / 'user_text_embeddings.npy')
    item_text_emb = np.load(config.OUTPUT_DIR / 'item_text_embeddings.npy')
    
    logger.info(f"Loaded {len(user2idx)} users, {len(item2idx)} items")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return (
        model, user2idx, item2idx, 
        train_df, val_df, test_df, interactions_df,
        train_items, user_text_emb, item_text_emb
    )


async def run_llm_classification(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Run LLM classification on all reviews."""
    output_path = config.OUTPUT_DIR / 'processed' / 'llm_aspect_labels.parquet'
    
    # Check if cached
    if output_path.exists():
        logger.info("Loading cached LLM classifications...")
        return pd.read_parquet(output_path)
    
    # Run classification with sampling for cost control
    # Sample 5K reviews for quick testing (~$0.12)
    df = await classify_all_reviews(interactions_df, output_path, sample_size=5000)
    return df


def evaluate_scorer(
    scorer,
    test_df: pd.DataFrame,
    train_items: Dict[int, set],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    k: int = 10
) -> Dict[str, float]:
    """Evaluate a scorer on test set."""
    all_items = np.arange(len(item2idx))
    
    hr_list = []
    ndcg_list = []
    mrr_list = []
    
    test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    for user_id, ground_truth_ids in tqdm(test_by_user.items(), desc="Evaluating", leave=False):
        if user_id not in user2idx:
            continue
        
        user_idx = user2idx[user_id]
        ground_truth = {item2idx[iid] for iid in ground_truth_ids if iid in item2idx}
        if not ground_truth:
            continue
        
        # Get candidates
        train_set = train_items.get(user_idx, set())
        candidates = np.array([i for i in all_items if i not in train_set])
        if len(candidates) == 0:
            continue
        
        # Score and rank
        scores = scorer.score(user_idx, candidates)
        ranked_items = candidates[np.argsort(-scores)]
        
        # HR@k
        top_k = set(ranked_items[:k])
        hr = 1.0 if len(top_k & ground_truth) > 0 else 0.0
        hr_list.append(hr)
        
        # NDCG@k
        relevance = [1.0 if item in ground_truth else 0.0 for item in ranked_items[:k]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
        
        # MRR
        for i, item in enumerate(ranked_items):
            if item in ground_truth:
                mrr_list.append(1.0 / (i + 1))
                break
        else:
            mrr_list.append(0.0)
    
    return {
        f'hr@{k}': np.mean(hr_list),
        f'ndcg@{k}': np.mean(ndcg_list),
        'mrr': np.mean(mrr_list)
    }


def grid_search_llm_gated(
    model, user_text_emb, item_text_emb, external_ratios,
    val_df, train_items, user2idx, item2idx
) -> Dict:
    """Grid search for LLM-gated scorer parameters."""
    logger.info("="*60)
    logger.info("Grid Search for LLM-Gated Parameters")
    logger.info("="*60)
    
    best_params = None
    best_ndcg = -1
    all_results = []
    
    # Reduced grid for faster execution
    alpha_values = [0.6, 0.7]
    cf_thresholds = [50]
    text_thresholds = [150]
    penalty_weights = [0.3, 0.5]
    
    total_combos = len(alpha_values) * len(cf_thresholds) * len(text_thresholds) * len(penalty_weights)
    pbar = tqdm(total=total_combos, desc="Grid search")
    
    for alpha in alpha_values:
        for cf_thresh in cf_thresholds:
            for text_thresh in text_thresholds:
                for penalty in penalty_weights:
                    scorer = LLMGatedHybridScorer(
                        cf_model=model,
                        user_text_embeddings=user_text_emb,
                        item_text_embeddings=item_text_emb,
                        item_external_ratio=external_ratios,
                        alpha=alpha,
                        cf_rank_threshold=cf_thresh,
                        text_rank_threshold=text_thresh,
                        penalty_weight=penalty
                    )
                    
                    metrics = evaluate_scorer(scorer, val_df, train_items, user2idx, item2idx)
                    
                    result = {
                        'alpha': alpha,
                        'cf_thresh': cf_thresh,
                        'text_thresh': text_thresh,
                        'penalty': penalty,
                        **metrics
                    }
                    all_results.append(result)
                    
                    if metrics['ndcg@10'] > best_ndcg:
                        best_ndcg = metrics['ndcg@10']
                        best_params = result
                    
                    pbar.update(1)
    
    pbar.close()
    
    logger.info(f"\nBest params: alpha={best_params['alpha']}, "
                f"cf_thresh={best_params['cf_thresh']}, "
                f"text_thresh={best_params['text_thresh']}, "
                f"penalty={best_params['penalty']}")
    logger.info(f"Best NDCG@10: {best_ndcg:.4f}")
    
    return best_params, all_results


def to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


async def main():
    """Run the full LLM-gated experiment."""
    logger.info("="*60)
    logger.info("LLM-Gated Hybrid Experiment")
    logger.info("="*60)
    
    # Load resources
    (model, user2idx, item2idx, 
     train_df, val_df, test_df, interactions_df,
     train_items, user_text_emb, item_text_emb) = load_resources()
    
    # Step 1: Run LLM classification
    logger.info("\n" + "="*40)
    logger.info("Step 1: LLM Aspect Classification")
    logger.info("="*40)
    
    classified_df = await run_llm_classification(interactions_df)
    
    # Log distribution
    dist = classified_df['llm_aspect'].value_counts()
    logger.info(f"LLM Aspect distribution:\n{dist}")
    
    # Step 2: Compute external ratios from LLM labels
    logger.info("\n" + "="*40)
    logger.info("Step 2: Computing External Ratios")
    logger.info("="*40)
    
    external_ratios = compute_item_external_ratio_from_llm(
        classified_df, item2idx, classified_df['llm_aspect']
    )
    
    # Also compute heuristic ratios for comparison
    logistics_ratios = compute_item_logistics_ratio(interactions_df, item2idx)
    
    logger.info(f"LLM external ratio: mean={external_ratios.mean():.3f}, max={external_ratios.max():.3f}")
    logger.info(f"Heuristic logistics ratio: mean={logistics_ratios.mean():.3f}, max={logistics_ratios.max():.3f}")
    
    # Step 3: Grid search on validation
    logger.info("\n" + "="*40)
    logger.info("Step 3: Grid Search on Validation")
    logger.info("="*40)
    
    best_params, grid_results = grid_search_llm_gated(
        model, user_text_emb, item_text_emb, external_ratios,
        val_df, train_items, user2idx, item2idx
    )
    
    # Step 4: Final comparison on test set
    logger.info("\n" + "="*40)
    logger.info("Step 4: Final Comparison on Test Set")
    logger.info("="*40)
    
    # 1. CF-only baseline
    cf_scorer = HybridScorer(
        cf_model=model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_penalties=np.zeros(len(item2idx)),
        alpha=1.0,
        lambda_penalty=0.0
    )
    cf_metrics = evaluate_scorer(cf_scorer, test_df, train_items, user2idx, item2idx)
    logger.info(f"CF-only:        HR@10={cf_metrics['hr@10']:.4f}, NDCG@10={cf_metrics['ndcg@10']:.4f}")
    
    # 2. Baseline + Text (no penalty)
    text_scorer = HybridScorer(
        cf_model=model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_penalties=np.zeros(len(item2idx)),
        alpha=0.7,
        lambda_penalty=0.0
    )
    text_metrics = evaluate_scorer(text_scorer, test_df, train_items, user2idx, item2idx)
    logger.info(f"Baseline+Text:  HR@10={text_metrics['hr@10']:.4f}, NDCG@10={text_metrics['ndcg@10']:.4f}")
    
    # 3. Heuristic-gated (previous approach)
    heuristic_scorer = AspectAwareHybridScorer(
        cf_model=model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_logistics_ratio=logistics_ratios,
        alpha=0.6,
        cf_rank_threshold=50,
        text_rank_threshold=150,
        penalty_weight=0.3
    )
    heuristic_metrics = evaluate_scorer(heuristic_scorer, test_df, train_items, user2idx, item2idx)
    logger.info(f"Heuristic-Gated: HR@10={heuristic_metrics['hr@10']:.4f}, NDCG@10={heuristic_metrics['ndcg@10']:.4f}")
    
    # 4. LLM-gated (new approach)
    llm_scorer = LLMGatedHybridScorer(
        cf_model=model,
        user_text_embeddings=user_text_emb,
        item_text_embeddings=item_text_emb,
        item_external_ratio=external_ratios,
        alpha=best_params['alpha'],
        cf_rank_threshold=best_params['cf_thresh'],
        text_rank_threshold=best_params['text_thresh'],
        penalty_weight=best_params['penalty']
    )
    llm_metrics = evaluate_scorer(llm_scorer, test_df, train_items, user2idx, item2idx)
    logger.info(f"LLM-Gated:      HR@10={llm_metrics['hr@10']:.4f}, NDCG@10={llm_metrics['ndcg@10']:.4f}")
    
    # Compute improvements
    cf_ndcg = cf_metrics['ndcg@10']
    improvements = {
        'baseline_text': (text_metrics['ndcg@10'] - cf_ndcg) / cf_ndcg * 100,
        'heuristic_gated': (heuristic_metrics['ndcg@10'] - cf_ndcg) / cf_ndcg * 100,
        'llm_gated': (llm_metrics['ndcg@10'] - cf_ndcg) / cf_ndcg * 100
    }
    
    logger.info("\n" + "="*40)
    logger.info("Summary")
    logger.info("="*40)
    logger.info(f"CF-only baseline NDCG@10: {cf_ndcg:.4f}")
    logger.info(f"Baseline+Text improvement: +{improvements['baseline_text']:.1f}%")
    logger.info(f"Heuristic-Gated improvement: +{improvements['heuristic_gated']:.1f}%")
    logger.info(f"LLM-Gated improvement: +{improvements['llm_gated']:.1f}%")
    
    # Analyze penalty distribution for LLM-gated
    logger.info("\n" + "="*40)
    logger.info("LLM-Gated Penalty Analysis")
    logger.info("="*40)
    
    # Sample analysis on test users
    sample_users = list(test_df['user_id'].unique()[:100])
    all_items_arr = np.arange(len(item2idx))
    total_candidates = 0
    total_ambiguous = 0
    total_penalized = 0
    
    for user_id in sample_users:
        if user_id not in user2idx:
            continue
        user_idx = user2idx[user_id]
        train_set = train_items.get(user_idx, set())
        candidates = np.array([i for i in all_items_arr if i not in train_set])
        
        _, _, _, penalties, is_ambiguous = llm_scorer.score(user_idx, candidates, return_components=True)
        total_candidates += len(candidates)
        total_ambiguous += is_ambiguous.sum()
        total_penalized += (penalties > 0).sum()
    
    logger.info(f"Sample of {len(sample_users)} users:")
    logger.info(f"  Total candidates: {total_candidates}")
    logger.info(f"  Ambiguous: {total_ambiguous} ({100*total_ambiguous/total_candidates:.2f}%)")
    logger.info(f"  Penalized: {total_penalized} ({100*total_penalized/total_candidates:.2f}%)")
    
    # Save results
    results = {
        'experiment': 'LLM-Gated Hybrid',
        'llm_aspect_distribution': to_serializable(dist.to_dict()),
        'best_params': to_serializable(best_params),
        'test_metrics': {
            'cf_only': to_serializable(cf_metrics),
            'baseline_text': to_serializable(text_metrics),
            'heuristic_gated': to_serializable(heuristic_metrics),
            'llm_gated': to_serializable(llm_metrics)
        },
        'improvements': to_serializable(improvements),
        'external_ratio_stats': {
            'llm_mean': float(external_ratios.mean()),
            'llm_max': float(external_ratios.max()),
            'heuristic_mean': float(logistics_ratios.mean()),
            'heuristic_max': float(logistics_ratios.max())
        },
        'penalty_analysis': {
            'sample_users': len(sample_users),
            'total_candidates': int(total_candidates),
            'ambiguous': int(total_ambiguous),
            'ambiguous_pct': float(100*total_ambiguous/total_candidates),
            'penalized': int(total_penalized),
            'penalized_pct': float(100*total_penalized/total_candidates)
        }
    }
    
    output_path = config.OUTPUT_DIR / 'llm_gated_hybrid_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    asyncio.run(main())
