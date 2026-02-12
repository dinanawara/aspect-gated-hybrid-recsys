"""
Aspect-Conditioned Collaborative Filtering Experiment.

Key idea: Different review aspects should have different influence on CF:
- Product quality reviews -> full weight (1.0)
- Logistics complaints -> low weight (0.3) 
- Seller issues -> low weight (0.3)
- Mixed -> partial weight (0.7)

This separates interaction dimensions rather than penalizing mismatch.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.aspect_classifier import AspectClassifier
from src.bpr_mf import BPRMF, BPRTrainer
from src.text_embeddings import TextEmbedder, build_item_embeddings, build_user_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, test_df, train_items, user2idx, item2idx, k_values=[5, 10, 20]):
    """Evaluate a trained model."""
    model.eval()
    device = next(model.parameters()).device
    
    all_items = torch.arange(len(item2idx)).to(device)
    
    metrics = {f'hr@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics['mrr'] = []
    
    test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    for user_id, ground_truth_ids in tqdm(test_by_user.items(), desc="Evaluating"):
        if user_id not in user2idx:
            continue
        
        user_idx = user2idx[user_id]
        ground_truth = {item2idx[iid] for iid in ground_truth_ids if iid in item2idx}
        if not ground_truth:
            continue
        
        train_set = train_items.get(user_idx, set())
        
        # Get scores for all items
        with torch.no_grad():
            user_emb = model.user_embeddings.weight[user_idx]
            scores = torch.matmul(model.item_embeddings.weight, user_emb).cpu().numpy()
        
        # Mask train items
        for item_idx in train_set:
            scores[item_idx] = -np.inf
        
        ranked_items = np.argsort(-scores)
        
        for k in k_values:
            top_k = set(ranked_items[:k])
            hr = 1.0 if len(top_k & ground_truth) > 0 else 0.0
            metrics[f'hr@{k}'].append(hr)
            
            # NDCG
            relevance = [1.0 if item in ground_truth else 0.0 for item in ranked_items[:k]]
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'ndcg@{k}'].append(ndcg)
        
        # MRR
        for i, item in enumerate(ranked_items):
            if item in ground_truth:
                metrics['mrr'].append(1.0 / (i + 1))
                break
        else:
            metrics['mrr'].append(0.0)
    
    return {k: np.mean(v) for k, v in metrics.items()}


def run_experiment():
    """Run the aspect-conditioned CF experiment."""
    logger.info("=" * 60)
    logger.info("Aspect-Conditioned CF Experiment")
    logger.info("=" * 60)
    
    # Load data
    full_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    
    with open(config.OUTPUT_DIR / 'processed' / 'id_mappings.json') as f:
        mappings = json.load(f)
    
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    logger.info(f"Dataset: {len(full_df)} interactions, {num_users} users, {num_items} items")
    
    # Step 1: Classify reviews by aspect
    aspect_path = config.OUTPUT_DIR / 'processed' / 'interactions_aspect.parquet'
    
    if aspect_path.exists():
        logger.info("Loading existing aspect classifications...")
        df = pd.read_parquet(aspect_path)
    else:
        logger.info("Running aspect classification...")
        classifier = AspectClassifier(use_llm=True)
        df = classifier.classify_batch(full_df)
        df.to_parquet(aspect_path, index=False)
    
    # Show aspect distribution
    logger.info("\nAspect Distribution:")
    for aspect, count in df['aspect'].value_counts().items():
        weight = AspectClassifier.ASPECT_WEIGHTS.get(aspect, 1.0)
        logger.info(f"  {aspect}: {count} ({count/len(df)*100:.1f}%) -> weight={weight}")
    
    # Step 2: Create train/val/test splits
    train_path = config.OUTPUT_DIR / 'processed' / 'train.parquet'
    
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'val.parquet')
        test_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'test.parquet')
    else:
        # Leave-one-out split
        train_rows, val_rows, test_rows = [], [], []
        
        if 'timestamp' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
        
        for user_id, group in df.groupby('user_id'):
            n = len(group)
            if n < 3:
                train_rows.append(group)
            else:
                train_rows.append(group.iloc[:-2])
                val_rows.append(group.iloc[-2:-1])
                test_rows.append(group.iloc[-1:])
        
        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)
        
        train_df.to_parquet(train_path)
        val_df.to_parquet(config.OUTPUT_DIR / 'processed' / 'val.parquet')
        test_df.to_parquet(config.OUTPUT_DIR / 'processed' / 'test.parquet')
    
    logger.info(f"\nSplits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Merge aspect weights into train_df
    if 'aspect_weight' not in train_df.columns:
        aspect_map = df.set_index(['user_id', 'item_id'])['aspect_weight'].to_dict()
        train_df['aspect_weight'] = train_df.apply(
            lambda r: aspect_map.get((r['user_id'], r['item_id']), 1.0), axis=1
        )
    
    # Build train items dict
    train_items = {}
    for _, row in train_df.iterrows():
        user_idx = user2idx[row['user_id']]
        item_idx = item2idx[row['item_id']]
        if user_idx not in train_items:
            train_items[user_idx] = set()
        train_items[user_idx].add(item_idx)
    
    # Add indices
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nUsing device: {device}")
    
    results = {}
    
    # =====================================================
    # Model 1: Baseline (no weighting)
    # =====================================================
    logger.info("\n" + "=" * 50)
    logger.info("Training Model 1: Baseline (uniform weights)")
    logger.info("=" * 50)
    
    baseline_model = BPRMF(num_users, num_items, embedding_dim=64).to(device)
    trainer = BPRTrainer(baseline_model, device=device)
    
    # Train without weights
    train_df_baseline = train_df.copy()
    train_df_baseline['weight'] = 1.0
    
    trainer.train(
        train_data=train_df_baseline,
        epochs=30,
        batch_size=1024,
        lr=0.001,
        weight_col='weight'
    )
    
    baseline_metrics = evaluate_model(baseline_model, test_df, train_items, user2idx, item2idx)
    results['Baseline'] = baseline_metrics
    logger.info(f"Baseline: HR@10={baseline_metrics['hr@10']:.4f}, NDCG@10={baseline_metrics['ndcg@10']:.4f}")
    
    # =====================================================
    # Model 2: Aspect-Weighted CF
    # =====================================================
    logger.info("\n" + "=" * 50)
    logger.info("Training Model 2: Aspect-Weighted CF")
    logger.info("=" * 50)
    
    aspect_model = BPRMF(num_users, num_items, embedding_dim=64).to(device)
    trainer = BPRTrainer(aspect_model, device=device)
    
    # Train with aspect weights
    trainer.train(
        train_data=train_df,
        epochs=30,
        batch_size=1024,
        lr=0.001,
        weight_col='aspect_weight'
    )
    
    aspect_metrics = evaluate_model(aspect_model, test_df, train_items, user2idx, item2idx)
    results['Aspect-Weighted'] = aspect_metrics
    logger.info(f"Aspect-Weighted: HR@10={aspect_metrics['hr@10']:.4f}, NDCG@10={aspect_metrics['ndcg@10']:.4f}")
    
    # =====================================================
    # Model 3: Aspect-Weighted + Text Hybrid
    # =====================================================
    logger.info("\n" + "=" * 50)
    logger.info("Training Model 3: Aspect + Text Hybrid")
    logger.info("=" * 50)
    
    # Load text embeddings
    user_text_path = config.OUTPUT_DIR / 'user_text_embeddings.npy'
    item_text_path = config.OUTPUT_DIR / 'item_text_embeddings.npy'
    
    if user_text_path.exists() and item_text_path.exists():
        user_text_emb = np.load(user_text_path)
        item_text_emb = np.load(item_text_path)
    else:
        logger.info("Building text embeddings...")
        embedder = TextEmbedder()
        item_text_emb = build_item_embeddings(full_df, embedder, item2idx)
        user_text_emb = build_user_embeddings(full_df, embedder, user2idx)
        
        # Normalize
        item_text_emb /= (np.linalg.norm(item_text_emb, axis=1, keepdims=True) + 1e-8)
        user_text_emb /= (np.linalg.norm(user_text_emb, axis=1, keepdims=True) + 1e-8)
        
        np.save(user_text_path, user_text_emb)
        np.save(item_text_path, item_text_emb)
    
    # Hybrid scoring function
    def hybrid_evaluate(cf_model, test_df, train_items, user2idx, item2idx,
                       user_text_emb, item_text_emb, alpha=0.7, k_values=[5, 10, 20]):
        """Evaluate with hybrid CF + text scoring."""
        cf_model.eval()
        device = next(cf_model.parameters()).device
        
        metrics = {f'hr@{k}': [] for k in k_values}
        metrics.update({f'ndcg@{k}': [] for k in k_values})
        metrics['mrr'] = []
        
        test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
        
        for user_id, ground_truth_ids in tqdm(test_by_user.items(), desc="Hybrid eval"):
            if user_id not in user2idx:
                continue
            
            user_idx = user2idx[user_id]
            ground_truth = {item2idx[iid] for iid in ground_truth_ids if iid in item2idx}
            if not ground_truth:
                continue
            
            train_set = train_items.get(user_idx, set())
            
            # CF scores
            with torch.no_grad():
                user_emb = cf_model.user_embeddings.weight[user_idx]
                cf_scores = torch.matmul(cf_model.item_embeddings.weight, user_emb).cpu().numpy()
            
            # Text similarity scores
            text_scores = item_text_emb @ user_text_emb[user_idx]
            
            # Normalize
            cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
            text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
            
            # Hybrid
            scores = alpha * cf_norm + (1 - alpha) * text_norm
            
            # Mask train items
            for item_idx in train_set:
                scores[item_idx] = -np.inf
            
            ranked_items = np.argsort(-scores)
            
            for k in k_values:
                top_k = set(ranked_items[:k])
                hr = 1.0 if len(top_k & ground_truth) > 0 else 0.0
                metrics[f'hr@{k}'].append(hr)
                
                relevance = [1.0 if item in ground_truth else 0.0 for item in ranked_items[:k]]
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                metrics[f'ndcg@{k}'].append(ndcg)
            
            for i, item in enumerate(ranked_items):
                if item in ground_truth:
                    metrics['mrr'].append(1.0 / (i + 1))
                    break
            else:
                metrics['mrr'].append(0.0)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    hybrid_metrics = hybrid_evaluate(
        aspect_model, test_df, train_items, user2idx, item2idx,
        user_text_emb, item_text_emb, alpha=0.7
    )
    results['Aspect+Text Hybrid'] = hybrid_metrics
    logger.info(f"Aspect+Text Hybrid: HR@10={hybrid_metrics['hr@10']:.4f}, NDCG@10={hybrid_metrics['ndcg@10']:.4f}")
    
    # =====================================================
    # Model 4: Baseline + Text (no aspect weighting)
    # =====================================================
    logger.info("\n" + "=" * 50)
    logger.info("Evaluating Model 4: Baseline + Text (no aspect)")
    logger.info("=" * 50)
    
    baseline_hybrid_metrics = hybrid_evaluate(
        baseline_model, test_df, train_items, user2idx, item2idx,
        user_text_emb, item_text_emb, alpha=0.7
    )
    results['Baseline+Text'] = baseline_hybrid_metrics
    logger.info(f"Baseline+Text: HR@10={baseline_hybrid_metrics['hr@10']:.4f}, NDCG@10={baseline_hybrid_metrics['ndcg@10']:.4f}")
    
    # =====================================================
    # Results Summary
    # =====================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    
    baseline_ndcg = results['Baseline']['ndcg@10']
    
    logger.info(f"{'Model':<25} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10} {'Δ NDCG':>10}")
    logger.info("-" * 65)
    
    for name, metrics in results.items():
        delta = (metrics['ndcg@10'] - baseline_ndcg) / baseline_ndcg * 100
        logger.info(f"{name:<25} {metrics['hr@10']:>10.4f} {metrics['ndcg@10']:>10.4f} "
                   f"{metrics['mrr']:>10.4f} {delta:>+10.1f}%")
    
    # Save results
    output = {
        'results': results,
        'aspect_distribution': df['aspect'].value_counts().to_dict() if 'aspect' in df.columns else {},
        'aspect_weights': AspectClassifier.ASPECT_WEIGHTS
    }
    
    with open(config.OUTPUT_DIR / 'aspect_experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {config.OUTPUT_DIR / 'aspect_experiment_results.json'}")
    
    return results


if __name__ == '__main__':
    run_experiment()
