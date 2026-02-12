"""
Run aspect classification with heuristics only (fast).
Then run the full experiment comparing aspect-weighted CF to baseline.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.aspect_classifier import AspectClassifier
from src.bpr_mf import BPRMF, BPRTrainer


def classify_aspects():
    """Classify all reviews by aspect using heuristics."""
    print("=" * 60)
    print("Step 1: Aspect Classification")
    print("=" * 60)
    
    df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    print(f"Loaded {len(df)} interactions")
    
    classifier = AspectClassifier(use_llm=False)
    
    aspects = []
    confidences = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        text = row.get('review_text', '')
        aspect, conf = classifier.classify_heuristic(str(text) if pd.notna(text) else '')
        aspects.append(aspect)
        confidences.append(conf)
    
    df['aspect'] = aspects
    df['aspect_confidence'] = confidences
    df['aspect_weight'] = df['aspect'].map(classifier.ASPECT_WEIGHTS)
    df['aspect_source'] = 'heuristic'
    
    print("\nAspect Distribution:")
    for aspect, count in df['aspect'].value_counts().items():
        weight = classifier.ASPECT_WEIGHTS.get(aspect, 1.0)
        pct = count / len(df) * 100
        print(f"  {aspect}: {count} ({pct:.1f}%) -> weight={weight}")
    
    output_path = config.OUTPUT_DIR / 'processed' / 'interactions_aspect.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Mean aspect weight: {df['aspect_weight'].mean():.3f}")
    
    return df


def evaluate_model(model, test_df, train_items, user2idx, item2idx, k_values=[5, 10, 20]):
    """Evaluate model on test set."""
    model.eval()
    device = next(model.parameters()).device
    
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
        
        with torch.no_grad():
            user_emb = model.user_embeddings.weight[user_idx]
            scores = torch.matmul(model.item_embeddings.weight, user_emb).cpu().numpy()
        
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


def run_experiment(df):
    """Run aspect-weighted CF experiment."""
    print("\n" + "=" * 60)
    print("Step 2: Training and Evaluation")
    print("=" * 60)
    
    with open(config.OUTPUT_DIR / 'processed' / 'id_mappings.json') as f:
        mappings = json.load(f)
    
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    # Create splits
    train_path = config.OUTPUT_DIR / 'processed' / 'train.parquet'
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'val.parquet')
        test_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'test.parquet')
    else:
        train_rows, val_rows, test_rows = [], [], []
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
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Merge aspect weights
    if 'aspect_weight' not in train_df.columns:
        aspect_map = df.set_index(['user_id', 'item_id'])['aspect_weight'].to_dict()
        train_df['aspect_weight'] = train_df.apply(
            lambda r: aspect_map.get((r['user_id'], r['item_id']), 1.0), axis=1
        )
    
    # Build train items
    train_items = {}
    for _, row in train_df.iterrows():
        user_idx = user2idx[row['user_id']]
        item_idx = item2idx[row['item_id']]
        if user_idx not in train_items:
            train_items[user_idx] = set()
        train_items[user_idx].add(item_idx)
    
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    # Model 1: Baseline
    print("\n" + "-" * 40)
    print("Training Baseline (uniform weights)")
    print("-" * 40)
    
    baseline = BPRMF(num_users, num_items, embedding_dim=64).to(device)
    trainer = BPRTrainer(baseline, device=device)
    
    train_baseline = train_df.copy()
    train_baseline['weight'] = 1.0
    
    trainer.train(train_baseline, epochs=30, batch_size=1024, lr=0.001, weight_col='weight')
    
    baseline_metrics = evaluate_model(baseline, test_df, train_items, user2idx, item2idx)
    results['Baseline'] = baseline_metrics
    print(f"Baseline: HR@10={baseline_metrics['hr@10']:.4f}, NDCG@10={baseline_metrics['ndcg@10']:.4f}")
    
    # Model 2: Aspect-Weighted
    print("\n" + "-" * 40)
    print("Training Aspect-Weighted CF")
    print("-" * 40)
    
    aspect_model = BPRMF(num_users, num_items, embedding_dim=64).to(device)
    trainer = BPRTrainer(aspect_model, device=device)
    
    trainer.train(train_df, epochs=30, batch_size=1024, lr=0.001, weight_col='aspect_weight')
    
    aspect_metrics = evaluate_model(aspect_model, test_df, train_items, user2idx, item2idx)
    results['Aspect-Weighted'] = aspect_metrics
    print(f"Aspect-Weighted: HR@10={aspect_metrics['hr@10']:.4f}, NDCG@10={aspect_metrics['ndcg@10']:.4f}")
    
    # Model 3: Hybrid with text
    print("\n" + "-" * 40)
    print("Evaluating Aspect + Text Hybrid")
    print("-" * 40)
    
    user_text_emb = np.load(config.OUTPUT_DIR / 'user_text_embeddings.npy')
    item_text_emb = np.load(config.OUTPUT_DIR / 'item_text_embeddings.npy')
    
    def hybrid_eval(cf_model, alpha=0.7):
        cf_model.eval()
        metrics = {'hr@10': [], 'ndcg@10': [], 'mrr': []}
        test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
        
        for user_id, ground_truth_ids in tqdm(test_by_user.items(), desc="Hybrid eval"):
            if user_id not in user2idx:
                continue
            user_idx = user2idx[user_id]
            ground_truth = {item2idx[iid] for iid in ground_truth_ids if iid in item2idx}
            if not ground_truth:
                continue
            train_set = train_items.get(user_idx, set())
            
            with torch.no_grad():
                user_emb = cf_model.user_embeddings.weight[user_idx]
                cf_scores = torch.matmul(cf_model.item_embeddings.weight, user_emb).cpu().numpy()
            
            text_scores = item_text_emb @ user_text_emb[user_idx]
            
            cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
            text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
            
            scores = alpha * cf_norm + (1 - alpha) * text_norm
            for item_idx in train_set:
                scores[item_idx] = -np.inf
            
            ranked = np.argsort(-scores)
            top_k = set(ranked[:10])
            
            metrics['hr@10'].append(1.0 if top_k & ground_truth else 0.0)
            
            rel = [1.0 if item in ground_truth else 0.0 for item in ranked[:10]]
            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(ground_truth))))
            metrics['ndcg@10'].append(dcg / idcg if idcg > 0 else 0.0)
            
            for i, item in enumerate(ranked):
                if item in ground_truth:
                    metrics['mrr'].append(1.0 / (i + 1))
                    break
            else:
                metrics['mrr'].append(0.0)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    hybrid_metrics = hybrid_eval(aspect_model, alpha=0.7)
    results['Aspect+Text'] = hybrid_metrics
    print(f"Aspect+Text: HR@10={hybrid_metrics['hr@10']:.4f}, NDCG@10={hybrid_metrics['ndcg@10']:.4f}")
    
    baseline_hybrid = hybrid_eval(baseline, alpha=0.7)
    results['Baseline+Text'] = baseline_hybrid
    print(f"Baseline+Text: HR@10={baseline_hybrid['hr@10']:.4f}, NDCG@10={baseline_hybrid['ndcg@10']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    baseline_ndcg = results['Baseline']['ndcg@10']
    print(f"{'Model':<20} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10} {'Δ NDCG':>10}")
    print("-" * 60)
    
    for name, m in results.items():
        delta = (m['ndcg@10'] - baseline_ndcg) / baseline_ndcg * 100
        print(f"{name:<20} {m['hr@10']:>10.4f} {m['ndcg@10']:>10.4f} {m['mrr']:>10.4f} {delta:>+10.1f}%")
    
    # Save
    with open(config.OUTPUT_DIR / 'aspect_experiment_results.json', 'w') as f:
        json.dump({'results': results, 'aspect_dist': df['aspect'].value_counts().to_dict()}, f, indent=2)
    
    return results


if __name__ == '__main__':
    aspect_path = config.OUTPUT_DIR / 'processed' / 'interactions_aspect.parquet'
    
    if aspect_path.exists():
        print("Loading existing aspect classifications...")
        df = pd.read_parquet(aspect_path)
    else:
        df = classify_aspects()
    
    run_experiment(df)
