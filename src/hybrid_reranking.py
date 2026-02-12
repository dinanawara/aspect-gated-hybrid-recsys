"""
Hybrid Reranking System.

Implements: score(u,i) = α·CF(u,i) + (1-α)·text_sim(u,i) - λ·penalty(i)

Components:
1. CF scores from trained BPR-MF
2. Text similarity from sentence-transformers
3. Dissonance penalty per item (fraction of bad reviews)
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.bpr_mf import BPRMF
from src.evaluation import hit_rate_at_k, ndcg_at_k, mrr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_item_dissonance_penalty(
    df: pd.DataFrame,
    item2idx: Dict[str, int],
    prefilter_flags: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Compute per-item dissonance penalty.
    
    penalty(i) = fraction of item i's reviews that are:
    - Logistics-only complaints (high rating but logistics issues)
    - High VADER-rating mismatch
    - Flagged by pre-filter
    
    Args:
        df: Interactions DataFrame
        item2idx: Item ID to index mapping
        prefilter_flags: Optional DataFrame with pre-filter flags
    
    Returns:
        numpy array of shape (num_items,) with penalty ∈ [0, 1]
    """
    logger.info("Computing item dissonance penalties...")
    
    num_items = len(item2idx)
    item_penalties = np.zeros(num_items)
    
    # If we have pre-filter flags, use them
    if prefilter_flags is not None and 'should_flag' in prefilter_flags.columns:
        merged = df.merge(
            prefilter_flags[['reviewerID', 'asin', 'should_flag', 'logistics_only']],
            left_on=['user_id', 'item_id'],
            right_on=['reviewerID', 'asin'],
            how='left'
        )
        
        for item_id, group in merged.groupby('item_id'):
            if item_id not in item2idx:
                continue
            idx = item2idx[item_id]
            
            n_reviews = len(group)
            if n_reviews == 0:
                continue
            
            # Penalty = fraction that are flagged OR logistics-only
            n_flagged = group['should_flag'].fillna(False).sum()
            n_logistics = group['logistics_only'].fillna(False).sum()
            
            # Combine: flagged reviews + logistics complaints (weighted)
            penalty = (n_flagged + 0.5 * n_logistics) / n_reviews
            item_penalties[idx] = min(1.0, penalty)
    
    else:
        # Fall back to simple rating variance as proxy
        for item_id, group in df.groupby('item_id'):
            if item_id not in item2idx:
                continue
            idx = item2idx[item_id]
            
            # High variance = potentially noisy
            if len(group) > 1:
                rating_std = group['rating'].std()
                # Normalize: std of 2 -> penalty of 0.5
                item_penalties[idx] = min(1.0, rating_std / 4.0)
    
    logger.info(f"Penalty stats: mean={item_penalties.mean():.3f}, "
                f"max={item_penalties.max():.3f}, "
                f"items_with_penalty={(item_penalties > 0).sum()}")
    
    return item_penalties


class HybridScorer:
    """
    Hybrid recommendation scorer.
    
    score(u,i) = α·norm(CF(u,i)) + (1-α)·norm(text_sim(u,i)) - λ·penalty(i)
    """
    
    def __init__(
        self,
        cf_model: BPRMF,
        user_text_embeddings: np.ndarray,
        item_text_embeddings: np.ndarray,
        item_penalties: np.ndarray,
        alpha: float = 0.8,
        lambda_penalty: float = 0.1
    ):
        """
        Args:
            cf_model: Trained BPR-MF model
            user_text_embeddings: Shape (num_users, embed_dim)
            item_text_embeddings: Shape (num_items, embed_dim)
            item_penalties: Shape (num_items,)
            alpha: Weight for CF score (1-alpha for text)
            lambda_penalty: Weight for dissonance penalty
        """
        self.cf_model = cf_model
        self.user_emb = user_text_embeddings
        self.item_emb = item_text_embeddings
        self.item_penalties = item_penalties
        self.alpha = alpha
        self.lambda_penalty = lambda_penalty
        
        # Pre-compute item norms for cosine similarity
        self.item_norms = np.linalg.norm(item_text_embeddings, axis=1, keepdims=True)
        self.user_norms = np.linalg.norm(user_text_embeddings, axis=1, keepdims=True)
    
    def compute_cf_scores(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Get CF scores for user-item pairs."""
        user_emb = self.cf_model.user_embeddings.weight[user_idx]
        item_emb = self.cf_model.item_embeddings.weight[item_indices]
        
        with torch.no_grad():
            scores = torch.matmul(item_emb, user_emb).cpu().numpy()
        
        return scores
    
    def compute_text_similarity(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between user and items."""
        user_vec = self.user_emb[user_idx]  # (embed_dim,)
        item_vecs = self.item_emb[item_indices]  # (n_items, embed_dim)
        
        # Cosine similarity (already normalized in text_embeddings.py)
        similarities = item_vecs @ user_vec
        
        return similarities
    
    def score(
        self,
        user_idx: int,
        item_indices: np.ndarray,
        return_components: bool = False
    ) -> np.ndarray:
        """
        Compute hybrid scores.
        
        Args:
            user_idx: User index
            item_indices: Array of item indices to score
            return_components: If True, return (scores, cf, text, penalty)
        """
        # Get component scores
        cf_scores = self.compute_cf_scores(user_idx, item_indices)
        text_scores = self.compute_text_similarity(user_idx, item_indices)
        penalties = self.item_penalties[item_indices]
        
        # Normalize to [0, 1] using min-max
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
        text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
        
        # Hybrid score
        hybrid = self.alpha * cf_norm + (1 - self.alpha) * text_norm - self.lambda_penalty * penalties
        
        if return_components:
            return hybrid, cf_norm, text_norm, penalties
        return hybrid
    
    def rank_items(
        self,
        user_idx: int,
        candidate_items: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """Rank items and return top-k."""
        scores = self.score(user_idx, candidate_items)
        top_k_local = np.argsort(-scores)[:k]
        return candidate_items[top_k_local]


class AspectAwareHybridScorer:
    """
    Aspect-Aware Hybrid Scorer.
    
    Key insight: Only apply logistics penalty when CF and text disagree.
    
    Logic:
    1. Compute CF ranks and text ranks for all candidates
    2. Identify "ambiguous" items: CF rank high but text rank low (or vice versa)
    3. For ambiguous items only, apply penalty based on logistics_ratio
    
    This avoids penalizing items where CF and text agree (strong signal),
    and focuses on cases where the discrepancy might be due to logistics issues.
    """
    
    def __init__(
        self,
        cf_model: BPRMF,
        user_text_embeddings: np.ndarray,
        item_text_embeddings: np.ndarray,
        item_logistics_ratio: np.ndarray,
        alpha: float = 0.7,
        cf_rank_threshold: int = 50,
        text_rank_threshold: int = 150,
        penalty_weight: float = 0.3
    ):
        """
        Args:
            cf_model: Trained BPR-MF model
            user_text_embeddings: Shape (num_users, embed_dim)
            item_text_embeddings: Shape (num_items, embed_dim)
            item_logistics_ratio: Shape (num_items,) - fraction of logistics/seller reviews
            alpha: Weight for CF score in final combination
            cf_rank_threshold: Consider item "CF-liked" if rank <= this
            text_rank_threshold: Consider item "text-disliked" if rank > this
            penalty_weight: Weight for logistics penalty on ambiguous items
        """
        self.cf_model = cf_model
        self.user_emb = user_text_embeddings
        self.item_emb = item_text_embeddings
        self.logistics_ratio = item_logistics_ratio
        self.alpha = alpha
        self.cf_rank_threshold = cf_rank_threshold
        self.text_rank_threshold = text_rank_threshold
        self.penalty_weight = penalty_weight
    
    def compute_cf_scores(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Get CF scores for user-item pairs."""
        user_emb = self.cf_model.user_embeddings.weight[user_idx]
        item_emb = self.cf_model.item_embeddings.weight[item_indices]
        
        with torch.no_grad():
            scores = torch.matmul(item_emb, user_emb).cpu().numpy()
        return scores
    
    def compute_text_similarity(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between user and items."""
        user_vec = self.user_emb[user_idx]
        item_vecs = self.item_emb[item_indices]
        return item_vecs @ user_vec
    
    def score(
        self,
        user_idx: int,
        item_indices: np.ndarray,
        return_components: bool = False
    ) -> np.ndarray:
        """
        Compute aspect-aware hybrid scores with gated penalty.
        
        Steps:
        1. Compute CF and text scores
        2. Compute ranks for CF and text
        3. Identify ambiguous items (CF rank <= threshold AND text rank > threshold)
        4. Apply penalty only to ambiguous items
        5. Combine: α·CF_norm + (1-α)·text_norm - penalty
        """
        # Get raw scores
        cf_scores = self.compute_cf_scores(user_idx, item_indices)
        text_scores = self.compute_text_similarity(user_idx, item_indices)
        
        # Compute ranks (1-indexed, lower = better)
        cf_ranks = np.argsort(np.argsort(-cf_scores)) + 1
        text_ranks = np.argsort(np.argsort(-text_scores)) + 1
        
        # Identify ambiguous items
        # Case 1: CF likes it (rank <= cf_threshold) but text doesn't (rank > text_threshold)
        cf_likes_text_dislikes = (cf_ranks <= self.cf_rank_threshold) & (text_ranks > self.text_rank_threshold)
        
        # Case 2: Text likes it but CF doesn't (also ambiguous, but less common)
        text_likes_cf_dislikes = (text_ranks <= self.cf_rank_threshold) & (cf_ranks > self.text_rank_threshold)
        
        is_ambiguous = cf_likes_text_dislikes | text_likes_cf_dislikes
        
        # Compute penalties (only for ambiguous items)
        logistics_ratios = self.logistics_ratio[item_indices]
        gated_penalty = np.where(is_ambiguous, logistics_ratios * self.penalty_weight, 0.0)
        
        # Normalize scores to [0, 1]
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
        text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
        
        # Final hybrid score
        hybrid = self.alpha * cf_norm + (1 - self.alpha) * text_norm - gated_penalty
        
        if return_components:
            return hybrid, cf_norm, text_norm, gated_penalty, is_ambiguous
        return hybrid
    
    def rank_items(
        self,
        user_idx: int,
        candidate_items: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """Rank items and return top-k."""
        scores = self.score(user_idx, candidate_items)
        top_k_local = np.argsort(-scores)[:k]
        return candidate_items[top_k_local]


class LLMGatedHybridScorer:
    """
    LLM-Gated Hybrid Scorer.
    
    Key insight: LLM decides whether review is about product or external factors.
    Only apply penalty when:
    1. LLM says "external" (not product quality)
    2. AND CF and text disagree (ambiguous signal)
    
    This makes LLM the decision-maker for aspect classification.
    """
    
    def __init__(
        self,
        cf_model: BPRMF,
        user_text_embeddings: np.ndarray,
        item_text_embeddings: np.ndarray,
        item_external_ratio: np.ndarray,
        alpha: float = 0.7,
        cf_rank_threshold: int = 50,
        text_rank_threshold: int = 150,
        penalty_weight: float = 0.3
    ):
        """
        Args:
            cf_model: Trained BPR-MF model
            user_text_embeddings: Shape (num_users, embed_dim)
            item_text_embeddings: Shape (num_items, embed_dim)
            item_external_ratio: Shape (num_items,) - fraction of external reviews (from LLM)
            alpha: Weight for CF score in final combination
            cf_rank_threshold: Consider item "CF-liked" if rank <= this
            text_rank_threshold: Consider item "text-disliked" if rank > this
            penalty_weight: Weight for external penalty on ambiguous items
        """
        self.cf_model = cf_model
        self.user_emb = user_text_embeddings
        self.item_emb = item_text_embeddings
        self.external_ratio = item_external_ratio
        self.alpha = alpha
        self.cf_rank_threshold = cf_rank_threshold
        self.text_rank_threshold = text_rank_threshold
        self.penalty_weight = penalty_weight
    
    def compute_cf_scores(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Get CF scores for user-item pairs."""
        user_emb = self.cf_model.user_embeddings.weight[user_idx]
        item_emb = self.cf_model.item_embeddings.weight[item_indices]
        
        with torch.no_grad():
            scores = torch.matmul(item_emb, user_emb).cpu().numpy()
        return scores
    
    def compute_text_similarity(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between user and items."""
        user_vec = self.user_emb[user_idx]
        item_vecs = self.item_emb[item_indices]
        return item_vecs @ user_vec
    
    def score(
        self,
        user_idx: int,
        item_indices: np.ndarray,
        return_components: bool = False
    ) -> np.ndarray:
        """
        Compute LLM-gated hybrid scores.
        
        Steps:
        1. Compute CF and text scores
        2. Compute ranks for CF and text
        3. Identify ambiguous items (CF and text disagree)
        4. Apply penalty only to ambiguous items with high external ratio (from LLM)
        5. Combine: α·CF_norm + (1-α)·text_norm - penalty
        """
        # Get raw scores
        cf_scores = self.compute_cf_scores(user_idx, item_indices)
        text_scores = self.compute_text_similarity(user_idx, item_indices)
        
        # Compute ranks (1-indexed, lower = better)
        cf_ranks = np.argsort(np.argsort(-cf_scores)) + 1
        text_ranks = np.argsort(np.argsort(-text_scores)) + 1
        
        # Identify ambiguous items (CF and text disagree)
        cf_likes_text_dislikes = (cf_ranks <= self.cf_rank_threshold) & (text_ranks > self.text_rank_threshold)
        text_likes_cf_dislikes = (text_ranks <= self.cf_rank_threshold) & (cf_ranks > self.text_rank_threshold)
        is_ambiguous = cf_likes_text_dislikes | text_likes_cf_dislikes
        
        # Compute penalties using LLM external ratios
        external_ratios = self.external_ratio[item_indices]
        gated_penalty = np.where(is_ambiguous, external_ratios * self.penalty_weight, 0.0)
        
        # Normalize scores to [0, 1]
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
        text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
        
        # Final hybrid score
        hybrid = self.alpha * cf_norm + (1 - self.alpha) * text_norm - gated_penalty
        
        if return_components:
            return hybrid, cf_norm, text_norm, gated_penalty, is_ambiguous
        return hybrid
    
    def rank_items(
        self,
        user_idx: int,
        candidate_items: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """Rank items and return top-k."""
        scores = self.score(user_idx, candidate_items)
        top_k_local = np.argsort(-scores)[:k]
        return candidate_items[top_k_local]


def compute_item_external_ratio_from_llm(
    df: pd.DataFrame,
    item2idx: Dict[str, int],
    llm_labels: pd.Series
) -> np.ndarray:
    """
    Compute per-item external ratio using LLM aspect labels.
    
    external_ratio(i) = fraction of item i's reviews labeled as 'external' by LLM
    
    Args:
        df: Interactions DataFrame with 'item_id' column
        item2idx: Item ID to index mapping
        llm_labels: Series with 'product', 'external', or 'mixed' labels
    
    Returns:
        numpy array of shape (num_items,) with ratio ∈ [0, 1]
    """
    logger.info("Computing item external ratios from LLM labels...")
    
    num_items = len(item2idx)
    external_ratios = np.zeros(num_items)
    
    df = df.copy()
    df['llm_aspect'] = llm_labels.values
    
    for item_id, group in df.groupby('item_id'):
        if item_id not in item2idx:
            continue
        idx = item2idx[item_id]
        n_reviews = len(group)
        if n_reviews == 0:
            continue
        
        # Count external reviews (pure external from LLM)
        n_external = (group['llm_aspect'] == 'external').sum()
        external_ratios[idx] = n_external / n_reviews
    
    logger.info(f"External ratio (LLM) stats: mean={external_ratios.mean():.3f}, "
                f"max={external_ratios.max():.3f}, "
                f"items_with_ratio>0.1={(external_ratios > 0.1).sum()}")
    
    return external_ratios


def compute_item_logistics_ratio(
    df: pd.DataFrame,
    item2idx: Dict[str, int],
    aspect_labels: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Compute per-item logistics ratio.
    
    logistics_ratio(i) = fraction of item i's reviews classified as logistics/seller
    
    Args:
        df: Interactions DataFrame with 'item_id' column
        item2idx: Item ID to index mapping
        aspect_labels: Optional Series with aspect labels per review
                      (indexes matching df). If None, uses heuristic.
    
    Returns:
        numpy array of shape (num_items,) with ratio ∈ [0, 1]
    """
    logger.info("Computing item logistics ratios...")
    
    num_items = len(item2idx)
    logistics_ratios = np.zeros(num_items)
    
    if aspect_labels is not None:
        # Use provided aspect labels
        df = df.copy()
        df['aspect'] = aspect_labels.values
        
        for item_id, group in df.groupby('item_id'):
            if item_id not in item2idx:
                continue
            idx = item2idx[item_id]
            n_reviews = len(group)
            if n_reviews == 0:
                continue
            
            # Count logistics and seller reviews
            n_logistics_seller = ((group['aspect'] == 'logistics') | 
                                  (group['aspect'] == 'seller_service')).sum()
            logistics_ratios[idx] = n_logistics_seller / n_reviews
    else:
        # Fallback: use heuristic classification on the fly
        from src.aspect_classifier import AspectClassifier
        classifier = AspectClassifier(use_llm=False)
        
        for item_id, group in tqdm(df.groupby('item_id'), desc="Computing logistics ratios"):
            if item_id not in item2idx:
                continue
            idx = item2idx[item_id]
            n_reviews = len(group)
            if n_reviews == 0:
                continue
            
            # Classify each review
            n_logistics_seller = 0
            for _, row in group.iterrows():
                text = str(row.get('reviewText', '')) + ' ' + str(row.get('summary', ''))
                aspect, _ = classifier.classify_heuristic(text)
                if aspect in ['logistics', 'seller_service']:
                    n_logistics_seller += 1
            
            logistics_ratios[idx] = n_logistics_seller / n_reviews
    
    logger.info(f"Logistics ratio stats: mean={logistics_ratios.mean():.3f}, "
                f"max={logistics_ratios.max():.3f}, "
                f"items_with_ratio>0.1={(logistics_ratios > 0.1).sum()}")
    
    return logistics_ratios


def evaluate_hybrid(
    scorer: HybridScorer,
    test_df: pd.DataFrame,
    train_items: Dict[int, set],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate hybrid scorer on test set.
    
    Args:
        scorer: HybridScorer instance
        test_df: Test interactions DataFrame
        train_items: Dict mapping user_idx to set of train item indices
        user2idx, item2idx: ID mappings
    """
    all_items = np.arange(len(item2idx))
    
    metrics = {f'hr@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics['mrr'] = []
    
    # Group test by user
    test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    for user_id, ground_truth_ids in tqdm(test_by_user.items(), desc="Evaluating"):
        if user_id not in user2idx:
            continue
        
        user_idx = user2idx[user_id]
        
        # Get ground truth indices
        ground_truth = {item2idx[iid] for iid in ground_truth_ids if iid in item2idx}
        if not ground_truth:
            continue
        
        # Get items user hasn't interacted with in training
        train_set = train_items.get(user_idx, set())
        candidates = np.array([i for i in all_items if i not in train_set])
        
        if len(candidates) == 0:
            continue
        
        # Score and rank
        scores = scorer.score(user_idx, candidates)
        ranked_items = candidates[np.argsort(-scores)]
        
        # Compute metrics
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
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}


def grid_search(
    cf_model: BPRMF,
    user_text_emb: np.ndarray,
    item_text_emb: np.ndarray,
    item_penalties: np.ndarray,
    val_df: pd.DataFrame,
    train_items: Dict[int, set],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    alpha_values: List[float] = [0.7, 0.8, 0.9, 0.95],
    lambda_values: List[float] = [0.0, 0.05, 0.1, 0.2]
) -> Tuple[Dict, List[Dict]]:
    """
    Grid search for best (alpha, lambda).
    
    Returns:
        best_params: {'alpha': x, 'lambda': y, 'metrics': {...}}
        all_results: List of all param combinations and results
    """
    logger.info("=" * 60)
    logger.info("Grid Search for Hybrid Parameters")
    logger.info("=" * 60)
    
    all_results = []
    best_params = None
    best_ndcg = -1
    
    for alpha in alpha_values:
        for lambda_p in lambda_values:
            logger.info(f"\nTrying alpha={alpha}, lambda={lambda_p}")
            
            scorer = HybridScorer(
                cf_model=cf_model,
                user_text_embeddings=user_text_emb,
                item_text_embeddings=item_text_emb,
                item_penalties=item_penalties,
                alpha=alpha,
                lambda_penalty=lambda_p
            )
            
            metrics = evaluate_hybrid(
                scorer, val_df, train_items, user2idx, item2idx,
                k_values=[10]
            )
            
            result = {
                'alpha': alpha,
                'lambda': lambda_p,
                **metrics
            }
            all_results.append(result)
            
            logger.info(f"  HR@10={metrics['hr@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")
            
            if metrics['ndcg@10'] > best_ndcg:
                best_ndcg = metrics['ndcg@10']
                best_params = result.copy()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Best params: alpha={best_params['alpha']}, lambda={best_params['lambda']}")
    logger.info(f"Best NDCG@10: {best_params['ndcg@10']:.4f}")
    logger.info("=" * 60)
    
    return best_params, all_results


def run_full_comparison(
    cf_model: BPRMF,
    user_text_emb: np.ndarray,
    item_text_emb: np.ndarray,
    item_penalties: np.ndarray,
    test_df: pd.DataFrame,
    train_items: Dict[int, set],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    best_alpha: float,
    best_lambda: float
) -> Dict[str, Dict]:
    """
    Compare all 4 models on test set:
    1. CF-only (alpha=1, lambda=0)
    2. Text-only (alpha=0, lambda=0)
    3. CF+Text (best alpha, lambda=0)
    4. CF+Text+Penalty (best alpha, best lambda) - our method
    """
    logger.info("\n" + "=" * 60)
    logger.info("Final Comparison on Test Set")
    logger.info("=" * 60)
    
    configs = [
        ("CF-only", 1.0, 0.0),
        ("Text-only", 0.0, 0.0),
        ("CF+Text", best_alpha, 0.0),
        ("CF+Text+Penalty", best_alpha, best_lambda),
    ]
    
    results = {}
    
    for name, alpha, lambda_p in configs:
        logger.info(f"\n{name} (α={alpha}, λ={lambda_p})")
        
        scorer = HybridScorer(
            cf_model=cf_model,
            user_text_embeddings=user_text_emb,
            item_text_embeddings=item_text_emb,
            item_penalties=item_penalties,
            alpha=alpha,
            lambda_penalty=lambda_p
        )
        
        metrics = evaluate_hybrid(
            scorer, test_df, train_items, user2idx, item2idx,
            k_values=[5, 10, 20]
        )
        
        results[name] = metrics
        logger.info(f"  HR@10={metrics['hr@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")
    
    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10}")
    logger.info("-" * 50)
    
    baseline_ndcg = results['CF-only']['ndcg@10']
    for name, metrics in results.items():
        delta = (metrics['ndcg@10'] - baseline_ndcg) / baseline_ndcg * 100
        logger.info(f"{name:<20} {metrics['hr@10']:>10.4f} {metrics['ndcg@10']:>10.4f} "
                   f"{metrics['mrr']:>10.4f} ({delta:+.1f}%)")
    
    return results


def create_splits(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Create train/val/test splits using leave-one-out per user."""
    from sklearn.model_selection import train_test_split
    
    # Sort by user and timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values(['user_id', 'timestamp'])
    
    train_rows, val_rows, test_rows = [], [], []
    
    for user_id, group in df.groupby('user_id'):
        n = len(group)
        if n < 3:
            # Put all in train
            train_rows.append(group)
        else:
            # Last item for test, second-to-last for val, rest for train
            train_rows.append(group.iloc[:-2])
            val_rows.append(group.iloc[-2:-1])
            test_rows.append(group.iloc[-1:])
    
    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    
    return train_df, val_df, test_df


def main():
    """Run the full hybrid reranking experiment."""
    import torch
    from src.bpr_mf import BPRMF
    
    logger.info("=" * 60)
    logger.info("Hybrid Reranking Experiment")
    logger.info("=" * 60)
    
    # Load data - create splits if they don't exist
    train_path = config.OUTPUT_DIR / 'processed' / 'train.parquet'
    val_path = config.OUTPUT_DIR / 'processed' / 'val.parquet'
    test_path = config.OUTPUT_DIR / 'processed' / 'test.parquet'
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
    else:
        logger.info("Creating train/val/test splits...")
        full_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
        train_df, val_df, test_df = create_splits(full_df)
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        test_df.to_parquet(test_path)
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    with open(config.OUTPUT_DIR / 'processed' / 'id_mappings.json') as f:
        mappings = json.load(f)
    
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    # Build train_items dict
    train_items = {}
    for _, row in train_df.iterrows():
        user_idx = user2idx[row['user_id']]
        item_idx = item2idx[row['item_id']]
        if user_idx not in train_items:
            train_items[user_idx] = set()
        train_items[user_idx].add(item_idx)
    
    # Load or train CF model (baseline)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    cf_model = BPRMF(num_users, num_items, embedding_dim=64).to(device)
    
    model_path = config.OUTPUT_DIR / 'models' / 'baseline_bpr.pt'
    if model_path.exists():
        logger.info("\nLoading CF model...")
        cf_model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        logger.info("\nTraining baseline CF model...")
        from src.bpr_mf import BPRTrainer
        
        # Prepare training data as DataFrame with indices
        train_indexed = train_df.copy()
        train_indexed['user_idx'] = train_indexed['user_id'].map(user2idx)
        train_indexed['item_idx'] = train_indexed['item_id'].map(item2idx)
        
        # Train using BPRTrainer.train() method
        trainer = BPRTrainer(cf_model, device=device)
        trainer.train(
            train_data=train_indexed,
            epochs=30,
            batch_size=1024,
            lr=0.001
        )
        
        # Save
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(cf_model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    
    cf_model.eval()
    
    # Load text embeddings
    logger.info("\nLoading text embeddings...")
    user_text_emb = np.load(config.OUTPUT_DIR / 'user_text_embeddings.npy')
    item_text_emb = np.load(config.OUTPUT_DIR / 'item_text_embeddings.npy')
    
    # Load dissonance penalty
    logger.info("\nLoading pre-filter flags for penalty computation...")
    prefilter_path = config.OUTPUT_DIR / 'processed' / 'prefilter_flags.parquet'
    prefilter_df = None
    if prefilter_path.exists():
        prefilter_df = pd.read_parquet(prefilter_path)
    
    # Compute item penalties
    full_df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    item_penalties = compute_item_dissonance_penalty(full_df, item2idx, prefilter_df)
    
    # Grid search on validation set
    best_params, all_results = grid_search(
        cf_model, user_text_emb, item_text_emb, item_penalties,
        val_df, train_items, user2idx, item2idx
    )
    
    # Final comparison on test set
    final_results = run_full_comparison(
        cf_model, user_text_emb, item_text_emb, item_penalties,
        test_df, train_items, user2idx, item2idx,
        best_params['alpha'], best_params['lambda']
    )
    
    # Save results
    output = {
        'best_params': best_params,
        'grid_search_results': all_results,
        'final_results': final_results
    }
    
    with open(config.OUTPUT_DIR / 'hybrid_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {config.OUTPUT_DIR / 'hybrid_results.json'}")
    
    return output


if __name__ == '__main__':
    main()
