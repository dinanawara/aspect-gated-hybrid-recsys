"""
BPR-MF (Bayesian Personalized Ranking - Matrix Factorization) Implementation.

Supports:
- Standard BPR-MF (baseline)
- Weighted BPR-MF (edge weights from dissonance analysis)
- Optional negative sampling strategies
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)


class BPRDataset(Dataset):
    """Dataset for BPR training with positive and negative sampling."""
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        user_col: str = 'user_idx',
        item_col: str = 'item_idx',
        weight_col: str = None,
        num_negatives: int = 1
    ):
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build user-positive items mapping
        self.user_positives = interactions.groupby(user_col)[item_col].apply(set).to_dict()
        
        # Store interactions
        self.users = interactions[user_col].values
        self.items = interactions[item_col].values
        
        # Edge weights (default 1.0)
        if weight_col and weight_col in interactions.columns:
            self.weights = interactions[weight_col].values
        else:
            self.weights = np.ones(len(interactions))
        
        # All items for negative sampling
        self.all_items = set(range(num_items))
    
    def __len__(self):
        return len(self.users)
    
    def _sample_negative(self, user: int) -> int:
        """Sample a negative item for a user."""
        positives = self.user_positives.get(user, set())
        negatives = list(self.all_items - positives)
        if not negatives:
            return np.random.randint(0, self.num_items)
        return np.random.choice(negatives)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, int, float]:
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_item = self._sample_negative(user)
        weight = self.weights[idx]
        return user, pos_item, neg_item, weight


class BPRMF(nn.Module):
    """
    BPR Matrix Factorization model.
    
    Learns user and item embeddings to predict preferences.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        reg_lambda: float = 0.01
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg_lambda = reg_lambda
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Item biases
        self.item_biases = nn.Embedding(num_items, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.item_biases.weight)
    
    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """Compute preference scores for user-item pairs."""
        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)
        item_bias = self.item_biases(items).squeeze(-1)
        
        scores = (user_emb * item_emb).sum(dim=-1) + item_bias
        return scores
    
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute weighted BPR loss.
        
        Loss = -log(sigmoid(pos_score - neg_score))
        """
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-8)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        loss = loss.mean()
        
        # L2 regularization
        reg_loss = self.reg_lambda * (
            self.user_embeddings(users).pow(2).mean() +
            self.item_embeddings(pos_items).pow(2).mean() +
            self.item_embeddings(neg_items).pow(2).mean()
        )
        
        return loss + reg_loss
    
    def get_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings."""
        return self.user_embeddings.weight.detach().cpu().numpy()
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings."""
        return self.item_embeddings.weight.detach().cpu().numpy()
    
    def predict_topk(
        self,
        user_idx: int,
        k: int = 10,
        exclude_items: set = None
    ) -> List[int]:
        """Predict top-k items for a user."""
        self.eval()
        with torch.no_grad():
            user = torch.tensor([user_idx], device=next(self.parameters()).device)
            all_items = torch.arange(self.num_items, device=user.device)
            
            scores = self.forward(
                user.expand(self.num_items),
                all_items
            ).cpu().numpy()
        
        # Exclude items if provided
        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = float('-inf')
        
        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()


class BPRTrainer:
    """Trainer for BPR-MF model."""
    
    def __init__(
        self,
        model: BPRMF,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 0.001,
        weight_col: str = None,
        early_stopping: int = 5,
        eval_fn = None
    ) -> Dict:
        """
        Train the BPR-MF model.
        
        Args:
            train_data: Training interactions
            val_data: Validation interactions
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_col: Column name for edge weights
            early_stopping: Patience for early stopping
            eval_fn: Optional evaluation function
        
        Returns:
            Training history
        """
        # Create dataset and loader
        dataset = BPRDataset(
            train_data,
            self.model.num_items,
            weight_col=weight_col
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        best_metric = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for users, pos_items, neg_items, weights in pbar:
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)
                weights = weights.float().to(self.device)
                
                optimizer.zero_grad()
                loss = self.model.bpr_loss(users, pos_items, neg_items, weights)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': total_loss / num_batches})
            
            avg_loss = total_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if val_data is not None and eval_fn is not None:
                metrics = eval_fn(self.model, val_data, train_data)
                history['val_metrics'].append(metrics)
                
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, HR@10={metrics.get('hr@10', 0):.4f}, NDCG@10={metrics.get('ndcg@10', 0):.4f}")
                
                # Early stopping
                current_metric = metrics.get('ndcg@10', 0)
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    # Save best model
                    self.save_model(config.MODELS_DIR / 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        return history
    
    def save_model(self, path: Path):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.model.embedding_dim
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def train_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by timestamp.
    
    Uses leave-last-out strategy per user.
    """
    # Sort by timestamp
    df = df.sort_values(['user_idx', 'timestamp'])
    
    train_list = []
    val_list = []
    test_list = []
    
    for user_idx, group in df.groupby('user_idx'):
        n = len(group)
        if n < 3:
            train_list.append(group)
            continue
        
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        
        train_list.append(group.iloc[:n_train])
        val_list.append(group.iloc[n_train:n_train+n_val])
        test_list.append(group.iloc[n_train+n_val:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    
    logger.info(f"Data split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    
    return train_df, val_df, test_df


def main():
    """Train BPR-MF models (baseline and weighted)."""
    logger.info("=" * 60)
    logger.info("Starting BPR-MF Training")
    logger.info("=" * 60)
    
    # Load data
    cf_path = config.OUTPUT_DIR / 'processed' / 'interactions_cf.csv'
    weights_path = config.OUTPUT_DIR / 'interactions_with_weights.parquet'
    
    if not cf_path.exists():
        logger.error(f"CF data not found at {cf_path}")
        logger.error("Run data_preprocessing.py first!")
        return
    
    # Load CF data
    df = pd.read_csv(cf_path)
    
    # Load weights if available
    has_weights = weights_path.exists()
    if has_weights:
        weights_df = pd.read_parquet(weights_path)
        df = df.merge(
            weights_df[['user_idx', 'item_idx', 'edge_weight']],
            on=['user_idx', 'item_idx'],
            how='left'
        )
        df['edge_weight'] = df['edge_weight'].fillna(1.0)
        logger.info("Loaded edge weights from dissonance analysis")
    
    # Load mappings
    mappings_path = config.OUTPUT_DIR / 'processed' / 'id_mappings.json'
    with open(mappings_path) as f:
        mappings = json.load(f)
    
    num_users = len(mappings['user2idx'])
    num_items = len(mappings['item2idx'])
    
    logger.info(f"Dataset: {num_users:,} users, {num_items:,} items, {len(df):,} interactions")
    
    # Split data
    train_df, val_df, test_df = train_split(df)
    
    # Save test set for evaluation
    test_df.to_csv(config.OUTPUT_DIR / 'test_interactions.csv', index=False)
    
    # Import evaluation (will be created next)
    from evaluation import evaluate_model
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    results = {}
    
    # Train baseline model (no weights)
    logger.info("\n" + "=" * 40)
    logger.info("Training BASELINE Model (unweighted)")
    logger.info("=" * 40)
    
    baseline_model = BPRMF(num_users, num_items, embedding_dim=64)
    baseline_trainer = BPRTrainer(baseline_model, device=device)
    baseline_history = baseline_trainer.train(
        train_df,
        val_df,
        epochs=50,
        batch_size=1024,
        lr=0.001,
        weight_col=None,
        eval_fn=lambda m, v, t: evaluate_model(m, v, t, device=device)
    )
    baseline_trainer.save_model(config.MODELS_DIR / 'baseline_bpr.pt')
    
    # Final evaluation on test set
    baseline_test_metrics = evaluate_model(baseline_model, test_df, train_df, device=device)
    results['baseline'] = baseline_test_metrics
    logger.info(f"Baseline Test Metrics: {baseline_test_metrics}")
    
    # Train weighted model (semantic-aware)
    if has_weights:
        logger.info("\n" + "=" * 40)
        logger.info("Training SEMANTIC-AWARE Model (weighted)")
        logger.info("=" * 40)
        
        weighted_model = BPRMF(num_users, num_items, embedding_dim=64)
        weighted_trainer = BPRTrainer(weighted_model, device=device)
        weighted_history = weighted_trainer.train(
            train_df,
            val_df,
            epochs=50,
            batch_size=1024,
            lr=0.001,
            weight_col='edge_weight',
            eval_fn=lambda m, v, t: evaluate_model(m, v, t, device=device)
        )
        weighted_trainer.save_model(config.MODELS_DIR / 'weighted_bpr.pt')
        
        # Final evaluation on test set
        weighted_test_metrics = evaluate_model(weighted_model, test_df, train_df, device=device)
        results['semantic_aware'] = weighted_test_metrics
        logger.info(f"Semantic-Aware Test Metrics: {weighted_test_metrics}")
    
    # Save results
    with open(config.OUTPUT_DIR / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    # Print comparison
    if 'semantic_aware' in results:
        logger.info("\nModel Comparison:")
        logger.info("-" * 40)
        for metric in ['hr@10', 'ndcg@10', 'mrr']:
            baseline_val = results['baseline'].get(metric, 0)
            weighted_val = results['semantic_aware'].get(metric, 0)
            diff = weighted_val - baseline_val
            pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
            logger.info(f"{metric}: Baseline={baseline_val:.4f}, Semantic={weighted_val:.4f}, Δ={diff:+.4f} ({pct:+.2f}%)")
    
    return results


if __name__ == '__main__':
    main()
