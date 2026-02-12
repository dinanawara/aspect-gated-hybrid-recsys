"""
Text Embedding Module for Hybrid Recommendation.

Builds:
1. Item text embeddings (from aggregated reviews)
2. User text profiles (from their review history)
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Embeds review text using sentence-transformers.
    Falls back to TF-IDF if sentence-transformers unavailable.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.use_tfidf = False
        self._load_model()
    
    def _load_model(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model with {self.embedding_dim}-dim embeddings")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to TF-IDF")
            self.use_tfidf = True
            self._init_tfidf()
    
    def _init_tfidf(self):
        """Initialize TF-IDF as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=128)
        self.embedding_dim = 128
        self.tfidf_fitted = False
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts."""
        if self.use_tfidf:
            return self._embed_tfidf(texts)
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            # Handle empty strings
            batch = [t if t and len(t.strip()) > 0 else "no text" for t in batch]
            batch_emb = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """Embed using TF-IDF + SVD."""
        texts = [t if t and len(t.strip()) > 0 else "no text" for t in texts]
        
        if not self.tfidf_fitted:
            logger.info("Fitting TF-IDF...")
            tfidf_matrix = self.tfidf.fit_transform(texts)
            self.svd.fit(tfidf_matrix)
            self.tfidf_fitted = True
            return self.svd.transform(tfidf_matrix)
        else:
            tfidf_matrix = self.tfidf.transform(texts)
            return self.svd.transform(tfidf_matrix)


def build_item_embeddings(
    df: pd.DataFrame,
    embedder: TextEmbedder,
    item2idx: Dict[str, int],
    max_reviews_per_item: int = 20
) -> np.ndarray:
    """
    Build item embeddings by averaging review embeddings.
    
    Args:
        df: DataFrame with 'item_id' and 'review_text'
        embedder: TextEmbedder instance
        item2idx: Item ID to index mapping
        max_reviews_per_item: Max reviews to consider per item
    
    Returns:
        numpy array of shape (num_items, embedding_dim)
    """
    logger.info("Building item text embeddings...")
    
    num_items = len(item2idx)
    item_embeddings = np.zeros((num_items, embedder.embedding_dim))
    item_counts = np.zeros(num_items)
    
    # Group reviews by item
    item_texts = {}
    for item_id, group in df.groupby('item_id'):
        if item_id not in item2idx:
            continue
        # Take top reviews (by length or random)
        reviews = group['review_text'].dropna().tolist()[:max_reviews_per_item]
        if reviews:
            # Concatenate reviews (or could average embeddings)
            item_texts[item_id] = " ".join(reviews)[:10000]  # Limit length
    
    # Embed all item texts
    item_ids = list(item_texts.keys())
    texts = [item_texts[iid] for iid in item_ids]
    
    if texts:
        embeddings = embedder.embed_texts(texts)
        
        for iid, emb in zip(item_ids, embeddings):
            idx = item2idx[iid]
            item_embeddings[idx] = emb
            item_counts[idx] = 1
    
    # Handle items with no reviews (use mean embedding)
    mean_emb = item_embeddings[item_counts > 0].mean(axis=0)
    item_embeddings[item_counts == 0] = mean_emb
    
    logger.info(f"Built embeddings for {int(item_counts.sum())}/{num_items} items")
    
    return item_embeddings


def build_user_embeddings(
    df: pd.DataFrame,
    embedder: TextEmbedder,
    user2idx: Dict[str, int],
    max_reviews_per_user: int = 10
) -> np.ndarray:
    """
    Build user embeddings by averaging their review embeddings.
    
    Args:
        df: DataFrame with 'user_id' and 'review_text'
        embedder: TextEmbedder instance
        user2idx: User ID to index mapping
        max_reviews_per_user: Max reviews to consider per user
    
    Returns:
        numpy array of shape (num_users, embedding_dim)
    """
    logger.info("Building user text embeddings...")
    
    num_users = len(user2idx)
    user_embeddings = np.zeros((num_users, embedder.embedding_dim))
    user_counts = np.zeros(num_users)
    
    # Group reviews by user
    user_texts = {}
    for user_id, group in df.groupby('user_id'):
        if user_id not in user2idx:
            continue
        reviews = group['review_text'].dropna().tolist()[:max_reviews_per_user]
        if reviews:
            user_texts[user_id] = " ".join(reviews)[:5000]
    
    # Embed all user texts
    user_ids = list(user_texts.keys())
    texts = [user_texts[uid] for uid in user_ids]
    
    if texts:
        embeddings = embedder.embed_texts(texts)
        
        for uid, emb in zip(user_ids, embeddings):
            idx = user2idx[uid]
            user_embeddings[idx] = emb
            user_counts[idx] = 1
    
    # Handle users with no reviews
    mean_emb = user_embeddings[user_counts > 0].mean(axis=0)
    user_embeddings[user_counts == 0] = mean_emb
    
    logger.info(f"Built embeddings for {int(user_counts.sum())}/{num_users} users")
    
    return user_embeddings


def compute_text_similarity_matrix(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute user-item text similarity scores.
    
    Returns sparse-friendly approach: just compute on demand.
    """
    if normalize:
        # L2 normalize for cosine similarity
        user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        user_embeddings = user_embeddings / (user_norms + 1e-8)
        item_embeddings = item_embeddings / (item_norms + 1e-8)
    
    # Return normalized embeddings for on-demand similarity
    return user_embeddings, item_embeddings


def main():
    """Build and save text embeddings."""
    logger.info("=" * 60)
    logger.info("Building Text Embeddings")
    logger.info("=" * 60)
    
    import json
    
    # Load data
    data_path = config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet'
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} interactions")
    
    # Load mappings
    mappings_path = config.OUTPUT_DIR / 'processed' / 'id_mappings.json'
    with open(mappings_path) as f:
        mappings = json.load(f)
    
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    
    # Initialize embedder
    embedder = TextEmbedder()
    
    # Build embeddings
    item_embeddings = build_item_embeddings(df, embedder, item2idx)
    user_embeddings = build_user_embeddings(df, embedder, user2idx)
    
    # Normalize for cosine similarity
    user_embeddings, item_embeddings = compute_text_similarity_matrix(
        user_embeddings, item_embeddings, normalize=True
    )
    
    # Save
    np.save(config.OUTPUT_DIR / 'item_text_embeddings.npy', item_embeddings)
    np.save(config.OUTPUT_DIR / 'user_text_embeddings.npy', user_embeddings)
    
    logger.info(f"Saved item embeddings: {item_embeddings.shape}")
    logger.info(f"Saved user embeddings: {user_embeddings.shape}")
    
    logger.info("=" * 60)
    logger.info("Text Embeddings Complete!")
    logger.info("=" * 60)
    
    return user_embeddings, item_embeddings


if __name__ == '__main__':
    main()
