"""
LLM-Based Aspect Classifier.

Uses LLM to classify reviews as:
- product: Criticizes/praises product quality
- external: Criticizes external factors (shipping, seller, packaging)
- mixed: Both product and external factors

This makes LLM the decision-maker for the gated penalty.
"""
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """Classify what this product review is primarily about.

Review: "{review_text}"
Rating: {rating}/5

Categories:
- product: The review focuses on the product itself (quality, features, performance, durability, design, value, functionality)
- external: The review focuses on external factors NOT related to the product itself (shipping speed, delivery issues, packaging damage, seller communication, Amazon service, return process, wrong item sent, late arrival)
- mixed: The review significantly discusses BOTH product quality AND external factors

Important: If the review mentions shipping/delivery/packaging AT ALL, classify as "external" or "mixed".

Respond with exactly one word: product, external, or mixed"""


class LLMClassifier:
    """Async LLM-based review classifier."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 20,  # Reduced to avoid rate limits
        temperature: float = 0.0
    ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def classify_single(
        self,
        review_text: str,
        rating: float
    ) -> str:
        """Classify a single review."""
        async with self.semaphore:
            try:
                prompt = CLASSIFICATION_PROMPT.format(
                    review_text=review_text[:1000],  # Truncate long reviews
                    rating=rating
                )
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().lower()
                
                # Normalize response
                if 'product' in result:
                    return 'product'
                elif 'external' in result:
                    return 'external'
                elif 'mixed' in result:
                    return 'mixed'
                else:
                    return 'product'  # Default to product
                    
            except Exception as e:
                logger.warning(f"Classification error: {e}")
                return 'product'  # Default on error
    
    async def classify_batch(
        self,
        reviews: List[Dict]
    ) -> List[str]:
        """Classify a batch of reviews concurrently."""
        tasks = [
            self.classify_single(r['text'], r['rating'])
            for r in reviews
        ]
        return await asyncio.gather(*tasks)


def compute_item_external_ratio(
    df: pd.DataFrame,
    item2idx: Dict[str, int],
    aspect_labels: pd.Series
) -> np.ndarray:
    """
    Compute per-item external ratio using LLM labels.
    
    external_ratio(i) = fraction of item i's reviews labeled as 'external'
    
    Args:
        df: Interactions DataFrame
        item2idx: Item ID to index mapping
        aspect_labels: Series with 'product', 'external', or 'mixed' labels
    
    Returns:
        numpy array of shape (num_items,) with ratio ∈ [0, 1]
    """
    logger.info("Computing item external ratios from LLM labels...")
    
    num_items = len(item2idx)
    external_ratios = np.zeros(num_items)
    
    df = df.copy()
    df['llm_aspect'] = aspect_labels.values
    
    for item_id, group in df.groupby('item_id'):
        if item_id not in item2idx:
            continue
        idx = item2idx[item_id]
        n_reviews = len(group)
        if n_reviews == 0:
            continue
        
        # Count external reviews (pure external, not mixed)
        n_external = (group['llm_aspect'] == 'external').sum()
        external_ratios[idx] = n_external / n_reviews
    
    logger.info(f"External ratio stats: mean={external_ratios.mean():.3f}, "
                f"max={external_ratios.max():.3f}, "
                f"items_with_ratio>0.1={(external_ratios > 0.1).sum()}")
    
    return external_ratios


async def classify_all_reviews(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Classify all reviews using LLM.
    
    Args:
        df: DataFrame with 'reviewText', 'summary', 'rating' columns
        output_path: Where to save results
        sample_size: If set, only classify a sample (for testing/cost control)
    
    Returns:
        DataFrame with added 'llm_aspect' column
    """
    logger.info(f"Classifying {len(df)} reviews with LLM...")
    
    # Check for cached results
    if output_path.exists():
        logger.info(f"Loading cached LLM classifications from {output_path}")
        cached = pd.read_parquet(output_path)
        if len(cached) == len(df):
            return cached
        logger.info("Cache size mismatch, re-classifying...")
    
    classifier = LLMClassifier()
    
    # Prepare review texts
    reviews = []
    for _, row in df.iterrows():
        text = str(row.get('reviewText', '') or '') + ' ' + str(row.get('summary', '') or '')
        text = text.strip() if text.strip() else "No review text"
        reviews.append({
            'text': text,
            'rating': row.get('rating', 3)
        })
    
    if sample_size and sample_size < len(reviews):
        logger.info(f"Sampling {sample_size} reviews for classification")
        indices = np.random.choice(len(reviews), sample_size, replace=False)
        sample_reviews = [reviews[i] for i in indices]
        
        # Classify sample
        labels = await classifier.classify_batch(sample_reviews)
        
        # Create full label array (default to 'product' for unsampled)
        df = df.copy()
        df['llm_aspect'] = 'product'
        for i, idx in enumerate(indices):
            df.iloc[idx, df.columns.get_loc('llm_aspect')] = labels[i]
    else:
        # Classify all in batches
        batch_size = 1000
        all_labels = []
        
        for i in tqdm(range(0, len(reviews), batch_size), desc="LLM Classification"):
            batch = reviews[i:i+batch_size]
            labels = await classifier.classify_batch(batch)
            all_labels.extend(labels)
        
        df = df.copy()
        df['llm_aspect'] = all_labels
    
    # Log distribution
    dist = df['llm_aspect'].value_counts()
    logger.info(f"Aspect distribution:\n{dist}")
    
    # Save results
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved LLM classifications to {output_path}")
    
    return df


async def main():
    """Run LLM classification on all reviews."""
    logger.info("="*60)
    logger.info("LLM Aspect Classification")
    logger.info("="*60)
    
    # Load interactions
    df = pd.read_parquet(config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet')
    logger.info(f"Loaded {len(df)} interactions")
    
    # Classify all reviews
    output_path = config.OUTPUT_DIR / 'processed' / 'llm_aspect_labels.parquet'
    
    # For cost control, you can set sample_size (e.g., 50000)
    # Set to None to classify all
    df = await classify_all_reviews(df, output_path, sample_size=None)
    
    return df


if __name__ == '__main__':
    asyncio.run(main())
