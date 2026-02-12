"""
Cheap Pre-Filter for Dissonance Detection.

Uses local signals to identify candidate dissonant reviews:
1. VADER sentiment vs rating mismatch
2. Keyword flags (logistics, incentivized, uncertainty)
3. Rating-polarity contradiction detection

This reduces the number of reviews that need expensive LLM processing.
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DissonancePreFilter:
    """
    Pre-filter to identify potentially dissonant reviews using cheap signals.
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Compile regex patterns for efficiency
        self.logistics_pattern = re.compile(
            r'\b(' + '|'.join(config.LOGISTICS_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.incentivized_pattern = re.compile(
            r'\b(' + '|'.join(config.INCENTIVIZED_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.uncertainty_pattern = re.compile(
            r'\b(' + '|'.join(config.UNCERTAINTY_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.positive_pattern = re.compile(
            r'\b(' + '|'.join(config.POSITIVE_WORDS) + r')\b',
            re.IGNORECASE
        )
        self.negative_pattern = re.compile(
            r'\b(' + '|'.join(config.NEGATIVE_WORDS) + r')\b',
            re.IGNORECASE
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores."""
        if not text or len(text) < 10:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        return self.vader.polarity_scores(text)
    
    def detect_keyword_flags(self, text: str) -> Dict[str, bool]:
        """Detect various keyword flags in the review text."""
        if not text:
            return {
                'has_logistics': False,
                'has_incentivized': False,
                'has_uncertainty': False,
                'has_positive': False,
                'has_negative': False
            }
        
        return {
            'has_logistics': bool(self.logistics_pattern.search(text)),
            'has_incentivized': bool(self.incentivized_pattern.search(text)),
            'has_uncertainty': bool(self.uncertainty_pattern.search(text)),
            'has_positive': bool(self.positive_pattern.search(text)),
            'has_negative': bool(self.negative_pattern.search(text))
        }
    
    def compute_rating_sentiment_mismatch(
        self,
        rating: float,
        sentiment_compound: float
    ) -> Tuple[float, str]:
        """
        Compute mismatch score between rating and sentiment.
        
        Returns:
            mismatch_score: 0-1, higher means more dissonant
            mismatch_type: category of mismatch
        """
        # Normalize rating to [-1, 1] scale
        # 1 star -> -1, 3 stars -> 0, 5 stars -> 1
        normalized_rating = (rating - 3) / 2
        
        # Compute absolute difference
        diff = abs(normalized_rating - sentiment_compound)
        
        # Categorize mismatch
        if diff < 0.3:
            mismatch_type = 'consistent'
        elif diff < 0.6:
            mismatch_type = 'mild_mismatch'
        else:
            mismatch_type = 'strong_mismatch'
        
        # Check for specific contradiction patterns
        if rating <= 2 and sentiment_compound > 0.5:
            mismatch_type = 'low_rating_positive_text'
            diff = max(diff, 0.7)
        elif rating >= 4 and sentiment_compound < -0.3:
            mismatch_type = 'high_rating_negative_text'
            diff = max(diff, 0.7)
        
        return min(diff, 1.0), mismatch_type
    
    def analyze_review(self, review: Dict) -> Dict:
        """
        Analyze a single review for potential dissonance.
        
        Returns enriched review dict with analysis results.
        """
        text = review.get('review_text', '')
        rating = review.get('rating', 3.0)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        
        # Keyword detection
        flags = self.detect_keyword_flags(text)
        
        # Mismatch computation
        mismatch_score, mismatch_type = self.compute_rating_sentiment_mismatch(
            rating, sentiment['compound']
        )
        
        # Compute priority score for LLM审审 (higher = more suspicious)
        priority_score = mismatch_score
        
        # Boost priority for suspicious patterns
        if flags['has_logistics'] and rating <= 2:
            priority_score += 0.3  # Might be shipping complaint, not product issue
        if flags['has_incentivized']:
            priority_score += 0.2  # Potentially biased
        if flags['has_uncertainty']:
            priority_score += 0.1  # Premature review
        
        # Strong contradiction indicators
        if flags['has_positive'] and rating <= 2:
            priority_score += 0.4
        if flags['has_negative'] and rating >= 4:
            priority_score += 0.4
        
        priority_score = min(priority_score, 1.0)
        
        return {
            **review,
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            **flags,
            'mismatch_score': mismatch_score,
            'mismatch_type': mismatch_type,
            'priority_score': priority_score,
            'needs_llm_review': priority_score >= 0.4
        }
    
    def batch_analyze(
        self,
        reviews: List[Dict],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Analyze a batch of reviews."""
        logger.info(f"Analyzing {len(reviews):,} reviews for dissonance signals...")
        
        results = []
        iterator = tqdm(reviews, desc="Pre-filtering") if show_progress else reviews
        
        for review in iterator:
            results.append(self.analyze_review(review))
        
        df = pd.DataFrame(results)
        
        # Log statistics
        logger.info(f"Pre-filter Statistics:")
        logger.info(f"  Total reviews: {len(df):,}")
        logger.info(f"  Needs LLM review: {df['needs_llm_review'].sum():,} ({df['needs_llm_review'].mean()*100:.1f}%)")
        logger.info(f"  Mismatch types:")
        for mtype, count in df['mismatch_type'].value_counts().items():
            logger.info(f"    {mtype}: {count:,} ({count/len(df)*100:.1f}%)")
        logger.info(f"  Has logistics keywords: {df['has_logistics'].sum():,}")
        logger.info(f"  Has incentivized keywords: {df['has_incentivized'].sum():,}")
        logger.info(f"  Has uncertainty keywords: {df['has_uncertainty'].sum():,}")
        
        return df


def select_for_llm_review(
    df: pd.DataFrame,
    max_reviews: int = 10000,
    stratify_by_rating: bool = True
) -> pd.DataFrame:
    """
    Select top-priority reviews for LLM analysis.
    
    Args:
        df: Pre-filtered DataFrame with priority scores
        max_reviews: Maximum number of reviews to select
        stratify_by_rating: If True, ensure representation across rating levels
    
    Returns:
        DataFrame of selected reviews for LLM processing
    """
    logger.info(f"Selecting up to {max_reviews:,} reviews for LLM analysis...")
    
    # Filter to candidates that need review
    candidates = df[df['needs_llm_review']].copy()
    
    if len(candidates) <= max_reviews:
        logger.info(f"Selected all {len(candidates):,} candidate reviews")
        return candidates
    
    if stratify_by_rating:
        # Stratified sampling to ensure representation
        selected = []
        per_rating = max_reviews // 5
        
        for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rating_df = candidates[candidates['rating'] == rating]
            n_select = min(len(rating_df), per_rating)
            
            # Select top priority within each rating
            top_priority = rating_df.nlargest(n_select, 'priority_score')
            selected.append(top_priority)
        
        result = pd.concat(selected, ignore_index=True)
        
        # Fill remaining slots with highest priority overall
        remaining = max_reviews - len(result)
        if remaining > 0:
            already_selected = set(result.index)
            unselected = candidates[~candidates.index.isin(already_selected)]
            extra = unselected.nlargest(remaining, 'priority_score')
            result = pd.concat([result, extra], ignore_index=True)
    else:
        # Simple top-K by priority
        result = candidates.nlargest(max_reviews, 'priority_score')
    
    logger.info(f"Selected {len(result):,} reviews for LLM analysis")
    logger.info(f"  Rating distribution:")
    for rating, count in result['rating'].value_counts().sort_index().items():
        logger.info(f"    {rating} stars: {count:,}")
    
    return result


def main():
    """Run pre-filtering on processed data."""
    logger.info("=" * 60)
    logger.info("Starting Dissonance Pre-Filter")
    logger.info("=" * 60)
    
    # Load processed data
    input_path = config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet'
    if not input_path.exists():
        logger.error(f"Processed data not found at {input_path}")
        logger.error("Run data_preprocessing.py first!")
        return
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} interactions")
    
    # Run pre-filter
    prefilter = DissonancePreFilter()
    reviews = df.to_dict('records')
    analyzed_df = prefilter.batch_analyze(reviews)
    
    # Save full analysis
    output_path = config.OUTPUT_DIR / 'prefilter_results.parquet'
    analyzed_df.to_parquet(output_path, index=False)
    logger.info(f"Saved pre-filter results to {output_path}")
    
    # Select for LLM review
    llm_candidates = select_for_llm_review(analyzed_df, max_reviews=10000)
    llm_path = config.OUTPUT_DIR / 'llm_candidates.parquet'
    llm_candidates.to_parquet(llm_path, index=False)
    logger.info(f"Saved LLM candidates to {llm_path}")
    
    logger.info("=" * 60)
    logger.info("Pre-Filter Complete!")
    logger.info("=" * 60)
    
    return analyzed_df


if __name__ == '__main__':
    main()
