"""
Aspect Classifier for Reviews.

Classifies reviews into dimensions:
- (A) Product quality
- (B) Logistics/shipping  
- (C) Seller/service
- (D) Mixed
- (E) Unclear

Step 1: Keyword heuristics for obvious cases
Step 2: LLM for ambiguous cases
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Keyword patterns for heuristic classification
LOGISTICS_KEYWORDS = [
    r'\bshipp(ing|ed|er)\b', r'\bdeliver(y|ed|ing)\b', r'\barriv(e|ed|al)\b',
    r'\bpackag(e|ing|ed)\b', r'\bbox\b', r'\bdamag(e|ed)\b', r'\blate\b',
    r'\bdays?\s+late\b', r'\bweeks?\b', r'\btracking\b', r'\bcarrier\b',
    r'\bfedex\b', r'\bups\b', r'\busps\b', r'\bamazon\s+prime\b',
    r'\bprime\s+shipping\b', r'\bfast\s+shipping\b', r'\bslow\s+shipping\b'
]

SELLER_KEYWORDS = [
    r'\bseller\b', r'\bvendor\b', r'\bmerchant\b', r'\bstore\b',
    r'\brefund(ed)?\b', r'\breturn(ed|ing)?\b', r'\bcustomer\s+service\b',
    r'\bsupport\b', r'\bwarranty\b', r'\breplacement\b', r'\bscam\b',
    r'\bfraud\b', r'\bfake\b', r'\bcounterfeit\b', r'\bknock\s*off\b'
]

PRODUCT_KEYWORDS = [
    r'\bquality\b', r'\bsound\b', r'\bbattery\b', r'\bscreen\b',
    r'\bperformance\b', r'\bfeature\b', r'\bdesign\b', r'\bbuild\b',
    r'\bcomfort(able)?\b', r'\bdurability\b', r'\bworks?\s+(great|well|perfectly)\b',
    r'\bdoesn\'?t\s+work\b', r'\bbroke(n)?\b', r'\bdefective\b',
    r'\bexcellent\b', r'\bamazing\b', r'\bterrible\b', r'\bawesome\b'
]


class AspectClassifier:
    """
    Classifies reviews by aspect dimension.
    Uses heuristics first, LLM for ambiguous cases.
    """
    
    ASPECT_LABELS = {
        'A': 'product_quality',
        'B': 'logistics',
        'C': 'seller_service',
        'D': 'mixed',
        'E': 'unclear'
    }
    
    # CF weights by aspect
    ASPECT_WEIGHTS = {
        'product_quality': 1.0,  # Full product signal
        'logistics': 0.3,        # Not product-relevant
        'seller_service': 0.3,   # Not product-relevant
        'mixed': 0.7,            # Partial signal
        'unclear': 1.0           # Benefit of doubt
    }
    
    def __init__(self, use_llm: bool = True, llm_model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.client = None
        
        # Compile regex patterns
        self.logistics_patterns = [re.compile(p, re.IGNORECASE) for p in LOGISTICS_KEYWORDS]
        self.seller_patterns = [re.compile(p, re.IGNORECASE) for p in SELLER_KEYWORDS]
        self.product_patterns = [re.compile(p, re.IGNORECASE) for p in PRODUCT_KEYWORDS]
    
    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count keyword matches."""
        return sum(1 for p in patterns if p.search(text))
    
    def classify_heuristic(self, text: str) -> Tuple[str, float]:
        """
        Classify using keyword heuristics.
        
        Returns:
            (aspect_label, confidence)
            confidence < 0.6 means "ambiguous" -> needs LLM
        """
        if not text or len(text.strip()) < 10:
            return 'unclear', 1.0
        
        logistics_count = self._count_matches(text, self.logistics_patterns)
        seller_count = self._count_matches(text, self.seller_patterns)
        product_count = self._count_matches(text, self.product_patterns)
        
        total = logistics_count + seller_count + product_count
        
        if total == 0:
            return 'unclear', 0.5  # Ambiguous - no keywords
        
        # Determine dominant aspect
        max_count = max(logistics_count, seller_count, product_count)
        
        # Check for mixed
        non_zero = sum(1 for c in [logistics_count, seller_count, product_count] if c > 0)
        if non_zero >= 2 and max_count < total * 0.7:
            # Multiple aspects, no clear dominant
            return 'mixed', 0.7
        
        # Single dominant aspect
        if logistics_count == max_count:
            conf = logistics_count / total if total > 0 else 0.5
            # High confidence only if logistics is clearly dominant and product is low
            if product_count <= 1 and logistics_count >= 2:
                return 'logistics', min(0.9, conf + 0.3)
            return 'logistics', conf
        
        if seller_count == max_count:
            conf = seller_count / total if total > 0 else 0.5
            if product_count <= 1 and seller_count >= 2:
                return 'seller_service', min(0.9, conf + 0.3)
            return 'seller_service', conf
        
        if product_count == max_count:
            conf = product_count / total if total > 0 else 0.5
            return 'product_quality', min(0.9, conf + 0.2)
        
        return 'unclear', 0.5
    
    async def classify_llm(self, reviews: List[Dict]) -> List[Dict]:
        """
        Classify ambiguous reviews using LLM.
        
        Args:
            reviews: List of dicts with 'text', 'rating' keys
        
        Returns:
            List of dicts with 'aspect' added
        """
        if not self.client:
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        semaphore = asyncio.Semaphore(10)
        
        async def classify_one(review: Dict) -> Dict:
            async with semaphore:
                prompt = f"""Classify this Amazon review into ONE category:

(A) Product quality - about the product itself, features, performance
(B) Logistics/shipping - about delivery, packaging, shipping time
(C) Seller/service - about the seller, customer service, refunds
(D) Mixed - mentions multiple categories significantly
(E) Unclear - can't determine

Review (Rating: {review.get('rating', 'N/A')}/5):
"{review['text'][:500]}"

Return ONLY the letter (A, B, C, D, or E):"""

                try:
                    response = await self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=5,
                        temperature=0
                    )
                    
                    answer = response.choices[0].message.content.strip().upper()
                    # Extract just the letter
                    label = answer[0] if answer and answer[0] in 'ABCDE' else 'E'
                    aspect = self.ASPECT_LABELS.get(label, 'unclear')
                    
                    return {**review, 'aspect': aspect, 'aspect_source': 'llm'}
                
                except Exception as e:
                    logger.warning(f"LLM error: {e}")
                    return {**review, 'aspect': 'unclear', 'aspect_source': 'error'}
        
        tasks = [classify_one(r) for r in reviews]
        results = await asyncio.gather(*tasks)
        return results
    
    def classify_batch(self, df: pd.DataFrame, text_col: str = 'review_text',
                       rating_col: str = 'rating') -> pd.DataFrame:
        """
        Classify a batch of reviews.
        Uses heuristics first, LLM for ambiguous cases.
        
        Args:
            df: DataFrame with reviews
            text_col: Column name for review text
            rating_col: Column name for rating
        
        Returns:
            DataFrame with 'aspect', 'aspect_weight', 'aspect_confidence' columns
        """
        logger.info(f"Classifying {len(df)} reviews by aspect...")
        
        aspects = []
        confidences = []
        sources = []
        
        ambiguous_indices = []
        ambiguous_reviews = []
        
        # Step 1: Heuristic classification
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Heuristic classification"):
            text = row.get(text_col, '')
            aspect, conf = self.classify_heuristic(str(text) if pd.notna(text) else '')
            
            if conf < 0.6 and self.use_llm:
                # Mark as ambiguous
                ambiguous_indices.append(idx)
                ambiguous_reviews.append({
                    'text': str(text)[:1000] if pd.notna(text) else '',
                    'rating': row.get(rating_col, 0),
                    'idx': idx
                })
                aspects.append(None)
                confidences.append(None)
                sources.append('pending_llm')
            else:
                aspects.append(aspect)
                confidences.append(conf)
                sources.append('heuristic')
        
        logger.info(f"Heuristic classified {len(df) - len(ambiguous_indices)}, "
                   f"{len(ambiguous_indices)} ambiguous ({len(ambiguous_indices)/len(df)*100:.1f}%)")
        
        # Step 2: LLM for ambiguous cases
        if ambiguous_reviews and self.use_llm:
            logger.info(f"Sending {len(ambiguous_reviews)} ambiguous reviews to LLM...")
            llm_results = asyncio.run(self.classify_llm(ambiguous_reviews))
            
            for result in llm_results:
                orig_idx = result['idx']
                list_idx = list(df.index).index(orig_idx)
                aspects[list_idx] = result['aspect']
                confidences[list_idx] = 0.8  # LLM confidence
                sources[list_idx] = result.get('aspect_source', 'llm')
        
        # Fill any remaining None values
        aspects = ['unclear' if a is None else a for a in aspects]
        confidences = [0.5 if c is None else c for c in confidences]
        
        # Add columns
        df = df.copy()
        df['aspect'] = aspects
        df['aspect_confidence'] = confidences
        df['aspect_source'] = sources
        df['aspect_weight'] = df['aspect'].map(self.ASPECT_WEIGHTS)
        
        # Log distribution
        logger.info("\nAspect Distribution:")
        for aspect, count in df['aspect'].value_counts().items():
            pct = count / len(df) * 100
            weight = self.ASPECT_WEIGHTS.get(aspect, 1.0)
            logger.info(f"  {aspect}: {count} ({pct:.1f}%) -> weight={weight}")
        
        return df


def main():
    """Run aspect classification on the dataset."""
    logger.info("=" * 60)
    logger.info("Aspect Classification Pipeline")
    logger.info("=" * 60)
    
    # Load interactions
    data_path = config.OUTPUT_DIR / 'processed' / 'interactions_full.parquet'
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} interactions")
    
    # Initialize classifier
    classifier = AspectClassifier(use_llm=True)
    
    # Classify
    df_classified = classifier.classify_batch(df)
    
    # Save
    output_path = config.OUTPUT_DIR / 'processed' / 'interactions_aspect.parquet'
    df_classified.to_parquet(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    
    print(df_classified['aspect'].value_counts())
    print(f"\nMean weight: {df_classified['aspect_weight'].mean():.3f}")
    print(f"Effective data reduction: {(1 - df_classified['aspect_weight'].mean()) * 100:.1f}%")
    
    return df_classified


if __name__ == '__main__':
    main()
