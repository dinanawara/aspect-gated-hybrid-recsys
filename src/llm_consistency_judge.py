"""
LLM Agent for Rating-Review Consistency Judgment.

Agent B: Consistency Judge
- Input: Review text + original star rating
- Output: Consistency label + confidence score

Supports multiple LLM backends:
- OpenAI (gpt-4o-mini, gpt-4o)
- Ollama (mistral, llama)
"""
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
from tqdm.asyncio import tqdm_asyncio
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConsistencyLabel(Enum):
    CONSISTENT = "consistent"
    MILD_INCONSISTENT = "mild_inconsistent"
    STRONGLY_DISSONANT = "strongly_dissonant"
    LOGISTICS_COMPLAINT = "logistics_complaint"
    INCENTIVIZED = "incentivized"
    UNCERTAIN = "uncertain"


@dataclass
class ConsistencyResult:
    """Result from consistency judgment."""
    label: ConsistencyLabel
    confidence: float  # 0-1
    reasoning: str
    adjusted_sentiment: float  # -1 to 1, product-focused sentiment
    should_downweight: bool


# Prompt template for Agent B
CONSISTENCY_JUDGE_PROMPT = """You are an expert at analyzing product reviews to detect rating-review inconsistencies.

TASK: Analyze if the review text is CONSISTENT with the star rating, focusing ONLY on product quality.

IMPORTANT RULES:
1. IGNORE complaints about: shipping, delivery, packaging, seller service, Amazon issues
2. FOCUS ONLY on: product functionality, quality, durability, value, features
3. A review complaining about shipping but praising the product should be marked as "logistics_complaint"
4. A review that seems incentivized or promotional should be marked as "incentivized"

STAR RATING: {rating}/5 stars

REVIEW TEXT:
"{review_text}"

Analyze this review and respond in JSON format:
{{
    "label": "consistent" | "mild_inconsistent" | "strongly_dissonant" | "logistics_complaint" | "incentivized" | "uncertain",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (max 50 words)",
    "product_sentiment": -1.0 to 1.0 (sentiment toward PRODUCT ONLY, ignoring logistics),
    "should_downweight": true/false (true if rating doesn't reflect product quality)
}}

Examples:
- 1 star + "Product works great but shipping was terrible" → logistics_complaint, product_sentiment: 0.7, should_downweight: true
- 5 stars + "Got this free, it's okay I guess" → incentivized, should_downweight: true
- 2 stars + "Broke after one week, waste of money" → consistent, product_sentiment: -0.8, should_downweight: false
- 4 stars + "Doesn't work at all, returning it" → strongly_dissonant, should_downweight: true

Respond with ONLY the JSON object, no other text."""


class LLMClient:
    """Client for LLM API calls."""
    
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self.max_tokens = config.LLM_MAX_TOKENS
        self.temperature = config.LLM_TEMPERATURE
        
        if self.provider == "openai":
            self.api_key = config.OPENAI_API_KEY
            self.base_url = "https://api.openai.com/v1"
            if not self.api_key:
                logger.warning("OpenAI API key not set. Please set OPENAI_API_KEY in .env")
        elif self.provider == "ollama":
            self.base_url = config.OLLAMA_BASE_URL
            self.model = config.OLLAMA_MODEL
    
    async def _call_openai(self, prompt: str, client: httpx.AsyncClient) -> str:
        """Call OpenAI API."""
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def _call_ollama(self, prompt: str, client: httpx.AsyncClient) -> str:
        """Call Ollama API."""
        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["response"]
    
    async def call(self, prompt: str, client: httpx.AsyncClient) -> str:
        """Call LLM with the given prompt."""
        if self.provider == "openai":
            return await self._call_openai(prompt, client)
        elif self.provider == "ollama":
            return await self._call_ollama(prompt, client)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


class ConsistencyJudge:
    """
    Agent B: Judges consistency between rating and review text.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
    
    def _parse_response(self, response: str) -> ConsistencyResult:
        """Parse LLM response into ConsistencyResult."""
        try:
            # Clean up response (remove markdown code blocks if present)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            data = json.loads(response)
            
            label = ConsistencyLabel(data.get("label", "uncertain"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            product_sentiment = float(data.get("product_sentiment", 0.0))
            should_downweight = bool(data.get("should_downweight", False))
            
            return ConsistencyResult(
                label=label,
                confidence=confidence,
                reasoning=reasoning,
                adjusted_sentiment=product_sentiment,
                should_downweight=should_downweight
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return ConsistencyResult(
                label=ConsistencyLabel.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Parse error: {str(e)}",
                adjusted_sentiment=0.0,
                should_downweight=False
            )
    
    async def judge_single(
        self,
        review_text: str,
        rating: float,
        client: httpx.AsyncClient
    ) -> ConsistencyResult:
        """Judge a single review."""
        prompt = CONSISTENCY_JUDGE_PROMPT.format(
            rating=int(rating),
            review_text=review_text[:2000]  # Truncate very long reviews
        )
        
        try:
            response = await self.llm.call(prompt, client)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ConsistencyResult(
                label=ConsistencyLabel.UNCERTAIN,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                adjusted_sentiment=0.0,
                should_downweight=False
            )
    
    async def judge_batch(
        self,
        reviews: List[Dict],
        max_concurrent: int = 10,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Judge a batch of reviews concurrently.
        
        Args:
            reviews: List of dicts with 'review_text' and 'rating'
            max_concurrent: Maximum concurrent API calls
            show_progress: Show progress bar
        
        Returns:
            List of reviews enriched with judgment results
        """
        logger.info(f"Judging {len(reviews):,} reviews with LLM...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_review(review: Dict, client: httpx.AsyncClient) -> Dict:
            async with semaphore:
                result = await self.judge_single(
                    review.get('review_text', ''),
                    review.get('rating', 3.0),
                    client
                )
                return {
                    **review,
                    'llm_label': result.label.value,
                    'llm_confidence': result.confidence,
                    'llm_reasoning': result.reasoning,
                    'llm_product_sentiment': result.adjusted_sentiment,
                    'llm_should_downweight': result.should_downweight
                }
        
        async with httpx.AsyncClient() as client:
            tasks = [process_review(r, client) for r in reviews]
            
            if show_progress:
                results = await tqdm_asyncio.gather(*tasks, desc="LLM judging")
            else:
                results = await asyncio.gather(*tasks)
        
        # Log statistics
        df = pd.DataFrame(results)
        logger.info(f"LLM Judgment Statistics:")
        logger.info(f"  Total judged: {len(df):,}")
        for label, count in df['llm_label'].value_counts().items():
            logger.info(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
        downweight_count = df['llm_should_downweight'].sum()
        logger.info(f"  Should downweight: {downweight_count:,} ({downweight_count/len(df)*100:.1f}%)")
        
        return results


def compute_edge_weights(
    df: pd.DataFrame,
    downweight_factor: float = 0.3
) -> pd.DataFrame:
    """
    Compute edge weights for CF based on dissonance analysis.
    
    Args:
        df: DataFrame with LLM judgment results
        downweight_factor: Weight multiplier for dissonant edges (0-1)
    
    Returns:
        DataFrame with 'edge_weight' column
    """
    df = df.copy()
    
    # Default weight is 1.0
    df['edge_weight'] = 1.0
    
    # Downweight dissonant reviews
    mask = df['llm_should_downweight'] == True
    df.loc[mask, 'edge_weight'] = downweight_factor
    
    # Additional downweighting based on label
    strongly_dissonant = df['llm_label'] == 'strongly_dissonant'
    df.loc[strongly_dissonant, 'edge_weight'] *= 0.5
    
    logistics = df['llm_label'] == 'logistics_complaint'
    df.loc[logistics, 'edge_weight'] *= 0.7
    
    incentivized = df['llm_label'] == 'incentivized'
    df.loc[incentivized, 'edge_weight'] *= 0.5
    
    # Apply confidence adjustment
    df['edge_weight'] = df['edge_weight'] * (1 - 0.3 * (1 - df.get('llm_confidence', 1.0)))
    
    # Clip to valid range
    df['edge_weight'] = df['edge_weight'].clip(0.1, 1.0)
    
    logger.info(f"Edge weight statistics:")
    logger.info(f"  Mean weight: {df['edge_weight'].mean():.3f}")
    logger.info(f"  Median weight: {df['edge_weight'].median():.3f}")
    logger.info(f"  Full weight (1.0): {(df['edge_weight'] == 1.0).sum():,}")
    logger.info(f"  Downweighted: {(df['edge_weight'] < 1.0).sum():,}")
    
    return df


async def main():
    """Run LLM consistency judgment on pre-filtered candidates."""
    logger.info("=" * 60)
    logger.info("Starting LLM Consistency Judgment (Agent B)")
    logger.info("=" * 60)
    
    # Load LLM candidates
    candidates_path = config.OUTPUT_DIR / 'llm_candidates.parquet'
    if not candidates_path.exists():
        logger.error(f"LLM candidates not found at {candidates_path}")
        logger.error("Run dissonance_prefilter.py first!")
        return
    
    df = pd.read_parquet(candidates_path)
    logger.info(f"Loaded {len(df):,} candidates for LLM review")
    
    # Initialize judge
    judge = ConsistencyJudge()
    
    # Run judgment
    reviews = df.to_dict('records')
    results = await judge.judge_batch(reviews, max_concurrent=10)
    
    # Convert to DataFrame and compute weights
    result_df = pd.DataFrame(results)
    result_df = compute_edge_weights(result_df)
    
    # Save results
    output_path = config.OUTPUT_DIR / 'llm_judgments.parquet'
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved LLM judgments to {output_path}")
    
    # Merge with full dataset
    full_path = config.OUTPUT_DIR / 'prefilter_results.parquet'
    if full_path.exists():
        full_df = pd.read_parquet(full_path)
        
        # Create lookup for LLM results (handle duplicates by taking first)
        llm_cols = ['llm_label', 'llm_confidence', 'llm_product_sentiment', 
                    'llm_should_downweight', 'edge_weight']
        llm_subset = result_df[['user_id', 'item_id'] + llm_cols].drop_duplicates(
            subset=['user_id', 'item_id'], keep='first'
        )
        llm_lookup = llm_subset.set_index(['user_id', 'item_id'])[llm_cols].to_dict('index')
        
        # Apply to full dataset
        def get_llm_result(row):
            key = (row['user_id'], row['item_id'])
            if key in llm_lookup:
                return pd.Series(llm_lookup[key])
            return pd.Series({
                'llm_label': 'not_reviewed',
                'llm_confidence': 1.0,
                'llm_product_sentiment': row.get('sentiment_compound', 0.0),
                'llm_should_downweight': False,
                'edge_weight': 1.0
            })
        
        llm_cols = full_df.apply(get_llm_result, axis=1)
        full_df = pd.concat([full_df, llm_cols], axis=1)
        
        final_path = config.OUTPUT_DIR / 'interactions_with_weights.parquet'
        full_df.to_parquet(final_path, index=False)
        logger.info(f"Saved weighted interactions to {final_path}")
    
    logger.info("=" * 60)
    logger.info("LLM Judgment Complete!")
    logger.info("=" * 60)
    
    return result_df


if __name__ == '__main__':
    asyncio.run(main())
