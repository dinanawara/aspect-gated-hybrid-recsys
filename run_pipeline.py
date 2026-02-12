"""
Main Pipeline for Robust Recommendation System.

Orchestrates the full experiment:
1. Data preprocessing
2. Dissonance pre-filtering
3. LLM consistency judgment
4. Model training (baseline + semantic-aware)
5. Evaluation

Run with: python run_pipeline.py [--skip-llm] [--quick]
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.OUTPUT_DIR / 'pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    """Step 1: Data preprocessing."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("=" * 60)
    
    from src.data_preprocessing import main as preprocess_main
    preprocess_main()


def run_prefilter():
    """Step 2: Dissonance pre-filtering."""
    logger.info("=" * 60)
    logger.info("STEP 2: Dissonance Pre-Filtering")
    logger.info("=" * 60)
    
    from src.dissonance_prefilter import main as prefilter_main
    prefilter_main()


async def run_llm_judgment():
    """Step 3: LLM consistency judgment."""
    logger.info("=" * 60)
    logger.info("STEP 3: LLM Consistency Judgment")
    logger.info("=" * 60)
    
    from src.llm_consistency_judge import main as llm_main
    await llm_main()


def run_training():
    """Step 4: Model training."""
    logger.info("=" * 60)
    logger.info("STEP 4: Model Training")
    logger.info("=" * 60)
    
    from src.bpr_mf import main as training_main
    training_main()


def run_evaluation():
    """Step 5: Final evaluation and reporting."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluation & Reporting")
    logger.info("=" * 60)
    
    from src.evaluation import print_comparison_table
    
    results_path = config.OUTPUT_DIR / 'experiment_results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print_comparison_table(results)
    else:
        logger.warning("No results found. Run training first.")


def main():
    parser = argparse.ArgumentParser(description="Run the Robust Recommendation Pipeline")
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM judgment step (use pre-filter only)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: use smaller sample sizes'
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only a specific step'
    )
    parser.add_argument(
        '--create-labeling-sample',
        action='store_true',
        help='Create sample for manual labeling (for agent validation)'
    )
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        config.TOP_VARIANCE_ITEMS = 500
        logger.info("Quick mode: using reduced sample sizes")
    
    # Create labeling sample
    if args.create_labeling_sample:
        from src.evaluation import create_manual_labeling_sample
        import pandas as pd
        
        prefilter_path = config.OUTPUT_DIR / 'prefilter_results.parquet'
        if prefilter_path.exists():
            df = pd.read_parquet(prefilter_path)
            create_manual_labeling_sample(df, n_samples=100)
        else:
            logger.error("Pre-filter results not found. Run step 2 first.")
        return
    
    # Run specific step or full pipeline
    if args.step:
        if args.step == 1:
            run_preprocessing()
        elif args.step == 2:
            run_prefilter()
        elif args.step == 3:
            if args.skip_llm:
                logger.info("Skipping LLM step as requested.")
            else:
                asyncio.run(run_llm_judgment())
        elif args.step == 4:
            run_training()
        elif args.step == 5:
            run_evaluation()
    else:
        # Full pipeline
        logger.info("Running full pipeline...")
        
        run_preprocessing()
        run_prefilter()
        
        if not args.skip_llm:
            asyncio.run(run_llm_judgment())
        else:
            logger.info("Skipping LLM judgment (--skip-llm)")
            # Create dummy weights for training
            import pandas as pd
            prefilter_path = config.OUTPUT_DIR / 'prefilter_results.parquet'
            if prefilter_path.exists():
                df = pd.read_parquet(prefilter_path)
                # Use pre-filter mismatch score as weight proxy
                df['edge_weight'] = 1.0 - (df['mismatch_score'] * 0.5)
                df['llm_label'] = df['mismatch_type']
                df['llm_should_downweight'] = df['mismatch_score'] > 0.5
                df.to_parquet(config.OUTPUT_DIR / 'interactions_with_weights.parquet')
                logger.info("Created weights from pre-filter (no LLM)")
        
        run_training()
        run_evaluation()
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
