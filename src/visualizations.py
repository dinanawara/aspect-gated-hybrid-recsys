"""
Visualization module for experiment results.

Creates publication-ready figures for:
1. Dissonance type distribution
2. Rating-sentiment scatter plot
3. Model comparison bar charts
4. Agent validation metrics
"""
import json
import logging
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

COLORS = {
    'baseline': '#3498db',
    'semantic': '#e74c3c',
    'consistent': '#27ae60',
    'dissonant': '#e74c3c',
    'logistics': '#f39c12',
    'incentivized': '#9b59b6'
}


def plot_dissonance_distribution(df: pd.DataFrame, output_path: Path):
    """Plot distribution of dissonance types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mismatch type distribution
    ax1 = axes[0]
    mismatch_counts = df['mismatch_type'].value_counts()
    colors = [COLORS.get(t.split('_')[0], '#95a5a6') for t in mismatch_counts.index]
    bars = ax1.bar(range(len(mismatch_counts)), mismatch_counts.values, color=colors)
    ax1.set_xticks(range(len(mismatch_counts)))
    ax1.set_xticklabels([t.replace('_', '\n') for t in mismatch_counts.index], rotation=0)
    ax1.set_ylabel('Number of Reviews')
    ax1.set_title('Distribution of Rating-Text Mismatch Types')
    
    # Add value labels on bars
    for bar, val in zip(bars, mismatch_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # Rating distribution by mismatch
    ax2 = axes[1]
    for mtype in ['consistent', 'mild_mismatch', 'strong_mismatch']:
        if mtype in df['mismatch_type'].values:
            subset = df[df['mismatch_type'] == mtype]
            ax2.hist(subset['rating'], bins=5, alpha=0.6, label=mtype.replace('_', ' ').title())
    ax2.set_xlabel('Star Rating')
    ax2.set_ylabel('Count')
    ax2.set_title('Rating Distribution by Mismatch Type')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'dissonance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved dissonance distribution plot to {output_path / 'dissonance_distribution.png'}")


def plot_sentiment_rating_scatter(df: pd.DataFrame, output_path: Path):
    """Scatter plot of sentiment vs rating with dissonance highlighting."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample for visibility if too many points
    sample_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    
    # Color by needs_llm_review
    colors = sample_df['needs_llm_review'].map({True: COLORS['dissonant'], False: COLORS['consistent']})
    
    scatter = ax.scatter(
        sample_df['rating'] + np.random.normal(0, 0.1, len(sample_df)),  # Jitter
        sample_df['sentiment_compound'],
        c=colors,
        alpha=0.3,
        s=20
    )
    
    # Add diagonal reference line (perfect alignment)
    x_line = np.array([1, 5])
    y_line = (x_line - 3) / 2  # Map 1-5 to -1 to 1
    ax.plot(x_line, y_line, 'k--', alpha=0.5, label='Perfect Alignment')
    
    # Add dissonance zones
    ax.axhspan(0.3, 1.0, xmin=0, xmax=0.4, alpha=0.1, color='red', label='High Rating Expected')
    ax.axhspan(-1.0, -0.3, xmin=0.6, xmax=1.0, alpha=0.1, color='red')
    
    ax.set_xlabel('Star Rating', fontsize=12)
    ax.set_ylabel('VADER Sentiment Score', fontsize=12)
    ax.set_title('Rating vs. Review Sentiment\n(Red = Flagged for LLM Review)', fontsize=14)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-1.1, 1.1)
    
    # Legend
    consistent_patch = mpatches.Patch(color=COLORS['consistent'], alpha=0.5, label='Consistent')
    dissonant_patch = mpatches.Patch(color=COLORS['dissonant'], alpha=0.5, label='Needs LLM Review')
    ax.legend(handles=[consistent_patch, dissonant_patch], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path / 'sentiment_rating_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sentiment-rating scatter to {output_path / 'sentiment_rating_scatter.png'}")


def plot_llm_judgment_results(df: pd.DataFrame, output_path: Path):
    """Plot LLM judgment distribution and edge weights."""
    if 'llm_label' not in df.columns:
        logger.warning("No LLM labels found, skipping LLM visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LLM label distribution
    ax1 = axes[0]
    label_counts = df['llm_label'].value_counts()
    colors = [COLORS.get(l.split('_')[0], '#95a5a6') for l in label_counts.index]
    bars = ax1.barh(range(len(label_counts)), label_counts.values, color=colors)
    ax1.set_yticks(range(len(label_counts)))
    ax1.set_yticklabels([l.replace('_', ' ').title() for l in label_counts.index])
    ax1.set_xlabel('Number of Reviews')
    ax1.set_title('LLM Consistency Judgments')
    
    # Edge weight distribution
    ax2 = axes[1]
    if 'edge_weight' in df.columns:
        ax2.hist(df['edge_weight'], bins=20, color=COLORS['baseline'], edgecolor='white')
        ax2.axvline(x=1.0, color='green', linestyle='--', label='Full Weight')
        ax2.axvline(x=df['edge_weight'].mean(), color='red', linestyle='--', 
                   label=f'Mean ({df["edge_weight"].mean():.2f})')
        ax2.set_xlabel('Edge Weight')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Edge Weights')
        ax2.legend()
    
    # Downweight by rating
    ax3 = axes[2]
    if 'llm_should_downweight' in df.columns:
        downweight_by_rating = df.groupby('rating')['llm_should_downweight'].mean()
        ax3.bar(downweight_by_rating.index, downweight_by_rating.values * 100, 
               color=COLORS['semantic'])
        ax3.set_xlabel('Star Rating')
        ax3.set_ylabel('% Downweighted')
        ax3.set_title('Downweight Rate by Rating')
        ax3.set_xticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    plt.savefig(output_path / 'llm_judgment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved LLM judgment results to {output_path / 'llm_judgment_results.png'}")


def plot_model_comparison(results: Dict, output_path: Path):
    """Bar chart comparing baseline vs semantic-aware model."""
    if 'baseline' not in results or 'semantic_aware' not in results:
        logger.warning("Need both baseline and semantic_aware results for comparison")
        return
    
    metrics = ['hr@5', 'hr@10', 'ndcg@5', 'ndcg@10', 'mrr']
    available_metrics = [m for m in metrics if m in results['baseline']]
    
    if not available_metrics:
        logger.warning("No metrics found in results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar comparison
    ax1 = axes[0]
    x = np.arange(len(available_metrics))
    width = 0.35
    
    baseline_vals = [results['baseline'].get(m, 0) for m in available_metrics]
    semantic_vals = [results['semantic_aware'].get(m, 0) for m in available_metrics]
    
    bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color=COLORS['baseline'])
    bars2 = ax1.bar(x + width/2, semantic_vals, width, label='Semantic-Aware', color=COLORS['semantic'])
    
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in available_metrics])
    ax1.legend()
    ax1.set_ylim(0, max(max(baseline_vals), max(semantic_vals)) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Improvement chart
    ax2 = axes[1]
    improvements = []
    for m in available_metrics:
        baseline = results['baseline'].get(m, 0)
        semantic = results['semantic_aware'].get(m, 0)
        if baseline > 0:
            imp = (semantic - baseline) / baseline * 100
        else:
            imp = 0
        improvements.append(imp)
    
    colors = [COLORS['semantic'] if i >= 0 else '#e74c3c' for i in improvements]
    bars = ax2.bar(x, improvements, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Relative Improvement over Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in available_metrics])
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison to {output_path / 'model_comparison.png'}")
    
    # Also create a summary table image
    create_results_table(results, available_metrics, output_path)


def create_results_table(results: Dict, metrics: list, output_path: Path):
    """Create a formatted results table as an image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Prepare data
    rows = ['Baseline', 'Semantic-Aware', 'Improvement']
    cols = [m.upper() for m in metrics]
    
    cell_data = []
    for model in ['baseline', 'semantic_aware']:
        row = [f"{results[model].get(m, 0):.4f}" for m in metrics]
        cell_data.append(row)
    
    # Improvement row
    imp_row = []
    for m in metrics:
        baseline = results['baseline'].get(m, 0)
        semantic = results['semantic_aware'].get(m, 0)
        if baseline > 0:
            imp = (semantic - baseline) / baseline * 100
            imp_row.append(f"{imp:+.2f}%")
        else:
            imp_row.append("N/A")
    cell_data.append(imp_row)
    
    table = ax.table(
        cellText=cell_data,
        rowLabels=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(cols),
        rowColours=['#e3f2fd', '#ffebee', '#e8f5e9']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    plt.title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path / 'results_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved results table to {output_path / 'results_table.png'}")


def plot_training_curves(history: Dict, output_path: Path, model_name: str = 'model'):
    """Plot training loss curves."""
    if 'train_loss' not in history:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    
    if 'val_metrics' in history and history['val_metrics']:
        val_ndcg = [m.get('ndcg@10', 0) for m in history['val_metrics']]
        ax2 = ax.twinx()
        ax2.plot(epochs[:len(val_ndcg)], val_ndcg, 'r-', label='Val NDCG@10')
        ax2.set_ylabel('NDCG@10', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(f'{model_name} Training Curves')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    if 'val_metrics' in history:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_training.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_example_dissonant_reviews(df: pd.DataFrame, output_path: Path, n_examples: int = 10):
    """Create a figure showing example dissonant reviews."""
    if 'llm_label' not in df.columns:
        # Use pre-filter mismatch instead
        dissonant = df[df['mismatch_type'].isin(['strong_mismatch', 'high_rating_negative_text', 'low_rating_positive_text'])]
    else:
        dissonant = df[df['llm_label'].isin(['strongly_dissonant', 'logistics_complaint', 'incentivized'])]
    
    if len(dissonant) == 0:
        return
    
    examples = dissonant.sample(min(n_examples, len(dissonant)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(14, n_examples * 0.8))
    ax.axis('off')
    
    text = "Example Dissonant Reviews\n" + "="*80 + "\n\n"
    
    for i, (_, row) in enumerate(examples.iterrows()):
        rating = row['rating']
        review = row.get('review_text', '')[:150] + '...'
        mtype = row.get('llm_label', row.get('mismatch_type', 'unknown'))
        
        text += f"{i+1}. [{rating}★] {mtype}\n"
        text += f"   \"{review}\"\n\n"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path / 'example_dissonant_reviews.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved example reviews to {output_path / 'example_dissonant_reviews.png'}")


def generate_all_visualizations():
    """Generate all visualizations from saved results."""
    output_path = config.OUTPUT_DIR
    output_path.mkdir(exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    # 1. Pre-filter results
    prefilter_path = output_path / 'prefilter_results.parquet'
    if prefilter_path.exists():
        df = pd.read_parquet(prefilter_path)
        plot_dissonance_distribution(df, output_path)
        plot_sentiment_rating_scatter(df, output_path)
        plot_example_dissonant_reviews(df, output_path)
    
    # 2. LLM judgments
    llm_path = output_path / 'interactions_with_weights.parquet'
    if llm_path.exists():
        df = pd.read_parquet(llm_path)
        plot_llm_judgment_results(df, output_path)
    
    # 3. Model comparison
    results_path = output_path / 'experiment_results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        plot_model_comparison(results, output_path)
    
    logger.info("All visualizations generated!")
    
    # List generated files
    viz_files = list(output_path.glob('*.png'))
    logger.info(f"Generated {len(viz_files)} visualization files:")
    for f in viz_files:
        logger.info(f"  - {f.name}")


if __name__ == '__main__':
    generate_all_visualizations()
