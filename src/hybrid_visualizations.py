"""
Visualization for Hybrid Reranking Results.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)


def create_hybrid_comparison(results_path: Path, output_dir: Path):
    """Create visualizations for hybrid approach results."""
    
    with open(results_path) as f:
        data = json.load(f)
    
    final_results = data['final_results']
    grid_results = data['grid_search_results']
    best_params = data['best_params']
    
    # 1. Bar chart comparing all approaches
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    models = list(final_results.keys())
    metrics = ['hr@10', 'ndcg@10', 'mrr']
    titles = ['Hit Rate @10', 'NDCG @10', 'MRR']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [final_results[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(metric.upper())
        ax.set_xticklabels(models, rotation=15, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=10)
        
        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('#00FF00')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Grid search heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    alphas = sorted(set(r['alpha'] for r in grid_results))
    lambdas = sorted(set(r['lambda'] for r in grid_results))
    
    for ax, metric in zip(axes, ['hr@10', 'ndcg@10']):
        grid = np.zeros((len(alphas), len(lambdas)))
        
        for r in grid_results:
            i = alphas.index(r['alpha'])
            j = lambdas.index(r['lambda'])
            grid[i, j] = r[metric]
        
        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f'{l:.2f}' for l in lambdas])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f'{a:.2f}' for a in alphas])
        ax.set_xlabel('λ (Penalty Weight)')
        ax.set_ylabel('α (CF Weight)')
        ax.set_title(f'{metric.upper()} Grid Search', fontweight='bold')
        
        # Add text annotations
        for i in range(len(alphas)):
            for j in range(len(lambdas)):
                ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center', 
                       fontsize=9, color='white' if grid[i, j] > grid.mean() else 'black')
        
        # Mark best
        best_i = alphas.index(best_params['alpha'])
        best_j = lambdas.index(best_params['lambda'])
        ax.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1, 
                                  fill=False, edgecolor='lime', linewidth=3))
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_grid_search.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Improvement summary
    fig, ax = plt.subplots(figsize=(8, 6))
    
    baseline_ndcg = final_results['CF-only']['ndcg@10']
    improvements = {
        m: (final_results[m]['ndcg@10'] - baseline_ndcg) / baseline_ndcg * 100
        for m in models
    }
    
    bars = ax.barh(list(improvements.keys()), list(improvements.values()),
                  color=['#808080', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('NDCG@10 Improvement (%)')
    ax.set_title('Relative Improvement over CF-only Baseline', fontweight='bold')
    
    for bar, val in zip(bars, improvements.values()):
        x = bar.get_width() + (0.5 if val >= 0 else -0.5)
        ax.annotate(f'{val:+.1f}%', xy=(x, bar.get_y() + bar.get_height()/2),
                   va='center', ha='left' if val >= 0 else 'right', fontsize=11)
    
    ax.set_xlim(-15, 20)
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved:")
    print(f"  - {output_dir / 'hybrid_comparison.png'}")
    print(f"  - {output_dir / 'hybrid_grid_search.png'}")
    print(f"  - {output_dir / 'hybrid_improvement.png'}")


def main():
    results_path = config.OUTPUT_DIR / 'hybrid_results.json'
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    create_hybrid_comparison(results_path, config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
