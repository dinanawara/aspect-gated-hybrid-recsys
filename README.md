# Aspect-Gated Hybrid Recommender System

A hybrid recommendation system combining **Collaborative Filtering** and **Text Semantics** with **aspect-aware gating** to filter noise from logistics/shipping complaints in reviews.

## 📊 Results

| Model | HR@10 | NDCG@10 | Δ vs CF-only |
|-------|-------|---------|--------------|
| CF-only (BPR-MF) | 0.1815 | 0.1304 | — |
| + Text Hybrid (α=0.7) | 0.2037 | 0.1405 | +7.7% |
| **+ Aspect-Gated Penalty** | **0.2105** | **0.1412** | **+8.3%** |
| + LLM-Gated Penalty | 0.2037 | 0.1404 | +7.7% |

**Dataset**: Amazon Electronics 5-core (210K interactions, 500 items, 31K users)

## 🎯 Key Idea

Traditional recommender systems assume star ratings reflect true product preferences. In practice, reviews contain:
- **Logistics complaints** (shipping, packaging) unrelated to product quality
- **Seller/service issues** (returns, customer service)
- **External factors** that don't reflect actual product quality

Our approach **gates the penalty** — only applying it when:
1. CF and text signals disagree (ambiguous items)
2. The item has high ratio of logistics/external complaints

## 🧠 Architecture

### Hybrid Scoring
```
score(u,i) = α·CF(u,i) + (1-α)·text_sim(u,i) - gated_penalty(i)
```

### Components
1. **BPR-MF**: Bayesian Personalized Ranking with 64-dim embeddings
2. **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
3. **Aspect Classifier**: Keywords/LLM to detect logistics vs product reviews
4. **Gated Penalty**: Applied only when CF-text disagree AND item has high external ratio

## 📁 Project Structure

```
.
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── run_pipeline.py                # Main orchestration script
├── run_aspect_aware_hybrid.py     # Heuristic-gated experiment
├── run_llm_gated_hybrid.py        # LLM-gated experiment
├── src/
│   ├── bpr_mf.py                  # BPR-MF model & trainer
│   ├── hybrid_reranking.py        # Hybrid scorers (HybridScorer, AspectAwareHybridScorer, LLMGatedHybridScorer)
│   ├── aspect_classifier.py       # Keyword-based aspect classifier
│   ├── llm_aspect_classifier.py   # LLM-based aspect classifier
│   ├── text_embeddings.py         # User/item text embeddings
│   ├── evaluation.py              # HR@K, NDCG@K, MRR
│   └── data_preprocessing.py      # Data loading & filtering
├── output/
│   ├── processed/                 # Train/val/test splits, mappings
│   ├── *_results.json             # Experiment results
│   └── *.npy                      # Text embeddings (generated)
└── models/
    └── baseline_bpr.pt            # Trained BPR-MF checkpoint
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the [Amazon Electronics 5-core dataset](http://jmcauley.ucsd.edu/data/amazon/) and place in `data/`:
```bash
mkdir -p data
# Download Electronics_5.json to data/
```

### 3. Run Experiments

```bash
# Full pipeline (preprocessing + training + evaluation)
python run_pipeline.py

# Aspect-aware hybrid experiment (heuristic-gated)
python run_aspect_aware_hybrid.py

# LLM-gated experiment (requires API key in .env)
python run_llm_gated_hybrid.py
```

## 📈 Evaluation Metrics

- **HR@K**: Hit Rate at K (fraction of test items in top-K)
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

## 🔬 Method Details

### Aspect-Aware Gating Logic

```python
# Only penalize items where:
# 1. CF likes it (rank <= 50) but text doesn't (rank > 150)
# 2. AND item has high logistics/external review ratio

is_ambiguous = (cf_rank <= 50) & (text_rank > 150)
gated_penalty = logistics_ratio * penalty_weight * is_ambiguous
```

### Why Gating Matters

Without gating, penalizing all items with logistics complaints can hurt good products that happen to have some shipping issues. Gating focuses the penalty only where CF and text disagree — exactly where the logistics noise is most likely to cause ranking errors.

## 📝 Key Findings

1. **Hybrid CF+Text improves over CF-only** by +7.7% NDCG
2. **Aspect-aware gating adds +0.6%** additional improvement (total +8.3%)
3. **Heuristic keywords outperformed LLM** for this task (LLM was too conservative)
4. ~12% of candidates are "ambiguous" (CF-text disagree), 3% get penalized

## 📄 License

MIT License
