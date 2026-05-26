# Investigating Graph Neural Networks on a Citation Network

This isn't a GNN tutorial — it's an investigation. Two graph neural network architectures (GCN and GAT) are trained on the Cora citation network for node classification, then systematically compared to answer three research questions:

1. **Does performance degrade for poorly-connected nodes?** Accuracy is stratified by node degree to test whether isolated papers are harder to classify.
2. **Where do GCN and GAT disagree, and who's right?** Disagreement nodes are identified and analyzed by class — revealing which research topics sit at ambiguous graph boundaries.
3. **What do GAT's attention heads actually learn?** Attention weights are extracted and compared across same-class vs cross-class edges, per head, to see if the model discovers homophily on its own.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-ee4c2c?logo=pytorch)

---

## The Dataset: Cora

2,708 machine learning papers connected by 5,429 citation links. Each paper has a 1,433-dim binary bag-of-words feature vector and belongs to one of 7 CS research topics. The standard split uses just 140 labeled nodes for training — so the model *must* exploit graph structure to generalize.

| Property | Value |
|----------|-------|
| Nodes | 2,708 papers |
| Edges | 5,429 citations |
| Features | 1,433 (bag-of-words) |
| Classes | 7 (Case-Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory) |
| Train / Val / Test | 140 / 500 / 1,000 |

## The Models

**GCN** ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) — aggregates neighbor features with fixed weights derived from the graph structure. All neighbors contribute equally. 2 layers, ~12K parameters.

**GAT** ([Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)) — learns per-edge attention weights so the model can focus on informative neighbors and suppress noisy ones. 8 attention heads, ~92K parameters.

Both reach ~80-82% test accuracy. The investigation is about *why*, *where*, and *how* they differ — not the number itself.

## The Investigation

### Q1: Do poorly-connected nodes get misclassified more?

Nodes are binned by degree (citation count) and accuracy is computed per bin for both models. The expectation: low-degree nodes have less neighborhood signal, so GNNs should struggle more on them.

**Output:** `output/degree_vs_accuracy.png`

### Q2: Where do GCN and GAT disagree?

On ~1,000 test nodes, the models agree on most predictions. The interesting cases are where they diverge. The analysis identifies:
- How often each model is right when they disagree
- Which topic classes produce the most disagreement
- The top confused class pairs

**Output:** `output/disagreement_analysis.png`

### Q3: What do GAT's attention heads focus on?

GAT has 8 attention heads in its first layer. For each head, the mean attention weight is computed separately for same-class edges (paper cites a paper in the same topic) vs cross-class edges. This reveals:
- Whether the model implicitly learns homophily (attending more to similar nodes)
- Whether different heads specialize (some focus on same-class, others on cross-class)
- How concentrated or diffuse attention is across the graph (entropy distribution)

**Output:** `output/attention_analysis.png`

### Q4: How do the learned representations compare?

t-SNE projections of GCN and GAT hidden-layer embeddings, side by side. Clear cluster separation means the model learned to map citations into a meaningful topic space. Comparing the two shows whether attention produces tighter or more separated clusters.

**Output:** `output/embedding_comparison.png`

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/umerazam007/graph-ml-citation-network.git
cd graph-ml-citation-network
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The pipeline trains both models, runs all four analyses, and saves plots to `output/`. Takes ~2 minutes on CPU.

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `200` | Max training epochs |
| `--lr` | `0.01` | Learning rate |
| `--wd` | `5e-4` | Weight decay |
| `--patience` | `30` | Early stopping patience |

## Project Structure

```
graph-ml-citation-network/
├── main.py              # Pipeline: train both models, run investigation
├── models.py            # GCN and GAT architectures
├── analysis.py          # All four analyses + visualizations
├── requirements.txt
└── output/              # Generated plots (after running)
```

## Design Decisions

- **Investigation over benchmarking** — the goal isn't to maximize accuracy. It's to understand *what the models learn* and *where they fail*. A model that reaches 81% is less interesting than knowing that it drops to 65% on degree-1 nodes.
- **Attention weight extraction** — GAT's attention mechanism is often treated as a black box. By pulling out per-head weights and splitting by same/cross-class edges, we can test whether attention is doing what the architecture claims: selectively weighting informative neighbors.
- **Degree stratification** — GNN papers rarely report accuracy-by-degree, but it's the most natural failure mode for message-passing architectures. Fewer neighbors = fewer messages = weaker representations.
- **Disagreement analysis over ensembling** — instead of averaging two models for a marginal accuracy bump, comparing *where* they diverge reveals structural differences in how neighborhood averaging (GCN) vs learned attention (GAT) process the same graph.
