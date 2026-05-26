# Investigating Graph Neural Networks on a Citation Network

Node classification on the **Cora citation network** using two GNN architectures (GCN and GAT), with a systematic investigation into where they fail, where they disagree, and what GAT's attention mechanism actually learns.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_2.12-Geometric_2.7-ee4c2c?logo=pytorch)

---

## Overview

Cora is a graph of 2,708 machine learning papers connected by 5,429 citation links. Each paper has a 1,433-dimensional bag-of-words feature vector and belongs to one of 7 CS research topics. The standard split uses just **140 labeled nodes** for training, forcing the model to exploit graph structure to generalize to 1,000 test nodes.

| Property | Value |
|----------|-------|
| Nodes | 2,708 papers |
| Edges | 5,429 citations |
| Features | 1,433 (bag-of-words) |
| Classes | 7 (Case-Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory) |
| Train / Val / Test | 140 / 500 / 1,000 |

## Approach

**GCN** ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) -- 2-layer Graph Convolutional Network. Aggregates neighbor features with fixed weights from the graph structure. All neighbors contribute equally. ~12K parameters.

**GAT** ([Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)) -- 8-head Graph Attention Network. Learns per-edge attention weights so the model can focus on informative neighbors. ~92K parameters.

Both trained with Adam (lr=0.01, weight decay=5e-4) and early stopping on validation accuracy.

## Results

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| GCN | ~12,500 | **~81%** |
| GAT | ~92,000 | **~80%** |

### Learned Embeddings (t-SNE)

![Embedding comparison](results/embedding_comparison.png)

Both models learn to separate the 7 topic classes in representation space. Visually, the cluster structure is similar between GCN and GAT, though t-SNE geometry should be read qualitatively -- distances between clusters are not directly meaningful.

---

## Key Findings

### 1. Low-degree nodes are the failure mode

![Degree vs accuracy](results/degree_vs_accuracy.png)

Papers with only 1-2 citations hit **~74-76% accuracy** -- roughly a 20-point drop compared to highly-cited papers (93-97%). This is consistent with known research on degree bias in GNNs ([Tang et al., 2020](https://arxiv.org/abs/2006.07337)): fewer neighbors means fewer messages, which means weaker representations. Message-passing architectures structurally disadvantage low-degree nodes, and this dataset confirms the pattern clearly.

### 2. Probabilistic Methods and Reinforcement Learning sit at the graph's semantic crossroads

![Disagreement analysis](results/disagreement_analysis.png)

GCN and GAT agree on **~90%** of test predictions. The ~10% disagreement is concentrated in specific topics: **Probabilistic Methods** and **Reinforcement Learning** together account for the majority of disagreements. These topics get confused with Case-Based, Neural Networks, and each other -- reflecting their position as methodological bridges that get cited across subfields. This isn't a model failure; it's a property of the citation graph's structure.

### 3. GAT's attention is near-uniform -- the mechanism isn't doing much on Cora

![Attention analysis](results/attention_analysis.png)

After **excluding self-loops** (which GATConv adds by default and which trivially inflate same-class attention), GAT's same-class bias drops to just **1.09x** -- a marginal preference. All 8 heads show similar deltas (~0.015), confirming no head specialization.

More strikingly, normalized attention entropy (H / ln(degree), where 1.0 = perfectly uniform) has a **median of 0.999**. Attention is essentially uniform across neighbors for nearly every node in the graph. GAT has the capacity to be selective, but on Cora it converges to treating all neighbors roughly equally -- functionally equivalent to what GCN does by design, but with 7x more parameters.

This complements the other findings: the redundant heads, the near-uniform entropy, and the marginal accuracy gap all point to the same conclusion -- Cora's homophily structure is simple enough that learned attention provides no real advantage over fixed neighborhood averaging.

---

## How to Run

### Prerequisites

- Python 3.8+
- PyTorch 2.12.0 (tested on CPU; CUDA works automatically if available)
- PyTorch Geometric 2.7.0

> **Note on PyG installation:** PyTorch Geometric depends on your exact PyTorch and CUDA versions. If `pip install torch-geometric` fails, check [the official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your setup.

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

Trains both models, runs all four analyses, and saves plots to `results/`. Takes ~2 minutes on CPU.

```bash
# Custom hyperparameters
python main.py --epochs 300 --lr 0.005 --patience 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `200` | Max training epochs |
| `--lr` | `0.01` | Learning rate |
| `--wd` | `5e-4` | Weight decay (L2) |
| `--patience` | `30` | Early stopping patience |

## Project Structure

```
graph-ml-citation-network/
├── main.py                # Entry point: train both models, run investigation
├── src/
│   ├── models.py          # GCN and GAT architectures
│   ├── train.py           # Data loading, training loop, evaluation
│   └── analysis.py        # All four analyses + visualization
├── results/
│   ├── degree_vs_accuracy.png
│   ├── disagreement_analysis.png
│   ├── attention_analysis.png
│   ├── embedding_comparison.png
│   └── metrics.md
├── requirements.txt
└── .gitignore
```

## What I Learned

The accuracy gap between GCN and GAT on Cora is negligible (~1%), but the investigation reveals *why*: GAT's attention mechanism converges to near-uniform weights, its 8 heads learn redundant strategies, and the same-class bias nearly vanishes once self-loops are properly excluded. GAT's machinery isn't broken -- it's unnecessary here. Cora's high homophily (81% of edges connect same-class nodes) means fixed averaging already captures most of the useful signal. The real predictor of classification difficulty isn't the model architecture -- it's node degree.

## References

- **Cora dataset:** McCallum et al., *Automating the Construction of Internet Portals with Machine Learning*, 2000
- **GCN:** Kipf & Welling, [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907), ICLR 2017
- **GAT:** Velickovic et al., [*Graph Attention Networks*](https://arxiv.org/abs/1710.10903), ICLR 2018
- **Degree bias in GNNs:** Tang et al., [*Investigating and Mitigating Degree-Related Biases in Graph Convolutional Networks*](https://arxiv.org/abs/2006.07337), 2020
- **PyTorch Geometric:** Fey & Lenssen, [*Fast Graph Representation Learning with PyTorch Geometric*](https://arxiv.org/abs/1903.02428), ICLR-W 2019
