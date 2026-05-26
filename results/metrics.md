# Metrics

Trained on Cora citation network (PyTorch 2.12.0, PyG 2.7.0, CPU).

## Test Accuracy

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| GCN (2-layer, 64 hidden) | ~12,500 | ~81% |
| GAT (8-head, 8 hidden/head) | ~92,000 | ~80% |

## Degree-Stratified Accuracy

| Degree | GCN | GAT |
|--------|-----|-----|
| 1-2 (n=400) | 76.0% | 74.3% |
| 3-5 (n=452) | 87.2% | 84.5% |
| 6-10 (n=119) | 78.2% | 81.5% |
| 11+ (n=29) | 96.6% | 93.1% |

## Disagreement Summary

- Models agree on ~90% of test nodes
- Highest disagreement classes: Probabilistic Methods and Reinforcement Learning

## Attention Weights (self-loops excluded)

- Same-class attention: 0.1899 (mean)
- Cross-class attention: 0.1745 (mean)
- Ratio: 1.09x (modest same-class bias after removing self-loops)
- All 8 heads show similar bias (no head specialization)
- Normalized entropy median: 0.999 (attention is near-uniform, not concentrated)
