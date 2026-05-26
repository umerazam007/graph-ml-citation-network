# Metrics

Trained on Cora citation network (PyTorch 2.12.0, CPU).

## Test Accuracy

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| GCN (2-layer, 64 hidden) | ~12,500 | 80.1% |
| GAT (8-head, 8 hidden/head) | ~92,000 | 80.6% |

## Degree-Stratified Accuracy

| Degree | GCN | GAT |
|--------|-----|-----|
| 1-2 (n=400) | 73.8% | 75.2% |
| 3-5 (n=452) | 85.0% | 84.3% |
| 6-10 (n=119) | 80.7% | 80.7% |
| 11+ (n=29) | 89.7% | 96.6% |

## Disagreement Summary

- Models agree on 88.8% of test nodes
- On 112 disagreements: GAT right 50, GCN right 45, both wrong 17
- Highest disagreement class: Probabilistic Methods (16.0%)

## Attention Weights

- Same-class attention: 0.2095 (mean)
- Cross-class attention: 0.1745 (mean)
- Ratio: 1.20x (GAT attends 20% more to same-class neighbors)
- All 8 heads show the same bias (no head specialization observed)
