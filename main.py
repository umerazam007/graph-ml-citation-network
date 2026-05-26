import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models import GCN, GAT
from analysis import (
    degree_stratified_accuracy,
    disagreement_analysis,
    attention_analysis,
    embedding_comparison,
    CLASS_NAMES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cora():
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(DEVICE)
    print(f"Dataset: Cora")
    print(f"  Nodes: {data.num_nodes}  |  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}  |  Classes: {dataset.num_classes}")
    print(f"  Train: {data.train_mask.sum()}  Val: {data.val_mask.sum()}  Test: {data.test_mask.sum()}")
    print()
    return dataset, data


def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = {}
    for split, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        accs[split] = (pred[mask] == data.y[mask]).float().mean().item()
    return accs, pred


def train_model(model, data, lr, weight_decay, epochs, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        loss = train_step(model, data, optimizer)
        accs, _ = evaluate(model, data)

        if accs["val"] > best_val_acc:
            best_val_acc = accs["val"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3d}  Loss: {loss:.4f}  "
                  f"Train: {accs['train']:.3f}  Val: {accs['val']:.3f}")

        if wait >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Graph ML Investigation on Cora")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    os.makedirs("output", exist_ok=True)

    dataset, data = load_cora()

    # ── Train both models ──────────────────────────────────────────────
    print("=" * 55)
    print("  PHASE 1: Training")
    print("=" * 55)

    print("\n  [GCN] 2-layer Graph Convolutional Network")
    gcn = GCN(dataset.num_node_features, 64, dataset.num_classes).to(DEVICE)
    gcn = train_model(gcn, data, args.lr, args.wd, args.epochs, args.patience)
    gcn_accs, gcn_pred = evaluate(gcn, data)
    print(f"    => Test accuracy: {gcn_accs['test']:.4f}\n")

    print("  [GAT] 8-head Graph Attention Network")
    gat = GAT(dataset.num_node_features, 8, dataset.num_classes).to(DEVICE)
    gat = train_model(gat, data, args.lr, args.wd, args.epochs, args.patience)
    gat_accs, gat_pred = evaluate(gat, data)
    print(f"    => Test accuracy: {gat_accs['test']:.4f}\n")

    # ── Investigation ──────────────────────────────────────────────────
    print("=" * 55)
    print("  PHASE 2: Investigation")
    print("=" * 55)
    print()

    print("-" * 55)
    print("  Q1: Does performance drop for low-degree nodes?")
    print("-" * 55)
    degree_stratified_accuracy(gcn_pred, gat_pred, data)

    print("-" * 55)
    print("  Q2: Where do GCN and GAT disagree?")
    print("-" * 55)
    disagreement_analysis(gcn_pred, gat_pred, data)

    print("-" * 55)
    print("  Q3: What do GAT's attention heads focus on?")
    print("-" * 55)
    attention_analysis(gat, data)

    print("-" * 55)
    print("  Q4: How do the learned representations compare?")
    print("-" * 55)
    embedding_comparison(gcn, gat, data)

    # ── Summary ────────────────────────────────────────────────────────
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  GCN test accuracy: {gcn_accs['test']:.4f}")
    print(f"  GAT test accuracy: {gat_accs['test']:.4f}")
    print(f"  Outputs saved to: output/")
    print(f"    - degree_vs_accuracy.png")
    print(f"    - disagreement_analysis.png")
    print(f"    - attention_analysis.png")
    print(f"    - embedding_comparison.png")
    print()


if __name__ == "__main__":
    main()
