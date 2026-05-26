import argparse
import os
import torch

from src.train import load_cora, build_model, train, evaluate
from src.analysis import (
    degree_stratified_accuracy,
    disagreement_analysis,
    attention_analysis,
    embedding_comparison,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Graph ML Investigation on Cora")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    os.makedirs("results", exist_ok=True)

    dataset, data = load_cora(DEVICE)

    # -- Train both models -------------------------------------------------
    print("=" * 55)
    print("  PHASE 1: Training")
    print("=" * 55)

    print("\n  [GCN] 2-layer Graph Convolutional Network")
    gcn = build_model("gcn", dataset, DEVICE)
    gcn = train(gcn, data, lr=args.lr, weight_decay=args.wd,
                epochs=args.epochs, patience=args.patience)
    gcn_accs, gcn_pred = evaluate(gcn, data)
    print(f"    => Test accuracy: {gcn_accs['test']:.4f}\n")

    print("  [GAT] 8-head Graph Attention Network")
    gat = build_model("gat", dataset, DEVICE)
    gat = train(gat, data, lr=args.lr, weight_decay=args.wd,
                epochs=args.epochs, patience=args.patience)
    gat_accs, gat_pred = evaluate(gat, data)
    print(f"    => Test accuracy: {gat_accs['test']:.4f}\n")

    # -- Investigation -----------------------------------------------------
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

    # -- Summary -----------------------------------------------------------
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  GCN test accuracy: {gcn_accs['test']:.4f}")
    print(f"  GAT test accuracy: {gat_accs['test']:.4f}")
    print(f"  Results saved to: results/")
    print()


if __name__ == "__main__":
    main()
