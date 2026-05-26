import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from collections import Counter


CLASS_NAMES = [
    "Case_Based", "Genetic_Alg", "Neural_Nets",
    "Probabilistic", "Reinforce_Learn", "Rule_Learning", "Theory",
]

COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#6b7280"]


def degree_stratified_accuracy(gcn_pred, gat_pred, data):
    """Do models fail on poorly-connected nodes?"""
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    y_true = data.y.cpu().numpy()
    gcn_p = gcn_pred.cpu().numpy()
    gat_p = gat_pred.cpu().numpy()

    test_idx = np.where(test_mask)[0]
    test_deg = deg[test_idx]

    bins = [(0, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 999, "11+")]
    bin_labels, gcn_accs, gat_accs, bin_counts = [], [], [], []

    for lo, hi, label in bins:
        mask = (test_deg >= lo) & (test_deg <= hi)
        if mask.sum() == 0:
            continue
        idx = test_idx[mask]
        gcn_acc = (gcn_p[idx] == y_true[idx]).mean()
        gat_acc = (gat_p[idx] == y_true[idx]).mean()
        bin_labels.append(label)
        gcn_accs.append(gcn_acc)
        gat_accs.append(gat_acc)
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bin_labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, gcn_accs, w, label="GCN", color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x + w / 2, gat_accs, w, label="GAT", color="#8b5cf6", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Node Degree (number of citations)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Do Poorly-Connected Papers Get Misclassified More?")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(bin_labels, bin_counts)])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/degree_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Degree-Stratified Accuracy:")
    for label, gc, ga, n in zip(bin_labels, gcn_accs, gat_accs, bin_counts):
        delta = ga - gc
        arrow = "^" if delta > 0.01 else ("v" if delta < -0.01 else "=")
        print(f"    Degree {label:>5s} (n={n:>3d}):  GCN {gc:.3f}  GAT {ga:.3f}  [{arrow} {delta:+.3f}]")
    print()


def disagreement_analysis(gcn_pred, gat_pred, data):
    """Where do GCN and GAT disagree, and who's right?"""
    test_mask = data.test_mask.cpu()
    y_true = data.y.cpu()
    gcn_p = gcn_pred.cpu()
    gat_p = gat_pred.cpu()

    test_idx = torch.where(test_mask)[0]
    disagree_mask = gcn_p[test_idx] != gat_p[test_idx]
    disagree_idx = test_idx[disagree_mask]

    total_test = len(test_idx)
    n_disagree = len(disagree_idx)

    gcn_right = (gcn_p[disagree_idx] == y_true[disagree_idx]).sum().item()
    gat_right = (gat_p[disagree_idx] == y_true[disagree_idx]).sum().item()
    neither = n_disagree - gcn_right - gat_right

    print(f"  Disagreement Analysis:")
    print(f"    Test nodes: {total_test}")
    print(f"    Agree:    {total_test - n_disagree} ({(total_test - n_disagree) / total_test:.1%})")
    print(f"    Disagree: {n_disagree} ({n_disagree / total_test:.1%})")
    print(f"      GCN right: {gcn_right}  |  GAT right: {gat_right}  |  Both wrong: {neither}")
    print()

    if n_disagree == 0:
        return

    true_classes = y_true[disagree_idx].numpy()
    gcn_classes = gcn_p[disagree_idx].numpy()
    gat_classes = gat_p[disagree_idx].numpy()

    class_disagree_count = Counter(true_classes)
    class_total = Counter(y_true[test_idx].numpy())

    print(f"    Disagreements by true class:")
    for cls_idx in sorted(class_disagree_count.keys()):
        count = class_disagree_count[cls_idx]
        total = class_total[cls_idx]
        print(f"      {CLASS_NAMES[cls_idx]:<18s}: {count:>3d}/{total:>3d} ({count / total:.1%})")
    print()

    confusion_pairs = []
    for i in range(n_disagree):
        true_c = CLASS_NAMES[true_classes[i]]
        gcn_c = CLASS_NAMES[gcn_classes[i]]
        gat_c = CLASS_NAMES[gat_classes[i]]
        if gcn_classes[i] != true_classes[i]:
            confusion_pairs.append((true_c, gcn_c, "GCN"))
        if gat_classes[i] != true_classes[i]:
            confusion_pairs.append((true_c, gat_c, "GAT"))

    pair_counts = Counter([(t, p) for t, p, _ in confusion_pairs])
    top_confusions = pair_counts.most_common(5)

    print(f"    Top confused pairs (true -> predicted):")
    for (true_c, pred_c), count in top_confusions:
        print(f"      {true_c} -> {pred_c}: {count}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels_plot = [CLASS_NAMES[i] for i in sorted(class_disagree_count.keys())]
    rates = [class_disagree_count[i] / class_total[i] for i in sorted(class_disagree_count.keys())]
    bar_colors = [COLORS[i] for i in sorted(class_disagree_count.keys())]

    axes[0].barh(labels_plot, rates, color=bar_colors, alpha=0.85)
    axes[0].set_xlabel("Disagreement Rate")
    axes[0].set_title("Which Topics Do GCN & GAT Disagree On?")
    axes[0].set_xlim(0, max(rates) * 1.3 if rates else 1)
    for i, v in enumerate(rates):
        axes[0].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)

    categories = ["GCN right", "GAT right", "Both wrong"]
    values = [gcn_right, gat_right, neither]
    pie_colors = ["#3b82f6", "#8b5cf6", "#6b7280"]
    axes[1].pie(values, labels=categories, colors=pie_colors, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 10})
    axes[1].set_title(f"When They Disagree (n={n_disagree}), Who's Right?")

    plt.tight_layout()
    plt.savefig("output/disagreement_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/disagreement_analysis.png")


def attention_analysis(gat_model, data):
    """What do GAT's attention weights actually focus on?"""
    attn_edge_index, attn_weights = gat_model.get_attention_weights(data.x, data.edge_index)

    attn_edge_index = attn_edge_index.cpu()
    attn_weights = attn_weights.cpu()

    avg_attn = attn_weights.mean(dim=1)
    labels = data.y.cpu()
    src_labels = labels[attn_edge_index[0]]
    dst_labels = labels[attn_edge_index[1]]
    same_class = (src_labels == dst_labels)

    attn_same = avg_attn[same_class].numpy()
    attn_diff = avg_attn[~same_class].numpy()

    print(f"  Attention Weight Analysis (Layer 1, {attn_weights.shape[1]} heads, averaged):")
    print(f"    Total edges: {len(avg_attn)}")
    print(f"    Same-class edges:  {same_class.sum().item()} ({same_class.float().mean():.1%})")
    print(f"    Cross-class edges: {(~same_class).sum().item()} ({(~same_class).float().mean():.1%})")
    print(f"    Mean attention on same-class neighbors:  {attn_same.mean():.4f}")
    print(f"    Mean attention on cross-class neighbors: {attn_diff.mean():.4f}")
    ratio = attn_same.mean() / attn_diff.mean() if attn_diff.mean() > 0 else float("inf")
    print(f"    Ratio (same/cross): {ratio:.2f}x")
    print()

    per_head_same = []
    per_head_diff = []
    n_heads = attn_weights.shape[1]
    for h in range(n_heads):
        head_w = attn_weights[:, h]
        per_head_same.append(head_w[same_class].mean().item())
        per_head_diff.append(head_w[~same_class].mean().item())

    print(f"    Per-head breakdown:")
    for h in range(n_heads):
        delta = per_head_same[h] - per_head_diff[h]
        marker = "*" if abs(delta) > 0.005 else " "
        print(f"      Head {h}: same={per_head_same[h]:.4f}  cross={per_head_diff[h]:.4f}  "
              f"delta={delta:+.4f} {marker}")
    print()

    deg = torch.zeros(data.num_nodes)
    deg.scatter_add_(0, attn_edge_index[1], avg_attn)
    deg_count = torch.zeros(data.num_nodes)
    deg_count.scatter_add_(0, attn_edge_index[1], torch.ones_like(avg_attn))
    avg_received = (deg / deg_count.clamp(min=1)).numpy()

    node_labels = labels.numpy()
    print(f"    Mean attention received by class:")
    for i, name in enumerate(CLASS_NAMES):
        mask = node_labels == i
        if mask.sum() > 0:
            print(f"      {name:<18s}: {avg_received[mask].mean():.4f}")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(attn_same, bins=50, alpha=0.7, color="#10b981", label="Same class", density=True)
    axes[0].hist(attn_diff, bins=50, alpha=0.7, color="#ef4444", label="Cross class", density=True)
    axes[0].set_xlabel("Attention Weight")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Does GAT Attend More to Same-Class Neighbors?")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    x = np.arange(n_heads)
    w = 0.35
    axes[1].bar(x - w / 2, per_head_same, w, label="Same class", color="#10b981", alpha=0.85)
    axes[1].bar(x + w / 2, per_head_diff, w, label="Cross class", color="#ef4444", alpha=0.85)
    axes[1].set_xlabel("Attention Head")
    axes[1].set_ylabel("Mean Attention")
    axes[1].set_title("Per-Head: Same vs Cross-Class Attention")
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    entropies = []
    for node_id in range(data.num_nodes):
        mask = attn_edge_index[1] == node_id
        if mask.sum() > 0:
            w_node = avg_attn[mask]
            w_norm = w_node / w_node.sum()
            entropy = -(w_norm * w_norm.clamp(min=1e-9).log()).sum().item()
            entropies.append(entropy)
    entropies = np.array(entropies)

    axes[2].hist(entropies, bins=50, color="#8b5cf6", alpha=0.8)
    axes[2].axvline(np.median(entropies), color="#f59e0b", linestyle="--", label=f"Median: {np.median(entropies):.2f}")
    axes[2].set_xlabel("Attention Entropy")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Attention Concentration (Low = Focused, High = Uniform)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/attention_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/attention_analysis.png")


def embedding_comparison(gcn_model, gat_model, data):
    """t-SNE of both models' learned representations side-by-side."""
    from sklearn.manifold import TSNE

    gcn_model.eval()
    gat_model.eval()
    with torch.no_grad():
        gcn_emb = gcn_model.get_embeddings(data.x, data.edge_index).cpu().numpy()
        gat_emb = gat_model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    labels = data.y.cpu().numpy()

    print("  Running t-SNE on GCN embeddings...")
    gcn_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(gcn_emb)
    print("  Running t-SNE on GAT embeddings...")
    gat_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(gat_emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        ax1.scatter(gcn_2d[mask, 0], gcn_2d[mask, 1], c=COLORS[i], label=name, s=10, alpha=0.65)
        ax2.scatter(gat_2d[mask, 0], gat_2d[mask, 1], c=COLORS[i], label=name, s=10, alpha=0.65)

    ax1.set_title("GCN Embeddings (t-SNE)", fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title("GAT Embeddings (t-SNE)", fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(markerscale=3, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("output/embedding_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/embedding_comparison.png\n")
