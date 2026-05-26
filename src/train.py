import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from src.models import GCN, GAT


def load_cora(device):
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(device)
    print(f"Dataset: Cora")
    print(f"  Nodes: {data.num_nodes}  |  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}  |  Classes: {dataset.num_classes}")
    print(f"  Train: {data.train_mask.sum()}  Val: {data.val_mask.sum()}  Test: {data.test_mask.sum()}")
    print()
    return dataset, data


def build_model(name, dataset, device):
    if name == "gcn":
        return GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
    elif name == "gat":
        return GAT(dataset.num_node_features, 8, dataset.num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")


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


def train(model, data, lr=0.01, weight_decay=5e-4, epochs=200, patience=30):
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
