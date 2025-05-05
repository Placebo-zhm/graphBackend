import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F


# 1. 数据预处理
def load_data(filename):
    with open(filename) as f:
        data = json.load(f)

    # 收集所有唯一节点
    nodes = set()
    edges = []
    for entry in data:
        source = entry["keyword"]
        nodes.add(source)
        for target, weight in entry["cite_num"].items():
            nodes.add(target)
            edges.append((source, target, weight))

    # 创建节点到索引的映射
    nodes = list(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # 构建边索引和边属性
    edge_index = []
    edge_attr = []
    for src, tgt, w in edges:
        edge_index.append([node_idx[src], node_idx[tgt]])
        edge_attr.append(float(w))

    # 转换为PyG数据格式
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    x = torch.ones(len(nodes), 1)  # 初始化节点特征

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), nodes


# 2. 定义GNN模型
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 3. 链路预测模型
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim * 2, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


# 4. 训练函数
def train(model, predictor, data, optimizer):
    model.train()
    predictor.train()

    # 生成负样本
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1)
    )

    optimizer.zero_grad()

    # 获取节点嵌入
    z = model(data.x, data.edge_index)

    # 计算正负样本得分
    pos_out = predictor(z, data.edge_index)
    neg_out = predictor(z, neg_edge_index)

    # 计算损失
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()


# 5. 可视化函数
def visualize(z, labels):
    z = z.detach().cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=50)
    for i, label in enumerate(labels):
        plt.annotate(label, (z_2d[i, 0], z_2d[i, 1]), fontsize=8)
    plt.title("Node Embeddings Visualization")
    plt.show()


# 主程序
def main():
    # 加载数据
    data, nodes = load_data("../data/jsondata/keyword_citation_count.json")

    # 初始化模型
    model = GNN(in_dim=1, hidden_dim=128, out_dim=64)
    predictor = LinkPredictor(64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.01
    )

    # 训练
    for epoch in range(1, 201):
        loss = train(model, predictor, data, optimizer)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    # 保存模型
    torch.save({
        "model_state": model.state_dict(),
        "predictor_state": predictor.state_dict()
    }, "gnn_model.pth")

    # 生成嵌入并可视化
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    visualize(z, nodes)


# 加载模型的函数
def load_model():
    checkpoint = torch.load("gnn_model.pth")
    model = GNN(1, 128, 64)
    predictor = LinkPredictor(64)
    model.load_state_dict(checkpoint["model_state"])
    predictor.load_state_dict(checkpoint["predictor_state"])
    return model, predictor


if __name__ == "__main__":
    main()