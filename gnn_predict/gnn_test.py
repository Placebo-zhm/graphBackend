import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges


# 1. 数据预处理
def load_data(filename):
    with open(filename) as f:
        data = json.load(f)

    # 收集所有唯一的节点
    nodes = set()
    edges = []
    for entry in data:
        source = entry["keyword"]
        nodes.add(source)
        for target, weight in entry["cite_num"].items():
            nodes.add(target)
            edges.append((source, target, weight))

    # 创建节点到索引的映射
    node_list = list(nodes)
    node2idx = {node: idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # 创建边索引和边属性（确保形状正确）
    edge_index = []
    edge_attr = []
    for src, dst, _ in edges:
        edge_index.append([node2idx[src], node2idx[dst]])  # 添加边索引

    # 转换为张量并调整形状
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 关键修复
    edge_attr = torch.tensor([w for _, _, w in edges], dtype=torch.float).view(-1, 1)

    # 构建PyG Data对象
    data = Data(
        x=torch.randn(num_nodes, 16),  # 随机初始化节点特征
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data, node2idx


# 2. 定义GNN模型
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_channels, 128)
        self.lin2 = torch.nn.Linear(128, 1)

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        out = torch.cat([src, dst], dim=1)
        out = self.lin1(out).relu()
        return self.lin2(out)


# 3. 训练函数
def train(model, predictor, data, optimizer):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)
    pos_pred = predictor(z, data.train_pos_edge_index)
    neg_pred = predictor(z, data.train_neg_edge_index)

    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


# 4. 主程序
def main():
    # 加载数据
    data, node2idx = load_data("../data/jsondata/keyword_citation_count.json")

    # 划分训练/测试边
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

    # 初始化模型
    model = GCNEncoder(in_channels=16, out_channels=64)
    predictor = LinkPredictor(in_channels=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    # 训练循环
    for epoch in range(1, 201):
        loss = train(model, predictor, data, optimizer)
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # 保存模型
    torch.save({
        'model_state': model.state_dict(),
        'predictor_state': predictor.state_dict()
    }, 'gnn_model.pth')

    # 提取嵌入
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).numpy()

    # 降维可视化
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10)
    plt.title("Node Embeddings Visualization")
    plt.savefig("embeddings.png")
    plt.close()


# 5. 加载模型函数
def load_model():
    model = GCNEncoder(16, 64)
    predictor = LinkPredictor(64)
    checkpoint = torch.load('gnn_model.pth')
    model.load_state_dict(checkpoint['model_state'])
    predictor.load_state_dict(checkpoint['predictor_state'])
    return model, predictor


if __name__ == "__main__":
    main()