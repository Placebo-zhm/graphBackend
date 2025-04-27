import json
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ================== 数据预处理 ==================
with open('../data/jsondata/keyword_citation_count.json', 'r') as f:
    json_data = json.load(f)

# 构建图结构
nodes = set()
edges = []
for entry in json_data:
    src = entry["keyword"]
    nodes.add(src)
    for dst, weight in entry["cite_num"].items():
        nodes.add(dst)
        edges.append((src, dst, weight))

# 节点映射和特征初始化
node_list = list(nodes)
node_to_idx = {node: idx for idx, node in enumerate(node_list)}
x = torch.eye(len(node_list))  # 独热编码特征

# 构建PyG数据对象
edge_index = [[node_to_idx[src], node_to_idx[dst]] for src, dst, _ in edges]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor([w for _, _, w in edges], dtype=torch.float)

graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ================== GNN模型定义 ==================
class EmbeddingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = EmbeddingModel(
    input_dim=len(node_list),
    hidden_dim=32,
    embed_dim=16
)

# ================== 模型训练 ==================
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    embeddings = model(graph_data.x, graph_data.edge_index)
    # 使用边权重重建任务进行训练
    pred_weights = (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1)
    loss = criterion(pred_weights, edge_attr)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# ================== 嵌入降维 ==================
model.eval()
with torch.no_grad():
    embeddings = model(graph_data.x, graph_data.edge_index).numpy()

# 使用t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
embeddings_2d = tsne.fit_transform(embeddings)

# ================== 可视化 ==================
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.6)

# 标注部分关键节点
#keywords_to_show = ["encoding", "deep learning", "virtual reality", "data mining"]
keywords_to_show = ["encoding"]
for idx, node in enumerate(node_list):
    if node in keywords_to_show:
        plt.annotate(node,
                     (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color='gray', alpha=0.3))

# 绘制实际存在的边
for src, dst in edge_index.t().numpy()[:200]:  # 只画部分边避免拥挤
    plt.plot([embeddings_2d[src, 0], embeddings_2d[dst, 0]],
             [embeddings_2d[src, 1], embeddings_2d[dst, 1]],
             linewidth=0.3,
             color='gray',
             alpha=0.4)

plt.title("GNN Node Embeddings Visualization (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()