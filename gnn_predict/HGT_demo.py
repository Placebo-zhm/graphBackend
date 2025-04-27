import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.loader import NeighborLoader

# Step 1: 准备一个简单的异构图数据

data = HeteroData()

# 添加节点特征（这里简单用随机向量模拟）
data['paper'].x = torch.randn(10, 16)  # 10篇论文，特征维度16
data['author'].x = torch.randn(5, 16)  # 5个作者
data['keyword'].x = torch.randn(6, 16)  # 6个关键词

# 添加边（注意：需要指明边的方向和类型）
data['author', 'writes', 'paper'].edge_index = torch.randint(0, 5, (2, 20))
data['paper', 'has', 'keyword'].edge_index = torch.randint(0, 10, (2, 30))
data['author', 'coauthor', 'author'].edge_index = torch.randint(0, 5, (2, 10))

# Step 2: 定义HGT模型

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, data_metadata, num_heads=2):
        super().__init__()
        self.hgt_conv1 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=data_metadata,
            heads=num_heads
        )
        self.hgt_conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            metadata=data_metadata,
            heads=num_heads
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.hgt_conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.hgt_conv2(x_dict, edge_index_dict)
        return x_dict

# Step 3: 初始化模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HGT(hidden_channels=16, out_channels=16, data_metadata=data.metadata()).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Step 4: 训练模型

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    # 这里只是简单对'keyword'节点做自监督学习
    loss = F.mse_loss(out['keyword'], data['keyword'].x)  # 假设用自己作为目标（模拟）
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 31):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Step 5: 提取关键词节点的嵌入
model.eval()
with torch.no_grad():
    embeddings = model(data.x_dict, data.edge_index_dict)
    keyword_embeddings = embeddings['keyword'].cpu()

print("Keyword 节点的嵌入向量：")
print(keyword_embeddings)
