import json
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HGTConv, Linear


# -------------------- 1. 数据预处理改进 --------------------
def load_data(file_path):
    """优化节点去重逻辑，增加异常处理"""
    nodes = []
    edges = []
    node_ids = set()

    try:
        with open(file_path, "r", encoding='utf-8-sig') as f:
            data = json.load(f)
            for item in data:
                p = item['p']

                # 处理起始节点
                start_node = p['start']
                if start_node['elementId'] not in node_ids:
                    nodes.append({
                        'id': start_node['elementId'],
                        'labels': start_node['labels'],
                        'properties': start_node['properties']
                    })
                    node_ids.add(start_node['elementId'])

                # 处理结束节点
                end_node = p['end']
                if end_node['elementId'] not in node_ids:
                    nodes.append({
                        'id': end_node['elementId'],
                        'labels': end_node['labels'],
                        'properties': end_node['properties']
                    })
                    node_ids.add(end_node['elementId'])

                # 提取关系信息（修复多segment处理）
                for segment in p['segments']:
                    rel = segment['relationship']
                    edges.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'label': rel['type']
                    })

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

    print("成功读取节点和边信息")
    return nodes, edges


def build_hetero_data(nodes, edges):
    """优化特征初始化，添加节点类型校验"""
    hetero_data = HeteroData()
    valid_node_types = {'Paper', 'Author', 'Keyword', 'C3', 'PY', 'SO'}

    # Step 1: 处理节点
    node_id_to_type = {}
    node_type_mapping = {nt: [] for nt in valid_node_types}  # {类型: [节点id列表]}

    for node in nodes:
        node_type = node['labels'][0]
        if node_type not in valid_node_types:
            print(f"Warning: 发现无效节点类型 {node_type}")
            continue

        node_id = node['id']
        if node_id in node_id_to_type:
            continue  # 确保去重

        node_id_to_type[node_id] = node_type
        node_type_mapping[node_type].append(node_id)

    # 初始化节点特征（随机特征+简单属性编码）
    for node_type, ids in node_type_mapping.items():
        if not ids:
            continue
        # 特征工程：结合properties生成特征（示例使用随机特征+属性长度）
        feat_dim = 32  # 可调整的特征维度
        num_nodes = len(ids)

        # 示例：将属性数量作为部分特征
        prop_sizes = [
            len(node['properties'])
            for node in nodes
            if node['id'] in ids
        ]
        prop_features = torch.tensor(prop_sizes, dtype=torch.float).view(-1, 1)

        # 组合随机特征
        rand_features = torch.randn(num_nodes, feat_dim - 1)
        hetero_data[node_type].x = torch.cat([prop_features, rand_features], dim=1)

    # Step 2: 处理边
    edge_type_mapping = defaultdict(list)  # {(src_type, rel_type, dst_type): edge_index}

    for edge in edges:
        # 获取节点元信息
        src_id = edge['start']['elementId']
        dst_id = edge['end']['elementId']
        src_type = node_id_to_type.get(src_id)
        dst_type = node_id_to_type.get(dst_id)
        rel_type = edge['label']

        # 校验节点有效性
        if not all([src_type, dst_type]):
            continue
        if src_type not in valid_node_types or dst_type not in valid_node_types:
            continue

        # 转换为局部索引
        try:
            src_idx = node_type_mapping[src_type].index(src_id)
            dst_idx = node_type_mapping[dst_type].index(dst_id)
        except ValueError:
            continue

        # 记录边索引
        edge_type_mapping[(src_type, rel_type, dst_type)].append([src_idx, dst_idx])

    # 构建边索引张量
    for (src_type, rel_type, dst_type), edges in edge_type_mapping.items():
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        hetero_data[src_type, rel_type, dst_type].edge_index = edge_index

    # 添加反向边（可选）
    hetero_data = ToUndirected()(hetero_data)

    print(f"异构数据统计:")
    print("节点类型:", [nt for nt in hetero_data.node_types])
    print("边类型:", hetero_data.edge_types)
    return hetero_data



# -------------------- 2. 模型改进 --------------------
class HGTModel(torch.nn.Module):
    def __init__(self, metadata, hidden_dim=128, out_dim=64):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        # 动态获取节点类型
        node_types = list(metadata[0])
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)

        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=8)
        self.conv2 = HGTConv(hidden_dim, out_dim, metadata=metadata, heads=4)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items()}
        x_dict = {nt: torch.relu(x) for nt, x in x_dict.items()}  # 添加激活函数
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {nt: self.dropout(x) for nt, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


# -------------------- 3. 训练优化 --------------------
def train_model(model, data, epochs=200):
    """添加学习率调度和早停机制"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z_dict = model(data.x_dict, data.edge_index_dict)

        # 改进损失函数：同时优化所有节点类型
        loss = sum(torch.mean(z.pow(2)) for z in z_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step(loss)

        # 早停机制
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return z_dict


# -------------------- 4. 可视化增强 --------------------
def visualize_embeddings(z_dict, labels, n_clusters=5):
    """优化聚类参数和可视化效果"""
    embeddings = z_dict['Keyword'].detach().cpu().numpy()

    # 自动确定聚类数量
    if n_clusters == 'auto':
        max_k = min(30, len(embeddings) - 1)  # 适当扩大搜索范围
        print(len(embeddings))
        inertias = []
        silhouette_scores = []

        # 计算不同k值的指标
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            inertias.append(kmeans.inertia_)

            # 仅当k>1时计算轮廓系数
            if k > 1:
                silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

        # 改进的肘部法则自动检测
        diffs = np.diff(inertias)
        diff_ratios = diffs[:-1] / diffs[1:]
        elbow_k = np.argmax(diff_ratios) + 2  # +2补偿索引偏移

        # 轮廓系数辅助验证
        if len(silhouette_scores) > 0:
            best_silhouette_k = np.argmax(silhouette_scores) + 2  # 因为k从2开始计算
            # 综合两种方法取最大值
            n_clusters = max(elbow_k, best_silhouette_k)
        else:
            n_clusters = elbow_k

        # 最终结果约束
        n_clusters = max(2, min(n_clusters, max_k))  # 确保在合理范围内
        print(f"自动确定的聚类数量: {n_clusters}")

    # t-SNE参数优化
    tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)

    # 可视化设置
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20', n_clusters)
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap=cmap,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        s=80
    )

    # 添加部分文本标签
    indices = np.random.choice(len(labels), size=min(20, len(labels)), replace=False)
    for i in indices:
        plt.text(
            embeddings_2d[i, 0] + 0.5,
            embeddings_2d[i, 1] + 0.5,
            labels[i],
            fontsize=8,
            alpha=0.7,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    plt.title(f'Keyword Clusters (t-SNE + K-means, K={n_clusters})', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------- 主程序改进 --------------------
if __name__ == "__main__":
    # 加载数据
    nodes, edges = load_data('../data/jsondata/relations.json')
    hetero_data = build_hetero_data(nodes, edges)

    # 模型初始化（必须在hetero_data之后）
    model = HGTModel(metadata=hetero_data.metadata())
    model_path = "hgt_model.pth"

    # 训练/加载模型
    if os.path.exists(model_path):
        print("加载预训练模型...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("开始训练新模型...")
        z_dict = train_model(model, hetero_data)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

    # 推理模式
    model.eval()
    with torch.no_grad():
        z_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)

    # 提取关键词标签
    keyword_labels = [
        node['properties'].get('name', f'Unknown_{i}')  # 处理缺失名称的情况
        for i, node in enumerate(nodes)
        if node['labels'][0] == 'Keyword'
    ]

    # 可视化（自动确定聚类数量）
    visualize_embeddings(z_dict, keyword_labels, n_clusters='auto')