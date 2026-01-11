import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
import os
import sys


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================== 原型链接预测器 ======================

# ====================== 修复链接预测器 ======================

# ====================== 进一步优化链接预测器 ======================

class PrototypeLinkPredictor(nn.Module):
    """基于原型的链接预测器 - 进一步优化"""

    def __init__(self, in_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim

        # 为每个关系创建原型
        self.relation_prototypes = nn.Parameter(torch.randn(num_relations, in_dim))

        # 投影网络：分别处理头和尾
        self.head_projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.tail_projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        # 组合网络
        self.combine_net = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim * 2),  # head + tail + relation
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )

        # 初始化参数
        nn.init.xavier_uniform_(self.relation_prototypes)
        for net in [self.head_projection, self.tail_projection, self.combine_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, head_emb, tail_emb, rel_ids):
        batch_size = head_emb.size(0)

        # 分别投影头和尾
        projected_head = self.head_projection(head_emb)
        projected_tail = self.tail_projection(tail_emb)

        # 获取对应关系原型
        rel_prototypes = self.relation_prototypes[rel_ids]  # (batch_size, in_dim)

        # 组合特征：头 + 尾 + 关系原型
        combined = torch.cat([projected_head, projected_tail, rel_prototypes], dim=1)

        # 计算最终得分
        scores = self.combine_net(combined)

        return scores.squeeze(1)


# 异构图神经网络
# ====================== 修复设备统一问题 ======================

class HetGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')

        self.bilstm = nn.LSTM(
            input_size=out_feats,
            hidden_size=out_feats // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, graph, inputs):
        # 确保图和输入在相同设备
        device = next(self.parameters()).device

        # 移动图到正确设备
        if graph.device != device:
            graph = graph.to(device)

        # 确保输入也在正确设备
        inputs = {k: v.to(device) if v.device != device else v
                  for k, v in inputs.items()}

        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        if 'entity' in h:
            ent_feats = h['entity']
            ent_feats = ent_feats.unsqueeze(1)
            ent_feats, _ = self.bilstm(ent_feats)
            ent_feats = ent_feats.squeeze(1)
            h['entity'] = ent_feats

        return h


# 链接预测模块（修正版）
class LinkPredictor(nn.Module):
    def __init__(self, in_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        # 双线性变换矩阵
        self.W = nn.Parameter(torch.randn(num_relations, in_dim))
        self.b = nn.Parameter(torch.randn(num_relations))

        # 初始化参数
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, head_emb, tail_emb, rel_ids):
        batch_size = head_emb.size(0)

        # 获取对应的关系权重
        W_r = self.W[rel_ids]  # (batch_size, in_dim)
        b_r = self.b[rel_ids]  # (batch_size)

        # 双线性评分函数: s = sum(h * W_r * t) + b_r
        # head_emb: (batch_size, in_dim)
        # tail_emb: (batch_size, in_dim)
        # W_r: (batch_size, in_dim)

        # 计算得分: sum(h * (W_r * t)) + b_r
        # 1. 逐元素相乘: W_r * t
        weighted_tail = tail_emb * W_r  # (batch_size, in_dim)

        # 2. 点积: h · (W_r * t)
        scores = torch.sum(head_emb * weighted_tail, dim=1)  # (batch_size)

        # 3. 加上偏置
        scores = scores + b_r

        return scores


# FB15K专用模型

# ====================== 添加缺失的contrastive_loss方法 ======================


class FB15K_XGradNet(nn.Module):
    def __init__(self, hetero_graph, num_entities, num_relations, feature_dim=128, hidden_dim=128, out_dim=64):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.out_dim = out_dim

        # 先保存图引用（不移动）
        self.graph = hetero_graph

        # 实体和关系嵌入 - 创建正确数量的嵌入
        self.entity_emb = nn.Embedding(num_entities, feature_dim)

        # 异构图神经网络
        rel_names = list(set(hetero_graph.etypes))
        self.hetgnn = HetGNN(feature_dim, hidden_dim, out_dim, rel_names)

        # 子图构建器 - 传递实体数量
        self.subgraph_builder = FB15K_SubgraphBuilder(hetero_graph, k_hop=2, num_entities=num_entities)

        # 链接预测器（使用原型思想改造）
        self.link_predictor = PrototypeLinkPredictor(out_dim, num_relations)

        # 对比学习权重
        self.contrast_weight = nn.Parameter(torch.tensor(0.3))

        # 初始化参数 - 使用更好的初始化
        nn.init.xavier_uniform_(self.entity_emb.weight)

    def forward(self, node_ids=None, return_subgraph=False):
        device = next(self.parameters()).device

        if node_ids is None:
            node_ids = torch.arange(self.num_entities, device=device)
        else:
            node_ids = node_ids.to(device)

        # 检查节点ID范围
        if torch.any(node_ids >= self.num_entities):
            invalid_ids = node_ids[node_ids >= self.num_entities]
            # 只保留有效ID
            valid_mask = node_ids < self.num_entities
            node_ids = node_ids[valid_mask]
            if len(node_ids) == 0:
                return torch.empty(0, self.out_dim, device=device)

        # 获取实体特征
        entity_features = self.entity_emb(node_ids)

        # 构建节点特征字典
        node_features = {'entity': entity_features}

        # 确保图在相同设备
        graph = self.graph
        if graph.device != device:
            try:
                graph = graph.to(device)
            except:
                device = torch.device('cpu')
                graph = graph.to(device)
                node_features = {k: v.to(device) for k, v in node_features.items()}
                node_ids = node_ids.to(device)

        # 通过异构图神经网络
        node_embeddings = self.hetgnn(graph, node_features)

        if return_subgraph:
            # 构建子图嵌入
            subgraph_embeddings = []
            for entity_id in node_ids.tolist():
                sub_emb = self.subgraph_builder.build_subgraph_embedding(
                    entity_id, node_embeddings['entity']
                )
                subgraph_embeddings.append(sub_emb)

            if subgraph_embeddings:
                subgraph_embeddings = torch.stack(subgraph_embeddings)
                return node_embeddings['entity'], subgraph_embeddings
            else:
                return node_embeddings['entity'], torch.empty(0, self.out_dim, device=device)
        else:
            return node_embeddings['entity']

    def predict_link(self, head_ids, tail_ids, rel_ids, use_subgraph=True):
        # 确保所有输入在相同设备
        device = next(self.parameters()).device
        head_ids = head_ids.to(device)
        tail_ids = tail_ids.to(device)
        rel_ids = rel_ids.to(device)

        # 检查ID范围
        max_id = self.num_entities - 1
        if torch.any(head_ids > max_id) or torch.any(tail_ids > max_id):
            valid_mask = (head_ids <= max_id) & (tail_ids <= max_id)
            if valid_mask.sum() == 0:
                return torch.zeros(len(head_ids), device=device)
            head_ids = head_ids[valid_mask]
            tail_ids = tail_ids[valid_mask]
            rel_ids = rel_ids[valid_mask]

        # 获取实体嵌入
        if use_subgraph:
            # 使用子图嵌入
            entity_embeddings, subgraph_embeddings = self.forward(return_subgraph=True)
            if len(subgraph_embeddings) == 0:
                return torch.zeros(len(head_ids), device=device)
            head_emb = subgraph_embeddings[head_ids]
            tail_emb = subgraph_embeddings[tail_ids]
        else:
            # 使用原始嵌入
            entity_embeddings = self.forward()
            if len(entity_embeddings) == 0:
                return torch.zeros(len(head_ids), device=device)
            head_emb = entity_embeddings[head_ids]
            tail_emb = entity_embeddings[tail_ids]

        # 计算链接预测得分（使用原型）
        scores = self.link_predictor(head_emb, tail_emb, rel_ids)

        return scores

    def contrastive_loss(self, head_ids, tail_ids, rel_ids):
        """对比学习损失"""
        device = next(self.parameters()).device

        # 确保输入在正确设备
        head_ids = head_ids.to(device)
        tail_ids = tail_ids.to(device)
        rel_ids = rel_ids.to(device)

        batch_size = len(head_ids)
        if batch_size == 0:
            return torch.tensor(0.0, device=device)

        # 获取正样本得分
        pos_scores = self.predict_link(head_ids, tail_ids, rel_ids, use_subgraph=True)

        # 生成负样本（破坏尾部实体）
        neg_tail_ids = torch.randint(0, self.num_entities, (batch_size,), device=device)
        neg_scores = self.predict_link(head_ids, neg_tail_ids, rel_ids, use_subgraph=True)

        # InfoNCE损失
        temperature = 0.1
        pos_exp = torch.exp(pos_scores / temperature)
        neg_exp = torch.exp(neg_scores / temperature)
        contrast_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()

        return contrast_loss

# ====================== 添加子图构建模块 ======================

# ====================== 修改子图构建模块 ======================

class FB15K_SubgraphBuilder:
    """为FB15K实体构建子图 - 修复ID范围问题"""

    def __init__(self, graph, k_hop=2, num_entities=None):
        self.g = graph
        self.k_hop = k_hop
        self.num_entities = num_entities

        # 构建邻居缓存以提高效率
        self.neighbor_cache = {}

    def get_k_hop_neighbors(self, entity_id):
        """获取实体的k跳邻居 - 修复API兼容性"""
        # 检查缓存
        if entity_id in self.neighbor_cache:
            return self.neighbor_cache[entity_id]

        try:
            # 方法1: 使用out_edges获取直接邻居
            if self.g.num_edges() > 0 and entity_id < self.g.num_nodes():
                # 获取所有出边
                out_nodes = self.g.successors(entity_id)
                neighbor_ids = out_nodes.tolist()
            else:
                neighbor_ids = []
        except:
            neighbor_ids = []

        # 去重并移除自身
        neighbor_ids = list(set(neighbor_ids))
        if entity_id in neighbor_ids:
            neighbor_ids.remove(entity_id)

        # 确保邻居ID在实体范围内
        if self.num_entities is not None:
            neighbor_ids = [nid for nid in neighbor_ids if nid < self.num_entities]

        # 缓存结果
        self.neighbor_cache[entity_id] = neighbor_ids

        return neighbor_ids

    def build_subgraph_embedding(self, entity_id, entity_embeddings, embedding_weights=None):
        """构建实体的子图嵌入 - 修复索引范围"""
        if embedding_weights is None:
            # 权重分配：中心实体0.7，邻居0.3
            embedding_weights = {
                'center': 0.7,
                'neighbors': 0.3
            }

        device = entity_embeddings.device

        # 检查实体ID是否在范围内
        if entity_id >= len(entity_embeddings):
            print(f"警告: 实体ID {entity_id} 超出嵌入矩阵范围 {len(entity_embeddings)}")
            return torch.zeros(entity_embeddings.shape[1], device=device)

        # 中心实体嵌入
        center_emb = entity_embeddings[entity_id] * embedding_weights['center']

        # 获取邻居
        neighbor_ids = self.get_k_hop_neighbors(entity_id)

        # 邻居嵌入
        if len(neighbor_ids) > 0:
            # 确保邻居ID在范围内
            valid_neighbors = [nid for nid in neighbor_ids if nid < len(entity_embeddings)]
            if valid_neighbors:
                neighbor_tensor = torch.tensor(valid_neighbors, device=device)
                neighbor_embs = entity_embeddings[neighbor_tensor]
                neighbor_emb = torch.mean(neighbor_embs, dim=0) * embedding_weights['neighbors']
            else:
                neighbor_emb = torch.zeros_like(center_emb, device=device)
        else:
            neighbor_emb = torch.zeros_like(center_emb, device=device)

        # 合并子图嵌入
        subgraph_emb = center_emb + neighbor_emb

        return subgraph_emb


# 加载FB15K数据
def load_fb15k(data_dir='data/FB15K'):
    # 加载实体和关系映射
    entity2id = {}
    relation2id = {}

    # 读取实体映射
    entity_file = os.path.join(data_dir, 'entity2id.txt')
    if not os.path.exists(entity_file):
        # 尝试其他可能的文件名
        entity_file = os.path.join(data_dir, 'entities.dict')
        if not os.path.exists(entity_file):
            # 尝试从train.txt推断实体和关系
            print(f"Warning: entity2id.txt not found, inferring from train.txt...")
            return load_fb15k_inferred(data_dir)

    with open(entity_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if '\t' in lines[0]:
            num_entities = int(lines[0].strip())
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    entity, eid = parts
                    entity2id[entity] = int(eid)
        else:
            for i, line in enumerate(lines):
                entity = line.strip()
                entity2id[entity] = i
            num_entities = len(entity2id)

    # 读取关系映射
    relation_file = os.path.join(data_dir, 'relation2id.txt')
    if not os.path.exists(relation_file):
        relation_file = os.path.join(data_dir, 'relations.dict')

    if os.path.exists(relation_file):
        with open(relation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if '\t' in lines[0]:
                num_relations = int(lines[0].strip())
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        rel, rid = parts
                        relation2id[rel] = int(rid)
            else:
                for i, line in enumerate(lines):
                    rel = line.strip()
                    relation2id[rel] = i
                num_relations = len(relation2id)
    else:
        print("Warning: relation2id.txt not found, inferring from train.txt...")
        relation2id = {}

    # 读取训练数据
    train_triples = []
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[:3]
                # 如果是字符串，转换为ID
                if h in entity2id:
                    h_id = entity2id[h]
                else:
                    # 添加新实体
                    h_id = len(entity2id)
                    entity2id[h] = h_id

                if t in entity2id:
                    t_id = entity2id[t]
                else:
                    t_id = len(entity2id)
                    entity2id[t] = t_id

                if r in relation2id:
                    r_id = relation2id[r]
                else:
                    r_id = len(relation2id)
                    relation2id[r] = r_id

                train_triples.append((h_id, r_id, t_id))

    # 读取验证数据
    valid_triples = []
    valid_file = os.path.join(data_dir, 'valid.txt')
    if os.path.exists(valid_file):
        with open(valid_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[:3]
                    if h in entity2id and t in entity2id and r in relation2id:
                        valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 读取测试数据
    test_triples = []
    test_file = os.path.join(data_dir, 'test.txt')
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[:3]
                    if h in entity2id and t in entity2id and r in relation2id:
                        test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    print(f"Loaded {num_entities} entities, {num_relations} relations")
    print(f"Training triples: {len(train_triples)}")
    print(f"Validation triples: {len(valid_triples)}")
    print(f"Test triples: {len(test_triples)}")

    return {
        'entity2id': entity2id,
        'relation2id': relation2id,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triples': train_triples,
        'valid_triples': valid_triples,
        'test_triples': test_triples
    }


def load_fb15k_inferred(data_dir):
    """从train.txt推断实体和关系"""
    entity2id = {}
    relation2id = {}
    train_triples = []

    # 读取训练数据
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[:3]

                # 添加实体
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)

                # 添加关系
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

                train_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 读取验证数据
    valid_triples = []
    valid_file = os.path.join(data_dir, 'valid.txt')
    if os.path.exists(valid_file):
        with open(valid_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[:3]
                    if h in entity2id and t in entity2id and r in relation2id:
                        valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 读取测试数据
    test_triples = []
    test_file = os.path.join(data_dir, 'test.txt')
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[:3]
                    if h in entity2id and t in entity2id and r in relation2id:
                        test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    print(f"Inferred {num_entities} entities, {num_relations} relations")
    print(f"Training triples: {len(train_triples)}")
    print(f"Validation triples: {len(valid_triples)}")
    print(f"Test triples: {len(test_triples)}")

    return {
        'entity2id': entity2id,
        'relation2id': relation2id,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triples': train_triples,
        'valid_triples': valid_triples,
        'test_triples': test_triples
    }


# 构建异构图
# ====================== 修改主要函数 ======================

# ====================== 修改构建函数确保实体ID一致 ======================

def build_fb15k_graph(train_triples, num_entities, num_relations, device='cpu'):
    """构建异构图 - 确保实体ID范围正确"""
    # 收集所有边
    head_list = []
    tail_list = []
    rel_list = []

    # 统计实际使用的实体ID
    used_entities = set()

    # 使用所有训练数据
    max_edges = len(train_triples)  # 使用所有边
    sample_triples = train_triples[:max_edges]

    for h, r, t in sample_triples:
        head_list.append(h)
        tail_list.append(t)
        rel_list.append(r)
        used_entities.add(h)
        used_entities.add(t)

    # 检查实体ID范围
    max_entity_id = max(used_entities) if used_entities else 0
    print(f"实际使用的最大实体ID: {max_entity_id}, 实体总数: {num_entities}")

    # 转换为张量
    head_tensor = torch.tensor(head_list, dtype=torch.long)
    tail_tensor = torch.tensor(tail_list, dtype=torch.long)
    rel_tensor = torch.tensor(rel_list, dtype=torch.long)

    # 直接创建图在目标设备上
    rel_name = 'has_relation'
    graph_data = {('entity', rel_name, 'entity'): (head_tensor, tail_tensor)}

    # 创建图
    g = dgl.heterograph(graph_data)

    # 添加关系ID作为边特征
    g.edges[rel_name].data['rel_id'] = rel_tensor

    # 移动图到指定设备
    try:
        g = g.to(device)
    except Exception as e:
        print(f"警告: 无法将图移动到 {device}: {e}，使用CPU")
        device = torch.device('cpu')
        g = g.to(device)

    print(f"Graph built with {g.num_edges()} edges on {device}")
    return g


# 训练函数


# ====================== 修改训练函数确保设备一致 ======================


# ====================== 修改训练函数确保设备一致 ======================

def train_fb15k(model, train_triples, num_epochs=50, lr=0.001, batch_size=1024, device='cpu',
                use_contrastive=True, phase=1):
    """训练函数 - 优化损失计算和训练策略"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr / 10)

    # 转换为张量并移动到设备
    train_triples_tensor = torch.tensor(train_triples, dtype=torch.long).to(device)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()

        # 随机打乱数据
        indices = torch.randperm(len(train_triples_tensor)).to(device)
        total_loss = 0.0
        num_batches = 0

        for start_idx in range(0, len(train_triples_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(train_triples_tensor))
            batch_indices = indices[start_idx:end_idx]
            batch = train_triples_tensor[batch_indices]

            head_ids = batch[:, 0]
            rel_ids = batch[:, 1]
            tail_ids = batch[:, 2]

            if len(head_ids) == 0:
                continue

            try:
                # 正样本得分
                pos_scores = model.predict_link(head_ids, tail_ids, rel_ids, use_subgraph=True)

                # 生成负样本（两种方式：破坏头部和尾部）
                neg_type = np.random.choice(['head', 'tail'])
                if neg_type == 'head':
                    neg_head_ids = torch.randint(0, model.num_entities, (len(batch),), device=device)
                    neg_scores = model.predict_link(neg_head_ids, tail_ids, rel_ids, use_subgraph=True)
                else:
                    neg_tail_ids = torch.randint(0, model.num_entities, (len(batch),), device=device)
                    neg_scores = model.predict_link(head_ids, neg_tail_ids, rel_ids, use_subgraph=True)

                # 使用间隔损失（margin-based loss）
                margin = 1.0
                link_loss = F.relu(margin + neg_scores - pos_scores).mean()

                total_loss_batch = link_loss

                # 添加对比学习损失（第二阶段使用）
                if use_contrastive and phase == 2:
                    contrast_loss = model.contrastive_loss(head_ids, tail_ids, rel_ids)
                    total_loss_batch = total_loss_batch + model.contrast_weight * contrast_loss

                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += total_loss_batch.item()
                num_batches += 1

            except Exception as e:
                continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)

            # 更新学习率
            scheduler.step()

            # 打印训练进度
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Batches: {num_batches}')

                # 打印一些正负样本的得分
                if epoch % 5 == 0 and num_batches > 0:
                    with torch.no_grad():
                        # 取第一个batch的第一个样本
                        sample_head = head_ids[:1]
                        sample_rel = rel_ids[:1]
                        sample_tail = tail_ids[:1]

                        pos_score = model.predict_link(sample_head, sample_tail, sample_rel, use_subgraph=True)

                        # 生成不同类型的负样本
                        neg_head = torch.randint(0, model.num_entities, (1,), device=device)
                        neg_tail = torch.randint(0, model.num_entities, (1,), device=device)

                        neg_score1 = model.predict_link(neg_head, sample_tail, sample_rel, use_subgraph=True)
                        neg_score2 = model.predict_link(sample_head, neg_tail, sample_rel, use_subgraph=True)

                        print(f"   正样本得分: {pos_score.item():.4f}")
                        print(f"   负样本1(破坏头): {neg_score1.item():.4f}, 负样本2(破坏尾): {neg_score2.item():.4f}")
                        print(f"   得分差(正-负1): {pos_score.item() - neg_score1.item():.4f}")

    return model, train_losses

# ====================== 修改评估函数 ======================


def evaluate_fb15k(model, test_triples, device='cpu', batch_size=512, eval_mode='subgraph'):
    """评估函数 - 改进版本"""
    model.eval()

    # 过滤无效的实体ID
    valid_triples = []
    max_entity_id = model.num_entities - 1

    for h, r, t in test_triples:
        if h <= max_entity_id and t <= max_entity_id:
            valid_triples.append((h, r, t))

    print(f"原始测试数据: {len(test_triples)}，有效数据: {len(valid_triples)}")

    if len(valid_triples) == 0:
        print("没有有效的测试数据")
        return 0.0

    test_triples_tensor = torch.tensor(valid_triples, dtype=torch.long).to(device)

    all_predictions = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for start_idx in range(0, len(test_triples_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(test_triples_tensor))
            batch = test_triples_tensor[start_idx:end_idx]

            if len(batch) == 0:
                continue

            head_ids = batch[:, 0]
            rel_ids = batch[:, 1]
            tail_ids = batch[:, 2]

            try:
                # 使用子图嵌入
                raw_scores = model.predict_link(head_ids, tail_ids, rel_ids, use_subgraph=True)

                # 保存原始得分
                all_scores.append(raw_scores.cpu())

                # 预测为正样本如果得分>0（因为使用间隔损失）
                predictions = (raw_scores > 0).float()

                all_predictions.append(predictions.cpu())
                all_labels.append(torch.ones_like(predictions).cpu())
            except Exception as e:
                continue

    if all_predictions:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)

        # 计算准确率
        correct = (all_predictions == all_labels).float()
        accuracy = correct.mean().item()

        # 计算平均得分
        avg_score = all_scores.mean().item()

        # 计算正样本的平均得分
        positive_mask = (all_predictions == 1)
        positive_scores = all_scores[positive_mask] if positive_mask.sum() > 0 else torch.tensor([0.0])
        avg_positive_score = positive_scores.mean().item()

        # 打印详细统计
        num_correct = correct.sum().item()
        num_total = len(correct)
        positive_predictions = all_predictions.sum().item()

        print(f'Test Results:')
        print(f'  Accuracy: {accuracy:.4f} ({num_correct}/{num_total})')
        print(f'  Positive predictions: {positive_predictions}/{num_total}')
        print(f'  Average score: {avg_score:.4f}')
        print(f'  Average positive score: {avg_positive_score:.4f}')

        return accuracy
    else:
        print("No predictions generated")
        return 0.0


# ====================== 添加原型可视化函数 ======================

def visualize_prototypes(model, relation_names=None, top_k=10):
    """可视化关系原型"""
    if not relation_names:
        relation_names = [f'rel_{i}' for i in range(min(model.num_relations, 20))]

    prototypes = model.link_predictor.relation_prototypes.detach().cpu().numpy()

    # 计算原型之间的距离
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(prototypes[:len(relation_names)])

    print("关系原型相似度矩阵（前20个关系）:")
    for i in range(min(len(relation_names), 20)):
        print(f"{relation_names[i]:20s}: ", end="")
        for j in range(min(len(relation_names), 10)):
            print(f"{similarity_matrix[i][j]:.3f} ", end="")
        print()

    return similarity_matrix
# 主函数

# ====================== 创建测试用的简单数据函数 ======================

def create_simple_test_data():
    """创建简单的测试数据"""
    print("创建简单测试数据...")

    # 少量实体和关系
    num_entities = 100
    num_relations = 10

    # 创建训练三元组
    train_triples = []
    for i in range(500):  # 500个训练三元组
        h = np.random.randint(0, num_entities)
        t = np.random.randint(0, num_entities)
        r = np.random.randint(0, num_relations)
        train_triples.append((h, r, t))

    # 创建测试三元组
    test_triples = []
    for i in range(100):  # 100个测试三元组
        h = np.random.randint(0, num_entities)
        t = np.random.randint(0, num_entities)
        r = np.random.randint(0, num_relations)
        test_triples.append((h, r, t))

    return {
        'entity2id': {},
        'relation2id': {},
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triples': train_triples,
        'valid_triples': [],
        'test_triples': test_triples
    }


# ====================== 添加性能分析函数 ======================
def analyze_model_performance(model, test_triples, device='cpu'):
    """分析模型性能"""
    model.eval()

    # 过滤无效的实体ID
    valid_triples = []
    max_entity_id = model.num_entities - 1

    for h, r, t in test_triples:
        if h <= max_entity_id and t <= max_entity_id:
            valid_triples.append((h, r, t))

    if len(valid_triples) < 100:
        sample_triples = valid_triples
    else:
        sample_triples = valid_triples[:100]

    test_triples_tensor = torch.tensor(sample_triples, dtype=torch.long).to(device)

    with torch.no_grad():
        head_ids = test_triples_tensor[:, 0]
        rel_ids = test_triples_tensor[:, 1]
        tail_ids = test_triples_tensor[:, 2]

        # 计算正样本得分
        pos_scores = model.predict_link(head_ids, tail_ids, rel_ids, use_subgraph=True)

        # 计算负样本得分（破坏尾部）
        neg_tail_ids = torch.randint(0, model.num_entities, (len(sample_triples),), device=device)
        neg_scores = model.predict_link(head_ids, neg_tail_ids, rel_ids, use_subgraph=True)

        print(f"\n性能分析（基于{len(sample_triples)}个样本）:")
        print(f"  正样本平均得分: {pos_scores.mean().item():.4f}")
        print(f"  负样本平均得分: {neg_scores.mean().item():.4f}")
        print(f"  得分差(正-负): {pos_scores.mean().item() - neg_scores.mean().item():.4f}")
        print(f"  正样本得分>0的比例: {(pos_scores > 0).float().mean().item():.4f}")
        print(f"  负样本得分<0的比例: {(neg_scores < 0).float().mean().item():.4f}")

        return pos_scores, neg_scores


def main():
    # 设置设备
    device = torch.device('cpu')
    print("使用CPU模式以确保兼容性")
    print(f'Using device: {device}')

    # 设置随机种子
    set_seed(42)

    # 使用简单测试数据
    print('Creating simple test data...')
    data = create_simple_test_data()

    num_entities = data['num_entities']
    num_relations = data['num_relations']
    train_triples = data['train_triples']
    test_triples = data['test_triples']

    print(f'Number of entities: {num_entities}')
    print(f'Number of relations: {num_relations}')
    print(f'Number of training triples: {len(train_triples)}')
    print(f'Number of test triples: {len(test_triples)}')

    # 构建异构图
    print('Building heterogeneous graph...')
    try:
        g = build_fb15k_graph(train_triples, num_entities, num_relations, device='cpu')
        print(f'Graph built with {g.num_edges()} edges')
    except Exception as e:
        print(f"Error building graph: {e}")
        print("创建简单测试图...")
        head_tensor = torch.tensor([0, 1, 2], dtype=torch.long)
        tail_tensor = torch.tensor([1, 2, 0], dtype=torch.long)
        graph_data = {('entity', 'has_relation', 'entity'): (head_tensor, tail_tensor)}
        g = dgl.heterograph(graph_data)

    # 创建模型 - 进一步增加维度
    print('Creating model...')
    model = FB15K_XGradNet(
        g,
        num_entities,
        num_relations,
        feature_dim=128,  # 进一步增加维度
        hidden_dim=128,
        out_dim=64
    )
    model = model.to(device)

    # 第一阶段：基础训练
    print('\n第一阶段：基础训练（不使用对比学习）')
    print(f"使用 {len(train_triples)} 条数据进行训练")

    model, train_losses1 = train_fb15k(
        model,
        train_triples,
        num_epochs=15,  # 增加epoch数
        lr=0.001,
        batch_size=32,
        device=device,
        use_contrastive=False,
        phase=1
    )

    # 分析第一阶段性能
    print("\n第一阶段训练完成，性能分析：")
    analyze_model_performance(model, test_triples, device=device)

    # 第二阶段：使用对比学习微调
    print('\n\n第二阶段：使用对比学习微调')
    model, train_losses2 = train_fb15k(
        model,
        train_triples,
        num_epochs=10,  # 减少epoch数
        lr=0.0003,  # 降低学习率
        batch_size=32,
        device=device,
        use_contrastive=True,
        phase=2
    )

    train_losses = train_losses1 + train_losses2

    # 最终评估
    print('\n最终评估：')
    accuracy = evaluate_fb15k(model, test_triples, device=device, batch_size=32)

    # 最终性能分析
    print("\n最终性能分析：")
    analyze_model_performance(model, test_triples, device=device)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_losses': train_losses,
        'accuracy': accuracy
    }, 'fb15k_igradnet_gpu_model.pth')
    print(f'\nModel saved to fb15k_igradnet_gpu_model.pth, Final Accuracy: {accuracy:.4f}')

    # 返回结果
    return model, accuracy, train_losses


def create_synthetic_data():
    """创建合成数据用于测试"""
    print("Creating synthetic data for testing...")

    # 创建少量实体和关系
    num_entities = 100
    num_relations = 10

    # 创建训练三元组
    train_triples = []
    for i in range(500):  # 500个训练三元组
        h = np.random.randint(0, num_entities)
        t = np.random.randint(0, num_entities)
        r = np.random.randint(0, num_relations)
        train_triples.append((h, r, t))

    # 创建测试三元组
    test_triples = []
    for i in range(100):  # 100个测试三元组
        h = np.random.randint(0, num_entities)
        t = np.random.randint(0, num_entities)
        r = np.random.randint(0, num_relations)
        test_triples.append((h, r, t))

    return {
        'entity2id': {},
        'relation2id': {},
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triples': train_triples,
        'valid_triples': [],
        'test_triples': test_triples
    }


if __name__ == '__main__':
    model, accuracy, train_losses = main()