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


# 异构图神经网络
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


# 链接预测模块（用于FB15K）
class LinkPredictor(nn.Module):
    def __init__(self, in_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.W = nn.Parameter(torch.randn(num_relations, in_dim, in_dim))
        self.b = nn.Parameter(torch.randn(num_relations, in_dim))

    def forward(self, head_emb, tail_emb, rel_id):
        W_r = self.W[rel_id]
        b_r = self.b[rel_id]
        # 简单的双线性评分函数
        score = torch.sum(head_emb * torch.matmul(tail_emb, W_r.T) + b_r, dim=1)
        return score


# FB15K专用模型
class FB15K_XGradNet(nn.Module):
    def __init__(self, hetero_graph, num_entities, num_relations, feature_dim=128, hidden_dim=128, out_dim=64):
        super().__init__()
        self.g = hetero_graph
        self.num_entities = num_entities
        self.num_relations = num_relations

        # 实体和关系嵌入
        self.entity_emb = nn.Embedding(num_entities, feature_dim)
        self.relation_emb = nn.Embedding(num_relations, feature_dim)

        # 异构图神经网络
        self.hetgnn = HetGNN(feature_dim, hidden_dim, out_dim, hetero_graph.etypes)

        # 链接预测器
        self.link_predictor = LinkPredictor(out_dim, num_relations)

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, node_ids=None):
        if node_ids is None:
            node_ids = torch.arange(self.num_entities, device=self.entity_emb.weight.device)

        # 获取实体特征
        entity_features = self.entity_emb(node_ids)

        # 构建节点特征字典
        node_features = {'entity': entity_features}

        # 通过异构图神经网络
        node_embeddings = self.hetgnn(self.g, node_features)

        return node_embeddings['entity']

    def predict_link(self, head_ids, tail_ids, rel_ids):
        # 获取实体嵌入
        entity_embeddings = self.forward()

        head_emb = entity_embeddings[head_ids]
        tail_emb = entity_embeddings[tail_ids]

        # 计算链接预测得分
        scores = self.link_predictor(head_emb, tail_emb, rel_ids)

        return scores


# 加载FB15K数据
def load_fb15k(data_dir='data/FB15K'):
    # 加载实体和关系映射
    entity2id = {}
    relation2id = {}

    # 读取实体映射
    with open(os.path.join(data_dir, 'entity2id.txt'), 'r') as f:
        lines = f.readlines()
        num_entities = int(lines[0].strip())
        for line in lines[1:]:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    # 读取关系映射
    with open(os.path.join(data_dir, 'relation2id.txt'), 'r') as f:
        lines = f.readlines()
        num_relations = int(lines[0].strip())
        for line in lines[1:]:
            rel, rid = line.strip().split('\t')
            relation2id[rel] = int(rid)

    # 读取训练数据
    train_triples = []
    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            train_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 读取验证数据
    valid_triples = []
    with open(os.path.join(data_dir, 'valid.txt'), 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 读取测试数据
    test_triples = []
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

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
def build_fb15k_graph(train_triples, num_entities, num_relations, device='cpu'):
    # 收集所有边
    head_list = []
    tail_list = []
    rel_list = []

    for h, r, t in train_triples:
        head_list.append(h)
        tail_list.append(t)
        rel_list.append(r)

    # 转换为张量
    head_tensor = torch.tensor(head_list, dtype=torch.long)
    tail_tensor = torch.tensor(tail_list, dtype=torch.long)
    rel_tensor = torch.tensor(rel_list, dtype=torch.long)

    # 构建异构图
    # 每种关系类型作为独立的边类型
    graph_data = {}

    # 为每种关系创建边
    for rel_id in range(num_relations):
        mask = (rel_tensor == rel_id)
        if mask.sum() > 0:
            rel_heads = head_tensor[mask]
            rel_tails = tail_tensor[mask]
            rel_name = f'rel_{rel_id}'
            graph_data[('entity', rel_name, 'entity')] = (rel_heads, rel_tails)

    # 创建异构图
    g = dgl.heterograph(graph_data)
    g = g.to(device)

    return g


# 训练函数
def train_fb15k(model, train_triples, num_epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 转换为张量
    train_triples = torch.tensor(train_triples, dtype=torch.long).to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 随机采样一批正样本
        batch_size = min(1024, len(train_triples))
        indices = torch.randint(0, len(train_triples), (batch_size,))
        batch = train_triples[indices]

        head_ids = batch[:, 0]
        rel_ids = batch[:, 1]
        tail_ids = batch[:, 2]

        # 正样本得分
        pos_scores = model.predict_link(head_ids, tail_ids, rel_ids)

        # 生成负样本（破坏尾部实体）
        neg_tail_ids = torch.randint(0, model.num_entities, (batch_size,)).to(device)
        neg_scores = model.predict_link(head_ids, neg_tail_ids, rel_ids)

        # 损失函数（最大化正负样本得分差距）
        loss = F.relu(1 + neg_scores - pos_scores).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    return model


# 评估函数
def evaluate_fb15k(model, test_triples, device='cpu'):
    model.eval()
    test_triples = torch.tensor(test_triples, dtype=torch.long).to(device)

    with torch.no_grad():
        head_ids = test_triples[:, 0]
        rel_ids = test_triples[:, 1]
        tail_ids = test_triples[:, 2]

        scores = model.predict_link(head_ids, tail_ids, rel_ids)
        predictions = (scores > 0.5).float()

        # 计算准确率
        accuracy = predictions.mean().item()

    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置随机种子
    set_seed(42)

    # 加载FB15K数据
    print('Loading FB15K data...')
    data = load_fb15k('data/FB15K')  # 修改为你的FB15K路径

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
    g = build_fb15k_graph(train_triples, num_entities, num_relations, device)

    # 创建模型
    print('Creating model...')
    model = FB15K_XGradNet(
        g,
        num_entities,
        num_relations,
        feature_dim=128,
        hidden_dim=128,
        out_dim=64
    )

    # 训练模型
    print('Training model...')
    model = train_fb15k(model, train_triples, num_epochs=100, lr=0.001, device=device)

    # 评估模型
    print('Evaluating model...')
    accuracy = evaluate_fb15k(model, test_triples, device=device)

    # 保存模型
    torch.save(model.state_dict(), 'fb15k_xgradnet_model.pth')
    print('Model saved to fb15k_xgradnet_model.pth')

    return model, accuracy


if __name__ == '__main__':
    model, accuracy = main()