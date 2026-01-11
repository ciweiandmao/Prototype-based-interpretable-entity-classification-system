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
class FB15K_XGradNet(nn.Module):
    def __init__(self, hetero_graph, num_entities, num_relations, feature_dim=128, hidden_dim=128, out_dim=64):
        super().__init__()
        self.g = hetero_graph
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.out_dim = out_dim

        # 实体和关系嵌入
        self.entity_emb = nn.Embedding(num_entities, feature_dim)

        # 异构图神经网络
        # 获取图中所有关系类型
        rel_names = list(set(hetero_graph.etypes))
        self.hetgnn = HetGNN(feature_dim, hidden_dim, out_dim, rel_names)

        # 链接预测器
        self.link_predictor = LinkPredictor(out_dim, num_relations)

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_emb.weight)

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
    if graph_data:
        g = dgl.heterograph(graph_data)
        g = g.to(device)
    else:
        # 创建空图作为占位符
        g = dgl.heterograph({('entity', 'rel_0', 'entity'): ([], [])})
        g = g.to(device)

    return g


# 训练函数
def train_fb15k(model, train_triples, num_epochs=50, lr=0.001, batch_size=1024, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 转换为张量
    train_triples_tensor = torch.tensor(train_triples, dtype=torch.long).to(device)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()

        # 随机打乱数据
        indices = torch.randperm(len(train_triples_tensor))
        total_loss = 0.0
        num_batches = 0

        for start_idx in range(0, len(train_triples_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(train_triples_tensor))
            batch_indices = indices[start_idx:end_idx]
            batch = train_triples_tensor[batch_indices]

            head_ids = batch[:, 0]
            rel_ids = batch[:, 1]
            tail_ids = batch[:, 2]

            # 正样本得分
            pos_scores = model.predict_link(head_ids, tail_ids, rel_ids)

            # 生成负样本（破坏尾部实体）
            neg_tail_ids = torch.randint(0, model.num_entities, (len(batch),)).to(device)
            neg_scores = model.predict_link(head_ids, neg_tail_ids, rel_ids)

            # 损失函数（最大化正负样本得分差距）
            loss = F.relu(1.0 + neg_scores - pos_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model, train_losses


# 评估函数
def evaluate_fb15k(model, test_triples, device='cpu', batch_size=1024):
    model.eval()
    test_triples_tensor = torch.tensor(test_triples, dtype=torch.long).to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for start_idx in range(0, len(test_triples_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(test_triples_tensor))
            batch = test_triples_tensor[start_idx:end_idx]

            head_ids = batch[:, 0]
            rel_ids = batch[:, 1]
            tail_ids = batch[:, 2]

            scores = model.predict_link(head_ids, tail_ids, rel_ids)
            predictions = (scores > 0).float()

            all_predictions.append(predictions.cpu())
            all_labels.append(torch.ones_like(predictions).cpu())  # 所有正样本标签为1

    if all_predictions:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # 计算准确率
        correct = (all_predictions == all_labels).float()
        accuracy = correct.mean().item()

        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy
    else:
        print("No predictions generated")
        return 0.0


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置随机种子
    set_seed(42)

    # 加载FB15K数据
    print('Loading FB15K data...')

    # 尝试不同的数据路径
    data_paths = [
        'data/FB15K',
        'FB15K',
        './FB15K',
        '../FB15K'
    ]

    data = None
    for data_path in data_paths:
        if os.path.exists(data_path):
            try:
                data = load_fb15k(data_path)
                print(f"Successfully loaded data from: {data_path}")
                break
            except Exception as e:
                print(f"Failed to load from {data_path}: {e}")

    if data is None:
        # 如果找不到数据文件，创建一个简单的示例数据
        print("FB15K data not found, creating synthetic data for testing...")
        data = create_synthetic_data()

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
        g = build_fb15k_graph(train_triples, num_entities, num_relations, device)
        print(f'Graph built with {len(g.etypes)} relation types')
    except Exception as e:
        print(f"Error building graph: {e}")
        return

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
    model, train_losses = train_fb15k(
        model,
        train_triples,
        num_epochs=50,  # 减少epoch数以加快训练
        lr=0.001,
        batch_size=512,  # 减小batch size
        device=device
    )

    # 评估模型
    print('Evaluating model...')
    accuracy = evaluate_fb15k(model, test_triples, device=device, batch_size=512)

    # 保存模型
    torch.save(model.state_dict(), 'fb15k_xgradnet_model.pth')
    print('Model saved to fb15k_xgradnet_model.pth')

    # 绘制训练损失
    if train_losses:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()

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