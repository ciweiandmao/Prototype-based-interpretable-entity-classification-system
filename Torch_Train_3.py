import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import defaultdict, Counter
import warnings
import random
from tqdm import tqdm
import gc
import os
import time

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


class RelationAwareGCN(nn.Module):
    """关系感知的GCN模型，专门用于实体类型预测"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=3, dropout=0.3):
        super(RelationAwareGCN, self).__init__()

        # 输入编码层（用于处理不同类型特征） - 修正维度
        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, h_feats),  # 改为h_feats而不是h_feats*2
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # GCN层 - 修正维度
        for i in range(num_layers):
            # 所有层都使用相同的输入输出维度
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 关系注意力层 - 修正维度
        self.relation_attention = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),  # 输入是h_feats*2
            nn.ReLU(),
            nn.Linear(h_feats, 1),
            nn.Sigmoid()
        )

        # 邻居类型聚合层 - 修正维度
        self.neighbor_aggregator = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),  # 改为h_feats而不是h_feats*2
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层 - 修正维度
        self.output_layer = nn.Sequential(
            nn.Linear(h_feats, num_classes),  # 直接输出到num_classes
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def aggregate_neighbor_types(self, g, node_features):
        """聚合邻居特征（简化版本）"""
        with g.local_scope():
            g.ndata['h'] = node_features

            # 聚合邻居特征（均值）
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neighbor_h'))
            neighbor_features = g.ndata['neighbor_h']

        # 拼接节点特征和邻居特征
        combined = torch.cat([node_features, neighbor_features], dim=1)
        aggregated = self.neighbor_aggregator(combined)

        return aggregated

    def apply_relation_attention(self, node_features, neighbor_features):
        """应用关系注意力机制"""
        # 拼接节点和邻居特征
        combined = torch.cat([node_features, neighbor_features], dim=1)

        # 计算注意力权重
        attention_weights = self.relation_attention(combined)

        # 加权邻居特征
        attended_neighbors = neighbor_features * attention_weights

        # 与节点特征结合
        enhanced = node_features + attended_neighbors

        return enhanced

    def forward(self, g, features):
        # 编码输入特征
        h = self.input_encoder(features)

        layer_outputs = []

        # 多层传播
        for i in range(self.num_layers):
            # 如果是第一层，先聚合邻居信息
            if i == 0:
                neighbor_agg = self.aggregate_neighbor_types(g, h)
                h_attended = self.apply_relation_attention(h, neighbor_agg)

            # GCN传播
            h_new = self.layers[i](g, h_attended if i == 0 else h)
            h_new = self.bns[i](h_new)
            h_new = F.relu(h_new)

            # 残差连接（除了第一层）
            if i > 0:
                h_new = h_new + h

            h = self.dropout(h_new)
            layer_outputs.append(h)

        # 使用最后一层的输出
        h_final = layer_outputs[-1]

        # 输出预测
        out = self.output_layer(h_final)

        return out


def load_training_data():
    """加载训练数据"""
    print("=" * 60)
    print("步骤1: 加载训练数据...")
    print("=" * 60)

    # 加载关系三元组
    triples = []
    entity_pairs = set()
    relation_counter = Counter()

    try:
        with open('data/FB15KET/xunlian.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="加载关系数据"):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    triples.append((head, relation, tail))
                    entity_pairs.add((head, tail))
                    relation_counter[relation] += 1
    except FileNotFoundError:
        print("错误: xunlian.txt 文件不存在")
        return None, None, None, None

    print(f"加载了 {len(triples)} 个关系三元组")
    print(f"发现 {len(relation_counter)} 种不同关系类型")
    print(f"发现 {len(entity_pairs)} 个不同的实体对")

    # 加载实体类型数据
    try:
        entity_types = pd.read_csv('data/FB15KET/Entity_All_typed.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误: Entity_All_typed.csv 文件不存在")
        return None, None, None, None

    print(f"加载了 {len(entity_types)} 个实体类型记录")

    # 统计类型分布
    if 'predicted_category' in entity_types.columns:
        type_counts = entity_types['predicted_category'].value_counts()
        print(f"发现 {len(type_counts)} 种不同实体类型")
        print(f"最常见的5种类型:")
        for i, (type_id, count) in enumerate(type_counts.head().items()):
            type_name = ""
            if 'predicted_category_name' in entity_types.columns:
                type_names = entity_types[entity_types['predicted_category'] == type_id][
                    'predicted_category_name'].unique()
                if len(type_names) > 0:
                    type_name = f" ({type_names[0]})"
            print(f"  类型 {type_id}{type_name}: {count} 个实体")

    return triples, entity_types, relation_counter, entity_pairs


def build_entity_graph(triples):
    """构建实体关系图"""
    print("\n构建实体关系图...")

    # 收集所有实体
    all_entities = set()
    for h, r, t in triples:
        all_entities.update([h, t])

    print(f"总实体数: {len(all_entities)}")

    # 创建实体到索引的映射
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}

    # 构建边
    src_nodes = []
    dst_nodes = []
    edge_types = []
    relation_to_idx = {}

    # 统计关系
    relations = set([r for _, r, _ in triples])
    for idx, rel in enumerate(relations):
        relation_to_idx[rel] = idx

    for h, r, t in tqdm(triples, desc="构建图边"):
        src_nodes.append(entity_to_idx[h])
        dst_nodes.append(entity_to_idx[t])
        edge_types.append(relation_to_idx[r])

    # 创建DGL图
    num_nodes = len(all_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    # 添加边类型特征
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)

    # 添加自环
    g = dgl.add_self_loop(g)

    print(f"图构建完成: 节点数={g.num_nodes()}, 边数={g.num_edges()}, 关系类型数={len(relations)}")

    return g, entity_to_idx, idx_to_entity, relation_to_idx


def extract_entity_features(triples, entity_types, entity_to_idx, relation_counter):
    """提取实体特征 - 简化版本"""
    print("\n提取实体特征...")

    # 获取实体类型映射
    entity_to_type = dict(zip(entity_types['entity_id'],
                              entity_types['predicted_category']))

    # 统计实体关系信息
    entity_relations = defaultdict(list)
    entity_in_degree = Counter()
    entity_out_degree = Counter()
    entity_neighbors = defaultdict(set)

    for h, r, t in tqdm(triples, desc="统计实体关系"):
        entity_relations[h].append(r)
        entity_relations[t].append(r)
        entity_out_degree[h] += 1
        entity_in_degree[t] += 1
        entity_neighbors[h].add(t)
        entity_neighbors[t].add(h)

    # 选择最常见的关系类型（限制数量）
    top_relations = [rel for rel, _ in relation_counter.most_common(20)]  # 减少到20个
    relation_to_feat_idx = {rel: i for i, rel in enumerate(top_relations)}

    # ========== 特征提取 ==========
    base_features = []
    relation_features = []

    # 为每个实体提取特征
    for entity in tqdm(entity_to_idx.keys(), desc="提取特征"):
        # 1. 基础特征（8个维度）
        base_feat = [
            1.0 if entity in entity_to_type else 0.0,  # has_label
            float(entity_in_degree.get(entity, 0)),  # in_degree
            float(entity_out_degree.get(entity, 0)),  # out_degree
            float(entity_in_degree.get(entity, 0) + entity_out_degree.get(entity, 0)),  # total_degree
            float(len(set(entity_relations.get(entity, [])))),  # unique_relations
            float(len(entity_neighbors.get(entity, []))),  # neighbor_count
        ]

        # 计算有标签的邻居数量
        labeled_neighbors = 0
        for neighbor in entity_neighbors.get(entity, []):
            if neighbor in entity_to_type:
                labeled_neighbors += 1

        # 邻居类型多样性（简化）
        neighbor_types = []
        for neighbor in entity_neighbors.get(entity, []):
            if neighbor in entity_to_type:
                neighbor_types.append(entity_to_type[neighbor])

        base_feat.append(float(labeled_neighbors))  # labeled_neighbor_count
        base_feat.append(float(len(set(neighbor_types))))  # unique_neighbor_types

        base_features.append(base_feat)

        # 2. 关系分布特征（20个维度）
        rel_feat = np.zeros(len(top_relations), dtype=np.float32)
        rel_counts = Counter(entity_relations.get(entity, []))
        total_rels = sum(rel_counts.values())

        if total_rels > 0:
            for rel, count in rel_counts.items():
                if rel in relation_to_feat_idx:
                    rel_idx = relation_to_feat_idx[rel]
                    rel_feat[rel_idx] = count / total_rels
        relation_features.append(rel_feat)

    # 组合所有特征
    base_features_np = np.array(base_features, dtype=np.float32)  # 8维
    relation_features_np = np.array(relation_features, dtype=np.float32)  # 20维

    # 标准化
    scaler = StandardScaler()
    base_features_scaled = scaler.fit_transform(base_features_np)

    # 合并特征
    all_features = np.concatenate([base_features_scaled, relation_features_np], axis=1)

    node_features = torch.tensor(all_features, dtype=torch.float32)

    print(f"特征提取完成:")
    print(f"  基础特征: {base_features_np.shape[1]} 维")
    print(f"  关系特征: {relation_features_np.shape[1]} 维")
    print(f"  总特征维度: {node_features.shape[1]}")

    # 特征名称
    feature_names = [
        'has_label', 'in_degree', 'out_degree', 'total_degree',
        'unique_relations', 'neighbor_count', 'labeled_neighbor_count',
        'unique_neighbor_types'
    ]
    feature_names.extend([f"rel_{i}" for i in range(len(top_relations))])

    return {
        'node_features': node_features,
        'entity_to_type': entity_to_type,
        'feature_names': feature_names,
        'scaler': scaler,
        'top_relations': top_relations
    }


def prepare_labels(entity_to_idx, entity_to_type, label_encoder=None):
    """准备标签数据"""
    print("\n准备标签数据...")

    # 获取有标签的实体
    labeled_entities = [e for e in entity_to_idx.keys() if e in entity_to_type]
    labeled_indices = [entity_to_idx[e] for e in labeled_entities]
    labels = [entity_to_type[e] for e in labeled_entities]

    print(f"有标签的实体: {len(labeled_entities)} / {len(entity_to_idx)}")

    # 编码标签
    if label_encoder is None:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
    else:
        encoded_labels = label_encoder.transform(labels)

    # 创建完整的标签张量
    full_labels = torch.full((len(entity_to_idx),), -1, dtype=torch.long)
    for i, idx in enumerate(labeled_indices):
        full_labels[idx] = encoded_labels[i]

    num_classes = len(label_encoder.classes_)

    # 计算类别权重
    class_counts = np.bincount(encoded_labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f"类别数量: {num_classes}")
    print(f"类别分布范围: {class_counts.min()} ~ {class_counts.max()}")

    return {
        'labels': full_labels,
        'labeled_indices': labeled_indices,
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'class_weights': class_weights
    }


def train_model(model, g, features, labels, train_mask, val_mask, class_weights):
    """训练模型 - 修正函数签名"""
    print("\n" + "=" * 60)
    print("步骤3: 训练模型...")
    print("=" * 60)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 移动数据到设备
    model = model.to(device)
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    class_weights = class_weights.to(device)

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # 训练函数 - 修正参数
    def train_epoch():
        model.train()
        optimizer.zero_grad()

        logits = model(g, features)  # 移除了neighbor_types参数
        loss = criterion(logits[train_mask], labels[train_mask])

        preds = logits[train_mask].argmax(dim=1)
        acc = (preds == labels[train_mask]).float().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item(), acc.item()

    # 评估函数 - 修正参数
    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        logits = model(g, features)  # 移除了neighbor_types参数

        loss = criterion(logits[mask], labels[mask])
        preds = logits[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean()

        # 计算F1分数
        preds_cpu = preds.cpu().numpy()
        true_cpu = labels[mask].cpu().numpy()
        f1 = f1_score(true_cpu, preds_cpu, average='weighted', zero_division=0)

        return loss.item(), acc.item(), f1, preds_cpu, true_cpu

    # 训练循环
    best_val_f1 = 0
    best_model_state = None
    patience = 30
    patience_counter = 0

    print("\n开始训练...")

    for epoch in range(200):  # 减少轮数
        train_loss, train_acc = train_epoch()
        val_loss, val_acc, val_f1, _, _ = evaluate(val_mask)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch + 1} 轮停止训练")
            break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, val_acc, val_f1


def save_model_and_data(model, model_config, data_dict, best_val_acc, best_val_f1):
    """保存模型和相关数据"""
    print("\n" + "=" * 60)
    print("步骤4: 保存模型和数据...")
    print("=" * 60)

    # 确保目录存在
    os.makedirs('models', exist_ok=True)

    # 准备保存数据 - 修复：确保scaler正确保存
    save_dict = {
        'model_state_dict': model.cpu().state_dict(),
        'model_config': model_config,
        'entity_to_idx': data_dict['entity_to_idx'],
        'idx_to_entity': data_dict['idx_to_entity'],
        'label_encoder': data_dict['label_encoder'],
        'node_features': data_dict['node_features'].cpu(),
        'feature_names': data_dict['feature_names'],
        'scaler': data_dict['scaler'],  # 确保scaler正确保存
        'top_relations': data_dict['top_relations'],
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1
    }

    # 保存模型
    model_path = 'models/entity_type_predictor_fixed.pth'
    torch.save(save_dict, model_path)
    print(f"✓ 模型已保存到: {model_path}")

    # 验证保存的数据
    print(f"保存的scaler类型: {type(data_dict['scaler'])}")
    if hasattr(data_dict['scaler'], 'n_features_in_'):
        print(f"Scaler训练特征维度: {data_dict['scaler'].n_features_in_}")

    return model_path


def main():
    """主训练函数"""
    print("=" * 80)
    print("实体类型预测模型训练")
    print("=" * 80)

    try:
        # 步骤1: 加载数据
        triples, entity_types, relation_counter, _ = load_training_data()
        if triples is None:
            return

        # 步骤2: 构建图
        g, entity_to_idx, idx_to_entity, _ = build_entity_graph(triples)

        # 步骤3: 提取特征
        feature_dict = extract_entity_features(triples, entity_types, entity_to_idx, relation_counter)

        # 步骤4: 准备标签
        label_dict = prepare_labels(entity_to_idx, feature_dict['entity_to_type'])

        # 合并数据字典
        data_dict = {
            'entity_to_idx': entity_to_idx,
            'idx_to_entity': idx_to_entity,
            'node_features': feature_dict['node_features'],
            'feature_names': feature_dict['feature_names'],
            'scaler': feature_dict['scaler'],
            'top_relations': feature_dict['top_relations']
        }
        data_dict.update(label_dict)

        # 步骤5: 划分数据集
        labeled_indices = np.array(data_dict['labeled_indices'])
        encoded_labels = label_dict['label_encoder'].transform(
            [feature_dict['entity_to_type'][idx_to_entity[i]] for i in labeled_indices]
        )

        train_idx, val_idx = train_test_split(
            labeled_indices,
            test_size=0.2,
            stratify=encoded_labels,
            random_state=42
        )

        train_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        val_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        print(f"训练集: {train_mask.sum().item()} 个节点")
        print(f"验证集: {val_mask.sum().item()} 个节点")

        # 步骤6: 创建模型 - 使用正确的维度
        in_feats = data_dict['node_features'].shape[1]  # 应该是28维 (8基础特征 + 20关系特征)
        print(f"输入特征维度: {in_feats}")

        # 调整隐藏层维度，确保计算正确
        h_feats = 128  # 减小隐藏层维度
        num_layers = 2  # 减少层数

        model = RelationAwareGCN(
            in_feats=in_feats,
            h_feats=h_feats,
            num_classes=data_dict['num_classes'],
            num_layers=num_layers,
            dropout=0.3
        )

        model_config = {
            'in_feats': in_feats,
            'h_feats': h_feats,
            'num_classes': data_dict['num_classes'],
            'num_layers': num_layers,
            'dropout': 0.3,
            'model_type': 'RelationAwareGCN'
        }

        print(f"模型配置:")
        print(f"  输入特征: {in_feats}")
        print(f"  隐藏层: {h_feats}")
        print(f"  类别数: {data_dict['num_classes']}")
        print(f"  层数: {num_layers}")

        # 步骤7: 训练模型 - 修正调用
        model, best_val_acc, best_val_f1 = train_model(
            model, g, data_dict['node_features'], data_dict['labels'],
            train_mask, val_mask, data_dict['class_weights']
        )

        # 步骤8: 保存模型
        model_path = save_model_and_data(model, model_config, data_dict, best_val_acc, best_val_f1)

        print(f"\n训练完成!")
        print(f"模型文件: {model_path}")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()