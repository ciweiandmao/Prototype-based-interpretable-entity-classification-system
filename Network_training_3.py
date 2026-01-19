import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import dgl
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter, defaultdict
import warnings
import random
from tqdm import tqdm
import gc
import os

from Torch_Train_2 import evaluate_single_entity, save_model, train_model, load_training_data, prepare_labels

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


class AttentionLayer(nn.Module):
    """注意力机制层，用于学习邻居特征的重要性"""

    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, neighbor_features):
        # node_features: [batch_size, feature_dim]
        # neighbor_features: [batch_size, num_neighbors, feature_dim]
        batch_size, num_neighbors, feature_dim = neighbor_features.shape

        # 扩展节点特征以匹配邻居特征
        node_features_expanded = node_features.unsqueeze(1).expand(-1, num_neighbors, -1)

        # 拼接节点特征和邻居特征
        combined = torch.cat([node_features_expanded, neighbor_features], dim=-1)

        # 计算注意力权重
        attention_weights = self.attention(combined).squeeze(-1)  # [batch_size, num_neighbors]
        attention_weights = F.softmax(attention_weights, dim=1)

        # 应用注意力权重
        weighted_neighbors = neighbor_features * attention_weights.unsqueeze(-1)
        aggregated = weighted_neighbors.sum(dim=1)

        return aggregated, attention_weights


class RelationPatternEncoder(nn.Module):
    """关系模式编码器，专门处理三元组类型模式"""

    def __init__(self, feature_dim):
        super(RelationPatternEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim)
        )

    def forward(self, features):
        return self.encoder(features)


class EnhancedResGCN(nn.Module):
    """增强的ResGCN模型，加入注意力机制"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=4, dropout=0.3):
        super(EnhancedResGCN, self).__init__()

        # 关系模式编码器
        self.relation_encoder = RelationPatternEncoder(in_feats)

        # 注意力层（用于邻居特征聚合）
        self.attention = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, 1),
            nn.Sigmoid()
        )

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 输入层
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True))
        self.bns.append(nn.BatchNorm1d(h_feats))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 输出层
        self.fc = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout)

    def apply_attention(self, g, node_features, neighbor_features):
        """应用注意力机制聚合邻居特征"""
        # 获取邻居特征
        with g.local_scope():
            g.ndata['h'] = node_features
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neighbor_h'))
            neighbor_features = g.ndata['neighbor_h']

        # 拼接当前节点特征和邻居特征
        combined = torch.cat([node_features, neighbor_features], dim=1)

        # 计算注意力权重
        attention_weights = self.attention(combined)

        # 加权聚合
        enhanced_features = node_features + attention_weights * neighbor_features

        return enhanced_features

    def forward(self, g, features):
        # 首先用关系编码器处理特征
        h = self.relation_encoder(features)

        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            # 对第2层及以后的层应用注意力机制
            if i >= 1:
                h = self.apply_attention(g, h, h)

            h_new = layer(g, h)
            h_new = bn(h_new)

            # 残差连接
            if i > 0:
                h_new = h_new + h

            h = F.relu(h_new)
            h = self.dropout(h)

        out = self.fc(h)
        return out


def extract_relation_patterns(triples, all_entities, top_relations):
    """提取关系模式特征：统计每个实体作为头/尾时出现的关系类型"""
    print("提取关系模式特征...")

    # 初始化统计字典
    head_relation_patterns = defaultdict(Counter)
    tail_relation_patterns = defaultdict(Counter)

    # 统计每个实体作为头和尾时的关系
    for h, r, t in tqdm(triples, desc="统计关系模式"):
        if r in top_relations:
            head_relation_patterns[h][r] += 1
            tail_relation_patterns[t][r] += 1

    # 创建特征向量
    head_pattern_features = np.zeros((len(all_entities), len(top_relations)), dtype=np.float32)
    tail_pattern_features = np.zeros((len(all_entities), len(top_relations)), dtype=np.float32)

    # 将实体映射到索引
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}

    # 填充特征矩阵
    for entity, patterns in head_relation_patterns.items():
        if entity in entity_to_idx:
            idx = entity_to_idx[entity]
            for rel, count in patterns.items():
                if rel in top_relations:
                    rel_idx = top_relations.index(rel)
                    head_pattern_features[idx, rel_idx] = count

    for entity, patterns in tail_relation_patterns.items():
        if entity in entity_to_idx:
            idx = entity_to_idx[entity]
            for rel, count in patterns.items():
                if rel in top_relations:
                    rel_idx = top_relations.index(rel)
                    tail_pattern_features[idx, rel_idx] = count

    # 归一化
    head_sums = head_pattern_features.sum(axis=1, keepdims=True)
    head_sums[head_sums == 0] = 1
    head_pattern_features = head_pattern_features / head_sums

    tail_sums = tail_pattern_features.sum(axis=1, keepdims=True)
    tail_sums[tail_sums == 0] = 1
    tail_pattern_features = tail_pattern_features / tail_sums

    return head_pattern_features, tail_pattern_features


def build_graph_and_features(triples, entity_types, relation_counter):
    """构建图和特征 - 增强版"""
    print("\n" + "=" * 60)
    print("步骤2: 构建图和特征...")
    print("=" * 60)

    from collections import defaultdict, Counter

    # 收集所有实体
    all_entities = set()
    for h, r, t in triples:
        all_entities.update([h, t])

    print(f"训练集中的总实体数: {len(all_entities)}")

    # 创建映射
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}

    # 统计每个实体的关系
    print("统计实体关系...")
    entity_relations = defaultdict(list)
    in_degree = Counter()
    out_degree = Counter()

    for h, r, t in tqdm(triples, desc="统计关系"):
        entity_relations[h].append(r)
        entity_relations[t].append(r)
        out_degree[h] += 1
        in_degree[t] += 1

    # 创建实体到类型的映射
    entity_to_type = dict(zip(entity_types['entity_id'],
                              entity_types['predicted_category']))

    # 选择最常见的关系类型
    top_relations = [rel for rel, _ in relation_counter.most_common(30)]
    relation_to_feat_idx = {rel: i for i, rel in enumerate(top_relations)}

    # ========== 关键改进：提取更丰富的特征 ==========
    print("\n提取增强特征...")

    features_list = []
    labels_list = []
    label_mask_list = []
    feature_names = []

    # 基础特征
    feature_names.extend([
        'has_label',
        'total_degree',
        'in_degree',
        'out_degree',
        'unique_relations',
    ])

    # 邻居类型统计特征（关键改进！）
    print("计算邻居类型统计特征...")

    # 首先构建邻居映射
    entity_neighbors = defaultdict(list)
    for h, r, t in triples:
        entity_neighbors[h].append(t)
        entity_neighbors[t].append(h)

    # 为每个实体计算邻居类型分布
    neighbor_type_stats = {}
    for entity in tqdm(all_entities, desc="计算邻居特征"):
        neighbors = entity_neighbors.get(entity, [])
        neighbor_types = []

        for neighbor in neighbors:
            if neighbor in entity_to_type:
                neighbor_types.append(entity_to_type[neighbor])

        if neighbor_types:
            # 邻居类型多样性
            unique_types = len(set(neighbor_types))
            # 最常见的邻居类型
            type_counts = Counter(neighbor_types)
            most_common_type = type_counts.most_common(1)[0][0] if type_counts else 0

            neighbor_type_stats[entity] = {
                'neighbor_count': len(neighbors),
                'labeled_neighbor_count': len(neighbor_types),
                'unique_neighbor_types': unique_types,
                'most_common_neighbor_type': most_common_type
            }
        else:
            neighbor_type_stats[entity] = {
                'neighbor_count': len(neighbors),
                'labeled_neighbor_count': 0,
                'unique_neighbor_types': 0,
                'most_common_neighbor_type': 0
            }

    # 添加邻居特征名称
    feature_names.extend([
        'neighbor_count',
        'labeled_neighbor_count',
        'unique_neighbor_types',
        'most_common_neighbor_type'
    ])

    # 提取关系模式特征（新增！）
    print("提取关系模式特征...")
    head_pattern_features, tail_pattern_features = extract_relation_patterns(
        triples, all_entities, top_relations
    )

    # 添加关系模式特征名称
    for i, rel in enumerate(top_relations):
        rel_name = rel.split('/')[-1][:10] if '/' in rel else rel[:10]
        feature_names.append(f'head_rel_{rel_name}')
        feature_names.append(f'tail_rel_{rel_name}')

    # 关系类型特征
    for rel in top_relations:
        rel_name = rel.split('/')[-1][:10] if '/' in rel else rel[:10]
        feature_names.append(f'rel_{rel_name}')

    print(f"总共提取 {len(feature_names)} 个特征")

    # 提取特征
    for i, entity in enumerate(tqdm(all_entities, desc="提取节点特征")):
        features = []

        # 1. 基础特征
        has_label = 1.0 if entity in entity_to_type else 0.0
        features.append(has_label)

        total_degree = in_degree.get(entity, 0) + out_degree.get(entity, 0)
        features.append(float(total_degree))
        features.append(float(in_degree.get(entity, 0)))
        features.append(float(out_degree.get(entity, 0)))

        rels = entity_relations.get(entity, [])
        unique_rels = len(set(rels))
        features.append(float(unique_rels))

        # 2. 邻居类型统计特征
        stats = neighbor_type_stats[entity]
        features.append(float(stats['neighbor_count']))
        features.append(float(stats['labeled_neighbor_count']))
        features.append(float(stats['unique_neighbor_types']))
        features.append(float(stats['most_common_neighbor_type']))

        # 3. 关系模式特征（新增！）
        idx = entity_to_idx[entity]
        features.extend(head_pattern_features[idx].tolist())
        features.extend(tail_pattern_features[idx].tolist())

        # 4. 关系类型分布特征
        rel_vector = [0.0] * len(top_relations)
        for rel in rels:
            if rel in relation_to_feat_idx:
                rel_vector[relation_to_feat_idx[rel]] += 1.0

        if sum(rel_vector) > 0:
            rel_vector = [v / sum(rel_vector) for v in rel_vector]
        features.extend(rel_vector)

        features_list.append(features)

        # 标签
        if entity in entity_to_type:
            labels_list.append(entity_to_type[entity])
            label_mask_list.append(idx)

    # 转换为张量
    features_np = np.array(features_list, dtype=np.float32)

    # 特征标准化
    print("标准化特征...")
    scaler = StandardScaler()

    if label_mask_list:
        labeled_features = features_np[label_mask_list]
        scaler.fit(labeled_features)

    features_np = scaler.transform(features_np)
    node_features = torch.tensor(features_np, dtype=torch.float32)

    print(f"特征矩阵形状: {node_features.shape}")

    # 清理内存
    del features_list, features_np
    gc.collect()

    # 构建DGL图
    print("\n构建DGL图...")
    src_nodes = []
    dst_nodes = []

    for h, r, t in tqdm(triples, desc="构建图边"):
        src_nodes.append(entity_to_idx[h])
        dst_nodes.append(entity_to_idx[t])

    num_nodes = len(all_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    # 添加自环
    g = dgl.add_self_loop(g)
    print(f"图构建完成: 节点数={g.num_nodes()}, 边数={g.num_edges()}")

    return (g, node_features, entity_to_idx, idx_to_entity, entity_to_type,
            label_mask_list, labels_list, scaler, feature_names, neighbor_type_stats)


def main():
    """主训练函数"""
    print("=" * 80)
    print("增强版ResGCN实体类型预测模型训练")
    print("加入关系模式特征和注意力机制")
    print("=" * 80)

    try:
        # 步骤1: 加载数据
        triples, entity_types, relation_counter = load_training_data()
        if triples is None:
            return

        # 步骤2: 构建图和特征
        (g, node_features, entity_to_idx, idx_to_entity, entity_to_type,
         label_mask_list, labels_list, scaler, feature_names, neighbor_type_stats) = build_graph_and_features(
            triples, entity_types, relation_counter
        )

        # 步骤3: 准备标签
        labels, label_encoder, num_classes, class_weights = prepare_labels(
            labels_list, label_mask_list, entity_to_idx
        )

        if labels is None:
            return

        # 划分数据集
        print("\n划分数据集...")
        labeled_indices = np.array(label_mask_list)
        encoded_labels = label_encoder.transform(labels_list)

        train_idx, val_idx = train_test_split(
            labeled_indices,
            test_size=0.001,
            stratify=encoded_labels,
            random_state=42
        )

        train_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        val_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        print(f"训练集: {train_mask.sum().item()} 个节点")
        print(f"验证集: {val_mask.sum().item()} 个节点")

        # 步骤4: 创建增强模型
        print("\n创建增强ResGCN模型（带注意力机制）...")
        in_feats = node_features.shape[1]
        h_feats = 128
        num_layers = 4

        model = EnhancedResGCN(
            in_feats=in_feats,
            h_feats=h_feats,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=0.3
        )

        model_config = {
            'in_feats': in_feats,
            'h_feats': h_feats,
            'num_classes': num_classes,
            'num_layers': num_layers,
            'dropout': 0.3,
            'model_type': 'EnhancedResGCN_with_Attention'
        }

        print(f"模型参数:")
        print(f"  输入特征: {in_feats}")
        print(f"  隐藏层: {h_feats}")
        print(f"  类别数: {num_classes}")
        print(f"  层数: {num_layers}")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 步骤5: 训练模型
        model, best_val_acc, best_val_f1 = train_model(
            model, g, node_features, labels, train_mask, val_mask, class_weights
        )

        # 步骤6: 保存模型
        model_path = save_model(
            model, model_config, entity_to_idx, label_encoder, scaler,
            node_features, feature_names, best_val_acc, best_val_f1,
            neighbor_type_stats, idx_to_entity
        )

        # 步骤7: 模拟测试单个实体
        evaluate_single_entity(
            model, g, node_features, entity_to_idx, idx_to_entity,
            label_encoder, entity_to_type, neighbor_type_stats, triples
        )

        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        print(f"模型文件: {model_path}")
        print(f"验证集准确率: {best_val_acc:.4f}")
        print(f"验证集F1分数: {best_val_f1:.4f}")
        print("\n新增功能说明:")
        print("1. 加入了关系模式特征（实体作为头/尾时的关系分布）")
        print("2. 加入了注意力机制，学习邻居特征的重要性")
        print("3. 增强了特征表示，结合了多种特征源")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from collections import defaultdict, Counter

    main()