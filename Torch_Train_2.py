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
from collections import Counter
import warnings
import random
from tqdm import tqdm
import gc
import os

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


class ResGCN(nn.Module):
    """ResGCN模型（与测试代码保持一致）"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=4, dropout=0.3):
        super(ResGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 输入层
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True))
        self.bns.append(nn.BatchNorm1d(h_feats))

        # 隐藏层（使用检测到的4层）
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 输出层
        self.fc = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features

        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            h_new = layer(g, h)
            h_new = bn(h_new)

            if i > 0:  # 残差连接
                h_new = h_new + h

            h = F.relu(h_new)
            h = self.dropout(h)

        out = self.fc(h)
        return out


def load_training_data():
    """加载训练数据"""
    print("=" * 60)
    print("步骤1: 加载训练数据...")
    print("=" * 60)

    # 加载关系三元组
    triples = []
    relation_counter = Counter()

    try:
        with open('data/FB15KET/xunlian.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="加载关系数据"):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    triples.append((head, relation, tail))
                    relation_counter[relation] += 1
    except FileNotFoundError:
        print("错误: xunlian.txt 文件不存在")
        return None, None, None

    print(f"加载了 {len(triples)} 个关系三元组")
    print(f"发现 {len(relation_counter)} 种不同关系类型")

    # 加载实体类型数据
    try:
        entity_types = pd.read_csv('data/FB15KET/Entity_All_typed.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误: Entity_All_typed.csv 文件不存在")
        return None, None, None

    print(f"加载了 {len(entity_types)} 个实体类型记录")

    return triples, entity_types, relation_counter


def build_graph_and_features(triples, entity_types, relation_counter):
    """构建图和特征"""
    print("\n" + "=" * 60)
    print("步骤2: 构建图和特征...")
    print("=" * 60)

    from collections import defaultdict, Counter  # 确保导入

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
    in_degree = Counter()  # 修复：改为小写
    out_degree = Counter()  # 修复：改为小写

    for h, r, t in tqdm(triples, desc="统计关系"):
        entity_relations[h].append(r)
        entity_relations[t].append(r)
        out_degree[h] += 1  # 修复：使用小写
        in_degree[t] += 1  # 修复：使用小写

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

    # 关系类型特征
    for rel in top_relations:
        rel_name = rel.split('/')[-1][:10] if '/' in rel else rel[:10]
        feature_names.append(f'rel_{rel_name}')

    print(f"总共提取 {len(feature_names)} 个特征")

    # 提取特征
    for entity in tqdm(all_entities, desc="提取节点特征"):
        features = []

        # 1. 基础特征
        has_label = 1.0 if entity in entity_to_type else 0.0
        features.append(has_label)

        total_degree = in_degree.get(entity, 0) + out_degree.get(entity, 0)  # 修复：使用小写
        features.append(float(total_degree))
        features.append(float(in_degree.get(entity, 0)))  # 修复：使用小写
        features.append(float(out_degree.get(entity, 0)))  # 修复：使用小写

        rels = entity_relations.get(entity, [])
        unique_rels = len(set(rels))
        features.append(float(unique_rels))

        # 2. 邻居类型统计特征（关键！）
        stats = neighbor_type_stats[entity]
        features.append(float(stats['neighbor_count']))
        features.append(float(stats['labeled_neighbor_count']))
        features.append(float(stats['unique_neighbor_types']))
        features.append(float(stats['most_common_neighbor_type']))

        # 3. 关系类型分布特征
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
            label_mask_list.append(entity_to_idx[entity])

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


def prepare_labels(labels_list, label_mask_list, entity_to_idx):
    """准备标签"""
    print("\n准备标签...")

    if not labels_list:
        print("错误：没有找到有标签的实体！")
        return None

    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_list)
    num_classes = len(label_encoder.classes_)

    # 创建完整的标签张量
    full_labels = torch.full((len(entity_to_idx),), -1, dtype=torch.long)
    for i, idx in enumerate(label_mask_list):
        full_labels[idx] = encoded_labels[i]

    labels = full_labels

    # 类别分布
    class_counts = np.bincount(encoded_labels)
    print(f"总类别数: {num_classes}")
    print(f"类别分布范围: {class_counts.min()} ~ {class_counts.max()}")

    # 类别权重
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
    class_weights = 1.0 / (class_counts_tensor + 1e-8)
    class_weights = class_weights / class_weights.sum()

    return labels, label_encoder, num_classes, class_weights


def train_model(model, g, features, labels, train_mask, val_mask, class_weights):
    """训练模型"""
    print("\n" + "=" * 60)
    print("步骤3: 训练模型...")
    print("=" * 60)

    # 设备
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

    # 训练函数
    def train_epoch(model, g, features, labels, train_mask, optimizer, criterion):
        model.train()
        optimizer.zero_grad()

        logits = model(g, features)
        loss = criterion(logits[train_mask], labels[train_mask])

        preds = logits[train_mask].argmax(dim=1)
        acc = (preds == labels[train_mask]).float().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item(), acc.item()

    # 评估函数
    @torch.no_grad()
    def evaluate(model, g, features, labels, mask, criterion):
        model.eval()
        logits = model(g, features)

        loss = criterion(logits[mask], labels[mask])
        preds = logits[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean()

        preds_cpu = preds.cpu().numpy()
        true_cpu = labels[mask].cpu().numpy()
        f1 = f1_score(true_cpu, preds_cpu, average='weighted', zero_division=0)

        return loss.item(), acc.item(), f1, preds.cpu().numpy(), true_cpu

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # 训练循环
    best_val_acc = 0
    best_val_f1 = 0
    best_model_state = None
    patience = 30
    patience_counter = 0

    print("\n开始训练...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
    print("-" * 70)

    for epoch in range(200):
        train_loss, train_acc = train_epoch(model, g, features, labels,
                                            train_mask, optimizer, criterion)

        val_loss, val_acc, val_f1, _, _ = evaluate(model, g, features,
                                                   labels, val_mask, criterion)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"{epoch + 1:<6} {train_loss:<12.4f} {train_acc:<12.4f} "
                  f"{val_loss:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f}")

        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch + 1} 轮停止训练")
            break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n加载最佳模型，验证集F1分数: {best_val_f1:.4f}")

    # 最终评估
    val_loss, val_acc, val_f1, val_preds, val_true = evaluate(
        model, g, features, labels, val_mask, criterion
    )

    print(f"\n验证集结果:")
    print(f"  损失: {val_loss:.4f}")
    print(f"  准确率: {val_acc:.4f}")
    print(f"  F1分数: {val_f1:.4f}")

    return model, val_acc, val_f1


def save_model(model, model_config, entity_to_idx, label_encoder, scaler,
               node_features, feature_names, best_val_acc, best_val_f1,
               neighbor_type_stats, idx_to_entity):
    """保存模型"""
    print("\n" + "=" * 60)
    print("步骤4: 保存模型...")
    print("=" * 60)

    # 确保目录存在
    os.makedirs('models', exist_ok=True)

    save_dict = {
        'model_state_dict': model.cpu().state_dict(),
        'model_config': model_config,
        'entity_to_idx': entity_to_idx,
        'idx_to_entity': idx_to_entity,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'node_features': node_features.cpu(),
        'feature_names': feature_names,
        'neighbor_type_stats': neighbor_type_stats,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1
    }

    model_path = 'models/entity_type_predictor_resgcn.pth'
    torch.save(save_dict, model_path)
    print(f"✓ 模型已保存到: {model_path}")

    # 保存配置信息
    config_info = f"""
模型训练配置:
=============
输入特征维度: {model_config['in_feats']}
隐藏层维度: {model_config['h_feats']}
类别数量: {model_config['num_classes']}
模型层数: {model_config['num_layers']}
Dropout率: {model_config['dropout']}
验证集准确率: {best_val_acc:.4f}
验证集F1分数: {best_val_f1:.4f}
特征数量: {len(feature_names)}
实体数量: {len(entity_to_idx)}
模型文件: {model_path}
"""

    with open('models/training_config.txt', 'w', encoding='utf-8') as f:
        f.write(config_info)

    print(config_info)

    return model_path


def evaluate_single_entity(model, g, features, entity_to_idx, idx_to_entity,
                           label_encoder, entity_to_type, neighbor_type_stats,
                           triples, test_entity=None):
    """评估单个实体（模拟测试场景）"""
    print("\n" + "=" * 60)
    print("步骤5: 模拟测试 - 单个实体评估")
    print("=" * 60)

    # 如果未指定测试实体，随机选择一个有标签的实体
    if test_entity is None:
        labeled_entities = [e for e in entity_to_type.keys() if e in entity_to_idx]
        if not labeled_entities:
            print("错误: 没有找到有标签的实体")
            return

        test_entity = random.choice(labeled_entities)

    print(f"测试实体: {test_entity}")

    if test_entity not in entity_to_idx:
        print(f"错误: 实体 {test_entity} 不在图中")
        return

    idx = entity_to_idx[test_entity]

    # 获取实体的关系和邻居
    entity_relations = []
    entity_neighbors = []

    for h, r, t in triples:
        if h == test_entity:
            entity_relations.append(r)
            entity_neighbors.append(t)
        elif t == test_entity:
            entity_relations.append(r)
            entity_neighbors.append(h)

    print(f"实体关系数: {len(entity_relations)}")
    print(f"实体邻居数: {len(entity_neighbors)}")

    # 显示邻居类型信息
    print(f"\n邻居类型信息:")
    neighbor_types = []
    for neighbor in entity_neighbors:
        if neighbor in entity_to_type:
            neighbor_type = entity_to_type[neighbor]
            neighbor_types.append(neighbor_type)
            print(f"  {neighbor}: 类型 {neighbor_type}")
        else:
            print(f"  {neighbor}: 类型未知")

    if neighbor_types:
        type_counts = Counter(neighbor_types)
        print(f"\n邻居类型统计:")
        for type_id, count in type_counts.most_common():
            print(f"  类型 {type_id}: {count} 个")

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits[idx], dim=0)

        # 获取top-3预测
        top_probs, top_indices = torch.topk(probs, 3)

        print(f"\n模型预测结果:")
        print(f"{'排名':<6} {'类型':<15} {'置信度':<12} {'是否真实'}")
        print("-" * 50)

        for i, (prob, cls_idx) in enumerate(zip(top_probs, top_indices)):
            pred_class = cls_idx.item()
            pred_type = label_encoder.inverse_transform([pred_class])[0]
            prob_val = prob.item()

            # 检查是否是真实类型
            true_type = entity_to_type.get(test_entity, "未知")
            true_type_encoded = label_encoder.transform([true_type])[0] if test_entity in entity_to_type else -1
            is_true = "✓" if pred_class == true_type_encoded else ""

            print(f"{i + 1:<6} {pred_type:<15} {prob_val:<12.2%} {is_true}")

        if test_entity in entity_to_type:
            print(f"\n真实类型: {true_type} (编码: {true_type_encoded})")

    # 分析特征重要性
    print(f"\n特征分析:")
    stats = neighbor_type_stats.get(test_entity, {})
    print(f"  邻居总数: {stats.get('neighbor_count', 0)}")
    print(f"  有标签的邻居数: {stats.get('labeled_neighbor_count', 0)}")
    print(f"  不同邻居类型数: {stats.get('unique_neighbor_types', 0)}")
    print(f"  最常见的邻居类型: {stats.get('most_common_neighbor_type', 0)}")


def main():
    """主训练函数"""
    print("=" * 80)
    print("ResGCN实体类型预测模型训练")
    print("使用 xunlian.txt 进行训练")
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

        # 步骤4: 创建模型
        print("\n创建ResGCN模型...")
        in_feats = node_features.shape[1]
        h_feats = 128
        num_layers = 4  # 使用4层

        model = ResGCN(
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
            'model_type': 'ResGCN'
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

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from collections import defaultdict, Counter

    main()