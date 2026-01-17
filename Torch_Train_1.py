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
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
import random
from tqdm import tqdm

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


def main():
    """主函数：训练实体类型预测模型（优化版）"""

    # ========== 1. 数据加载与预处理（增强版） ==========
    print("=" * 60)
    print("步骤1: 加载和预处理数据...")
    print("=" * 60)

    # 加载关系三元组
    triples = []
    relation_counter = Counter()
    with open('data/FB15KET/xunlian.txt', 'r') as f:
        for line in tqdm(f, desc="加载关系数据"):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                triples.append((head, relation, tail))
                relation_counter[relation] += 1

    print(f"加载了 {len(triples)} 个关系三元组")
    print(f"发现 {len(relation_counter)} 种不同关系类型")

    # 加载实体类型数据
    entity_types = pd.read_csv('data/FB15KET/Entity_All_typed.csv')
    print(f"加载了 {len(entity_types)} 个实体类型记录")

    # 创建实体到类型的映射
    entity_to_type = dict(zip(entity_types['entity_id'],
                              entity_types['predicted_category']))

    # 统计类型分布
    type_counts = entity_types['predicted_category'].value_counts()
    print(f"\n实体类型分布统计:")
    print(f"总类别数: {type_counts.shape[0]}")
    print(f"类别分布:")
    print(type_counts.describe())

    # 显示类别不平衡情况
    print(f"\n前10个最常见的类型:")
    for i, (type_id, count) in enumerate(type_counts.head(10).items()):
        print(f"  类型 {type_id}: {count} 个样本 ({count / len(entity_types):.2%})")

    # ========== 2. 构建图结构（优化特征提取） ==========
    print("\n" + "=" * 60)
    print("步骤2: 构建图结构并提取丰富特征...")
    print("=" * 60)

    # 收集所有实体
    all_entities = set()
    for h, r, t in triples:
        all_entities.update([h, t])

    # 创建映射：实体/关系到索引
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}

    # 统计每个实体的关系类型分布
    entity_relations = {entity: [] for entity in all_entities}
    for h, r, t in triples:
        entity_relations[h].append(r)
        entity_relations[t].append(r)

    # 选择最常见的关系类型作为特征
    top_relations = [rel for rel, _ in relation_counter.most_common(50)]  # 取前50种关系
    relation_to_feat_idx = {rel: i for i, rel in enumerate(top_relations)}

    # ========== 3. 提取丰富特征 ==========
    print("提取丰富特征...")

    # 先构建networkx图以便计算中心性等特征
    print("构建NetworkX图计算复杂特征...")
    G_nx = nx.Graph()
    for h, r, t in triples[:50000]:  # 限制边数以防内存问题
        G_nx.add_edge(h, t, relation=r)

    # 计算度中心性（限制在有标签的实体上以减少计算量）
    labeled_entities = set(entity_to_type.keys())
    labeled_entities_in_graph = [e for e in labeled_entities if e in G_nx]

    if len(labeled_entities_in_graph) > 0:
        print(f"计算 {len(labeled_entities_in_graph)} 个有标签实体的中心性特征...")
        degree_centrality = nx.degree_centrality(G_nx)
        # 为节省时间，只计算部分中心性
        # betweenness_centrality = nx.betweenness_centrality(G_nx.subgraph(list(G_nx.nodes())[:1000]))
    else:
        degree_centrality = {}

    # 准备特征矩阵
    node_features = []
    labels = []
    label_mask = []
    feature_names = []

    # 基础特征
    feature_names.extend([
        'has_label',  # 是否有标签
        'degree',  # 总度数
        'in_degree',  # 入度
        'out_degree',  # 出度
        'unique_relations',  # 不同关系类型数
        'total_relations'  # 总关系数
    ])

    # 添加关系类型特征
    for rel in top_relations:
        feature_names.append(f'rel_{rel.split("/")[-1][:10]}')

    print(f"总共提取 {len(feature_names)} 个特征")

    for entity, idx in tqdm(entity_to_idx.items(), desc="提取节点特征"):
        features = []

        # 1. 是否有标签
        has_label = 1.0 if entity in entity_to_type else 0.0
        features.append(has_label)

        # 2. 度特征
        rels = entity_relations[entity]
        features.append(float(len(rels)))  # 总关系数作为度近似

        # 入度和出度（简化为统计关系）
        incoming = sum(1 for h, r, t in triples if t == entity)
        outgoing = sum(1 for h, r, t in triples if h == entity)
        features.append(float(incoming))
        features.append(float(outgoing))

        # 3. 关系多样性
        unique_rels = len(set(rels))
        features.append(float(unique_rels))
        features.append(float(len(rels)))

        # 4. 关系类型分布特征（one-hot简化版）
        rel_vector = [0.0] * len(top_relations)
        for rel in rels:
            if rel in relation_to_feat_idx:
                rel_vector[relation_to_feat_idx[rel]] += 1.0

        # 归一化
        if sum(rel_vector) > 0:
            rel_vector = [v / sum(rel_vector) for v in rel_vector]
        features.extend(rel_vector)

        # 5. 中心性特征（如果有）
        if entity in degree_centrality:
            features.append(degree_centrality[entity])
            if 'centrality' not in feature_names:
                feature_names.append('degree_centrality')
        elif has_label:  # 只有有标签的实体需要中心性
            features.append(0.0)
            if 'centrality' not in feature_names:
                feature_names.append('degree_centrality')

        node_features.append(features)

        # 标签
        if entity in entity_to_type:
            labels.append(entity_to_type[entity])
            label_mask.append(idx)
        else:
            labels.append(-1)

    # 确保所有特征向量长度一致
    max_len = max(len(f) for f in node_features)
    for i in range(len(node_features)):
        while len(node_features[i]) < max_len:
            node_features[i].append(0.0)

    # 转换为张量
    node_features = torch.tensor(node_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    print(f"特征矩阵形状: {node_features.shape}")

    # 特征标准化
    print("标准化特征...")
    scaler = StandardScaler()
    node_features_np = node_features.numpy()

    # 只使用有标签的样本进行标准化，避免数据泄露
    labeled_indices = np.array(label_mask)
    if len(labeled_indices) > 0:
        labeled_features = node_features_np[labeled_indices]
        scaler.fit(labeled_features)

    node_features_np = scaler.transform(node_features_np)
    node_features = torch.tensor(node_features_np, dtype=torch.float32)

    # ========== 4. 处理类别不平衡 ==========
    print("\n" + "=" * 60)
    print("步骤3: 处理类别不平衡...")
    print("=" * 60)

    # 只对有标签的节点进行编码和划分
    labeled_indices = torch.tensor(label_mask)
    labeled_labels = labels[labeled_indices]

    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labeled_labels.numpy())
    num_classes = len(label_encoder.classes_)

    # 更新标签
    for i, idx in enumerate(labeled_indices):
        labels[idx] = encoded_labels[i]

    # 类别分布
    print(f"类别分布（编码后）:")
    class_dist = Counter(encoded_labels)
    for cls, count in class_dist.most_common(10):
        print(f"  类别 {cls}: {count} 个样本 ({count / len(encoded_labels):.2%})")

    # 使用类别权重处理不平衡
    class_counts = torch.bincount(torch.tensor(encoded_labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    print(f"\n类别权重: {class_weights.tolist()}")

    # 划分数据集（分层抽样）
    print("\n划分数据集...")
    train_idx, temp_idx = train_test_split(
        labeled_indices.numpy(),
        test_size=0.3,
        stratify=encoded_labels,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=encoded_labels[np.isin(labeled_indices.numpy(), temp_idx)],
        random_state=42
    )

    # 创建掩码
    train_mask = torch.zeros(node_features.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(node_features.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(node_features.shape[0], dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"训练集: {train_mask.sum().item()} 个节点")
    print(f"验证集: {val_mask.sum().item()} 个节点")
    print(f"测试集: {test_mask.sum().item()} 个节点")
    print(f"总类别数: {num_classes}")

    # ========== 5. 构建DGL图（用于GNN） ==========
    print("\n" + "=" * 60)
    print("步骤4: 构建DGL图...")
    print("=" * 60)

    # 构建DGL图
    src_nodes = []
    dst_nodes = []

    # 使用所有三元组
    for h, r, t in triples:
        src_nodes.append(entity_to_idx[h])
        dst_nodes.append(entity_to_idx[t])

    # 创建图
    num_nodes = len(all_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    # 添加自环
    g = dgl.add_self_loop(g)

    print(f"图构建完成: 节点数={g.num_nodes()}, 边数={g.num_edges()}")

    # ========== 6. 定义增强的GNN模型 ==========
    print("\n" + "=" * 60)
    print("步骤5: 定义增强的GNN模型...")
    print("=" * 60)

    class EnhancedEntityGNN(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes, dropout=0.4):
            super(EnhancedEntityGNN, self).__init__()

            # 特征增强层
            self.feature_enhancer = nn.Sequential(
                nn.Linear(in_feats, in_feats * 2),
                nn.BatchNorm1d(in_feats * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_feats * 2, h_feats),
                nn.BatchNorm1d(h_feats),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # GCN层
            self.gcn1 = dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
            self.bn1 = nn.BatchNorm1d(h_feats)

            self.gcn2 = dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
            self.bn2 = nn.BatchNorm1d(h_feats)

            # 注意力机制
            self.attention = nn.MultiheadAttention(h_feats, num_heads=4, batch_first=True)
            self.attention_norm = nn.LayerNorm(h_feats)

            # 分类头
            self.classifier = nn.Sequential(
                nn.Linear(h_feats * 2, h_feats),
                nn.BatchNorm1d(h_feats),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_feats, h_feats // 2),
                nn.BatchNorm1d(h_feats // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_feats // 2, num_classes)
            )

            self.dropout = nn.Dropout(dropout)

        def forward(self, g, features):
            # 特征增强
            h = self.feature_enhancer(features)

            # 图卷积
            h_gcn = self.gcn1(g, h)
            h_gcn = self.bn1(h_gcn)
            h_gcn = F.relu(h_gcn)
            h_gcn = self.dropout(h_gcn)

            h_gcn = self.gcn2(g, h_gcn)
            h_gcn = self.bn2(h_gcn)
            h_gcn = F.relu(h_gcn)
            h_gcn = self.dropout(h_gcn)

            # 注意力机制（全局特征）
            h_reshaped = h_gcn.unsqueeze(0)  # [1, N, D]
            attn_output, _ = self.attention(h_reshaped, h_reshaped, h_reshaped)
            attn_output = attn_output.squeeze(0)
            attn_output = self.attention_norm(attn_output + h_gcn)  # 残差连接

            # 特征融合
            combined = torch.cat([h_gcn, attn_output], dim=1)

            # 分类
            out = self.classifier(combined)
            return out

    # 更简单的ResGCN（如果上面模型太复杂）
    class ResGCN(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes, num_layers=3, dropout=0.3):
            super(ResGCN, self).__init__()

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

    # 选择模型
    MODEL_TYPE = "ResGCN"  # 可选: "Enhanced" 或 "ResGCN"

    in_feats = node_features.shape[1]
    h_feats = 256 if MODEL_TYPE == "Enhanced" else 128

    if MODEL_TYPE == "Enhanced":
        model = EnhancedEntityGNN(in_feats, h_feats, num_classes)
        print("使用增强版GNN模型（带注意力机制）")
    else:
        model = ResGCN(in_feats, h_feats, num_classes, num_layers=4)
        print("使用ResGCN模型（带残差连接）")

    print(f"输入特征维度: {in_feats}")
    print(f"隐藏层维度: {h_feats}")
    print(f"输出类别数: {num_classes}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 7. 训练配置和训练 ==========
    print("\n" + "=" * 60)
    print("步骤6: 训练模型...")
    print("=" * 60)

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 移动数据到设备
    model = model.to(device)
    g = g.to(device)
    node_features = node_features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # 创建类别权重张量
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 训练函数
    def train_epoch(model, g, features, labels, train_mask, optimizer, criterion, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()

        # 前向传播
        logits = model(g, features)

        # 计算损失（带类别权重）
        loss = criterion(logits[train_mask], labels[train_mask])

        # 计算准确率
        preds = logits[train_mask].argmax(dim=1)
        correct = (preds == labels[train_mask]).sum().item()
        total = train_mask.sum().item()
        acc = correct / total

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item(), acc

    # 评估函数
    @torch.no_grad()
    def evaluate(model, g, features, labels, mask, criterion):
        model.eval()
        logits = model(g, features)

        # 计算损失
        loss = criterion(logits[mask], labels[mask])

        # 计算准确率和其他指标
        preds = logits[mask].argmax(dim=1)
        true_labels = labels[mask]

        acc = (preds == true_labels).float().mean().item()

        # 计算F1分数
        preds_cpu = preds.cpu().numpy()
        true_cpu = true_labels.cpu().numpy()

        # 计算加权F1（处理类别不平衡）
        f1 = f1_score(true_cpu, preds_cpu, average='weighted', zero_division=0)

        return loss.item(), acc, f1, preds, true_labels

    # 训练配置
    learning_rate = 0.001
    weight_decay = 5e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    # 使用带类别权重的损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    # 训练循环
    best_val_acc = 0
    best_val_f1 = 0
    best_model_state = None
    patience = 40
    patience_counter = 0

    print("\n开始训练...")
    print(
        f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12} {'LR':<12}")
    print("-" * 80)

    for epoch in range(300):
        # 训练
        train_loss, train_acc = train_epoch(model, g, node_features, labels,
                                            train_mask, optimizer, criterion, epoch)

        # 验证
        val_loss, val_acc, val_f1, _, _ = evaluate(model, g, node_features,
                                                   labels, val_mask, criterion)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{epoch + 1:<6} {train_loss:<12.4f} {train_acc:<12.4f} "
                  f"{val_loss:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f} "
                  f"{current_lr:<12.6f}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch + 1} 轮停止训练")
            break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n加载最佳模型:")
        print(f"  最佳验证准确率: {best_val_acc:.4f}")
        print(f"  最佳验证F1分数: {best_val_f1:.4f}")

    # ========== 8. 模型评估 ==========
    print("\n" + "=" * 60)
    print("步骤7: 评估模型...")
    print("=" * 60)

    # 在测试集上评估
    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(
        model, g, node_features, labels, test_mask, criterion
    )

    print(f"测试集结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  F1分数: {test_f1:.4f}")

    # 详细分类报告
    print("\n详细分类报告:")
    report = classification_report(
        test_true.cpu().numpy(),
        test_preds.cpu().numpy(),
        target_names=[f"Class_{i}" for i in range(min(num_classes, 15))],
        zero_division=0,
        digits=4
    )
    print(report)

    # ========== 9. 错误分析和改进建议 ==========
    print("\n" + "=" * 60)
    print("步骤8: 错误分析和改进建议")
    print("=" * 60)

    # 分析错误预测
    test_preds_cpu = test_preds.cpu().numpy()
    test_true_cpu = test_true.cpu().numpy()

    # 计算混淆矩阵（主要类别）
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_true_cpu, test_preds_cpu)

    print(f"混淆矩阵形状: {cm.shape}")
    print(f"\n主要错误分析:")

    # 找出被错误预测最多的类别
    error_rate_by_class = []
    for i in range(min(num_classes, 20)):  # 只看前20类
        if i < cm.shape[0] and i < cm.shape[1]:
            total = cm[i].sum()
            correct = cm[i, i] if i < min(cm.shape[0], cm.shape[1]) else 0
            error = total - correct
            error_rate = error / total if total > 0 else 0
            error_rate_by_class.append((i, error_rate, total))

    # 按错误率排序
    error_rate_by_class.sort(key=lambda x: x[1], reverse=True)

    print("\n错误率最高的5个类别:")
    for cls, error_rate, total in error_rate_by_class[:5]:
        print(f"  类别 {cls}: 错误率 {error_rate:.2%} ({total}个样本)")

    # ========== 10. 预测示例和可视化 ==========
    print("\n" + "=" * 60)
    print("步骤9: 预测示例")
    print("=" * 60)

    def predict_with_confidence(entity_id, top_k=5):
        """带置信度的预测"""
        if entity_id not in entity_to_idx:
            print(f"实体 {entity_id} 不在图中")
            return None

        idx = entity_to_idx[entity_id]
        model.eval()

        with torch.no_grad():
            logits = model(g, node_features)
            probs = F.softmax(logits[idx], dim=0)

            # 获取top-k预测
            top_probs, top_classes = torch.topk(probs, top_k)

            print(f"实体: {entity_id}")
            print(f"{'排名':<6} {'类型':<15} {'置信度':<12} {'是否真实'}")
            print("-" * 50)

            for i, (prob, cls_idx) in enumerate(zip(top_probs, top_classes)):
                cls_name = label_encoder.inverse_transform([cls_idx.cpu().item()])[0]
                prob_val = prob.item()

                # 检查是否是真实类型
                is_true = ""
                if entity_id in entity_to_type:
                    true_type = entity_to_type[entity_id]
                    true_type_encoded = label_encoder.transform([true_type])[0]
                    if cls_idx.item() == true_type_encoded:
                        is_true = "✓"

                print(f"{i + 1:<6} {cls_name:<15} {prob_val:<12.2%} {is_true}")

        return top_probs, top_classes

    # 预测一些示例实体
    print("\n预测示例:")
    example_entities = list(entity_to_type.keys())[:10]
    for i, entity in enumerate(example_entities):
        print(f"\n示例 {i + 1}:")
        predict_with_confidence(entity, top_k=3)

    # ========== 11. 保存模型和结果 ==========
    print("\n" + "=" * 60)
    print("步骤10: 保存模型和结果")
    print("=" * 60)

    # 保存完整模型
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_feats': in_feats,
            'h_feats': h_feats,
            'num_classes': num_classes,
            'model_type': MODEL_TYPE
        },
        'node_features': node_features.cpu(),
        'entity_to_idx': entity_to_idx,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'class_weights': class_weights,
        'optimizer_state': optimizer.state_dict(),
    }

    torch.save(save_dict, 'entity_type_predictor_enhanced.pth')

    print("✓ 模型已保存到 'entity_type_predictor_enhanced.pth'")

    # 保存测试结果
    results_df = pd.DataFrame({
        '实体ID': [idx_to_entity.get(idx, f'unknown_{idx}') for idx in test_idx],
        '真实类型': test_true_cpu,
        '预测类型': test_preds_cpu,
        '是否正确': (test_true_cpu == test_preds_cpu).astype(int)
    })

    results_df.to_csv('entity_type_predictions.csv', index=False)
    print("✓ 预测结果已保存到 'entity_type_predictions.csv'")

    print("\n" + "=" * 60)
    print("训练完成！总结:")
    print("=" * 60)
    print(f"1. 模型类型: {MODEL_TYPE}")
    print(f"2. 验证集准确率: {best_val_acc:.4f}")
    print(f"3. 验证集F1分数: {best_val_f1:.4f}")
    print(f"4. 测试集准确率: {test_acc:.4f}")
    print(f"5. 测试集F1分数: {test_f1:.4f}")
    print(f"6. 总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 进一步改进建议
    print("\n" + "=" * 60)
    print("如需进一步提高准确率，建议:")
    print("=" * 60)
    print("1. 添加更多特征:")
    print("   - 实体名称的嵌入特征（如果有实体名称）")
    print("   - 更复杂的图特征（聚类系数、PageRank等）")
    print("   - 关系路径特征")
    print("\n2. 使用预训练模型:")
    print("   - 使用TransE、RotatE等知识图谱嵌入作为特征")
    print("   - 使用BERT等语言模型处理实体描述")
    print("\n3. 模型改进:")
    print("   - 尝试更深的GNN（4-6层）")
    print("   - 使用RGCN处理异构图")
    print("   - 集成多个模型")
    print("\n4. 数据增强:")
    print("   - 对少数类进行过采样")
    print("   - 使用图数据增强（边删除、特征掩码）")

    return model, g, node_features, entity_to_idx, label_encoder


if __name__ == "__main__":
    try:
        model, graph, features, entity_map, label_encoder = main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()