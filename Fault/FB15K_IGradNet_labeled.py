# FB15K_IGradNet_complete.py
import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import argparse
from collections import defaultdict
import json
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ============== 异构图神经网络 (HetGNN) ==============
class HetGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 第一层异构图卷积
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')

        # 第二层异构图卷积
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')

        # BiLSTM用于信息融合
        self.bilstm = nn.LSTM(
            input_size=out_feats,
            hidden_size=out_feats // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, graph, inputs):
        # 第一层卷积计算
        h = self.conv1(graph, inputs)
        # 应用ReLU激活函数
        h = {k: F.relu(v) for k, v in h.items()}
        # 第二层卷积计算
        h = self.conv2(graph, h)

        # BiLSTM处理实体节点表示
        if 'entity' in h:
            entity_feats = h['entity']
            # 增加batch维度
            entity_feats = entity_feats.unsqueeze(1)
            # 通过BiLSTM
            entity_feats, _ = self.bilstm(entity_feats)
            # 移除batch维度
            entity_feats = entity_feats.squeeze(1)
            h['entity'] = entity_feats

        return h


# ============== 二阶邻居子图构建模块 ==============
class SubgraphBuilder:
    def __init__(self, hetero_graph):
        self.g = hetero_graph

    def get_two_hop_neighbors(self, node_id):
        """获取节点的二阶邻居，返回子图"""
        # 获取一阶邻居 (通过关系连接的其他实体)
        neighbor_entities = []

        # 遍历所有关系类型
        for etype in self.g.etypes:
            if etype.startswith('rel_'):
                # 获取通过该关系连接的邻居
                try:
                    predecessors = self.g.predecessors(node_id, etype=etype)
                    successors = self.g.successors(node_id, etype=etype)

                    # 添加邻居
                    neighbor_entities.extend(predecessors.tolist())
                    neighbor_entities.extend(successors.tolist())

                    # 对于每个一阶邻居，获取其二阶邻居
                    for neighbor in list(predecessors.tolist()) + list(successors.tolist()):
                        # 获取该邻居的其他邻居
                        for etype2 in self.g.etypes:
                            if etype2.startswith('rel_'):
                                pred2 = self.g.predecessors(neighbor, etype=etype2)
                                succ2 = self.g.successors(neighbor, etype=etype2)
                                neighbor_entities.extend(pred2.tolist())
                                neighbor_entities.extend(succ2.tolist())
                except:
                    continue

        # 移除自身ID
        if node_id in neighbor_entities:
            neighbor_entities.remove(node_id)

        # 去重
        neighbor_entities = list(set(neighbor_entities))

        return neighbor_entities

    def build_subgraph_embedding(self, node_id, node_feats, embedding_weights=None):
        """构建节点的子图嵌入"""
        if embedding_weights is None:
            # 默认权重分配
            embedding_weights = {
                'center': 0.6,  # 中心实体节点
                'neighbors': 0.4,  # 邻居实体
            }

        # 获取二阶邻居节点
        neighbor_entities = self.get_two_hop_neighbors(node_id)

        # 构建子图嵌入
        center_emb = node_feats['entity'][node_id] * embedding_weights['center']

        # 邻居嵌入
        if len(neighbor_entities) > 0:
            neighbor_emb = torch.mean(node_feats['entity'][neighbor_entities], dim=0) * embedding_weights['neighbors']
        else:
            neighbor_emb = torch.zeros_like(center_emb)

        # 合并子图嵌入
        subgraph_emb = center_emb + neighbor_emb

        # 保存组成部分，用于可解释性分析
        components = {
            'center': center_emb / embedding_weights['center'] if embedding_weights['center'] > 0 else center_emb,
            'neighbors': neighbor_emb / embedding_weights['neighbors'] if embedding_weights[
                                                                              'neighbors'] > 0 else neighbor_emb
        }

        return subgraph_emb, components


# ============== 原型网络分类模块 ==============
class PrototypeNetwork(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 初始化类原型 (将在训练中更新)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))

        # 温度参数，控制softmax的平滑度
        self.temperature = 0.1

    def forward(self, features):
        # 计算与各原型的相似度
        distances = torch.cdist(features, self.prototypes, p=2)  # 欧氏距离

        # 基于距离计算相似度 (距离越小，相似度越高)
        similarities = torch.log((distances ** 2 + 1) / (distances ** 2 + 1e-4))

        # 计算分类概率
        logits = similarities / self.temperature
        probs = F.softmax(logits, dim=1)

        return probs, similarities

    def update_prototypes(self, features, labels):
        """更新类原型"""
        for c in range(self.num_classes):
            # 选择属于类别c的样本
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                # 计算该类别的原型 (类中心)
                self.prototypes.data[c] = class_features.mean(dim=0)


# ============== IGradNet模型整合所有组件 ==============
class IGradNet(nn.Module):
    def __init__(self, hetero_graph, feature_dim, hidden_dim=128, out_dim=64, num_classes=8):
        super().__init__()
        self.g = hetero_graph
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        # 异构图神经网络
        self.hetgnn = HetGNN(feature_dim, hidden_dim, out_dim, hetero_graph.etypes)

        # 子图构建器
        self.subgraph_builder = SubgraphBuilder(hetero_graph)

        # 原型网络
        self.prototype_net = PrototypeNetwork(out_dim, num_classes)

        # 结构贡献权重
        self.structure_weights = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, node_features):
        # 通过异构图神经网络处理
        node_embeddings = self.hetgnn(self.g, node_features)
        return node_embeddings

    def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
        subgraph_embeddings = []
        components_list = []

        # 为每个节点构建子图嵌入
        for node_id in node_ids:
            weights = {
                'center': self.structure_weights[0].item(),
                'neighbors': self.structure_weights[1].item()
            }
            subgraph_emb, components = self.subgraph_builder.build_subgraph_embedding(
                node_id, node_embeddings, weights
            )
            subgraph_embeddings.append(subgraph_emb)
            components_list.append(components)

        # 将列表转换为张量
        subgraph_embeddings = torch.stack(subgraph_embeddings)

        # 通过原型网络进行分类
        probs, similarities = self.prototype_net(subgraph_embeddings)

        # 如果处于训练模式且提供了标签，更新原型
        if update_prototypes and labels is not None:
            self.prototype_net.update_prototypes(subgraph_embeddings, labels)

        # 预测类别
        _, predicted_classes = torch.max(probs, dim=1)

        # 为可解释性分析准备数据
        explanations = {
            'similarities': similarities,
            'components': components_list,
            'structure_weights': self.structure_weights
        }

        return probs, predicted_classes, explanations


# ============== 训练函数 ==============
def train(model, node_features, train_idx, val_idx, labels, num_epochs=50, lr=0.001, weight_decay=1e-4, device=None):
    """训练IGradNet模型"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 将模型移到设备
    model = model.to(device)

    # 将数据转移到设备
    train_idx_tensor = torch.tensor(train_idx).to(device)
    val_idx_tensor = torch.tensor(val_idx).to(device)

    # 确保所有节点特征在正确设备上
    for k in node_features:
        if node_features[k].device != device:
            node_features[k] = node_features[k].to(device)

    # 确保标签在正确设备上
    if labels.device != device:
        labels = labels.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 训练记录
    best_val_acc = 0
    best_epoch = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        # 前向传播
        node_embeddings = model(node_features)

        # 计算分类结果
        probs, predicted, explanations = model.classify(
            node_embeddings, train_idx_tensor, update_prototypes=True, labels=labels[train_idx_tensor]
        )

        # 计算损失
        loss = F.cross_entropy(probs, labels[train_idx_tensor])

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 更新学习率
        scheduler.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                node_embeddings = model(node_features)
                val_probs, val_predicted, _ = model.classify(
                    node_embeddings, val_idx_tensor, update_prototypes=False
                )

                # 计算验证准确率
                val_acc = (val_predicted == labels[val_idx_tensor]).float().mean().item()
                val_accuracies.append(val_acc)

                # 打印训练进度
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

                # 检查是否是最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    # 保存最佳模型
                    torch.save(model.state_dict(), 'best_model_fb15k.pth')

        # 早停判断
        if epoch - best_epoch > 30:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, train_losses, val_accuracies


# ============== 评估函数 ==============
def evaluate(model, node_features, test_idx, true_labels, category_names, device=None):
    """评估模型性能"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 将数据转移到设备
    for k in node_features:
        node_features[k] = node_features[k].to(device)
    true_labels = true_labels.to(device)

    # 获取测试预测
    with torch.no_grad():
        node_embeddings = model(node_features)
        test_probs, test_predicted, explanations = model.classify(
            node_embeddings, test_idx, update_prototypes=False
        )

    # 转移到CPU进行评估
    test_predicted = test_predicted.cpu()
    true_labels_cpu = true_labels[test_idx].cpu()

    # 计算评估指标
    accuracy = accuracy_score(true_labels_cpu, test_predicted)
    macro_f1 = f1_score(true_labels_cpu, test_predicted, average='macro')
    weighted_f1 = f1_score(true_labels_cpu, test_predicted, average='weighted')

    print(f"\n=== 评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"宏平均F1 (Macro-F1): {macro_f1:.4f}")
    print(f"加权平均F1 (Weighted-F1): {weighted_f1:.4f}")

    # 打印分类报告
    print(f"\n分类报告:")
    print(classification_report(true_labels_cpu, test_predicted, target_names=category_names))

    # 返回结果
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': test_predicted,
        'true_labels': true_labels_cpu,
        'explanations': explanations
    }


# ============== 数据处理器 ==============
class FB15KLabeledDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = self._read_mapping('entity2id.txt')
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = self._read_mapping('relation2id.txt')

    def _read_mapping(self, filename):
        """读取映射文件"""
        mapping = {}
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    def _read_triples(self, filename):
        """读取三元组文件"""
        triples = []
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h in self.entity2id and r in self.relation2id and t in self.entity2id:
                        triples.append((h, r, t))
        return triples

    def load_labeled_data(self, labels_file='labels_final.json'):
        """加载标注数据"""
        print("加载标注数据...")

        # 加载标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            entity_labels = json.load(f)

        # 读取所有三元组文件
        all_triples = []
        for filename in ['train.txt', 'valid.txt', 'test.txt']:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                triples = self._read_triples(filename)
                all_triples.extend(triples)

        # 统计有标签的实体
        labeled_entities = set()
        for entity in entity_labels.keys():
            if entity in self.entity2id:
                labeled_entities.add(entity)

        print(f"有标签的实体数量: {len(labeled_entities)}")

        # 只保留两个实体都有标签的三元组
        labeled_triples = []
        for h, r, t in all_triples:
            if h in labeled_entities and t in labeled_entities:
                labeled_triples.append((h, r, t))

        print(f"有标签的三元组数量: {len(labeled_triples)}")

        # 统计类别分布
        category_counts = defaultdict(int)
        for entity, label in entity_labels.items():
            if entity in self.entity2id:
                category_counts[label] += 1

        print("\n类别分布:")
        for label in sorted(category_counts.keys()):
            print(f"类别{label}: {category_counts[label]} 个实体")

        return {
            'entity_labels': entity_labels,
            'labeled_triples': labeled_triples,
            'category_counts': dict(category_counts)
        }


# ============== 可解释性分析器 ==============
class ExplainabilityAnalyzer:
    def __init__(self, model, category_names):
        self.model = model
        self.category_names = category_names

    def analyze_prediction(self, node_id, node_features, entity_name=None, true_label=None):
        """分析单个实体的预测"""
        # 获取预测
        probs, pred_class, explanations = self.model.classify(
            node_features, [node_id], update_prototypes=False
        )

        # 提取相似度
        similarities = explanations['similarities'][0]

        # 获取结构贡献
        components = explanations['components'][0]
        structure_contributions = {
            'center': torch.norm(components['center']).item(),
            'neighbors': torch.norm(components['neighbors']).item()
        }

        # 归一化
        total_contrib = sum(structure_contributions.values())
        if total_contrib > 0:
            structure_contributions = {k: v / total_contrib for k, v in structure_contributions.items()}

        # 相似度字典
        similarity_dict = {}
        for i in range(len(self.category_names)):
            similarity_dict[self.category_names[i]] = similarities[i].item()

        result = {
            'entity': entity_name if entity_name else f"Entity_{node_id}",
            'prediction': {
                'class': pred_class.item(),
                'class_name': self.category_names[pred_class.item()],
                'probability': probs[0, pred_class.item()].item()
            },
            'true_label': {
                'class': true_label,
                'class_name': self.category_names[true_label] if true_label is not None else None
            } if true_label is not None else None,
            'similarities': similarity_dict,
            'structure_contributions': structure_contributions
        }

        return result

    def visualize_similarity_heatmap(self, similarities, entity_names=None):
        """可视化相似度热力图"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if entity_names is None:
            entity_names = [f"Entity_{i}" for i in range(similarities.shape[0])]

        plt.figure(figsize=(12, 8))
        sns.heatmap(similarities.numpy(),
                    xticklabels=self.category_names,
                    yticklabels=entity_names[:similarities.shape[0]],
                    cmap="YlOrRd",
                    annot=True,
                    fmt=".2f")
        plt.title("实体与类别原型相似度热力图")
        plt.xlabel("类别")
        plt.ylabel("实体")
        plt.tight_layout()
        plt.savefig("prototype_similarity_heatmap.png", dpi=300)
        plt.close()
        print("相似度热力图已保存为 prototype_similarity_heatmap.png")


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description='FB15K实体分类使用IGradNet模型（基于标注数据）')
    parser.add_argument('--data_dir', type=str, default='data/FB15K', help='FB15K数据目录')
    parser.add_argument('--labels_file', type=str, default='labels_final.json', help='标签文件路径')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--out_dim', type=int, default=64, help='输出维度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--analyze_samples', action='store_true', help='分析样本预测')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    # 1. 加载标注数据
    print("\n" + "=" * 60)
    print("FB15K 实体分类 - IGradNet 模型")
    print("=" * 60)

    processor = FB15KLabeledDataProcessor(args.data_dir)
    data = processor.load_labeled_data(args.labels_file)

    entity_labels = data['entity_labels']
    labeled_triples = data['labeled_triples']

    # 2. 构建异构图
    print("\n构建异构图...")

    # 准备边数据
    edge_dict = {}
    for h, r, t in labeled_triples:
        h_id = processor.entity2id[h]
        r_id = processor.relation2id[r]
        t_id = processor.entity2id[t]

        rel_name = f'rel_{r_id}'
        if rel_name not in edge_dict:
            edge_dict[rel_name] = ([], [])
        edge_dict[rel_name][0].append(h_id)
        edge_dict[rel_name][1].append(t_id)

    print(f"构建 {len(edge_dict)} 种关系类型的异构图")

    # 构建图
    g = dgl.heterograph(edge_dict)

    # 3. 创建节点特征和标签
    num_entities = len(processor.entity2id)
    feature_dim = args.hidden_dim

    # 使用随机初始化特征
    torch.manual_seed(args.seed)
    entity_features = torch.randn(num_entities, feature_dim)

    # 设置节点特征
    g.nodes['entity'].data['feature'] = entity_features

    # 创建标签张量
    labels = torch.full((num_entities,), -1, dtype=torch.long)
    for entity_name, label in entity_labels.items():
        if entity_name in processor.entity2id:
            entity_id = processor.entity2id[entity_name]
            labels[entity_id] = label - 1  # 转换为0-based

    # 只保留有标签的实体
    labeled_indices = torch.where(labels != -1)[0].tolist()
    print(f"有标签的实体数量: {len(labeled_indices)}")

    if len(labeled_indices) == 0:
        print("错误：没有找到有标签的实体！")
        return

    # 4. 划分数据集 (7:1:2)
    print("\n划分数据集...")
    train_size = int(len(labeled_indices) * 0.7)
    val_size = int(len(labeled_indices) * 0.1)
    test_size = len(labeled_indices) - train_size - val_size

    np.random.seed(args.seed)
    np.random.shuffle(labeled_indices)

    train_idx = labeled_indices[:train_size]
    val_idx = labeled_indices[train_size:train_size + val_size]
    test_idx = labeled_indices[train_size + val_size:]

    print(f"训练集: {len(train_idx)} 个实体")
    print(f"验证集: {len(val_idx)} 个实体")
    print(f"测试集: {len(test_idx)} 个实体")

    # 5. 创建模型
    num_classes = len(set(entity_labels.values()))
    category_names = [f"类别{i + 1}" for i in range(num_classes)]

    print(f"\n创建IGradNet模型...")
    print(f"特征维度: {feature_dim}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"输出维度: {args.out_dim}")
    print(f"类别数量: {num_classes}")

    model = IGradNet(
        g,
        feature_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_classes=num_classes
    )

    # 准备节点特征
    node_features = {'entity': g.nodes['entity'].data['feature']}

    # 6. 训练模型
    print("\n" + "=" * 60)
    print("开始训练模型")
    print("=" * 60)

    trained_model, train_losses, val_accuracies = train(
        model, node_features, train_idx, val_idx, labels,
        num_epochs=args.epochs, lr=0.001, weight_decay=1e-4, device=device
    )

    # 绘制训练曲线
    if train_losses and val_accuracies:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(5, 5 * len(val_accuracies) + 1, 5), val_accuracies)
        plt.title('验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300)
        plt.close()
        print("训练曲线已保存为 training_curve.png")

    # 7. 评估模型
    print("\n" + "=" * 60)
    print("评估模型性能")
    print("=" * 60)

    results = evaluate(
        trained_model, node_features, test_idx, labels, category_names, device=device
    )

    # 8. 可解释性分析
    if args.analyze_samples:
        print("\n" + "=" * 60)
        print("可解释性分析")
        print("=" * 60)

        analyzer = ExplainabilityAnalyzer(trained_model, category_names)

        # 分析几个样本
        sample_indices = test_idx[:5]  # 分析前5个测试样本
        for i, node_id in enumerate(sample_indices, 1):
            entity_name = processor.id2entity.get(node_id, f"Entity_{node_id}")
            true_label = labels[node_id].item() if labels[node_id] != -1 else None

            analysis = analyzer.analyze_prediction(node_id, node_features, entity_name, true_label)

            print(f"\n样本 {i}: {entity_name}")
            print(
                f"预测类别: {analysis['prediction']['class_name']} (置信度: {analysis['prediction']['probability']:.3f})")

            if true_label is not None:
                print(f"真实类别: {analysis['true_label']['class_name']}")
                print(f"预测{'正确' if analysis['prediction']['class'] == true_label else '错误'}")

            print(f"结构贡献: 自身特征 {analysis['structure_contributions']['center']:.3f}, "
                  f"邻居特征 {analysis['structure_contributions']['neighbors']:.3f}")

            # 显示相似度最高的3个类别
            similarities = analysis['similarities']
            top_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"最相似的3个类别:")
            for cat, sim in top_similarities:
                print(f"  {cat}: {sim:.3f}")

    # 可视化相似度热力图（部分样本）
    if results['explanations']['similarities'] is not None:
        sample_similarities = results['explanations']['similarities'][:20]  # 前20个样本
        sample_names = []
        for i in range(min(20, len(test_idx))):
            node_id = test_idx[i]
            entity_name = processor.id2entity.get(node_id, f"Entity_{node_id}")
            sample_names.append(entity_name[:20] + "..." if len(entity_name) > 20 else entity_name)

        analyzer = ExplainabilityAnalyzer(trained_model, category_names)
        analyzer.visualize_similarity_heatmap(sample_similarities, sample_names)

    # 9. 保存模型
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)

    model_info = {
        'model_state_dict': trained_model.state_dict(),
        'entity2id': processor.entity2id,
        'relation2id': processor.relation2id,
        'entity_labels': entity_labels,
        'category_names': category_names,
        'test_indices': test_idx,
        'results': {
            'accuracy': results['accuracy'],
            'macro_f1': results['macro_f1'],
            'weighted_f1': results['weighted_f1']
        }
    }

    torch.save(model_info, 'fb15k_igradnet_trained.pth')

    print(f"模型已保存为 'fb15k_igradnet_trained.pth'")
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"宏平均F1: {results['macro_f1']:.4f}")
    print(f"加权平均F1: {results['weighted_f1']:.4f}")

    return trained_model, results, processor, g, node_features


# ============== 推理函数 ==============
def predict_single_entity(model, entity_id, node_features, category_names, processor=None):
    """预测单个实体的类别"""
    model.eval()
    with torch.no_grad():
        node_embeddings = model(node_features)
        probs, predicted, explanations = model.classify(
            node_embeddings, [entity_id], update_prototypes=False
        )

    prediction = {
        'entity_id': entity_id,
        'entity_name': processor.id2entity[
            entity_id] if processor and entity_id in processor.id2entity else f"Entity_{entity_id}",
        'predicted_class': predicted.item(),
        'predicted_class_name': category_names[predicted.item()],
        'probability': probs[0, predicted.item()].item(),
        'all_probabilities': probs[0].tolist()
    }

    return prediction


if __name__ == "__main__":
    # 运行主函数
    model, results, processor, g, node_features = main()

    # 示例：如何使用训练好的模型进行预测
    print("\n" + "=" * 60)
    print("示例：使用训练好的模型进行预测")
    print("=" * 60)

    # 加载保存的模型
    model_info = torch.load('fb15k_igradnet_trained.pth', map_location='cpu')

    # 重新创建模型结构
    num_classes = len(model_info['category_names'])
    feature_dim = model_info['model_state_dict']['hetgnn.conv1.mods.rel_0.weight'].shape[1]

    # 创建新模型（需要图结构）
    # 注意：在实际使用中，您需要保存图结构或重新构建图

    print("训练完成！模型已保存并可以进行预测。")


# 辅助函数：加载和预测
def load_and_predict(model_path='fb15k_igradnet_trained.pth', entity_id=0):
    """加载模型并预测"""
    # 加载模型信息
    model_info = torch.load(model_path, map_location='cpu')

    # 注意：这个函数是简化的，实际使用时需要重新构建图结构
    # 这里只演示如何获取预测结果

    print(f"模型信息:")
    print(f"类别数量: {len(model_info['category_names'])}")
    print(f"类别名称: {model_info['category_names']}")
    print(f"测试准确率: {model_info['results']['accuracy']:.4f}")

    return model_info