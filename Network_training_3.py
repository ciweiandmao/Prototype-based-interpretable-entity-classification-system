import time

import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import subprocess
import sys
import argparse
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
import gc


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_cuda_status(device):
    """检查CUDA状态"""
    if device.type == 'cuda':
        print(f"\n=== CUDA状态检查 ===")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(device)}")

        # 检查模型是否在GPU上
        print(f"\n=== 模型设备检查 ===")

        # 检查CUDA计算是否被启用
        print(f"CUDA计算已启用: {torch.cuda.is_initialized()}")

        # 检查内存使用
        print(f"GPU内存分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

        # 执行一个简单的CUDA计算测试
        test_tensor = torch.randn(1000, 1000, device=device)
        result = test_tensor @ test_tensor.T
        print(f"CUDA计算测试: 完成 {result.shape} 矩阵乘法")

        return True
    return False

# 内存优化版本的HetGNN
# 修复 MemoryEfficientHetGNN 类的 forward 方法
class MemoryEfficientHetGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.rel_names = rel_names

        # 确保所有层都在正确的设备上初始化
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats, norm='right', weight=True, bias=True)
            for rel in rel_names}, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats, norm='right', weight=True, bias=True)
            for rel in rel_names}, aggregate='mean')

        self.fc = nn.Linear(out_feats, out_feats)

        # 强制将参数移动到GPU（如果可用）
        self._ensure_device()

    def _ensure_device(self):
        """确保所有参数在正确设备上"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, graph, inputs, batch_nodes=None):
        """内存优化的前向传播"""
        # 确保图在正确设备上
        device = graph.device

        if batch_nodes is None or len(batch_nodes) == 0:
            return {'entity': torch.tensor([], device=device)}

        # 确保batch_nodes在正确设备上
        batch_nodes_tensor = torch.tensor(batch_nodes, device=device, dtype=torch.int64)

        try:
            # 不使用混合精度，保持统一的dtype
            subgraph = dgl.sampling.sample_neighbors(
                graph, {'entity': batch_nodes_tensor}, fanout=5
            )

            if subgraph.num_edges() == 0:
                # 如果没有邻居，直接返回中心节点的特征
                h = {'entity': inputs['entity'][batch_nodes_tensor]}
                h['entity'] = self.fc(h['entity'])
                return h

            # 将子图移动到GPU（如果尚未在GPU上）
            if subgraph.device != device:
                subgraph = subgraph.to(device)

            # 获取子图中的节点特征
            sub_inputs = {}
            for ntype in inputs:
                if ntype in subgraph.ntypes:
                    original_node_ids = subgraph.nodes(ntype)
                    # 确保节点ID在正确设备上
                    if original_node_ids.device != device:
                        original_node_ids = original_node_ids.to(device)
                    sub_inputs[ntype] = inputs[ntype][original_node_ids]

            # 使用图卷积
            h = self.conv1(subgraph, sub_inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(subgraph, h)

            if 'entity' in h:
                # 处理batch_nodes对应的特征
                entity_feats = h['entity']

                # 使用向量化操作而不是循环
                batch_nodes_tensor_long = batch_nodes_tensor.long()
                subgraph_entity_nodes = subgraph.nodes('entity')

                # 创建掩码
                if subgraph_entity_nodes.device != device:
                    subgraph_entity_nodes = subgraph_entity_nodes.to(device)

                # 确保dtype一致
                subgraph_entity_nodes = subgraph_entity_nodes.long()
                batch_nodes_tensor_long = batch_nodes_tensor_long.long()

                # 检查哪些节点是batch_nodes
                is_batch_node = torch.isin(subgraph_entity_nodes, batch_nodes_tensor_long)

                if is_batch_node.sum() > 0:
                    # 确保dtype一致
                    batch_feats = entity_feats[is_batch_node]
                    # 确保batch_feats是float32
                    if batch_feats.dtype != torch.float32:
                        batch_feats = batch_feats.float()

                    batch_feats = self.fc(batch_feats)

                    # 确保赋值时的dtype一致
                    if entity_feats.dtype != batch_feats.dtype:
                        entity_feats = entity_feats.to(batch_feats.dtype)

                    entity_feats[is_batch_node] = batch_feats

                h['entity'] = entity_feats

            return h

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU内存不足，跳过批次...")
                return {'entity': torch.tensor([], device=device)}
            else:
                raise e
# 批量化的子图构建模块
# 修复 BatchSubgraphBuilder 类
# 修复 BatchSubgraphBuilder.get_neighbors_batch 方法
class BatchSubgraphBuilder:
    def __init__(self, hetero_graph):
        self.g = hetero_graph
        self.neighbor_cache = {}

    def get_neighbors_batch(self, node_ids, fanout=5):  # 减小fanout
        """批量获取节点的邻居"""
        batch_neighbors = {}
        device = self.g.device

        for node_id in node_ids:
            if node_id not in self.neighbor_cache:
                try:
                    # 获取少量邻居
                    node_tensor = torch.tensor([node_id], device=device, dtype=torch.int64)
                    frontiers = dgl.sampling.sample_neighbors(
                        self.g, {'entity': node_tensor}, fanout=fanout
                    )
                    neighbors = frontiers.nodes('entity')
                    if neighbors.device.type == 'cuda':
                        neighbors = neighbors.cpu()
                    neighbors = neighbors.numpy()
                    self.neighbor_cache[node_id] = neighbors
                except:
                    self.neighbor_cache[node_id] = np.array([])

            batch_neighbors[node_id] = self.neighbor_cache[node_id]

        return batch_neighbors

    def build_subgraph_embeddings_batch(self, node_ids, node_feats, embedding_weights=None,
                                        batch_size=8):  # 减小batch_size
        """批量构建子图嵌入"""
        if embedding_weights is None:
            embedding_weights = {
                'center': 0.7,
                'relations': 0.3
            }

        subgraph_embeddings = []
        components_list = []

        # 分批处理，使用更小的batch
        for i in range(0, len(node_ids), batch_size):
            batch_nodes = node_ids[i:i + batch_size]
            batch_neighbors = self.get_neighbors_batch(batch_nodes)

            for node_id in batch_nodes:
                if node_id not in batch_neighbors:
                    continue

                neighbors = batch_neighbors[node_id]

                # 中心节点嵌入
                center_emb = node_feats['entity'][node_id] * embedding_weights['center']

                # 关系邻居嵌入
                relation_emb = torch.zeros_like(center_emb)
                if len(neighbors) > 0:
                    # 只使用少量邻居
                    max_neighbors = 5
                    if len(neighbors) > max_neighbors:
                        np.random.seed(node_id % 1000)
                        neighbors = np.random.choice(neighbors, max_neighbors, replace=False)

                    # 排除中心节点自身
                    other_neighbors = []
                    for n in neighbors:
                        if n != node_id:
                            other_neighbors.append(n)

                    if other_neighbors:
                        other_neighbors_tensor = torch.tensor(other_neighbors,
                                                              device=node_feats['entity'].device)
                        neighbor_feats = node_feats['entity'][other_neighbors_tensor]
                        relation_emb = torch.mean(neighbor_feats, dim=0) * embedding_weights['relations']

                # 合并子图嵌入
                subgraph_emb = center_emb + relation_emb
                subgraph_embeddings.append(subgraph_emb)

                # 保存组成部分
                components = {
                    'center': center_emb / embedding_weights['center'] if embedding_weights[
                                                                              'center'] > 0 else center_emb,
                    'relations': relation_emb / embedding_weights['relations'] if embedding_weights[
                                                                                      'relations'] > 0 else relation_emb
                }
                components_list.append(components)

                # 及时清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 将列表转换为张量
        if subgraph_embeddings:
            subgraph_embeddings = torch.stack(subgraph_embeddings)
        else:
            subgraph_embeddings = torch.tensor([], device=node_feats['entity'].device)

        return subgraph_embeddings, components_list


# 特征重要性分析模块
class FeatureImportance(nn.Module):
    def __init__(self, feature_dim, category_dim=9):
        super().__init__()
        self.feature_dim = feature_dim
        self.category_dim = category_dim
        self.other_dim = feature_dim - category_dim

        # 更小的特征重要性权重矩阵
        self.W_imp = nn.Parameter(torch.randn(feature_dim, 1) * 0.01)

    def forward(self, node_features):
        # 分批计算特征重要性
        batch_size = 256
        imp_scores_list = []

        for i in range(0, node_features.size(0), batch_size):
            batch = node_features[i:i + batch_size]
            batch_scores = torch.sigmoid(batch @ self.W_imp)
            imp_scores_list.append(batch_scores)

        imp_scores = torch.cat(imp_scores_list, dim=0)

        # 分离类别得分特征和其他特征的重要性
        if self.category_dim > 0 and imp_scores.size(0) > 0:
            category_imp = imp_scores[:, :self.category_dim].mean()
            other_imp = imp_scores[:, self.category_dim:].mean()
        else:
            category_imp = torch.tensor(0.0)
            other_imp = imp_scores.mean() if imp_scores.size(0) > 0 else torch.tensor(0.0)

        # 归一化
        total_imp = category_imp + other_imp
        if total_imp > 0:
            category_imp = category_imp / total_imp
            other_imp = other_imp / total_imp

        return category_imp, other_imp, imp_scores


# 内存优化的原型网络
class MemoryEfficientPrototypeNetwork(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 更小的原型初始化
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.1)

        # 温度参数
        self.temperature = 0.1

        # 缓存最近的原型更新
        self.prototype_updates = {}

    def forward(self, features, batch_size=256):
        """分批计算相似度"""
        if features.size(0) == 0:
            return torch.tensor([]), torch.tensor([])

        similarities_list = []

        for i in range(0, features.size(0), batch_size):
            batch = features[i:i + batch_size]

            # 计算与各原型的相似度
            distances = torch.cdist(batch, self.prototypes, p=2)

            # 基于距离计算相似度
            batch_similarities = torch.log((distances ** 2 + 1) / (distances ** 2 + 1e-4))
            similarities_list.append(batch_similarities)

        similarities = torch.cat(similarities_list, dim=0)

        # 计算分类概率
        logits = similarities / self.temperature
        probs = F.softmax(logits, dim=1)

        return probs, similarities

    def update_prototypes_batch(self, features, labels, batch_size=512):
        """分批更新类原型"""
        unique_labels = torch.unique(labels)

        for c in unique_labels:
            # 分批处理属于类别c的样本
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                class_mean = torch.zeros(self.feature_dim, device=features.device)

                # 分批计算均值
                for i in range(0, class_features.size(0), batch_size):
                    batch = class_features[i:i + batch_size]
                    class_mean += batch.sum(dim=0)

                class_mean = class_mean / mask.sum()

                # 平滑更新原型
                alpha = 0.1  # 学习率
                self.prototypes.data[c] = (1 - alpha) * self.prototypes.data[c] + alpha * class_mean


# 内存优化的EntityGradNet模型
class MemoryEfficientEntityGradNet(nn.Module):
    def __init__(self, hetero_graph, feature_dims, hidden_dim=24, out_dim=12, num_classes=9, category_dim=9):
        super().__init__()
        self.g = hetero_graph
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim  # 进一步减小维度
        self.out_dim = out_dim  # 进一步减小维度
        self.num_classes = num_classes

        # 超内存优化的异构图神经网络
        self.hetgnn = MemoryEfficientHetGNN(feature_dims, hidden_dim, out_dim, hetero_graph.etypes)

        # 超内存优化的子图构建器
        self.subgraph_builder = BatchSubgraphBuilder(hetero_graph)

        # 特征重要性分析（简化）
        self.feature_importance = FeatureImportance(out_dim, category_dim)

        # 原型网络（简化）
        self.prototype_net = MemoryEfficientPrototypeNetwork(out_dim, num_classes)

        # 结构贡献权重
        self.structure_weights = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, node_features, batch_nodes=None):
        # 通过异构图神经网络处理
        node_embeddings = self.hetgnn(self.g, node_features, batch_nodes)

        return node_embeddings

    def classify_batch(self, node_embeddings, node_ids, batch_size=8, update_prototypes=False,
                       labels=None):  # 减小batch_size
        """批量分类"""
        if not node_ids:
            return torch.tensor([], device=node_embeddings['entity'].device), \
                torch.tensor([], device=node_embeddings['entity'].device, dtype=torch.long), \
                {}

        subgraph_embeddings = []
        components_list = []

        # 分批构建子图嵌入
        for i in range(0, len(node_ids), batch_size):
            batch_nodes = node_ids[i:i + batch_size]

            weights = {
                'center': self.structure_weights[0].item(),
                'relations': self.structure_weights[1].item()
            }

            batch_embeddings, batch_components = self.subgraph_builder.build_subgraph_embeddings_batch(
                batch_nodes, node_embeddings, weights, batch_size=batch_size
            )

            if batch_embeddings is not None and len(batch_embeddings) > 0:
                subgraph_embeddings.append(batch_embeddings)
            components_list.extend(batch_components)

            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 合并所有批次的嵌入
        if subgraph_embeddings:
            subgraph_embeddings = torch.cat(subgraph_embeddings, dim=0)
        else:
            subgraph_embeddings = torch.tensor([], device=node_embeddings['entity'].device)

        if subgraph_embeddings.size(0) == 0:
            return torch.tensor([], device=node_embeddings['entity'].device), \
                torch.tensor([], device=node_embeddings['entity'].device, dtype=torch.long), \
                {}

        # 计算特征重要性
        category_imp, other_imp, feature_scores = self.feature_importance(subgraph_embeddings)

        # 通过原型网络进行分类
        probs, similarities = self.prototype_net(subgraph_embeddings, batch_size=batch_size)

        # 如果处于训练模式且提供了标签，更新原型
        if update_prototypes and labels is not None and len(labels) > 0:
            self.prototype_net.update_prototypes_batch(subgraph_embeddings, labels, batch_size=batch_size)

        # 预测类别
        if probs.size(0) > 0:
            _, predicted_classes = torch.max(probs, dim=1)
        else:
            predicted_classes = torch.tensor([], device=probs.device, dtype=torch.long)

        # 为可解释性分析准备数据
        explanations = {
            'similarities': similarities,
            'category_importance': category_imp,
            'other_importance': other_imp,
            'feature_scores': feature_scores,
            'components': components_list,
            'structure_weights': self.structure_weights
        }

        return probs, predicted_classes, explanations


# 可解释性分析与可视化函数
class MemoryEfficientExplainabilityAnalyzer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def analyze_prediction_batch(self, node_ids, node_embeddings, true_labels=None, batch_size=32):
        """批量分析预测和解释"""
        results = []

        # 分批处理
        for i in range(0, len(node_ids), batch_size):
            batch_nodes = node_ids[i:i + batch_size]

            # 获取预测和解释
            probs, pred_classes, explanations = self.model.classify_batch(
                node_embeddings, batch_nodes, batch_size=batch_size, update_prototypes=False
            )

            if probs.size(0) == 0:
                continue

            # 处理每个节点的结果
            for j, node_id in enumerate(batch_nodes):
                if j >= probs.size(0):
                    continue

                # 提取解释数据
                similarities = explanations['similarities'][j] if j < explanations['similarities'].size(0) else None
                category_imp = explanations['category_importance'].item()
                other_imp = explanations['other_importance'].item()
                components = explanations['components'][j] if j < len(explanations['components']) else {}

                # 计算结构贡献
                structure_contributions = {}
                if components:
                    # 确保张量在CPU上计算范数
                    center_component = components.get('center')
                    relations_component = components.get('relations')

                    if center_component is not None:
                        if center_component.device.type == 'cuda':
                            center_component = center_component.cpu()
                        structure_contributions['center'] = torch.norm(center_component).item()

                    if relations_component is not None:
                        if relations_component.device.type == 'cuda':
                            relations_component = relations_component.cpu()
                        structure_contributions['relations'] = torch.norm(relations_component).item()

                    # 归一化结构贡献
                    total_contrib = sum(structure_contributions.values())
                    if total_contrib > 0:
                        structure_contributions = {k: v / total_contrib for k, v in structure_contributions.items()}

                # 格式化相似度
                similarity_dict = {}
                if similarities is not None:
                    # 确保相似度在CPU上
                    if similarities.device.type == 'cuda':
                        similarities_cpu = similarities.cpu()
                    else:
                        similarities_cpu = similarities

                    similarity_dict = {self.class_names[i]: similarities_cpu[i].item()
                                       for i in range(min(len(self.class_names), similarities_cpu.size(0)))}

                # 构建解释结果
                result = {
                    'node_id': node_id,
                    'prediction': {
                        'class': self.class_names[pred_classes[j].item()] if j < pred_classes.size(0) else None,
                        'probability': probs[j, pred_classes[j]].item() if j < pred_classes.size(0) and pred_classes[
                            j] < probs.size(1) else None
                    },
                    'true_label': self.class_names[true_labels[j]] if true_labels is not None and j < len(
                        true_labels) else None,
                    'prototype_similarities': similarity_dict,
                    'feature_importance': {
                        'category_scores': category_imp,
                        'other_features': other_imp
                    },
                    'structure_contributions': structure_contributions
                }

                results.append(result)

            # 清理内存
            del probs, pred_classes, explanations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


# 分批训练和评估函数
# 修改 train_batch 函数中的打乱部分
# 修改 train_batch 函数中批量节点的处理
# 在 train_batch 和 evaluate_batch 函数中，确保图在正确的设备上
def train_batch(model, node_features, train_idx, val_idx, labels, num_epochs=20, lr=0.001,
                             weight_decay=1e-4, device=None, train_batch_size=16, eval_batch_size=32):
    """超内存优化的分批训练"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 检查CUDA状态
    check_cuda_status(device)

    # 确保模型在正确的设备上
    model = model.to(device)
    print(f"\n训练开始时模型设备: {next(model.parameters()).device}")

    # 确保图的设备与训练设备一致
    if hasattr(model, 'g') and model.g.device != device:
        print(f"图设备({model.g.device})与训练设备({device})不匹配，正在移动图...")
        model.g = model.g.to(device)

    # 将索引转换为numpy数组（CPU操作）
    train_idx_array = np.array(train_idx)
    val_idx_array = np.array(val_idx)

    # 确保所有节点特征在正确设备上
    for k in node_features:
        if node_features[k].device != device:
            print(f"将{k}特征移动到{device}")
            node_features[k] = node_features[k].to(device)
        # 确保特征dtype一致
        if node_features[k].dtype != torch.float32:
            node_features[k] = node_features[k].float()

    # 确保标签在正确设备上
    if labels.device != device:
        print(f"将标签移动到{device}")
        labels = labels.to(device)

    # 不使用混合精度训练，保持统一的float32
    use_amp = False  # 禁用混合精度

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 训练记录
    best_val_acc = 0
    best_epoch = 0
    train_losses = []
    val_accuracies = []

    # 计算总批次数
    total_batches = (len(train_idx_array) + train_batch_size - 1) // train_batch_size

    print(f"\n开始训练，总批次数: {total_batches}")

    for epoch in range(num_epochs):
        model.train()

        # 打乱训练数据
        np.random.shuffle(train_idx_array)
        epoch_loss = 0
        processed_samples = 0
        start_time = time.time()

        # 分批训练
        for batch_idx in range(0, len(train_idx_array), train_batch_size):
            batch_indices = train_idx_array[batch_idx:batch_idx + train_batch_size]
            batch_nodes = batch_indices.tolist()
            batch_labels = labels[batch_indices]

            if len(batch_nodes) == 0:
                continue

            current_batch = batch_idx // train_batch_size + 1

            try:
                # 不使用混合精度
                # 前向传播
                node_embeddings = model(node_features, batch_nodes)

                # 检查是否成功获取嵌入
                if 'entity' not in node_embeddings or node_embeddings['entity'].size(0) == 0:
                    continue

                # 计算分类结果
                probs, predicted, explanations = model.classify_batch(
                    node_embeddings, batch_nodes,
                    batch_size=train_batch_size,
                    update_prototypes=True,
                    labels=batch_labels
                )

                if probs.size(0) == 0 or len(batch_labels) == 0:
                    continue

                # 计算损失
                loss = F.cross_entropy(probs, batch_labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += loss.item() * len(batch_nodes)
                processed_samples += len(batch_nodes)

                # 清理中间变量释放内存
                del node_embeddings, probs, predicted, explanations, loss

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\r批次 {current_batch}/{total_batches} GPU内存不足，跳过...", end="")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # 显示进度
            progress_percent = (batch_idx + len(batch_nodes)) / len(train_idx_array) * 100
            print(
                f"\rEpoch {epoch + 1}/{num_epochs} - 进度: {progress_percent:.1f}% (批次 {current_batch}/{total_batches})",
                end="")

            # 每处理几个batch就清理一次内存
            if batch_idx % (train_batch_size * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算本epoch耗时
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / processed_samples if processed_samples > 0 else 0
        train_losses.append(avg_loss)

        print(f"\nEpoch {epoch + 1} 完成，耗时: {epoch_time:.1f}秒，平均损失: {avg_loss:.4f}")

        # 验证
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                val_start_time = time.time()

                # 计算验证总批次数
                val_total_batches = (len(val_idx_array) + eval_batch_size - 1) // eval_batch_size

                # 分批验证
                for val_batch_idx in range(0, len(val_idx_array), eval_batch_size):
                    batch_indices = val_idx_array[val_batch_idx:val_batch_idx + eval_batch_size]
                    batch_nodes = batch_indices.tolist()
                    batch_labels = labels[batch_indices]

                    if len(batch_nodes) == 0:
                        continue

                    val_current_batch = val_batch_idx // eval_batch_size + 1

                    try:
                        # 前向传播
                        node_embeddings = model(node_features, batch_nodes)
                        val_probs, val_predicted, _ = model.classify_batch(
                            node_embeddings, batch_nodes,
                            batch_size=eval_batch_size,
                            update_prototypes=False
                        )

                        if val_predicted.size(0) > 0 and len(batch_labels) > 0:
                            val_correct += (val_predicted == batch_labels).sum().item()
                            val_total += len(batch_nodes)

                        # 清理内存
                        del node_embeddings, val_probs, val_predicted

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"\r验证批次 {val_current_batch}/{val_total_batches} GPU内存不足，跳过...", end="")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                    # 显示验证进度
                    val_progress_percent = (val_batch_idx + len(batch_nodes)) / len(val_idx_array) * 100
                    print(
                        f"\rEpoch {epoch + 1} - 验证进度: {val_progress_percent:.1f}% (批次 {val_current_batch}/{val_total_batches})",
                        end="")

                    # 每处理几个batch就清理一次内存
                    if val_batch_idx % (eval_batch_size * 10) == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                val_time = time.time() - val_start_time
                print(f"\n验证完成，耗时: {val_time:.1f}秒")

                # 计算验证准确率
                val_acc = val_correct / val_total if val_total > 0 else 0
                val_accuracies.append(val_acc)

                # 打印训练进度
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, 处理样本: {processed_samples}")

                # 检查是否是最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch

        # 更新学习率
        scheduler.step()

        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 早停判断
        if epoch - best_epoch > 10:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, train_losses, val_accuracies

def evaluate_batch(model, node_features, test_idx, true_labels, class_names, device=None, batch_size=128):
    """分批评估模型性能"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 将数据转移到设备
    for k in node_features:
        node_features[k] = node_features[k].to(device)
    true_labels = true_labels.to(device)

    # 存储预测结果
    all_predictions = []
    all_true_labels = []

    # 分批测试
    test_idx_array = np.array(test_idx)
    with torch.no_grad():
        for i in range(0, len(test_idx_array), batch_size):
            batch_indices = test_idx_array[i:i + batch_size]
            batch_nodes = batch_indices.tolist()
            batch_labels = true_labels[batch_indices]

            if len(batch_nodes) == 0:
                continue

            # 前向传播
            node_embeddings = model(node_features, batch_nodes)
            test_probs, test_predicted, _ = model.classify_batch(
                node_embeddings, batch_nodes,
                batch_size=batch_size,
                update_prototypes=False
            )

            if test_predicted.size(0) > 0:
                all_predictions.append(test_predicted.cpu())
                all_true_labels.append(batch_labels.cpu())

            # 清理内存
            del node_embeddings, test_probs, test_predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 合并结果
    if all_predictions:
        all_predictions = torch.cat(all_predictions)
        all_true_labels = torch.cat(all_true_labels)
    else:
        all_predictions = torch.tensor([])
        all_true_labels = torch.tensor([])

    # 计算评估指标
    if len(all_predictions) > 0:
        accuracy = accuracy_score(all_true_labels, all_predictions)
        macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
        micro_f1 = f1_score(all_true_labels, all_predictions, average='micro')
    else:
        accuracy = macro_f1 = micro_f1 = 0.0

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    # 使用可解释性分析器
    explainer = MemoryEfficientExplainabilityAnalyzer(model, class_names)

    # 只分析前几个样本以节省内存
    sample_size = min(10, len(test_idx_array))
    if sample_size > 0:
        sample_indices = test_idx_array[:sample_size]
        sample_true_labels = true_labels[sample_indices]
        # 确保标签在CPU上
        if sample_true_labels.device.type == 'cuda':
            sample_true_labels_cpu = sample_true_labels.cpu().numpy()
        else:
            sample_true_labels_cpu = sample_true_labels.numpy()

        explanations = explainer.analyze_prediction_batch(
            sample_indices.tolist(), node_features,
            sample_true_labels_cpu,
            batch_size=batch_size
        )
    else:
        explanations = []

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'explainer': explainer,
        'explanations': explanations
    }
# 数据加载和预处理函数（保持原有）
# 修改 load_and_preprocess_data 函数中的采样部分
def load_and_preprocess_data(triples_file, entity_types_file):
    """加载三元组数据和实体类型数据"""
    print("加载三元组数据...")
    triples = []
    entity_counts = defaultdict(int)

    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append(parts)
                    entity_counts[parts[0]] += 1
                    entity_counts[parts[2]] += 1

    print(f"加载了 {len(triples)} 个三元组")
    print(f"涉及 {len(entity_counts)} 个实体")

    # 加载实体类型数据
    print("加载实体类型数据...")
    entity_types = pd.read_csv(entity_types_file)
    print(f"加载了 {len(entity_types)} 个实体类型数据")

    # 创建实体ID映射（只包含在triples中的实体和有类型的实体）
    all_entities = set(entity_counts.keys())
    all_relations = set()

    for h, r, t in triples:
        all_relations.add(r)

    # 添加有类型的实体
    for entity_id in entity_types['entity_id']:
        all_entities.add(entity_id)

    # 创建ID映射
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(all_relations)}

    print(f"总实体数: {len(entity_to_idx)}")
    print(f"总关系数: {len(relation_to_idx)}")

    # 采样部分数据以减少内存使用
    max_triples = 500000  # 限制三元组数量
    if len(triples) > max_triples:
        print(f"采样 {max_triples} 个三元组以减少内存使用")
        np.random.seed(42)
        # 使用索引进行采样，而不是直接采样三元组
        sampled_indices = np.random.choice(len(triples), max_triples, replace=False)
        sampled_triples = [triples[i] for i in sampled_indices]
        triples = sampled_triples

    return triples, entity_types, entity_to_idx, relation_to_idx


# 修改 build_heterogeneous_graph 函数，让图在正确的设备上

def build_heterogeneous_graph(triples, entity_to_idx, relation_to_idx, device):
    """构建异构图（内存优化版本）"""
    print("构建异构图...")

    # 为每种关系创建边列表
    edge_dict = {}

    for h, r, t in triples:
        if h not in entity_to_idx or t not in entity_to_idx:
            continue

        h_idx = entity_to_idx[h]
        t_idx = entity_to_idx[t]
        r_idx = relation_to_idx[r]

        # 正向关系
        rel_name = f"rel_{r_idx}"
        if rel_name not in edge_dict:
            edge_dict[rel_name] = ([], [])
        edge_dict[rel_name][0].append(h_idx)
        edge_dict[rel_name][1].append(t_idx)

    print(f"构建了 {len(edge_dict)} 种关系")

    # 合并所有边
    all_edges = {}
    for rel_name, (src, dst) in edge_dict.items():
        if len(src) > 0 and len(dst) > 0:
            all_edges[('entity', rel_name, 'entity')] = (
                torch.tensor(src, dtype=torch.int64),
                torch.tensor(dst, dtype=torch.int64)
            )

    # 构建异构图
    if all_edges:
        g = dgl.heterograph(all_edges)
        print(f"图构建完成，节点数: {g.num_nodes('entity')}, 边数: {g.num_edges()}")

        # 将图移动到指定设备
        try:
            g = g.to(device)
            print(f"成功将图移动到设备: {device}")
        except Exception as e:
            print(f"将图移动到设备时出错: {e}")
            print("在CPU上保留图")
    else:
        print("错误：没有有效的边构建图")
        raise ValueError("无法构建异构图：没有有效的边")

    return g


def create_node_features(entity_types, entity_to_idx, feature_dim=64):
    """创建节点特征（内存优化）"""
    print("创建节点特征...")

    # 初始化特征矩阵
    num_entities = len(entity_to_idx)

    # 使用更小的特征维度
    if feature_dim > 128:
        feature_dim = 128
    print(f"使用特征维度: {feature_dim}")

    node_features = np.zeros((num_entities, feature_dim), dtype=np.float32)  # 使用float32

    # 处理有类型信息的实体
    processed_count = 0
    for _, row in entity_types.iterrows():
        entity_id = row['entity_id']
        if entity_id in entity_to_idx:
            idx = entity_to_idx[entity_id]

            # 使用类别得分作为特征
            category_scores = []
            for i in range(1, 10):  # category_1_score 到 category_9_score
                score_col = f'category_{i}_score'
                if score_col in row:
                    score = row[score_col]
                    if pd.notna(score):
                        category_scores.append(float(score))
                    else:
                        category_scores.append(0.0)

            # 填充特征
            if len(category_scores) == 9:
                # 只使用前9维存储类别得分
                node_features[idx, :min(9, feature_dim)] = category_scores[:min(9, feature_dim)]

            processed_count += 1

            # 每1000个实体打印一次进度
            if processed_count % 1000 == 0:
                print(f"已处理 {processed_count} 个实体")

    print(f"总共处理了 {processed_count} 个有类型信息的实体")

    # 标准化特征
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    return torch.tensor(node_features, dtype=torch.float32)  # 确保是float32

def create_labels(entity_types, entity_to_idx, num_classes=9):
    """创建标签"""
    print("创建标签...")

    # 初始化标签数组（-1表示没有标签）
    num_entities = len(entity_to_idx)
    labels = np.full(num_entities, -1, dtype=np.int64)

    # 设置有类型实体的标签
    labeled_count = 0
    for _, row in entity_types.iterrows():
        entity_id = row['entity_id']
        if entity_id in entity_to_idx:
            idx = entity_to_idx[entity_id]
            # 使用predicted_category作为标签
            if 'predicted_category' in row and pd.notna(row['predicted_category']):
                try:
                    label = int(row['predicted_category'])
                    # 确保类别在0-8范围内
                    if 0 <= label < num_classes:
                        labels[idx] = label
                        labeled_count += 1
                except:
                    continue

    print(f"有标签的实体数: {labeled_count}")

    return torch.tensor(labels, dtype=torch.long)


# 运行EntityGradNet模型的主函数
def run_memory_efficient_entitygradnet(gpu_id=0, data_path='data/FB15KET'):
    # 打印CUDA是否可用以及可用GPU数量
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 设置随机种子确保结果可复现
    set_seed(42)

    # 设置设备
    if torch.cuda.is_available() and gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            print(f"指定的GPU ID {gpu_id} 超出范围，使用GPU 0")
            gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        # 清空GPU缓存
        torch.cuda.empty_cache()
        # 设置当前设备
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
        print("警告: CUDA不可用或指定使用CPU，使用CPU")

    # 设置环境变量以指定使用的GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 检查张量是否正确放置在GPU上的测试
    test_tensor = torch.FloatTensor([1.0, 2.0, 3.0]).to(device)
    print(f"测试张量设备: {test_tensor.device}")

    # 1. 加载和预处理数据
    triples_file = os.path.join(data_path, 'xunlian.txt')
    entity_types_file = os.path.join(data_path, 'Entity_All_typed.csv')

    triples, entity_types, entity_to_idx, relation_to_idx = load_and_preprocess_data(
        triples_file, entity_types_file
    )

    # 2. 构建异构图（在指定设备上）
    g = build_heterogeneous_graph(triples, entity_to_idx, relation_to_idx, device)

    # 3. 创建节点特征和标签
    node_features_tensor = create_node_features(entity_types, entity_to_idx, feature_dim=64)
    labels_tensor = create_labels(entity_types, entity_to_idx, num_classes=9)

    # 设置节点特征前，确保图在正确的设备上
    if g.device != device:
        print(f"图设备({g.device})与目标设备({device})不匹配，正在移动图...")
        g = g.to(device)

    # 将特征和标签移动到设备
    node_features_tensor = node_features_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    # 设置节点特征
    g.nodes['entity'].data['entity_feature'] = node_features_tensor
    g.nodes['entity'].data['label'] = labels_tensor

    print(f"图设备: {g.device}")
    print(f"特征设备: {node_features_tensor.device}")
    print(f"标签设备: {labels_tensor.device}")

    # 4. 划分数据集 (只使用有标签的实体)
    # 确保标签在CPU上再进行numpy操作
    if labels_tensor.device.type == 'cuda':
        labels_np = labels_tensor.cpu().numpy()
    else:
        labels_np = labels_tensor.numpy()

    labeled_indices = np.where(labels_np != -1)[0]
    print(f"有标签的实体索引数: {len(labeled_indices)}")

    if len(labeled_indices) == 0:
        raise ValueError("没有找到有标签的实体，无法进行训练")

    # 随机打乱
    np.random.seed(42)
    np.random.shuffle(labeled_indices)

    # 划分训练、验证、测试集
    total_labeled = len(labeled_indices)
    test_size = min(int(total_labeled * 0.2), 1000)  # 限制测试集大小
    val_size = min(int((total_labeled - test_size) * 0.125), 500)  # 限制验证集大小

    test_idx = labeled_indices[:test_size].tolist()
    val_idx = labeled_indices[test_size:test_size + val_size].tolist()
    train_idx = labeled_indices[test_size + val_size:].tolist()

    # 限制训练集大小
    max_train_size = 5000
    if len(train_idx) > max_train_size:
        train_idx = train_idx[:max_train_size]

    print(f"训练集大小: {len(train_idx)}")
    print(f"验证集大小: {len(val_idx)}")
    print(f"测试集大小: {len(test_idx)}")

    # 5. 创建超内存优化的EntityGradNet模型
    feature_dim = node_features_tensor.shape[1]
    hidden_dim = 16  # 使用更小的维度
    out_dim = 8  # 使用更小的维度
    num_classes = 9
    category_dim = min(9, feature_dim)  # 类别特征维度

    # 初始化超内存优化模型
    model = MemoryEfficientEntityGradNet(g, feature_dim, hidden_dim, out_dim, num_classes, category_dim)
    model = model.to(device)

    # 确认模型是否在GPU上
    model_device = next(model.parameters()).device
    print(f"模型初始化后设备: {model_device}")

    # 准备节点特征（分批加载到GPU）
    node_features = {
        'entity': node_features_tensor
    }

    # 分批将特征移动到GPU
    print("分批将特征移动到GPU...")
    for k in node_features:
        if node_features[k].device != device:
            node_features[k] = node_features[k].to(device)

    # 确保标签在正确的设备上
    if labels_tensor.device != device:
        labels_tensor = labels_tensor.to(device)

    # 更新图中的标签
    g.nodes['entity'].data['label'] = labels_tensor

    # 6. 训练模型（分批训练）
    # 6. 训练模型（超内存优化训练）
    print(f"\n开始超内存优化训练，使用设备: {device}")
    trained_model, train_losses, val_accuracies = train_batch(
        model, node_features, train_idx, val_idx, g.nodes['entity'].data['label'],
        num_epochs=15, lr=0.005, weight_decay=1e-4, device=device,
        train_batch_size=64, eval_batch_size=128  # 使用更小的batch
    )

    # 7. 评估模型（分批评估）
    class_names = [
        '类别1', '类别2', '类别3', '类别4', '类别5',
        '类别6', '类别7', '类别8', '类别9'
    ]

    results = evaluate_batch(
        trained_model, node_features, test_idx, g.nodes['entity'].data['label'],
        class_names, device=device, batch_size=64
    )

    # 8. 保存模型
    torch.save(trained_model.state_dict(), 'entitygradnet_model_memory_efficient.pth')

    # 9. 示例解释
    print("\n=== 示例解释 ===")
    if len(test_idx) > 0:
        sample_size = min(3, len(test_idx))
        sample_indices = test_idx[:sample_size]

        for i, idx in enumerate(sample_indices):
            if i >= len(results['explanations']):
                break

            explanation = results['explanations'][i]
            if explanation and explanation['prediction']['class']:
                print(f"\n示例 {i + 1}:")
                print(f"实体ID: {explanation['node_id']}")
                print(
                    f"预测类别: {explanation['prediction']['class']} (概率: {explanation['prediction']['probability']:.4f})")
                if explanation['true_label']:
                    print(f"真实类别: {explanation['true_label']}")

    return trained_model, results, g, node_features


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行内存优化的EntityGradNet模型')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID（0,1,2,...）')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--data', type=str, default='data/FB15KET', help='数据目录路径')
    parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    args = parser.parse_args()

    if args.cpu:
        print("强制使用CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_id = None
    elif args.gpu is not None:
        print(f"使用指定的GPU: {args.gpu}")
        gpu_id = args.gpu
    else:
        # 默认使用GPU 0
        gpu_id = 0

    # 检查CUDA是否可用
    if not torch.cuda.is_available() and gpu_id is not None:
        print("警告: CUDA不可用，强制使用CPU")
        gpu_id = None

    # 运行内存优化的模型训练
    try:
        model, results, g, node_features = run_memory_efficient_entitygradnet(
            gpu_id=gpu_id,
            data_path=args.data
        )

        # 打印最终设备信息
        print(f"\n=== 训练完成后设备信息 ===")
        if torch.cuda.is_available():
            print(f"训练使用的设备: {next(model.parameters()).device}")
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback

        traceback.print_exc()