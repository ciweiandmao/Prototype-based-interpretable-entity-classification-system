import pandas as pd
import numpy as np
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import os
import subprocess
import sys
import argparse


class FB15KETDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_type_path = os.path.join(data_dir, 'Entity_All_typed.csv')
        self.train_path = os.path.join(data_dir, 'train.txt')
        self.valid_path = os.path.join(data_dir, 'valid.txt')
        self.test_path = os.path.join(data_dir, 'test.txt')

        # 9个类别名称映射
        self.category_names = {
            1: "人物和生命（Person & Life）",
            2: "组织与机构（Organization）",
            3: "地点与地理（Location）",
            4: "创作与娱乐作品（Creative Work）",
            5: "事件与活动（Event）",
            6: "学科与概念（Concept & Subject）",
            7: "物品与产品（Product & Object）",
            8: "属性与度量（Attribute & Measurement）",
            9: "其他（Others）"
        }

    def load_entity_types(self):
        """加载实体类型信息"""
        df = pd.read_csv(self.entity_type_path)
        # 选择类别得分最高的作为标签
        score_cols = [f'category_{i}_score' for i in range(1, 10)]
        df['predicted_category'] = df[score_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)

        # 构建实体ID到类别标签的映射
        entity_to_label = dict(zip(df['entity_id'], df['predicted_category']))

        # 获取类别得分作为特征
        entity_to_scores = {}
        for _, row in df.iterrows():
            scores = [row[f'category_{i}_score'] for i in range(1, 10)]
            entity_to_scores[row['entity_id']] = scores

        return entity_to_label, entity_to_scores

    def load_triplets(self):
        """加载所有三元组（训练+验证+测试）"""
        triplets = []

        for file_path in [self.train_path, self.valid_path, self.test_path]:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = line.strip().split('\t')
                    triplets.append((h, r, t))

        return triplets

    def build_heterogeneous_graph(self):
        """构建异构图"""
        # 1. 加载数据
        entity_to_label, entity_to_scores = self.load_entity_types()
        triplets = self.load_triplets()

        # 2. 创建ID映射
        entity_ids = set()
        relation_ids = set()

        for h, r, t in triplets:
            entity_ids.add(h)
            entity_ids.add(t)
            relation_ids.add(r)

        entity_id_map = {eid: i for i, eid in enumerate(sorted(entity_ids))}
        relation_id_map = {rid: i for i, rid in enumerate(sorted(relation_ids))}

        # 3. 构建图数据
        src_nodes, dst_nodes, rel_ids = [], [], []
        for h, r, t in triplets:
            src_nodes.append(entity_id_map[h])
            dst_nodes.append(entity_id_map[t])
            rel_ids.append(relation_id_map[r])

        # 4. 创建异构图（这里简化为同构图，因为只有一种节点类型）
        # 但实际上关系是异构的，我们可以为每种关系类型创建不同的边类型
        g = dgl.heterograph({
            ('entity', 'relation', 'entity'): (torch.tensor(src_nodes), torch.tensor(dst_nodes))
        })

        # 5. 创建节点特征
        num_entities = len(entity_ids)
        node_features = np.zeros((num_entities, 9))  # 9维类别得分

        for eid, idx in entity_id_map.items():
            if eid in entity_to_scores:
                node_features[idx] = entity_to_scores[eid]

        # 6. 创建节点标签
        node_labels = np.zeros(num_entities, dtype=int) - 1  # -1表示未知标签
        for eid, idx in entity_id_map.items():
            if eid in entity_to_label:
                node_labels[idx] = entity_to_label[eid] - 1  # 转换为0-8索引

        return g, node_features, node_labels, entity_id_map, relation_id_map


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

        # BiLSTM处理学生节点表示
        if 'stu' in h:
            stu_feats = h['stu']
            # 增加batch维度
            stu_feats = stu_feats.unsqueeze(1)
            # 通过BiLSTM
            stu_feats, _ = self.bilstm(stu_feats)
            # 移除batch维度
            stu_feats = stu_feats.squeeze(1)
            h['stu'] = stu_feats

        return h


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


class SubgraphBuilder:
    def __init__(self, hetero_graph):
        self.g = hetero_graph

    def get_two_hop_neighbors(self, node_id):
        """获取节点的二阶邻居，返回子图"""
        # 获取一阶邻居 (宿舍和选修课)
        dorm_neighbors = self.g.successors(node_id, etype='live')
        course_neighbors = self.g.successors(node_id, etype='choose')

        # 从一阶邻居获取二阶邻居 (学生)
        student_neighbors = []
        for dorm_id in dorm_neighbors:
            neighbors = self.g.successors(dorm_id, etype='lived-by')
            student_neighbors.extend(neighbors.tolist())

        for course_id in course_neighbors:
            neighbors = self.g.successors(course_id, etype='choosed-by')
            student_neighbors.extend(neighbors.tolist())

        # 移除自身ID
        if node_id in student_neighbors:
            student_neighbors.remove(node_id)

        # 去重
        student_neighbors = list(set(student_neighbors))

        # 构建子图
        nodes = {
            'stu': torch.tensor([node_id] + student_neighbors),
            'dorm': dorm_neighbors,
            'course': course_neighbors
        }

        return nodes

    def build_subgraph_embedding(self, node_id, node_feats, embedding_weights=None):
        """构建节点的子图嵌入"""
        if embedding_weights is None:
            # 默认权重分配
            embedding_weights = {
                'center': 0.4,  # 中心学生节点
                'dorm': 0.2,  # 宿舍关系
                'course': 0.2,  # 课程关系
                'peers': 0.2  # 同学关系
            }

        # 获取二阶邻居节点
        neighbor_nodes = self.get_two_hop_neighbors(node_id)

        # 构建子图嵌入
        center_emb = node_feats['stu'][node_id] * embedding_weights['center']

        # 宿舍邻居嵌入
        if len(neighbor_nodes['dorm']) > 0:
            dorm_emb = torch.mean(node_feats['dorm'][neighbor_nodes['dorm']], dim=0) * embedding_weights['dorm']
        else:
            dorm_emb = torch.zeros_like(center_emb)

        # 课程邻居嵌入
        if len(neighbor_nodes['course']) > 0:
            course_emb = torch.mean(node_feats['course'][neighbor_nodes['course']], dim=0) * embedding_weights['course']
        else:
            course_emb = torch.zeros_like(center_emb)

        # 同学邻居嵌入
        if len(neighbor_nodes['stu']) > 1:  # > 1 因为我们排除了中心节点自身
            peers_emb = torch.mean(node_feats['stu'][neighbor_nodes['stu'][1:]], dim=0) * embedding_weights['peers']
        else:
            peers_emb = torch.zeros_like(center_emb)

        # 合并子图嵌入
        subgraph_emb = center_emb + dorm_emb + course_emb + peers_emb

        # 保存组成部分，用于可解释性分析
        components = {
            'center': center_emb / embedding_weights['center'] if embedding_weights['center'] > 0 else center_emb,
            'dorm': dorm_emb / embedding_weights['dorm'] if embedding_weights['dorm'] > 0 else dorm_emb,
            'course': course_emb / embedding_weights['course'] if embedding_weights['course'] > 0 else course_emb,
            'peers': peers_emb / embedding_weights['peers'] if embedding_weights['peers'] > 0 else peers_emb
        }

        return subgraph_emb, components


class FeatureImportance(nn.Module):
    def __init__(self, feature_dim, grade_dim=30):
        super().__init__()
        self.feature_dim = feature_dim
        self.grade_dim = grade_dim
        self.text_dim = feature_dim - grade_dim

        # 特征重要性权重矩阵
        self.W_imp = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, node_features):
        # 计算特征重要性分数
        imp_scores = torch.sigmoid(node_features @ self.W_imp)

        # 分离成绩特征和文本特征的重要性
        grade_imp = imp_scores[:, :self.grade_dim].mean()
        text_imp = imp_scores[:, self.grade_dim:].mean()

        # 归一化，确保总和为1
        total_imp = grade_imp + text_imp
        grade_imp = grade_imp / total_imp
        text_imp = text_imp / total_imp

        return grade_imp, text_imp, imp_scores
class XGradNet(nn.Module):
    def __init__(self, hetero_graph, feature_dims, hidden_dim=128, out_dim=64, num_classes=6, grade_dim=30):
        super().__init__()
        self.g = hetero_graph
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        # 异构图神经网络
        self.hetgnn = HetGNN(feature_dims, hidden_dim, out_dim, hetero_graph.etypes)

        # 子图构建器
        self.subgraph_builder = SubgraphBuilder(hetero_graph)

        # 特征重要性分析
        self.feature_importance = FeatureImportance(out_dim, grade_dim)

        # 原型网络
        self.prototype_net = PrototypeNetwork(out_dim, num_classes)

        # 结构贡献权重 - 提高center权重到0.82，其他权重按比例调整
        self.structure_weights = nn.Parameter(torch.tensor([0.82, 0.06, 0.06, 0.06]))

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
                'dorm': self.structure_weights[1].item(),
                'course': self.structure_weights[2].item(),
                'peers': self.structure_weights[3].item()
            }
            subgraph_emb, components = self.subgraph_builder.build_subgraph_embedding(
                node_id, node_embeddings, weights
            )
            subgraph_embeddings.append(subgraph_emb)
            components_list.append(components)

        # 将列表转换为张量
        subgraph_embeddings = torch.stack(subgraph_embeddings)

        # 计算特征重要性
        grade_imp, text_imp, feature_scores = self.feature_importance(subgraph_embeddings)

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
            'grade_importance': grade_imp,
            'text_importance': text_imp,
            'feature_scores': feature_scores,
            'components': components_list,
            'structure_weights': self.structure_weights
        }

        return probs, predicted_classes, explanations


class FB15KETXGradNet(XGradNet):
    """针对FB15KET数据集适配的XGradNet"""

    def __init__(self, hetero_graph, feature_dims, hidden_dim=128, out_dim=64, num_classes=9, device=None):
        super().__init__(hetero_graph, feature_dims, hidden_dim, out_dim, num_classes)

        # FB15KET中关系类型更多样，需要更复杂的处理
        self.relation_embeddings = nn.Embedding(
            len(hetero_graph.etypes), hidden_dim
        )

        # 修改子图构建器，考虑不同关系类型
        self.subgraph_builder = FB15KETSubgraphBuilder(hetero_graph)

    def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
        """重写分类方法，适应FB15KET数据特点"""
        subgraph_embeddings = []
        components_list = []

        for node_id in node_ids:
            # 获取节点的关系感知子图
            subgraph_emb, components = self.subgraph_builder.build_relation_aware_subgraph(
                node_id, node_embeddings, self.relation_embeddings
            )
            subgraph_embeddings.append(subgraph_emb)
            components_list.append(components)

        subgraph_embeddings = torch.stack(subgraph_embeddings)

        # 原型网络分类
        probs, similarities = self.prototype_net(subgraph_embeddings)

        if update_prototypes and labels is not None:
            self.prototype_net.update_prototypes(subgraph_embeddings, labels)

        _, predicted_classes = torch.max(probs, dim=1)

        explanations = {
            'similarities': similarities,
            'components': components_list,
            'relation_weights': self.relation_embeddings.weight
        }

        return probs, predicted_classes, explanations


class FB15KETSubgraphBuilder:
    """针对FB15KET的子图构建器"""

    def __init__(self, hetero_graph):
        self.g = hetero_graph

    def build_relation_aware_subgraph(self, node_id, node_feats, relation_embeddings):
        """构建关系感知的子图嵌入"""
        # 获取节点的所有邻居和关系
        src, dst, eids = self.g.out_edges(node_id, form='all')

        if len(src) == 0:
            # 无邻居，返回节点自身特征
            return node_feats[node_id], {'self': 1.0}

        # 收集邻居特征，按关系类型加权
        neighbor_embs = []
        relation_weights = []

        for i, (s, d, eid) in enumerate(zip(src, dst, eids)):
            # 获取关系类型
            rel_type = self.g.edata['etype'][eid] if 'etype' in self.g.edata else 0
            rel_emb = relation_embeddings(torch.tensor([rel_type]))

            # 邻居特征与关系嵌入结合
            neighbor_feat = node_feats[d]
            combined = torch.cat([neighbor_feat, rel_emb.squeeze(0)], dim=0)
            neighbor_embs.append(combined)

            # 关系权重（可学习或基于频率）
            relation_weights.append(1.0)

        # 加权聚合
        weights = torch.softmax(torch.tensor(relation_weights), dim=0)
        aggregated = sum(w * emb for w, emb in zip(weights, neighbor_embs))

        # 与自身特征结合
        self_weight = 0.6  # 自身权重较高
        neighbor_weight = 0.4

        final_emb = self_weight * node_feats[node_id] + neighbor_weight * aggregated

        # 构建解释信息
        components = {
            'self_contribution': self_weight,
            'neighbor_contribution': neighbor_weight,
            'relation_distribution': weights.tolist()
        }

        return final_emb, components