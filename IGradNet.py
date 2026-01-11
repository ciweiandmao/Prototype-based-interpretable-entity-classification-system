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

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义异构图神经网络 (HetGNN)
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
            hidden_size=out_feats//2,
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

# 二阶邻居子图构建模块
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
                'dorm': 0.2,    # 宿舍关系
                'course': 0.2,  # 课程关系
                'peers': 0.2    # 同学关系
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

# 特征重要性分析模块
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

# 原型网络分类模块
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
        similarities = torch.log((distances**2 + 1) / (distances**2 + 1e-4))
        
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

# XGradNet模型整合所有组件
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

# 可解释性分析与可视化函数
class ExplainabilityAnalyzer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def analyze_prediction(self, node_id, node_embeddings, true_label=None):
        """分析单个节点的预测和解释"""
        # 获取预测和解释
        probs, pred_class, explanations = self.model.classify(
            node_embeddings, [node_id], update_prototypes=False
        )
        
        # 提取解释数据
        similarities = explanations['similarities'][0]  # 第一个（也是唯一一个）节点的相似度
        grade_imp = explanations['grade_importance'].item()
        text_imp = explanations['text_importance'].item()
        components = explanations['components'][0]
        
        # 计算结构贡献
        structure_contributions = {
            'center': torch.norm(components['center']).item(),
            'dorm': torch.norm(components['dorm']).item(),
            'course': torch.norm(components['course']).item(),
            'peers': torch.norm(components['peers']).item()
        }
        
        # 归一化结构贡献
        total_contrib = sum(structure_contributions.values())
        if total_contrib > 0:
            structure_contributions = {k: v/total_contrib for k, v in structure_contributions.items()}
        
        # 格式化相似度
        similarity_dict = {self.class_names[i]: similarities[i].item() for i in range(len(self.class_names))}
        
        # 构建解释结果
        result = {
            'prediction': {
                'class': self.class_names[pred_class.item()],
                'probability': probs[0, pred_class.item()].item()
            },
            'true_label': self.class_names[true_label] if true_label is not None else None,
            'prototype_similarities': similarity_dict,
            'feature_importance': {
                'academic_scores': grade_imp,
                'text_features': text_imp
            },
            'structure_contributions': structure_contributions
        }
        
        return result
    
    def visualize_similarity_heatmap(self, similarities, student_ids=None):
        """将原型相似度可视化为热力图"""
        if student_ids is None:
            student_ids = [f"S{i+1}" for i in range(similarities.shape[0])]
        
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(similarities, annot=True, fmt=".2f", cmap="YlGnBu",
        #            xticklabels=self.class_names, 
        #            yticklabels=student_ids)
        # plt.title("原型相似性热力图")
        # plt.xlabel("毕业去向类别")
        # plt.ylabel("学生ID")
        # plt.tight_layout()
        # plt.savefig("prototype_similarity_heatmap.png")
        # plt.close()

    def visualize_feature_importance(self, feature_importance_list, student_ids=None):
        """可视化特征重要性"""
        if student_ids is None:
            student_ids = [f"S{i+1}" for i in range(len(feature_importance_list))]
        
        # 提取学业成绩和文本特征的重要性
        academic_scores = [imp['academic_scores'] for imp in feature_importance_list]
        text_features = [imp['text_features'] for imp in feature_importance_list]
        
        data = pd.DataFrame({
            '学生ID': student_ids,
            '学业成绩': academic_scores,
            '文本特征': text_features
        })
        
        # plt.figure(figsize=(12, 8))
        # data_melted = pd.melt(data, id_vars=['学生ID'], var_name='特征类型', value_name='重要性得分')
        # chart = sns.barplot(x='学生ID', y='重要性得分', hue='特征类型', data=data_melted)
        # chart.set_title("特征重要性分析")
        # chart.set_xlabel("学生ID")
        # chart.set_ylabel("重要性得分")
        # plt.tight_layout()
        # plt.savefig("feature_importance.png")
        # plt.close()
    
    def visualize_structure_contributions(self, structure_contrib_list, student_ids=None):
        """可视化结构贡献热力图"""
        if student_ids is None:
            student_ids = [f"S{i+1}" for i in range(len(structure_contrib_list))]
        
        # 构建结构贡献矩阵
        contrib_data = []
        for contrib in structure_contrib_list:
            contrib_data.append([
                contrib['center'], 
                contrib['dorm'], 
                contrib['course'], 
                contrib['peers']
            ])
        
        contrib_matrix = np.array(contrib_data)
        
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(contrib_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
        #            xticklabels=['个人特征', '宿舍关系', '选课关系', '同学关系'], 
        #            yticklabels=student_ids)
        # plt.title("结构贡献热力图")
        # plt.xlabel("结构组成部分")
        # plt.ylabel("学生ID")
        # plt.tight_layout()
        # plt.savefig("structure_contributions_heatmap.png")
        # plt.close()
    
    def visualize_confusion_matrix(self, true_labels, predicted_labels):
        """可视化混淆矩阵"""
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        #            xticklabels=self.class_names, 
        #            yticklabels=self.class_names)
        # plt.title("混淆矩阵")
        # plt.xlabel("预测标签")
        # plt.ylabel("真实标签")
        # plt.tight_layout()
        # plt.savefig("confusion_matrix.png")
        # plt.close()

# 训练和评估函数
def train(model, node_features, train_idx, val_idx, labels, num_epochs=50, lr=0.001, weight_decay=1e-4, device=None):
    """训练XGradNet模型"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 再次确保模型在正确的设备上
    model = model.to(device)
    print(f"训练开始时模型设备: {next(model.parameters()).device}")
    
    # 将数据转移到设备
    train_idx_tensor = torch.tensor(train_idx).to(device)
    val_idx_tensor = torch.tensor(val_idx).to(device)
    
    # 确保所有节点特征在正确设备上
    for k in node_features:
        if node_features[k].device != device:
            print(f"将{k}特征移动到{device}")
            node_features[k] = node_features[k].to(device)
    
    # 确保标签在正确设备上
    if labels.device != device:
        print(f"将标签移动到{device}")
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
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")
                
                # 检查是否是最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    # 这里可以保存最佳模型
                    # torch.save(model.state_dict(), 'best_model.pth')
        
        # 早停判断
        if epoch - best_epoch > 30:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_accuracies

def evaluate(model, node_features, test_idx, true_labels, class_names, device=None):
    """评估模型性能并生成解释"""
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
    micro_f1 = f1_score(true_labels_cpu, test_predicted, average='micro')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    
    # 使用可解释性分析器进行分析和可视化
    explainer = ExplainabilityAnalyzer(model, class_names)
    
    # 可视化混淆矩阵
    explainer.visualize_confusion_matrix(true_labels_cpu, test_predicted)
    
    # 返回结果和解释
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'predictions': test_predicted,
        'true_labels': true_labels_cpu,
        'explainer': explainer,
        'explanations': explanations
    }

# 检查GPU使用情况并选择最合适的GPU
def get_free_gpu():
    try:
        # 获取nvidia-smi输出
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,memory.used,memory.total"])
        gpu_stats = gpu_stats.decode("utf-8").strip().split("\n")[1:]
        
        free_memory = []
        for line in gpu_stats:
            parts = line.split(",")
            idx = int(parts[0])
            used_memory = int(parts[1].strip().split()[0])
            total_memory = int(parts[2].strip().split()[0])
            free_percent = (total_memory - used_memory) / total_memory
            free_memory.append((idx, free_percent))
        
        # 按空闲百分比排序
        free_memory.sort(key=lambda x: x[1], reverse=True)
        
        print("GPU使用情况:")
        for idx, free in free_memory:
            print(f"GPU {idx}: {free*100:.2f}% 空闲")
        
        # 返回最空闲的GPU
        return free_memory[0][0]
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return 0

# 运行XGradNet模型的主函数
def run_xgradnet(gpu_id=0):
    # 打印CUDA是否可用以及可用GPU数量
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 设置随机种子确保结果可复现
    set_seed(42)
    
    # 强制使用CUDA
    if torch.cuda.is_available() and gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            print(f"指定的GPU ID {gpu_id} 超出范围，使用GPU 0")
            gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        # 清空GPU缓存
        torch.cuda.empty_cache()
        # 设置当前设备，确保所有新创建的张量都在这个设备上
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
    
    # 加载成绩特征和标签
    scores = pd.read_excel('data/improved_mock_scores.xlsx')
    # scores = pd.read_excel('data/student_scores_features.xlsx')
    scores_features = scores.values
    y = pd.read_excel('data/student_labels.xlsx')
    labels = np.squeeze(y.values)
    
    # 加载文本特征
    text_features_path = 'data/student_text_features3.xlsx'
    text_df = pd.read_excel(text_features_path)
    
    # 打印两个数据集的形状，检查差异
    print(f"成绩特征数量: {len(scores_features)}")
    print(f"文本特征数量: {len(text_df)}")
    
    # 创建学生ID索引
    student_ids = list(range(len(scores_features)))
    
    # 处理文本特征
    combined_texts = []
    for i in range(len(scores_features)):
        # 检查学生ID是否在文本数据中
        if i < len(text_df):
            row = text_df.iloc[i]
            text = ' '.join([str(row[col]) for col in text_df.columns if isinstance(row[col], str)])
        else:
            # 缺失的学生添加空文本
            text = ""
        combined_texts.append(text)
    
    # TF-IDF和PCA处理
    vectorizer = TfidfVectorizer(max_features=1000)
    text_features_tfidf = vectorizer.fit_transform(combined_texts).toarray()
    pca = PCA(n_components=128)
    text_features = pca.fit_transform(text_features_tfidf)
    
    # 现在text_features应该有918行，与scores_features匹配
    print(f"处理后文本特征数量: {len(text_features)}")
    
    # 合并特征
    stu_features = np.concatenate((scores_features, text_features), axis=1)
    scaler = StandardScaler()
    stu_features_scaled = scaler.fit_transform(stu_features)
    
    # 2. 构建异构图
    stu_dorm_df = pd.read_csv("data/csv_data/stu_dorm.csv")
    stu_course_df = pd.read_csv("data/csv_data/stu_course.csv")
    
    # 重新映射ID
    def remap_ids(stu_dorm_df, stu_course_df):
        stu_ids = pd.concat([stu_dorm_df['stu_src'], stu_course_df['stu_src']]).unique()
        stu_map = {old_id: new_id for new_id, old_id in enumerate(stu_ids)}
        
        dorm_ids = stu_dorm_df['dorm_dst'].unique()
        dorm_map = {old_id: new_id for new_id, old_id in enumerate(dorm_ids)}
        
        course_ids = stu_course_df['course_dst'].unique()
        course_map = {old_id: new_id for new_id, old_id in enumerate(course_ids)}
        
        stu_dorm_df = stu_dorm_df.copy()
        stu_course_df = stu_course_df.copy()
        
        stu_dorm_df['stu_src'] = stu_dorm_df['stu_src'].map(stu_map)
        stu_dorm_df['dorm_dst'] = stu_dorm_df['dorm_dst'].map(dorm_map)
        
        stu_course_df['stu_src'] = stu_course_df['stu_src'].map(stu_map)
        stu_course_df['course_dst'] = stu_course_df['course_dst'].map(course_map)
        
        return stu_dorm_df, stu_course_df, len(stu_ids), len(dorm_ids), len(course_ids)
    
    remapped_stu_dorm_df, remapped_stu_course_df, num_stus, num_dorms, num_courses = remap_ids(stu_dorm_df, stu_course_df)
    
    # 构建异构图前，确保特征数量正确
    print(f"学生特征数量: {len(stu_features_scaled)}")
    print(f"重映射后学生节点数量: {num_stus}")

    # 如果节点数量与特征数量不匹配，则调整节点数量或特征数量
    if num_stus != len(stu_features_scaled):
        print(f"检测到节点数量与特征数量不匹配，进行调整...")
        if num_stus > len(stu_features_scaled):
            # 如果节点数量更多，添加零特征向量
            additional_features = np.zeros((num_stus - len(stu_features_scaled), stu_features_scaled.shape[1]))
            stu_features_scaled = np.vstack([stu_features_scaled, additional_features])
        else:
            # 如果特征数量更多，截断特征数量以匹配节点数量
            stu_features_scaled = stu_features_scaled[:num_stus, :]
        print(f"调整后学生特征数量: {len(stu_features_scaled)}")

    # 修改标签数量以匹配节点数量
    if len(labels) != num_stus:
        print(f"标签数量({len(labels)})与节点数量({num_stus})不匹配，进行调整...")
        if num_stus > len(labels):
            # 如果节点更多，用最后一个标签填充
            additional_labels = np.full(num_stus - len(labels), labels[-1])
            labels = np.concatenate([labels, additional_labels])
        else:
            # 如果标签更多，截断标签
            labels = labels[:num_stus]
        print(f"调整后标签数量: {len(labels)}")

    # 构建异构图
    g = dgl.heterograph({
        ('stu', 'live', 'dorm'): (torch.tensor(remapped_stu_dorm_df['stu_src']).to(device), 
                                  torch.tensor(remapped_stu_dorm_df['dorm_dst']).to(device)),
        ('dorm', 'lived-by', 'stu'): (torch.tensor(remapped_stu_dorm_df['dorm_dst']).to(device), 
                                      torch.tensor(remapped_stu_dorm_df['stu_src']).to(device)),
        ('stu', 'choose', 'course'): (torch.tensor(remapped_stu_course_df['stu_src']).to(device), 
                                     torch.tensor(remapped_stu_course_df['course_dst']).to(device)),
        ('course', 'choosed-by', 'stu'): (torch.tensor(remapped_stu_course_df['course_dst']).to(device), 
                                         torch.tensor(remapped_stu_course_df['stu_src']).to(device))
    })
    
    # 将图移动到指定设备 (DGL 0.5+ 支持图移动到GPU)
    try:
        g = g.to(device)
        print(f"成功将图移动到设备: {device}")
    except Exception as e:
        print(f"将图移动到设备时出错: {e}")
        print("尝试只将图的特征移动到设备")
    
    # 创建特征张量并移动到指定设备
    stu_features_tensor = torch.tensor(stu_features_scaled, dtype=torch.float32).to(device)
    course_features_tensor = torch.zeros(num_courses, stu_features_scaled.shape[1], dtype=torch.float32).to(device)
    dorm_features_tensor = torch.zeros(num_dorms, stu_features_scaled.shape[1], dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # 设置节点特征
    g.nodes['stu'].data['stu_feature'] = stu_features_tensor
    g.nodes['course'].data['course_feature'] = course_features_tensor
    g.nodes['dorm'].data['dorm_feature'] = dorm_features_tensor
    g.nodes['stu'].data['label'] = labels_tensor
    
    # 3. 划分数据集 (7:1:2)
    all_indices = np.arange(len(labels))
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    test_size = int(len(labels) * 0.2)
    val_size = int((len(labels) - test_size) * 0.125)
    
    test_idx = all_indices[:test_size]
    val_idx = all_indices[test_size:test_size+val_size]
    train_idx = all_indices[test_size+val_size:]
    
    # 4. 创建XGradNet模型
    feature_dim = stu_features_scaled.shape[1]
    hidden_dim = 128
    out_dim = 64
    num_classes = 6
    grade_dim = scores_features.shape[1]  # 成绩特征的维度
    
    # 初始化模型并明确移动到GPU
    model = XGradNet(g, feature_dim, hidden_dim, out_dim, num_classes, grade_dim)
    model = model.to(device)  # 明确将模型移动到设备
    
    # 确认模型是否在GPU上
    model_device = next(model.parameters()).device
    print(f"模型初始化后设备: {model_device}")
    
    # 如果模型不在指定设备上，再次尝试移动
    if str(model_device) != str(device) and device.type == 'cuda':
        print(f"模型设备不匹配，尝试重新移动模型到 {device}")
        model = model.to(device)  # 再次尝试移动
        print(f"移动后模型设备: {next(model.parameters()).device}")
    
    # 准备节点特征
    node_features = {
        'stu': g.nodes['stu'].data['stu_feature'],
        'course': g.nodes['course'].data['course_feature'],
        'dorm': g.nodes['dorm'].data['dorm_feature']
    }
    
    # 显式检查和修复特征设备问题
    for k, v in node_features.items():
        if v.device != device:
            print(f"警告: {k}特征不在正确设备上 (当前: {v.device}，目标: {device})，正在修复...")
            node_features[k] = v.to(device)
    
    # 训练前确认所有张量都在GPU上
    print("\n=== 设备检查 ===")
    for k, v in node_features.items():
        print(f"{k} 特征设备: {v.device}")
    print(f"图设备: {g.device if hasattr(g, 'device') else '不适用'}")
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 检查内存使用情况
    if torch.cuda.is_available() and device.type == 'cuda':
        gpu_idx = device.index
        print(f"\n当前GPU内存使用: {torch.cuda.memory_allocated(gpu_idx) / 1024**2:.2f} MB")
        print(f"当前GPU内存缓存: {torch.cuda.memory_reserved(gpu_idx) / 1024**2:.2f} MB")
    
    # 5. 训练模型
    print(f"\n开始训练，使用设备: {device}")
    trained_model, train_losses, val_accuracies = train(
        model, node_features, train_idx, val_idx, g.nodes['stu'].data['label'],
        num_epochs=50, lr=0.001, weight_decay=1e-4, device=device
    )
    
    # 6. 评估模型
    class_names = ['国内升学', '出国深造', '企业就业', '自主创业', '参军或进入政府机关', '待就业']
    
    results = evaluate(
        trained_model, node_features, test_idx, g.nodes['stu'].data['label'], class_names, device=device
    )
    
    # 7. 保存模型
    torch.save(trained_model.state_dict(), 'xgradnet_model.pth')
    
    return trained_model, results, g, node_features

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行XGradNet模型')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID（0,1,2,...）')
    parser.add_argument('--auto', action='store_true', help='自动选择最空闲的GPU')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    args = parser.parse_args()
    
    if args.cpu:
        print("强制使用CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_id = None
    elif args.gpu is not None:
        print(f"使用指定的GPU: {args.gpu}")
        gpu_id = args.gpu
    elif args.auto:
        print("自动选择最空闲的GPU")
        gpu_id = get_free_gpu()
        print(f"已选择 GPU {gpu_id}")
    else:
        # 默认使用GPU 0
        gpu_id = 0
        
    # 检查CUDA是否可用
    if not torch.cuda.is_available() and gpu_id is not None:
        print("警告: CUDA不可用，强制使用CPU")
        gpu_id = None
    
    # 运行模型训练
    model, results, g, node_features = run_xgradnet(gpu_id=gpu_id)
    
    # 打印最终设备信息
    print(f"\n=== 训练完成后设备信息 ===")
    if torch.cuda.is_available():
        print(f"训练使用的设备: {next(model.parameters()).device}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")