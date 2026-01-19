import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
import dgl.nn as dglnn
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re
from collections import defaultdict, Counter
import warnings
import os
from tqdm import tqdm
import gc
import random

warnings.filterwarnings('ignore')


# ========== 定义与训练完全相同的RelationAwareGCN模型 ==========
class RelationAwareGCN(nn.Module):
    """关系感知的GCN模型，专门用于实体类型预测"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=3, dropout=0.3):
        super(RelationAwareGCN, self).__init__()

        # 输入编码层
        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # GCN层
        for i in range(num_layers):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 关系注意力层
        self.relation_attention = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, 1),
            nn.Sigmoid()
        )

        # 邻居类型聚合层
        self.neighbor_aggregator = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(h_feats, num_classes),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def aggregate_neighbor_types(self, g, node_features):
        """聚合邻居特征"""
        with g.local_scope():
            g.ndata['h'] = node_features

            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neighbor_h'))
            neighbor_features = g.ndata['neighbor_h']

        combined = torch.cat([node_features, neighbor_features], dim=1)
        aggregated = self.neighbor_aggregator(combined)

        return aggregated

    def apply_relation_attention(self, node_features, neighbor_features):
        """应用关系注意力机制"""
        combined = torch.cat([node_features, neighbor_features], dim=1)

        attention_weights = self.relation_attention(combined)
        attended_neighbors = neighbor_features * attention_weights

        enhanced = node_features + attended_neighbors

        return enhanced

    def forward(self, g, features):
        # 编码输入特征
        h = self.input_encoder(features)

        layer_outputs = []

        # 多层传播
        for i in range(self.num_layers):
            if i == 0:
                neighbor_agg = self.aggregate_neighbor_types(g, h)
                h_attended = self.apply_relation_attention(h, neighbor_agg)

            # GCN传播
            h_new = self.layers[i](g, h_attended if i == 0 else h)
            h_new = self.bns[i](h_new)
            h_new = F.relu(h_new)

            if i > 0:
                h_new = h_new + h

            h = self.dropout(h_new)
            layer_outputs.append(h)

        # 使用最后一层的输出
        h_final = layer_outputs[-1]

        # 输出预测
        out = self.output_layer(h_final)

        return out


# ========== 加载模型和相关数据 ==========
# ========== 加载模型和相关数据 ==========
def load_trained_model(model_path):
    """加载训练好的RelationAwareGCN模型"""
    print(f"加载RelationAwareGCN模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型文件失败: {e}")
        return None

    if 'model_config' not in checkpoint:
        print("错误: 模型文件格式不正确，缺少model_config")
        return None

    model_config = checkpoint['model_config']
    state_dict = checkpoint['model_state_dict']

    # 创建模型
    try:
        print(f"创建 {model_config['num_layers']} 层RelationAwareGCN模型...")

        model = RelationAwareGCN(
            in_feats=model_config['in_feats'],
            h_feats=model_config['h_feats'],
            num_classes=model_config['num_classes'],
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.3)
        )

        # 加载状态字典
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ 模型加载成功")

    except Exception as e:
        print(f"创建模型失败: {e}")
        print("尝试使用strict=False加载...")

        try:
            model = RelationAwareGCN(
                in_feats=model_config['in_feats'],
                h_feats=model_config['h_feats'],
                num_classes=model_config['num_classes'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.3)
            )

            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("✓ 模型加载成功（使用strict=False）")

        except Exception as e2:
            print(f"重新匹配也失败: {e2}")
            return None

    # 提取其他必要数据
    entity_to_idx = checkpoint.get('entity_to_idx', {})
    idx_to_entity = checkpoint.get('idx_to_entity', {})
    label_encoder = checkpoint.get('label_encoder')
    feature_names = checkpoint.get('feature_names', [])

    # 修复：正确处理scaler
    scaler = checkpoint.get('scaler')
    if scaler is None:
        print("警告: 模型文件中没有scaler，特征将不进行标准化")
    else:
        # 检查scaler的维度
        expected_features = model_config['in_feats']
        if hasattr(scaler, 'n_features_in_'):
            scaler_features = scaler.n_features_in_
            print(f"Scaler期望特征维度: {scaler_features}")
            print(f"模型期望特征维度: {expected_features}")

    top_relations = checkpoint.get('top_relations', [])
    node_features = checkpoint.get('node_features')

    print(f"\n模型配置:")
    print(f"  模型类型: {model_config.get('model_type', 'RelationAwareGCN')}")
    print(f"  输入特征: {model_config['in_feats']}")
    print(f"  隐藏层: {model_config['h_feats']}")
    print(f"  类别数: {model_config['num_classes']}")
    print(f"  层数: {model_config['num_layers']}")
    print(f"  实体数: {len(entity_to_idx)}")
    print(f"  特征数量: {len(feature_names)}")
    print(f"  验证集准确率: {checkpoint.get('best_val_acc', 0):.4f}")
    print(f"  验证集F1分数: {checkpoint.get('best_val_f1', 0):.4f}")

    return {
        'model': model,
        'model_config': model_config,
        'entity_to_idx': entity_to_idx,
        'idx_to_entity': idx_to_entity,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'scaler': scaler,
        'top_relations': top_relations,
        'node_features': node_features,
        'best_val_acc': checkpoint.get('best_val_acc', 0),
        'best_val_f1': checkpoint.get('best_val_f1', 0)
    }


# ========== 加载真实标签数据 ==========
def load_ground_truth(csv_path):
    """加载真实标签数据"""
    print(f"\n加载真实标签数据: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"错误: 真实标签文件不存在: {csv_path}")
        return {}, {}

    try:
        entity_types = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            entity_types = pd.read_csv(csv_path, encoding='latin-1')
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return {}, {}

    entity_to_true_type = {}
    entity_to_type_name = {}

    if 'entity_id' in entity_types.columns and 'predicted_category' in entity_types.columns:
        entity_to_true_type = dict(zip(
            entity_types['entity_id'],
            entity_types['predicted_category']
        ))
    else:
        print("警告: CSV文件中缺少必要的列")

    if 'predicted_category_name' in entity_types.columns:
        entity_to_type_name = dict(zip(
            entity_types['entity_id'],
            entity_types['predicted_category_name']
        ))

    print(f"加载了 {len(entity_to_true_type)} 个实体的真实标签")

    return entity_to_true_type, entity_to_type_name


# ========== 解析测试文件 ==========
def parse_test_file_detailed(test_file_path):
    """解析详细的测试文件格式"""
    print(f"\n解析测试文件: {test_file_path}")

    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return set(), {}

    target_entities = set()
    all_entities_in_file = set()
    entity_relations = defaultdict(list)

    with open(test_file_path, 'r', encoding='utf-8') as f:
        current_entity = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('实体:'):
                match = re.search(r'实体:\s*([^\s(]+)', line)
                if match:
                    current_entity = match.group(1)
                    target_entities.add(current_entity)
                continue

            if line.startswith('---') or line.startswith('=='):
                continue

            if line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                all_entities_in_file.update([head, tail])

                if current_entity:
                    if head == current_entity or tail == current_entity:
                        entity_relations[current_entity].append((head, relation, tail))

    print(f"解析结果:")
    print(f"  目标测试实体: {len(target_entities)} 个")
    print(f"  文件中出现的所有实体: {len(all_entities_in_file)} 个")

    if len(target_entities) != 100:
        print(f"警告: 期望100个测试实体，但找到了 {len(target_entities)} 个")

    return target_entities, entity_relations


# ========== 构建测试子图 ==========
def build_test_graph(target_entities, entity_relations, entity_to_idx):
    """为测试实体构建子图"""
    print(f"\n为测试实体构建子图...")

    all_entities_in_test = set()
    for entity in target_entities:
        all_entities_in_test.add(entity)
        triples = entity_relations.get(entity, [])
        for head, rel, tail in triples:
            all_entities_in_test.add(head)
            all_entities_in_test.add(tail)

    print(f"测试子图中包含 {len(all_entities_in_test)} 个实体")

    valid_entities = [e for e in all_entities_in_test if e in entity_to_idx]
    print(f"其中 {len(valid_entities)} 个实体在模型索引中")

    target_in_valid = [e for e in target_entities if e in valid_entities]
    print(f"目标实体在valid_entities中的数量: {len(target_in_valid)} / {len(target_entities)}")

    if len(target_in_valid) == 0:
        print("错误: 没有目标实体在模型索引中!")
        return None, None, None

    if len(valid_entities) == 0:
        print("错误: 没有测试实体在模型索引中!")
        return None, None, None

    subgraph_entity_to_idx = {entity: idx for idx, entity in enumerate(valid_entities)}

    src_nodes = []
    dst_nodes = []

    for entity in target_entities:
        triples = entity_relations.get(entity, [])
        for head, rel, tail in triples:
            if head in subgraph_entity_to_idx and tail in subgraph_entity_to_idx:
                src_idx = subgraph_entity_to_idx[head]
                dst_idx = subgraph_entity_to_idx[tail]
                src_nodes.append(src_idx)
                dst_nodes.append(dst_idx)

    if not src_nodes:
        src_nodes = list(range(len(valid_entities)))
        dst_nodes = list(range(len(valid_entities)))

    num_nodes = len(valid_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    g = dgl.add_self_loop(g)

    print(f"测试子图构建完成: {g.num_nodes()} 个节点, {g.num_edges()} 条边")

    return g, subgraph_entity_to_idx, valid_entities


# ========== 创建测试特征 ==========
# ========== 创建测试特征 ==========
def create_test_features(valid_entities, model_data, entity_relations, entity_to_true_type):
    """为测试实体创建特征 - 修复标准化问题"""
    print(f"\n为测试实体创建特征...")

    entity_to_idx = model_data['entity_to_idx']
    feature_names = model_data.get('feature_names', [])
    scaler = model_data.get('scaler')
    saved_features = model_data.get('node_features')
    top_relations = model_data.get('top_relations', [])

    feature_dim = model_data['model_config']['in_feats']
    print(f"模型期望特征维度: {feature_dim}")

    if top_relations:
        relation_to_idx = {rel: i for i, rel in enumerate(top_relations)}
        print(f"使用 {len(top_relations)} 个关系类型")

    features = []
    entity_order = []

    for entity in tqdm(valid_entities, desc="创建特征"):
        entity_order.append(entity)

        feat_vector = np.zeros(feature_dim, dtype=np.float32)

        # 如果实体在保存的特征中，直接使用
        if saved_features is not None and entity in entity_to_idx:
            idx = entity_to_idx[entity]
            if idx < saved_features.shape[0]:
                saved_feat = saved_features[idx]
                if len(saved_feat) == feature_dim:
                    feat_vector = saved_feat.numpy()
                elif len(saved_feat) < feature_dim:
                    feat_vector[:len(saved_feat)] = saved_feat.numpy()
                    print(f"警告: 保存的特征维度 {len(saved_feat)} 小于期望维度 {feature_dim}")
        else:
            # 基于测试文件创建特征
            triples = entity_relations.get(entity, [])

            # 计算基础特征
            has_label = 1.0 if entity in entity_to_true_type else 0.0

            # 统计度数
            in_degree = 0
            out_degree = 0
            neighbors = set()
            relation_counts = Counter()

            for h, r, t in triples:
                if h == entity:
                    out_degree += 1
                    neighbors.add(t)
                if t == entity:
                    in_degree += 1
                    neighbors.add(h)
                relation_counts[r] += 1

            total_degree = in_degree + out_degree
            unique_relations = len(set(relation_counts.keys()))
            neighbor_count = len(neighbors)

            # 计算有标签的邻居数
            labeled_neighbors = 0
            for neighbor in neighbors:
                if neighbor in entity_to_true_type:
                    labeled_neighbors += 1

            # 计算邻居类型多样性
            neighbor_types = []
            for neighbor in neighbors:
                if neighbor in entity_to_true_type:
                    neighbor_types.append(entity_to_true_type[neighbor])
            unique_neighbor_types = len(set(neighbor_types))

            # 基础特征部分 (前8个维度)
            base_feat = [
                has_label,
                float(in_degree),
                float(out_degree),
                float(total_degree),
                float(unique_relations),
                float(neighbor_count),
                float(labeled_neighbors),
                float(unique_neighbor_types)
            ]

            # 关系分布特征部分
            if top_relations:
                rel_feat = np.zeros(len(top_relations), dtype=np.float32)
                total_rels = sum(relation_counts.values())

                if total_rels > 0:
                    for rel, count in relation_counts.items():
                        if rel in relation_to_idx:
                            rel_idx = relation_to_idx[rel]
                            rel_feat[rel_idx] = count / total_rels

                # 组合特征
                if feature_dim == len(base_feat) + len(rel_feat):
                    feat_vector = np.concatenate([base_feat, rel_feat])
                else:
                    print(
                        f"警告: 特征维度不匹配，基础特征 {len(base_feat)} + 关系特征 {len(rel_feat)} != 期望 {feature_dim}")
                    # 使用基础特征
                    if feature_dim >= len(base_feat):
                        feat_vector[:len(base_feat)] = base_feat
            else:
                # 如果没有top_relations，只使用基础特征
                if feature_dim == len(base_feat):
                    feat_vector = np.array(base_feat, dtype=np.float32)
                else:
                    print(f"警告: 特征维度不匹配，基础特征 {len(base_feat)} != 期望 {feature_dim}")
                    if feature_dim >= len(base_feat):
                        feat_vector[:len(base_feat)] = base_feat

        features.append(feat_vector)

    features_np = np.array(features, dtype=np.float32)
    print(f"特征矩阵原始形状: {features_np.shape}")

    # 标准化特征 - 修复标准化问题
    if scaler is not None:
        print("标准化特征...")
        try:
            # 检查scaler维度
            if hasattr(scaler, 'n_features_in_'):
                scaler_dim = scaler.n_features_in_
                print(f"Scaler训练时特征维度: {scaler_dim}")
                print(f"当前特征维度: {features_np.shape[1]}")

                if scaler_dim == features_np.shape[1]:
                    features_np = scaler.transform(features_np)
                else:
                    print(f"警告: Scaler维度 {scaler_dim} 与特征维度 {features_np.shape[1]} 不匹配")
                    print("跳过标准化步骤")
            else:
                # 尝试直接转换
                features_np = scaler.transform(features_np)
        except Exception as e:
            print(f"标准化失败: {e}")
            print("跳过标准化步骤")
    else:
        print("没有可用的scaler，跳过标准化")

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    print(f"特征矩阵最终形状: {features_tensor.shape}")

    return features_tensor, entity_order


# ========== 进行预测 ==========
def predict_entities_for_targets(model, g, features, entity_order, label_encoder, target_entities):
    """专门为目标实体进行预测"""
    print(f"\n为目标实体进行预测...")

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits, dim=1)

    predictions = {}

    for i, entity in enumerate(entity_order):
        if entity in target_entities:
            entity_probs = probs[i]

            top_probs, top_indices = torch.topk(entity_probs, 3)

            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                pred_class = idx.item()

                if label_encoder is not None:
                    try:
                        pred_type = label_encoder.inverse_transform([pred_class])[0]
                    except:
                        pred_type = str(pred_class)
                else:
                    pred_type = str(pred_class)

                top_predictions.append({
                    'type': pred_type,
                    'class': pred_class,
                    'probability': prob.item()
                })

            main_pred = top_predictions[0]

            predictions[entity] = {
                'predicted_type': main_pred['type'],
                'predicted_class': main_pred['class'],
                'confidence': main_pred['probability'],
                'top3': top_predictions
            }

    print(f"完成了 {len(predictions)} 个目标实体的预测")
    print(f"成功预测的目标实体示例: {list(predictions.keys())[:5]}")

    return predictions


# ========== 评估预测结果 ==========
def evaluate_predictions(predictions, entity_to_true_type, target_entities):
    """评估预测结果"""
    print(f"\n评估预测结果...")

    results = []
    correct_count = 0
    total_count = 0

    entities_to_evaluate = []
    for entity in target_entities:
        if entity in predictions and entity in entity_to_true_type:
            entities_to_evaluate.append(entity)

    print(f"可以评估的实体: {len(entities_to_evaluate)}")

    if not entities_to_evaluate:
        print("错误: 没有找到可以评估的实体!")
        return None, 0

    for entity in entities_to_evaluate:
        pred_info = predictions[entity]
        true_type = entity_to_true_type[entity]
        pred_type = pred_info['predicted_type']

        is_correct = (str(pred_type) == str(true_type))

        result = {
            'entity': entity,
            'true_type': true_type,
            'predicted_type': pred_type,
            'is_correct': is_correct,
            'confidence': pred_info['confidence']
        }

        if len(pred_info['top3']) > 1:
            result['top2_type'] = pred_info['top3'][1]['type']
            result['top2_prob'] = pred_info['top3'][1]['probability']

        if len(pred_info['top3']) > 2:
            result['top3_type'] = pred_info['top3'][2]['type']
            result['top3_prob'] = pred_info['top3'][2]['probability']

        results.append(result)

        total_count += 1
        if is_correct:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\n评估结果:")
    print(f"  目标实体总数: {len(target_entities)}")
    print(f"  可以评估的目标实体: {total_count}")
    print(f"  正确预测数: {correct_count}")
    print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if total_count > 0:
        true_labels = [str(r['true_type']) for r in results]
        pred_labels = [str(r['predicted_type']) for r in results]

        print(f"  加权F1分数: {f1_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}")

    return results, accuracy


# ========== 测试单个实体的辅助函数 ==========
def test_single_entity_manual():
    """手动测试单个实体"""
    print("=" * 80)
    print("手动测试单个实体")
    print("=" * 80)

    model_path = 'models/entity_type_predictor_fixed.pth'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'

    try:
        # 加载模型
        print("\n1. 加载RelationAwareGCN模型...")
        model_data = load_trained_model(model_path)
        if model_data is None:
            print("模型加载失败，退出测试")
            return

        # 加载真实标签
        print("\n2. 加载真实标签...")
        entity_to_true_type, _ = load_ground_truth(ground_truth_path)

        # 从有标签的实体中随机选择一个
        labeled_entities = [e for e in entity_to_true_type.keys() if e in model_data['entity_to_idx']]
        if not labeled_entities:
            print("错误: 没有找到有标签的实体")
            return

        test_entity = random.choice(labeled_entities)
        print(f"\n3. 随机选择的测试实体: {test_entity}")

        # 显示实体的真实类型
        if test_entity in entity_to_true_type:
            true_type = entity_to_true_type[test_entity]
            print(f"真实类型: {true_type}")

        # 获取实体索引
        if test_entity in model_data['entity_to_idx']:
            idx = model_data['entity_to_idx'][test_entity]

            # 使用模型进行预测
            model = model_data['model']
            model.eval()

            # 构建单节点图
            g = dgl.graph(([0], [0]), num_nodes=1)
            g = dgl.add_self_loop(g)

            # 获取该实体的特征
            if model_data['node_features'] is not None and idx < model_data['node_features'].shape[0]:
                features = model_data['node_features'][idx:idx + 1]
                print(f"使用保存的特征，维度: {features.shape}")
            else:
                print("警告: 无法获取实体特征")
                return

            with torch.no_grad():
                logits = model(g, features)
                probs = F.softmax(logits[0], dim=0)

                # 获取top-3预测
                top_probs, top_indices = torch.topk(probs, 5)  # 显示前5个

                print(f"\n预测结果:")
                print(f"{'排名':<6} {'类型ID':<15} {'置信度':<12} {'是否真实'}")
                print("-" * 50)

                for i, (prob, cls_idx) in enumerate(zip(top_probs, top_indices)):
                    pred_class = cls_idx.item()

                    if model_data['label_encoder'] is not None:
                        try:
                            pred_type = model_data['label_encoder'].inverse_transform([pred_class])[0]
                        except:
                            pred_type = str(pred_class)
                    else:
                        pred_type = str(pred_class)

                    prob_val = prob.item()

                    # 检查是否是真实类型
                    is_true = ""
                    if test_entity in entity_to_true_type:
                        true_type = entity_to_true_type[test_entity]
                        try:
                            if model_data['label_encoder']:
                                true_type_encoded = model_data['label_encoder'].transform([true_type])[0]
                                if pred_class == true_type_encoded:
                                    is_true = "✓"
                        except:
                            pass

                    print(f"{i + 1:<6} {pred_type:<15} {prob_val:<12.2%} {is_true}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 保存结果 ==========
def save_results(results, accuracy, model_info, target_entities, output_dir='test_results'):
    """保存评估结果"""
    print(f"\n保存评估结果...")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'relation_aware_gcn_results_{timestamp}.csv')

    if results:
        df = pd.DataFrame(results)

        columns_order = ['entity', 'true_type', 'predicted_type', 'is_correct', 'confidence']
        extra_cols = [col for col in df.columns if col not in columns_order]
        df = df[columns_order + extra_cols]

        df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✓ 详细结果已保存到: {results_file}")

    summary_file = os.path.join(output_dir, f'relation_aware_gcn_summary_{timestamp}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RelationAwareGCN模型测试结果汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"模型信息:\n")
        f.write(f"  模型类型: {model_info['model_config'].get('model_type', 'RelationAwareGCN')}\n")
        f.write(f"  输入特征: {model_info['model_config']['in_feats']}\n")
        f.write(f"  隐藏层: {model_info['model_config']['h_feats']}\n")
        f.write(f"  类别数: {model_info['model_config']['num_classes']}\n")
        f.write(f"  层数: {model_info['model_config']['num_layers']}\n")
        f.write(f"  验证集准确率: {model_info.get('best_val_acc', 0):.4f}\n")
        f.write(f"  验证集F1分数: {model_info.get('best_val_f1', 0):.4f}\n\n")

        f.write(f"测试配置:\n")
        f.write(f"  目标实体数: {len(target_entities)}\n")
        f.write(f"  测试时间: {timestamp}\n\n")

        if results:
            f.write(f"测试结果:\n")
            f.write(f"  评估实体数: {len(results)}\n")
            f.write(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

            correct_count = sum(1 for r in results if r['is_correct'])
            incorrect_count = len(results) - correct_count

            f.write(f"  正确预测: {correct_count} 个 ({correct_count / len(results) * 100:.1f}%)\n")
            f.write(f"  错误预测: {incorrect_count} 个 ({incorrect_count / len(results) * 100:.1f}%)\n")

    print(f"✓ 汇总报告已保存到: {summary_file}")

    return results_file, summary_file


# ========== 主测试函数 ==========
def test_relation_aware_gcn():
    """RelationAwareGCN模型测试主函数"""
    print("=" * 80)
    print("RelationAwareGCN模型测试")
    print("=" * 80)

    model_path = 'models/entity_type_predictor_fixed.pth'
    test_file_path = 'data/FB15KET/TEST_PART_DETAILED.txt'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'
    output_dir = 'test_results'

    try:
        # 步骤1: 加载模型
        print("\n1. 加载RelationAwareGCN模型...")
        model_data = load_trained_model(model_path)
        if model_data is None:
            print("模型加载失败，退出测试")
            return

        # 步骤2: 加载真实标签
        print("\n2. 加载真实标签...")
        entity_to_true_type, entity_to_type_name = load_ground_truth(ground_truth_path)

        if not entity_to_true_type:
            print("警告: 没有加载到真实标签，无法进行准确评估")

        # 步骤3: 解析测试文件
        print("\n3. 解析测试文件...")
        target_entities, entity_relations = parse_test_file_detailed(test_file_path)

        if not target_entities:
            print("没有找到目标测试实体，退出测试")
            return

        print(f"确认: 找到 {len(target_entities)} 个目标测试实体")

        # 步骤4: 构建测试子图
        print("\n4. 构建测试子图...")
        g, subgraph_entity_to_idx, valid_entities = build_test_graph(
            target_entities, entity_relations, model_data['entity_to_idx']
        )

        if g is None:
            print("构建图失败，退出测试")
            return

        # 步骤5: 创建测试特征
        print("\n5. 创建测试特征...")
        features, entity_order = create_test_features(
            valid_entities, model_data, entity_relations, entity_to_true_type
        )

        # 步骤6: 进行预测
        print("\n6. 为目标实体进行预测...")
        predictions = predict_entities_for_targets(
            model_data['model'], g, features, entity_order,
            model_data['label_encoder'], target_entities
        )

        if not predictions:
            print("严重错误: 没有成功预测任何目标实体!")
            return

        # 步骤7: 评估预测结果
        if entity_to_true_type:
            print("\n7. 评估目标实体预测结果...")
            results, accuracy = evaluate_predictions(predictions, entity_to_true_type, target_entities)

            if results is None:
                print("评估失败，退出测试")
                return

            # 步骤8: 保存结果
            print("\n8. 保存结果...")
            results_file, summary_file = save_results(results, accuracy, model_data, target_entities, output_dir)

            print(f"\n" + "=" * 80)
            print("测试完成!")
            print("=" * 80)
            print(f"目标实体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # 显示预测示例
            print(f"\n预测示例:")
            if results:
                correct_examples = [r for r in results if r['is_correct']][:2]
                incorrect_examples = [r for r in results if not r['is_correct']][:2]

                if correct_examples:
                    print(f"正确预测示例:")
                    for i, r in enumerate(correct_examples):
                        print(
                            f"  {i + 1}. {r['entity']}: 真实={r['true_type']}, 预测={r['predicted_type']}, 置信度={r['confidence']:.3f}")

                if incorrect_examples:
                    print(f"\n错误预测示例:")
                    for i, r in enumerate(incorrect_examples):
                        print(
                            f"  {i + 1}. {r['entity']}: 真实={r['true_type']}, 预测={r['predicted_type']}, 置信度={r['confidence']:.3f}")
        else:
            print("\n警告: 没有真实标签数据，跳过评估步骤")
            print(f"完成了 {len(predictions)} 个实体的预测")

            print(f"\n预测示例 (前5个):")
            for i, (entity, pred) in enumerate(list(predictions.items())[:5]):
                print(f"  {i + 1}. {entity}: 预测类型={pred['predicted_type']}, 置信度={pred['confidence']:.3f}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("RelationAwareGCN实体类型预测模型测试工具")
    print("=" * 80)

    print("\n测试模式:")
    print("1. 完整测试 (使用TEST_PART_DETAILED.txt测试目标实体)")
    print("2. 随机单个实体测试")

    try:
        choice = input("请输入选项 (1-2): ").strip()

        if choice == '1':
            test_relation_aware_gcn()
        elif choice == '2':
            test_single_entity_manual()
        else:
            print("无效选项，使用默认完整测试")
            test_relation_aware_gcn()

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()