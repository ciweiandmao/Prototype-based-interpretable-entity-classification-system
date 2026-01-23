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
import hashlib

warnings.filterwarnings('ignore')


# ========== 定义与训练完全相同的GraphSAGE模型 ==========
class TypeAwareGraphSAGE(nn.Module):
    """类型感知的GraphSAGE模型，专门用于实体类型预测"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.3):
        super(TypeAwareGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入编码层
        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GraphSAGE层
        self.sage_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层
        self.sage_layers.append(dglnn.SAGEConv(
            in_feats=h_feats,
            out_feats=h_feats * 2,
            aggregator_type='mean',
            feat_drop=dropout
        ))
        self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 中间层
        for i in range(1, num_layers - 1):
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats * 2,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 输出层
        if num_layers > 1:
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 关系类型编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_feats // 2, num_classes)
        )

    def forward(self, g, features):
        h = self.input_encoder(features)

        # 保存每层的输出用于特征融合
        layer_outputs = [h]

        # GraphSAGE传播
        for i in range(self.num_layers):
            h = self.sage_layers[i](g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # 多层特征融合
        if len(layer_outputs) > 1:
            # 使用最后一层和第一层的特征
            h_final = torch.cat([layer_outputs[0], layer_outputs[-1]], dim=1)
            h_final = self.relation_encoder(h_final)
        else:
            h_final = layer_outputs[0]

        # 最终分类
        out = self.classifier(h_final)

        return out


# ========== 新增：图像特征提取函数 ==========
def extract_image_features_for_entity(entity_id, base_path="D:/Z-Downloader/download"):
    """
    提取实体的图像特征
    与训练代码保持一致
    """
    # 将实体ID转换为文件夹名格式
    if entity_id.startswith('/m/'):
        folder_name = f"m.{entity_id[3:].replace('/', '.')}"
    else:
        folder_name = entity_id.replace('/', '.').strip('.')

    image_dir = os.path.join(base_path, folder_name)

    # 初始化特征值（全部为0）
    features = np.zeros(10, dtype=np.float32)

    if not os.path.exists(image_dir):
        # 没有图像目录，返回全0特征
        return features

    try:
        # 获取目录下所有文件
        all_files = [f for f in os.listdir(image_dir)
                     if os.path.isfile(os.path.join(image_dir, f))]

        # 只考虑常见的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in all_files
                       if os.path.splitext(f)[1].lower() in image_extensions]

        num_images = len(image_files)

        if num_images == 0:
            return features

        # 提取简单统计特征
        file_sizes = []
        for img_file in image_files[:20]:  # 只检查前20个文件避免耗时过长
            try:
                file_path = os.path.join(image_dir, img_file)
                size = os.path.getsize(file_path)
                file_sizes.append(size)
            except:
                continue

        if file_sizes:
            file_sizes = np.array(file_sizes)
            features[0] = num_images
            features[1] = np.mean(file_sizes) / 1024
            features[2] = np.std(file_sizes) / 1024 if len(file_sizes) > 1 else 0
            features[3] = np.min(file_sizes) / 1024
            features[4] = np.max(file_sizes) / 1024
            features[5] = float(len(file_sizes)) / num_images  # 可访问文件比例
            features[6] = 1.0  # 有图像标志
        else:
            features[6] = 0.0  # 无有效图像标志

        # 添加一些随机噪声特征
        features[7] = np.random.random() * 0.1  # 很小的随机值
        # 使用hashlib确保一致性
        features[8] = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 100 / 100.0
        features[9] = len(folder_name) / 100.0  # 文件夹名长度归一化

    except Exception as e:
        # 出错时返回默认特征
        print(f"警告: 提取实体 {entity_id} 的图像特征时出错: {e}")

    return features


def create_mock_features_for_entity(entity, entity_relations, entity_to_true_type,
                                    feature_dim, top_relations, type_to_idx, all_types):
    """为新实体创建特征（包含图像特征）"""
    import hashlib

    triples = entity_relations.get(entity, [])

    # 1. 基础特征
    has_label = 1.0 if entity in entity_to_true_type else 0.0

    in_degree = sum(1 for h, r, t in triples if t == entity)
    out_degree = sum(1 for h, r, t in triples if h == entity)

    neighbors = set()
    for h, r, t in triples:
        if h == entity:
            neighbors.add(t)
        if t == entity:
            neighbors.add(h)

    total_degree = in_degree + out_degree
    unique_relations = len(set(r for h, r, t in triples))
    neighbor_count = len(neighbors)

    # 邻居类型信息
    labeled_neighbors = sum(1 for n in neighbors if n in entity_to_true_type)
    neighbor_types = [entity_to_true_type[n] for n in neighbors if n in entity_to_true_type]
    unique_neighbor_types = len(set(neighbor_types))

    most_common_neighbor_type = 0
    if neighbor_types:
        type_counts = Counter(neighbor_types)
        most_common = type_counts.most_common(1)[0][0]
        most_common_neighbor_type = type_to_idx.get(most_common, 0)

    base_feat = [
        has_label,
        float(in_degree),
        float(out_degree),
        float(total_degree),
        float(unique_relations),
        float(neighbor_count),
        float(labeled_neighbors),
        float(unique_neighbor_types),
        float(most_common_neighbor_type)
    ]

    # 2. 关系模式特征
    if top_relations:
        rel_pattern_feat = np.zeros(len(top_relations) * 2, dtype=np.float32)
        # 添加一些随机分布
        np.random.seed(int(hashlib.md5(entity.encode()).hexdigest(), 16) % 10000)
        rel_pattern_feat = np.random.random(len(rel_pattern_feat)) * 0.1
    else:
        rel_pattern_feat = np.array([], dtype=np.float32)

    # 3. 邻居类型特征
    if all_types:
        neighbor_type_feat = np.zeros(len(all_types), dtype=np.float32)
        if neighbor_types:
            for ntype in neighbor_types[:3]:  # 只考虑前3个邻居
                if ntype in type_to_idx:
                    type_idx = type_to_idx[ntype]
                    neighbor_type_feat[type_idx] = 0.3  # 固定值
    else:
        neighbor_type_feat = np.array([], dtype=np.float32)

    # 4. 图像特征
    img_features = extract_image_features_for_entity(entity)

    # 组合所有特征
    all_feat = np.concatenate([
        base_feat,
        rel_pattern_feat,
        neighbor_type_feat,
        img_features
    ])

    # 确保维度正确
    if len(all_feat) > feature_dim:
        all_feat = all_feat[:feature_dim]
    elif len(all_feat) < feature_dim:
        padded = np.zeros(feature_dim, dtype=np.float32)
        padded[:len(all_feat)] = all_feat
        all_feat = padded

    return all_feat


# ========== 加载模型和相关数据 ==========
def load_sage_model(model_path):
    """加载训练好的GraphSAGE模型"""
    print(f"加载GraphSAGE模型: {model_path}")

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
        print(f"创建 {model_config['num_layers']} 层GraphSAGE模型...")

        model = TypeAwareGraphSAGE(
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
            model = TypeAwareGraphSAGE(
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
    scaler = checkpoint.get('scaler')
    top_relations = checkpoint.get('top_relations', [])
    node_features = checkpoint.get('node_features')
    type_to_idx = checkpoint.get('type_to_idx', {})
    all_types = checkpoint.get('all_types', [])
    is_multimodal = checkpoint.get('is_multimodal', False)

    print(f"\n模型配置:")
    print(f"  模型类型: {model_config.get('model_type', 'TypeAwareGraphSAGE')}")
    print(f"  输入特征: {model_config['in_feats']}")
    print(f"  隐藏层: {model_config['h_feats']}")
    print(f"  类别数: {model_config['num_classes']}")
    print(f"  层数: {model_config['num_layers']}")
    print(f"  实体数: {len(entity_to_idx)}")
    print(f"  特征数量: {len(feature_names)}")
    print(f"  关系类型数: {len(top_relations)}")
    print(f"  实体类型数: {len(all_types)}")
    print(f"  是否多模态: {is_multimodal}")
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
        'type_to_idx': type_to_idx,
        'all_types': all_types,
        'is_multimodal': is_multimodal,
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


# ========== 创建测试特征（匹配GraphSAGE训练特征） ==========
def create_test_features_for_sage(valid_entities, model_data, entity_relations, entity_to_true_type):
    """为测试实体创建特征 - 直接使用训练时保存的特征"""
    print(f"\n为GraphSAGE模型创建测试特征...")

    entity_to_idx = model_data['entity_to_idx']
    scaler = model_data.get('scaler')
    saved_features = model_data.get('node_features')

    feature_dim = model_data['model_config']['in_feats']
    print(f"模型期望特征维度: {feature_dim}")

    features = []
    entity_order = []

    for entity in tqdm(valid_entities, desc="获取特征"):
        entity_order.append(entity)

        if entity in entity_to_idx and saved_features is not None:
            idx = entity_to_idx[entity]
            if idx < saved_features.shape[0]:
                # 直接使用保存的特征
                feat_vector = saved_features[idx].numpy()
                if len(feat_vector) != feature_dim:
                    print(f"警告: 实体 {entity} 的特征维度 {len(feat_vector)} 不匹配 {feature_dim}")
                    # 填充0值到正确维度
                    if len(feat_vector) < feature_dim:
                        padded = np.zeros(feature_dim, dtype=np.float32)
                        padded[:len(feat_vector)] = feat_vector
                        feat_vector = padded
                    else:
                        feat_vector = feat_vector[:feature_dim]
            else:
                print(f"警告: 实体 {entity} 索引 {idx} 超出范围")
                feat_vector = np.zeros(feature_dim, dtype=np.float32)
        else:
            print(f"警告: 实体 {entity} 不在保存的特征中")
            feat_vector = np.zeros(feature_dim, dtype=np.float32)

        features.append(feat_vector)

    features_np = np.array(features, dtype=np.float32)
    print(f"特征矩阵形状: {features_np.shape}")

    # 注意：由于特征已经是训练时标准化过的，这里不应该再次标准化
    # 除非模型保存的是未标准化的原始特征

    features_tensor = torch.tensor(features_np, dtype=torch.float32)

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
    if predictions:
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


# ========== 保存结果 ==========
def save_results(results, accuracy, model_info, target_entities, output_dir='test_results'):
    """保存评估结果"""
    print(f"\n保存评估结果...")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'graphsage_results_{timestamp}.csv')

    if results:
        df = pd.DataFrame(results)

        columns_order = ['entity', 'true_type', 'predicted_type', 'is_correct', 'confidence']
        extra_cols = [col for col in df.columns if col not in columns_order]
        df = df[columns_order + extra_cols]

        df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✓ 详细结果已保存到: {results_file}")

    summary_file = os.path.join(output_dir, f'graphsage_summary_{timestamp}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("GraphSAGE实体类型预测模型测试结果汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"模型信息:\n")
        f.write(f"  模型类型: {model_info['model_config'].get('model_type', 'TypeAwareGraphSAGE')}\n")
        f.write(f"  输入特征: {model_info['model_config']['in_feats']}\n")
        f.write(f"  隐藏层: {model_info['model_config']['h_feats']}\n")
        f.write(f"  类别数: {model_info['model_config']['num_classes']}\n")
        f.write(f"  层数: {model_info['model_config']['num_layers']}\n")
        f.write(f"  是否多模态: {model_info.get('is_multimodal', False)}\n")
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


# ========== 分析预测结果 ==========
def analyze_predictions(results):
    """分析预测结果的详细信息"""
    if not results:
        return

    print(f"\n预测结果分析:")

    # 置信度分析
    confidences = [r['confidence'] for r in results]
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]

    print(f"  平均置信度: {np.mean(confidences):.4f}")
    print(f"  正确预测平均置信度: {np.mean(correct_confidences) if correct_confidences else 0:.4f}")
    print(f"  错误预测平均置信度: {np.mean(incorrect_confidences) if incorrect_confidences else 0:.4f}")

    # 类型分布
    true_types = [r['true_type'] for r in results]
    type_counts = Counter(true_types)
    print(f"  涉及实体类型数: {len(type_counts)}")

    # 计算每个类型的准确率
    type_accuracies = {}
    for type_id in type_counts:
        type_results = [r for r in results if r['true_type'] == type_id]
        accuracy = sum(1 for r in type_results if r['is_correct']) / len(type_results)
        type_accuracies[type_id] = accuracy

    # 显示准确率最高和最低的类型
    sorted_types = sorted(type_accuracies.items(), key=lambda x: x[1], reverse=True)
    print(f"  准确率最高的类型 (前3):")
    for type_id, acc in sorted_types[:3]:
        print(f"    类型 {type_id}: {acc:.2%} ({type_counts[type_id]}个样本)")

    if len(sorted_types) > 3:
        print(f"  准确率最低的类型 (后3):")
        for type_id, acc in sorted_types[-3:]:
            print(f"    类型 {type_id}: {acc:.2%} ({type_counts[type_id]}个样本)")


# ========== 主测试函数 ==========
def test_graphsage_model():
    """GraphSAGE模型测试主函数"""
    print("=" * 80)
    print("GraphSAGE实体类型预测模型测试（多模态版）")
    print("=" * 80)

    model_path = 'models/entity_type_predictor_sage.pth'
    test_file_path = 'data/FB15KET/TEST_PART_DETAILED.txt'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'
    output_dir = 'test_results'

    try:
        # 步骤1: 加载模型
        print("\n1. 加载GraphSAGE模型...")
        model_data = load_sage_model(model_path)
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

        # 步骤5: 创建测试特征（GraphSAGE专用，包含图像特征）
        print("\n5. 创建测试特征（多模态）...")
        features, entity_order = create_test_features_for_sage(
            valid_entities, model_data, entity_relations, entity_to_true_type
        )

        if features is None:
            print("特征创建失败，退出测试")
            return

        # 显示特征可视化信息
        print(f"\n特征矩阵分析:")
        print(f"  维度: {features.shape[1]}维")
        print(f"  样本: {features.shape[0]}个实体")

        # 分析图像特征部分
        if model_data.get('is_multimodal', False) and features.shape[1] >= 10:
            img_slice = features[:, -10:]
            img_present = (img_slice[:, 6] > 0).sum().item()  # has_images_flag列
            print(f"  图像特征统计:")
            print(f"    包含图像的实体: {img_present}/{len(entity_order)} ({img_present / len(entity_order):.1%})")
            print(f"    平均图像数量: {img_slice[:, 0].mean():.1f}")

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

            # 步骤8: 分析结果
            analyze_predictions(results)

            # 步骤9: 保存结果
            print("\n8. 保存结果...")
            results_file, summary_file = save_results(results, accuracy, model_data, target_entities, output_dir)

            print(f"\n" + "=" * 80)
            print("测试完成!")
            print("=" * 80)
            print(f"目标实体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # 显示预测示例
            if results:
                print(f"\n预测示例:")
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


# ========== 单个实体测试函数 ==========
def test_single_entity_sage():
    """测试单个实体 - GraphSAGE版本"""
    print("=" * 80)
    print("GraphSAGE模型 - 单个实体测试（多模态版）")
    print("=" * 80)

    model_path = 'models/entity_type_predictor_sage.pth'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'

    try:
        # 加载模型
        print("\n1. 加载GraphSAGE模型...")
        model_data = load_sage_model(model_path)
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

                # 检查是否是图像特征
                if model_data.get('is_multimodal', False):
                    print("使用多模态特征（包含图像统计特征）")
            else:
                print("警告: 无法获取实体特征")
                return

            with torch.no_grad():
                logits = model(g, features)
                probs = F.softmax(logits[0], dim=0)

                # 获取top-5预测
                top_probs, top_indices = torch.topk(probs, 5)

                print(f"\nGraphSAGE预测结果:")
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


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("GraphSAGE实体类型预测模型测试工具（多模态版）")
    print("=" * 80)

    print("\n测试模式:")
    print("1. 完整测试 (使用TEST_PART_DETAILED.txt测试100个目标实体)")
    print("2. 随机单个实体测试")

    try:
        choice = input("请输入选项 (1-2): ").strip()

        if choice == '1':
            test_graphsage_model()
        elif choice == '2':
            test_single_entity_sage()
        else:
            print("无效选项，使用默认完整测试")
            test_graphsage_model()

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()