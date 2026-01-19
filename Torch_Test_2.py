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
import json
import os
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')


# ========== 定义与训练完全相同的ResGCN模型 ==========
class ResGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=4, dropout=0.3):
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


# ========== 加载模型和相关数据 ==========
def load_resgcn_model(model_path):
    """加载ResGCN模型和相关数据"""
    print(f"加载ResGCN模型: {model_path}")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型文件失败: {e}")
        return None

    # 检查模型配置
    if 'model_config' not in checkpoint:
        print("错误: 模型文件格式不正确，缺少model_config")
        return None

    model_config = checkpoint['model_config']
    state_dict = checkpoint['model_state_dict']

    # 创建模型
    try:
        print(f"创建 {model_config['num_layers']} 层ResGCN模型...")
        model = ResGCN(
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
        return None

    # 提取其他必要数据
    entity_to_idx = checkpoint.get('entity_to_idx', {})
    idx_to_entity = checkpoint.get('idx_to_entity', {})
    label_encoder = checkpoint.get('label_encoder')
    feature_names = checkpoint.get('feature_names', [])
    scaler = checkpoint.get('scaler')
    neighbor_type_stats = checkpoint.get('neighbor_type_stats', {})
    node_features = checkpoint.get('node_features')

    print(f"\n模型配置:")
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
        'neighbor_type_stats': neighbor_type_stats,
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

    # 创建实体到类型的映射
    entity_to_true_type = {}
    entity_to_type_name = {}

    if 'entity_id' in entity_types.columns and 'predicted_category' in entity_types.columns:
        entity_to_true_type = dict(zip(
            entity_types['entity_id'],
            entity_types['predicted_category']
        ))
    else:
        print("警告: CSV文件中缺少必要的列 'entity_id' 或 'predicted_category'")

    # 创建实体到类型名称的映射（如果有）
    if 'predicted_category_name' in entity_types.columns:
        entity_to_type_name = dict(zip(
            entity_types['entity_id'],
            entity_types['predicted_category_name']
        ))

    print(f"加载了 {len(entity_to_true_type)} 个实体的真实标签")

    return entity_to_true_type, entity_to_type_name


# ========== 解析测试文件 ==========
def parse_test_file_detailed(test_file_path):
    """解析详细的测试文件格式，只提取标题行的实体作为目标实体"""
    print(f"\n解析测试文件: {test_file_path}")

    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return set(), {}

    target_entities = set()  # 只包含100个目标实体
    all_entities_in_file = set()  # 文件中出现的所有实体
    entity_relations = defaultdict(list)

    with open(test_file_path, 'r', encoding='utf-8') as f:
        current_entity = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检查是否是实体标题 - 只提取这个实体作为目标实体
            if line.startswith('实体:'):
                # 提取实体ID
                match = re.search(r'实体:\s*([^\s(]+)', line)
                if match:
                    current_entity = match.group(1)
                    target_entities.add(current_entity)
                continue

            # 检查是否是分隔线
            if line.startswith('---') or line.startswith('=='):
                continue

            # 检查是否是注释
            if line.startswith('#'):
                continue

            # 处理三元组数据
            parts = line.split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                all_entities_in_file.update([head, tail])

                # 只添加与当前目标实体相关的三元组
                if current_entity:
                    if head == current_entity or tail == current_entity:
                        entity_relations[current_entity].append((head, relation, tail))

    print(f"解析结果:")
    print(f"  目标测试实体: {len(target_entities)} 个 (应该是100个)")
    print(f"  文件中出现的所有实体: {len(all_entities_in_file)} 个")

    # 验证数量
    if len(target_entities) != 100:
        print(f"警告: 期望100个测试实体，但找到了 {len(target_entities)} 个")

    return target_entities, entity_relations


# ========== 构建测试子图 ==========
def build_test_graph(target_entities, entity_relations, entity_to_idx):
    """为测试实体构建子图 - 修复版本"""
    print(f"\n为测试实体构建子图...")

    # 收集所有相关的实体（包括目标实体及其邻居）
    all_entities_in_test = set()
    for entity in target_entities:
        all_entities_in_test.add(entity)
        triples = entity_relations.get(entity, [])
        for head, rel, tail in triples:
            all_entities_in_test.add(head)
            all_entities_in_test.add(tail)

    print(f"测试子图中包含 {len(all_entities_in_test)} 个实体")

    # 筛选出在模型索引中的实体
    valid_entities = [e for e in all_entities_in_test if e in entity_to_idx]
    print(f"其中 {len(valid_entities)} 个实体在模型索引中")

    # 特别检查目标实体是否在valid_entities中
    target_in_valid = [e for e in target_entities if e in valid_entities]
    print(f"目标实体在valid_entities中的数量: {len(target_in_valid)} / {len(target_entities)}")

    if len(target_in_valid) == 0:
        print("严重错误: 没有目标实体在模型索引中!")
        print("检查目标实体示例:")
        for i, entity in enumerate(list(target_entities)[:10]):
            in_idx = entity in entity_to_idx
            print(f"  {i + 1}. {entity}: 在entity_to_idx中 = {in_idx}")
        return None, None, None

    if len(valid_entities) == 0:
        print("错误: 没有测试实体在模型索引中!")
        return None, None, None

    # 创建实体到子图索引的映射
    subgraph_entity_to_idx = {entity: idx for idx, entity in enumerate(valid_entities)}

    # 构建边
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
        print("警告: 没有找到有效的边，创建自环图")
        src_nodes = list(range(len(valid_entities)))
        dst_nodes = list(range(len(valid_entities)))

    # 创建DGL图
    num_nodes = len(valid_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    # 添加自环
    g = dgl.add_self_loop(g)

    print(f"测试子图构建完成: {g.num_nodes()} 个节点, {g.num_edges()} 条边")

    return g, subgraph_entity_to_idx, valid_entities


def predict_entities_for_new_entities(model, g, features, entity_order, label_encoder, target_entities):
    """专门为目标实体进行预测"""
    print(f"\n为目标实体进行预测...")

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits, dim=1)

    predictions = {}

    for i, entity in enumerate(entity_order):
        # 只保存目标实体的预测结果
        if entity in target_entities:
            entity_probs = probs[i]

            # 获取top-3预测
            top_probs, top_indices = torch.topk(entity_probs, 3)

            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                pred_class = idx.item()

                # 解码类型
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

            # 主要预测结果
            main_pred = top_predictions[0]

            predictions[entity] = {
                'predicted_type': main_pred['type'],
                'predicted_class': main_pred['class'],
                'confidence': main_pred['probability'],
                'top3': top_predictions
            }

    print(f"完成了 {len(predictions)} 个目标实体的预测")
    print(f"成功预测的目标实体: {list(predictions.keys())[:10]}...")  # 显示前10个

    return predictions

def handle_new_entities(target_entities, entity_to_idx):
    """处理不在训练集中的新实体"""
    print(f"\n处理新实体...")

    new_entities = [e for e in target_entities if e not in entity_to_idx]
    existing_entities = [e for e in target_entities if e in entity_to_idx]

    print(f"目标实体统计:")
    print(f"  在训练集中的实体: {len(existing_entities)}")
    print(f"  新实体（不在训练集中）: {len(new_entities)}")

    if new_entities:
        print(f"\n新实体示例 (前5个):")
        for i, entity in enumerate(new_entities[:5]):
            print(f"  {i + 1}. {entity}")

    # 对于新实体，我们需要创建虚拟的entity_to_idx映射
    # 扩展entity_to_idx来包含新实体
    extended_entity_to_idx = entity_to_idx.copy()
    next_idx = len(entity_to_idx)

    for entity in new_entities:
        if entity not in extended_entity_to_idx:
            extended_entity_to_idx[entity] = next_idx
            next_idx += 1

    print(f"扩展后的实体索引包含 {len(extended_entity_to_idx)} 个实体")

    return extended_entity_to_idx, new_entities, existing_entities

# ========== 创建测试特征 ==========
def create_test_features(valid_entities, model_data, entity_relations, entity_to_true_type):
    """为测试实体创建特征（包含邻居类型信息）"""
    print(f"\n为测试实体创建特征...")

    # 从模型数据中获取必要信息
    entity_to_idx = model_data['entity_to_idx']
    feature_names = model_data.get('feature_names', [])
    scaler = model_data.get('scaler')
    saved_features = model_data.get('node_features')
    neighbor_type_stats = model_data.get('neighbor_type_stats', {})

    # 获取特征维度
    feature_dim = model_data['model_config']['in_feats']

    # 创建特征矩阵
    features = []
    entity_order = []

    for entity in tqdm(valid_entities, desc="创建特征"):
        entity_order.append(entity)

        # 初始化特征向量
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

        # 否则，基于关系和邻居信息创建特征
        else:
            triples = entity_relations.get(entity, [])

            # 1. 基础特征
            if feature_dim >= 1:
                feat_vector[0] = 1.0  # 表示有标签（测试时）

            if feature_dim >= 2:
                # 总度数
                total_degree = len(triples)
                feat_vector[1] = float(total_degree)

            # 2. 邻居类型统计特征（关键！）
            # 计算邻居
            neighbors = []
            for h, r, t in triples:
                neighbor = t if h == entity else h
                neighbors.append(neighbor)

            # 统计邻居类型
            neighbor_types = []
            for neighbor in neighbors:
                if neighbor in entity_to_true_type:
                    neighbor_types.append(entity_to_true_type[neighbor])

            # 填充邻居特征
            if feature_dim >= 3:
                # 邻居总数
                feat_vector[2] = float(len(neighbors))

            if feature_dim >= 4:
                # 有标签的邻居数
                feat_vector[3] = float(len(neighbor_types))

            if feature_dim >= 5:
                # 不同邻居类型数
                unique_types = len(set(neighbor_types)) if neighbor_types else 0
                feat_vector[4] = float(unique_types)

            if feature_dim >= 6 and neighbor_types:
                # 最常见的邻居类型
                type_counts = Counter(neighbor_types)
                most_common = type_counts.most_common(1)[0][0]
                feat_vector[5] = float(most_common)

        features.append(feat_vector)

    features_np = np.array(features, dtype=np.float32)

    # 如果存在scaler，进行标准化
    if scaler is not None:
        print("标准化特征...")
        features_np = scaler.transform(features_np)

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    print(f"特征矩阵形状: {features_tensor.shape}")

    return features_tensor, entity_order


# ========== 进行预测 ==========
def predict_entities(model, g, features, entity_order, label_encoder):
    """使用ResGCN模型预测实体类型"""
    print(f"\n使用ResGCN进行预测...")

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits, dim=1)

    predictions = {}

    for i, entity in tqdm(enumerate(entity_order), desc="预测实体", total=len(entity_order)):
        entity_probs = probs[i]

        # 获取top-3预测
        top_probs, top_indices = torch.topk(entity_probs, 3)

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            pred_class = idx.item()

            # 解码类型
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

        # 主要预测结果
        main_pred = top_predictions[0]

        predictions[entity] = {
            'predicted_type': main_pred['type'],
            'predicted_class': main_pred['class'],
            'confidence': main_pred['probability'],
            'top3': top_predictions
        }

    print(f"完成了 {len(predictions)} 个实体的预测")

    return predictions


# ========== 评估预测结果 ==========
def evaluate_predictions(predictions, entity_to_true_type, target_entities=None):
    """评估预测结果"""
    print(f"\n评估预测结果...")

    results = []
    correct_count = 0
    total_count = 0

    # 确定要评估的实体
    if target_entities:
        print(f"目标实体总数: {len(target_entities)}")
        print(f"在predictions中的目标实体: {len([e for e in target_entities if e in predictions])}")
        print(f"有真实标签的目标实体: {len([e for e in target_entities if e in entity_to_true_type])}")

        # 找出既有预测又有真实标签的实体
        entities_to_evaluate = []
        for entity in target_entities:
            if entity in predictions and entity in entity_to_true_type:
                entities_to_evaluate.append(entity)

        print(f"可以评估的实体: {len(entities_to_evaluate)}")

    else:
        entities_to_evaluate = [e for e in predictions.keys() if e in entity_to_true_type]
        print(f"可以评估的实体: {len(entities_to_evaluate)}")

    if not entities_to_evaluate:
        print("错误: 没有找到可以评估的实体!")
        print("可能的原因:")
        print("  1. 目标实体没有真实标签")
        print("  2. 目标实体不在predictions中")
        print("  3. entity_to_true_type为空")
        return None, 0

    # 显示一些无法评估的实体示例
    if target_entities:
        missing_entities = [e for e in target_entities if e not in entity_to_true_type]
        if missing_entities:
            print(f"\n缺少真实标签的实体示例 (前5个):")
            for i, entity in enumerate(missing_entities[:5]):
                print(f"  {i + 1}. {entity}")

        missing_predictions = [e for e in target_entities if e not in predictions]
        if missing_predictions:
            print(f"\n缺少预测结果的实体示例 (前5个):")
            for i, entity in enumerate(missing_predictions[:5]):
                print(f"  {i + 1}. {entity}")

    for entity in entities_to_evaluate:
        pred_info = predictions[entity]
        true_type = entity_to_true_type[entity]
        pred_type = pred_info['predicted_type']

        # 转换为字符串比较
        is_correct = (str(pred_type) == str(true_type))

        result = {
            'entity': entity,
            'true_type': true_type,
            'predicted_type': pred_type,
            'is_correct': is_correct,
            'confidence': pred_info['confidence'],
            'is_target_entity': target_entities is not None and entity in target_entities
        }

        # 添加top-2和top-3预测
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

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\n评估结果:")
    if target_entities:
        print(f"  目标实体总数: {len(target_entities)}")
        print(f"  可以评估的目标实体: {total_count}")

    print(f"  正确预测数: {correct_count}")
    print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 计算F1分数
    if total_count > 0:
        true_labels = [str(r['true_type']) for r in results]
        pred_labels = [str(r['predicted_type']) for r in results]

        print(f"  加权F1分数: {f1_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}")

    return results, accuracy


# ========== 分析预测结果 ==========
def analyze_predictions(results, target_entities):
    """分析预测结果"""
    print(f"\n分析预测结果...")

    if not results:
        print("没有结果可以分析")
        return

    df = pd.DataFrame(results)

    # 1. 整体统计
    print(f"\n1. 整体统计:")
    print(f"   总预测数: {len(df)}")
    print(f"   准确率: {df['is_correct'].mean():.4f} ({df['is_correct'].mean() * 100:.2f}%)")

    # 2. 置信度分析
    print(f"\n2. 置信度分析:")
    print(f"   平均置信度: {df['confidence'].mean():.4f}")
    print(f"   正确预测的平均置信度: {df[df['is_correct']]['confidence'].mean():.4f}")
    print(f"   错误预测的平均置信度: {df[~df['is_correct']]['confidence'].mean():.4f}")

    # 3. 置信度分布
    print(f"\n3. 置信度分布:")
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        count = len(df[(df['confidence'] >= low) & (df['confidence'] < high)])
        percentage = count / len(df) * 100
        print(f"   [{low:.1f}-{high:.1f}): {count} 个 ({percentage:.1f}%)")

    # 4. 类型分布
    print(f"\n4. 类型分布:")
    type_counts = df['true_type'].value_counts()
    print(f"   总类型数: {len(type_counts)}")
    print(f"   最常见的5个类型:")
    for i, (type_id, count) in enumerate(type_counts.head().items()):
        print(f"     {i + 1}. 类型 {type_id}: {count} 个")

    return df


# ========== 保存结果 ==========
def save_results(results, accuracy, model_info, target_entities, output_dir='test_results'):
    """保存评估结果"""
    print(f"\n保存评估结果...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果到CSV
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'resgcn_results_{timestamp}.csv')

    if results:
        df = pd.DataFrame(results)

        # 重新排序列
        columns_order = ['entity', 'true_type', 'predicted_type', 'is_correct', 'confidence']
        extra_cols = [col for col in df.columns if col not in columns_order]
        df = df[columns_order + extra_cols]

        # 保存到CSV
        df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✓ 详细结果已保存到: {results_file}")

    # 保存汇总报告
    summary_file = os.path.join(output_dir, f'resgcn_summary_{timestamp}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ResGCN模型测试结果汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"模型信息:\n")
        f.write(f"  模型类型: {model_info['model_config'].get('model_type', 'ResGCN')}\n")
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

            # 按正确性统计
            correct_count = sum(1 for r in results if r['is_correct'])
            incorrect_count = len(results) - correct_count

            f.write(f"  正确预测: {correct_count} 个 ({correct_count / len(results) * 100:.1f}%)\n")
            f.write(f"  错误预测: {incorrect_count} 个 ({incorrect_count / len(results) * 100:.1f}%)\n\n")

            # 显示部分错误预测
            incorrect_results = [r for r in results if not r['is_correct']]
            if incorrect_results:
                f.write("错误预测示例 (前10个):\n")
                f.write("-" * 60 + "\n")
                for i, r in enumerate(incorrect_results[:10]):
                    f.write(f"{i + 1}. 实体: {r['entity']}\n")
                    f.write(f"   真实类型: {r['true_type']}\n")
                    f.write(f"   预测类型: {r['predicted_type']}\n")
                    f.write(f"   置信度: {r['confidence']:.4f}\n")
                    if 'top2_type' in r:
                        f.write(f"   第二预测: {r['top2_type']} ({r['top2_prob']:.4f})\n")
                    f.write("\n")

    print(f"✓ 汇总报告已保存到: {summary_file}")

    return results_file, summary_file


def filter_correct_predictions(test_file_path, predictions, entity_to_true_type, output_file_path):
    """
    过滤出预测正确的实体及其关系，保存到新文件

    Args:
        test_file_path: 原始测试文件路径
        predictions: 预测结果字典 {entity: {predicted_type: ..., confidence: ...}}
        entity_to_true_type: 真实标签字典
        output_file_path: 输出文件路径
    """
    #print(f"\n过滤预测正确的实体...")

    # 找出预测正确的实体
    correct_entities = set()
    for entity, pred_info in predictions.items():
        if entity in entity_to_true_type:
            true_type = entity_to_true_type[entity]
            pred_type = pred_info['predicted_type']
            if str(pred_type) == str(true_type):
                correct_entities.add(entity)

    #print(f"预测正确的实体数: {len(correct_entities)}")
    #print(f"预测错误的实体数: {len(predictions) - len(correct_entities)}")

    # 读取原始文件
    with open(test_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 处理文件，只保留预测正确的实体及其关系
    output_lines = []
    current_entity = None
    keep_current_section = False
    in_entity_section = False

    for line in lines:
        line = line.strip()

        # 检查是否是实体标题行
        if line.startswith('实体:'):
            # 提取实体ID
            import re
            match = re.search(r'实体:\s*([^\s(]+)', line)
            if match:
                entity_id = match.group(1)
                current_entity = entity_id

                # 检查这个实体是否预测正确
                if entity_id in correct_entities:
                    keep_current_section = True
                    in_entity_section = True
                    output_lines.append(line)  # 保留标题行
                else:
                    keep_current_section = False
                    in_entity_section = False
                continue

        # 检查是否是分隔线（实体部分结束）
        if line.startswith('---') and len(line) > 10:  # 长分隔线
            if keep_current_section and in_entity_section:
                output_lines.append(line)  # 保留分隔线
            in_entity_section = False
            continue

        # 如果是注释或空行，保留格式
        if not line or line.startswith('#'):
            if keep_current_section:
                output_lines.append(line)
            continue

        # 处理三元组数据
        parts = line.split('\t')
        if len(parts) == 3:
            if keep_current_section:
                output_lines.append(line)
        else:
            # 其他格式的行
            if keep_current_section:
                output_lines.append(line)

    # 保存到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

    #print(f"预测正确的实体已保存到: {output_file_path}")

    # 统计信息
    original_entity_count = 0
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('实体:'):
                original_entity_count += 1

    #print(f"\n统计信息:")
    #print(f"  原始文件实体数: {original_entity_count}")
    #print(f"  预测正确实体数: {len(correct_entities)}")
    #print(f"  保留比例: {len(correct_entities) / original_entity_count * 100:.1f}%")

    return correct_entities, output_file_path


# ========== 主测试函数 ==========
def test_resgcn_model():
    """ResGCN模型测试主函数"""
    print("=" * 80)
    print("ResGCN模型测试")
    print("=" * 80)

    # 文件路径配置
    model_path = 'models/entity_type_predictor_resgcn_50.pth'  # 新训练的模型
    test_file_path = 'data/FB15KET/TEST_PART_DETAILED.txt'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'
    output_dir = 'test_results'

    try:
        # 步骤1: 加载模型
        print("\n1. 加载ResGCN模型...")
        model_data = load_resgcn_model(model_path)
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

        # 步骤4: 构建测试子图（处理新实体）
        print("\n4. 处理实体并构建测试子图...")

        # 先处理新实体
        extended_entity_to_idx, new_entities, existing_entities = handle_new_entities(
            target_entities, model_data['entity_to_idx']
        )

        # 如果所有目标实体都是新实体，模型可能无法有效预测
        if len(existing_entities) == 0:
            print("警告: 所有目标实体都是新实体（不在训练集中）")
            print("模型可能无法准确预测这些实体的类型")

        # 使用扩展后的entity_to_idx构建图
        g, subgraph_entity_to_idx, valid_entities = build_test_graph(
            target_entities, entity_relations, extended_entity_to_idx
        )
        # 步骤5: 创建测试特征
        print("\n5. 创建测试特征...")
        features, entity_order = create_test_features(
            valid_entities, model_data, entity_relations, entity_to_true_type
        )

        # 步骤6: 进行预测（专门为目标实体）
        print("\n6. 为目标实体进行预测...")
        predictions = predict_entities_for_new_entities(
            model_data['model'], g, features, entity_order,
            model_data['label_encoder'], target_entities
        )

        # 检查预测结果
        if not predictions:
            print("严重错误: 没有成功预测任何目标实体!")
            print("检查目标实体是否在entity_order中...")

            # 检查每个目标实体
            for entity in list(target_entities)[:10]:
                in_order = entity in entity_order
                in_idx = entity in model_data['entity_to_idx']
                print(f"  {entity}: 在entity_order中={in_order}, 在entity_to_idx中={in_idx}")

            return

        # 步骤7: 评估预测结果（只评估100个目标实体）
        # 在主测试函数中，在评估之前添加调试
        print("\n7. 评估目标实体预测结果...")
        if entity_to_true_type:
            print(f"调试信息:")
            print(f"  predictions中的实体数: {len(predictions)}")
            print(f"  entity_to_true_type中的实体数: {len(entity_to_true_type)}")
            print(f"  目标实体数: {len(target_entities)}")

            # 检查交集
            common_entities = set(predictions.keys()) & set(entity_to_true_type.keys()) & set(target_entities)
            print(f"  三者共有的实体数: {len(common_entities)}")

            if not common_entities:
                print("错误: 没有找到可以评估的实体!")
                print("可能原因分析:")

                # 检查哪些目标实体在predictions中
                in_predictions = [e for e in target_entities if e in predictions]
                print(f"  在predictions中的目标实体: {len(in_predictions)}")
                if in_predictions:
                    print(f"    示例: {in_predictions[:3]}")

                # 检查哪些目标实体在entity_to_true_type中
                in_ground_truth = [e for e in target_entities if e in entity_to_true_type]
                print(f"  有真实标签的目标实体: {len(in_ground_truth)}")
                if in_ground_truth:
                    print(f"    示例: {in_ground_truth[:3]}")

                return

            results, accuracy = evaluate_predictions(predictions, entity_to_true_type, target_entities)

            # ... 其余代码 ...
            if results is None:
                print("评估失败，退出测试")
                return

            # 步骤8: 分析结果
            print("\n8. 分析预测结果...")
            df = analyze_predictions(results, target_entities)

            # 步骤9: 保存结果
            print("\n9. 保存结果...")
            results_file, summary_file = save_results(results, accuracy, model_data, target_entities, output_dir)

            print(f"\n" + "=" * 80)
            print("测试完成!")
            print("=" * 80)
            print(f"目标实体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            #print(f"\n结果文件:")
            #print(f"  详细结果: {results_file}")
            #print(f"  汇总报告: {summary_file}")

            if results and accuracy > 0:
                #print(f"\n10. 过滤预测正确的实体...")

                # 过滤出预测正确的实体
                test_file_path = 'data/FB15KET/TEST_PART_DETAILED.txt'
                output_file_path = 'data/FB15KET/TEST_PART_DETAILED_TRUE.txt'

                correct_entities, output_path = filter_correct_predictions(
                    test_file_path, predictions, entity_to_true_type, output_file_path
                )

                # 可选：生成错误预测文件
                error_file_path = 'data/FB15KET/TEST_PART_DETAILED_FALSE.txt'

                # 找出预测错误的实体
                error_entities = set()
                for entity, pred_info in predictions.items():
                    if entity in entity_to_true_type:
                        true_type = entity_to_true_type[entity]
                        pred_type = pred_info['predicted_type']
                        if str(pred_type) != str(true_type):
                            error_entities.add(entity)

                # 保存错误预测的实体
                if error_entities:
                    with open(error_file_path, 'w', encoding='utf-8') as f:
                        f.write("# 预测错误的实体列表\n")
                        f.write("=" * 60 + "\n\n")
                        for entity in sorted(error_entities):
                            true_type = entity_to_true_type.get(entity, "未知")
                            pred_type = predictions[entity]['predicted_type']
                            confidence = predictions[entity]['confidence']
                            f.write(f"实体: {entity}\n")
                            f.write(f"真实类型: {true_type}\n")
                            f.write(f"预测类型: {pred_type}\n")
                            f.write(f"置信度: {confidence:.4f}\n")
                            f.write("-" * 40 + "\n\n")

                    #print(f"预测错误的实体列表已保存到: {error_file_path}")

                #print(f"\n过滤完成!")
                #print(f"  正确预测文件: {output_path}")
                #print(f"  包含 {len(correct_entities)} 个实体")


            # 显示预测示例
            #print(f"\n预测示例:")
            correct_examples = [r for r in results if r['is_correct']][:3]
            incorrect_examples = [r for r in results if not r['is_correct']][:3]
            '''
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
            '''
        else:
            print("\n警告: 没有真实标签数据，跳过评估步骤")
            print(f"完成了 {len(predictions)} 个实体的预测")

            # 显示一些预测结果
            print(f"\n预测示例 (前5个):")
            for i, (entity, pred) in enumerate(list(predictions.items())[:5]):
                print(f"  {i + 1}. {entity}: 预测类型={pred['predicted_type']}, 置信度={pred['confidence']:.3f}")

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 单个实体测试函数 ==========
def test_single_entity():
    """测试单个实体"""
    print("=" * 80)
    print("ResGCN模型 - 单个实体测试")
    print("=" * 80)

    model_path = 'models/entity_type_predictor_resgcn.pth'
    ground_truth_path = 'data/FB15KET/Entity_All_typed.csv'

    try:
        # 加载模型
        print("\n1. 加载ResGCN模型...")
        model_data = load_resgcn_model(model_path)
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

        # 需要实体关系信息来构建子图
        # 这里简化处理，直接使用模型中的特征
        if test_entity in model_data['entity_to_idx']:
            idx = model_data['entity_to_idx'][test_entity]

            # 使用模型进行预测
            model = model_data['model']
            model.eval()

            # 需要构建一个最小图
            num_nodes = len(model_data['entity_to_idx'])
            g = dgl.graph(([0], [0]), num_nodes=1)  # 单节点图
            g = dgl.add_self_loop(g)

            # 获取该实体的特征
            if model_data['node_features'] is not None and idx < model_data['node_features'].shape[0]:
                features = model_data['node_features'][idx:idx + 1]
            else:
                print("警告: 无法获取实体特征，使用随机特征")
                features = torch.randn(1, model_data['model_config']['in_feats'])

            with torch.no_grad():
                logits = model(g, features)
                probs = F.softmax(logits[0], dim=0)

                # 获取top-3预测
                top_probs, top_indices = torch.topk(probs, 3)

                print(f"\n预测结果:")
                print(f"{'排名':<6} {'类型':<15} {'置信度':<12} {'是否真实'}")
                print("-" * 50)

                for i, (prob, cls_idx) in enumerate(zip(top_probs, top_indices)):
                    pred_class = cls_idx.item()
                    pred_type = model_data['label_encoder'].inverse_transform([pred_class])[0]
                    prob_val = prob.item()

                    # 检查是否是真实类型
                    true_type = entity_to_true_type.get(test_entity, "未知")
                    try:
                        true_type_encoded = model_data['label_encoder'].transform([true_type])[0]
                        is_true = "✓" if pred_class == true_type_encoded else "✗"
                    except:
                        is_true = "?"

                    print(f"{i + 1:<6} {pred_type:<15} {prob_val:<12.2%} {is_true}")

                print(f"\n真实类型: {true_type}")

                # 显示邻居统计信息
                if 'neighbor_type_stats' in model_data and test_entity in model_data['neighbor_type_stats']:
                    stats = model_data['neighbor_type_stats'][test_entity]
                    print(f"\n邻居统计信息:")
                    print(f"  邻居总数: {stats.get('neighbor_count', 0)}")
                    print(f"  有标签的邻居数: {stats.get('labeled_neighbor_count', 0)}")
                    print(f"  不同邻居类型数: {stats.get('unique_neighbor_types', 0)}")
                    print(f"  最常见的邻居类型: {stats.get('most_common_neighbor_type', 0)}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 主程序入口 ==========
if __name__ == "__main__":
    import random

    print("ResGCN实体类型预测模型测试工具")
    print("=" * 80)

    print("\n选择测试模式:")
    print("1. 完整测试 (使用TEST_PART_DETAILED.txt测试100个目标实体)")
    print("2. 单个实体测试 (随机选择一个实体)")

    try:
        choice = input("请输入选项 (1-2): ").strip()

        if choice == '1':
            test_resgcn_model()
        elif choice == '2':
            test_single_entity()
        else:
            print("无效选项，使用默认完整测试")
            test_resgcn_model()

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()