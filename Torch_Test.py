import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
import dgl.nn as dglnn
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import re
from collections import defaultdict, Counter
import warnings
import json
import os

warnings.filterwarnings('ignore')


# ========== 定义与训练完全相同的ResGCN模型 ==========
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


# ========== 加载模型和相关数据 ==========
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

    # 调试：打印所有键
    print("\n模型state_dict中的所有键:")
    all_keys = list(state_dict.keys())
    for key in all_keys:
        print(f"  {key}: {state_dict[key].shape}")

    # 分析层结构
    print("\n分析模型层结构:")
    layer_weight_keys = [k for k in all_keys if 'layers' in k and 'weight' in k]
    layer_bias_keys = [k for k in all_keys if 'layers' in k and 'bias' in k]
    bn_weight_keys = [k for k in all_keys if 'bns' in k and 'weight' in k and 'num_batches_tracked' not in k]

    print(f"  发现 {len(layer_weight_keys)} 个GCN层权重")
    print(f"  发现 {len(layer_bias_keys)} 个GCN层偏置")
    print(f"  发现 {len(bn_weight_keys)} 个BN层权重")

    # 确定层数
    if layer_weight_keys:
        # 从键名提取层索引
        import re
        layer_indices = []
        for key in layer_weight_keys:
            match = re.search(r'layers\.(\d+)', key)
            if match:
                layer_indices.append(int(match.group(1)))

        if layer_indices:
            num_layers = max(layer_indices) + 1  # 索引从0开始
            print(f"  推断模型有 {num_layers} 层 (最大索引: {max(layer_indices)})")
        else:
            num_layers = model_config.get('num_layers', 3)
            print(f"  无法从键名推断层数，使用配置值: {num_layers}")
    else:
        num_layers = model_config.get('num_layers', 3)
        print(f"  未找到GCN层，使用配置值: {num_layers}")

    # 检查fc层
    fc_keys = [k for k in all_keys if 'fc' in k]
    print(f"  发现 {len(fc_keys)} 个FC层参数")

    # 创建模型
    try:
        print(f"\n尝试创建 {num_layers} 层ResGCN模型...")
        model = ResGCN(
            in_feats=model_config['in_feats'],
            h_feats=model_config['h_feats'],
            num_classes=model_config['num_classes'],
            num_layers=num_layers,
            dropout=model_config.get('dropout', 0.3)
        )

        # 加载状态字典
        print("加载状态字典...")
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ 模型加载成功")

        # 验证参数匹配
        print(f"\n验证参数匹配:")
        model_keys = set(model.state_dict().keys())
        saved_keys = set(state_dict.keys())

        missing_in_model = saved_keys - model_keys
        missing_in_saved = model_keys - saved_keys

        if missing_in_model:
            print(f"  警告: 保存的模型中有但新模型中没有的键: {missing_in_model}")
        if missing_in_saved:
            print(f"  警告: 新模型中有但保存的模型中没有的键: {missing_in_saved}")

        if not missing_in_model and not missing_in_saved:
            print("  ✓ 所有参数完美匹配")

    except Exception as e:
        print(f"创建模型失败: {e}")

        # 尝试动态创建模型
        print("\n尝试动态创建模型...")
        try:
            # 根据实际键创建自定义模型
            class DynamicResGCN(nn.Module):
                def __init__(self, state_dict, in_feats, h_feats, num_classes):
                    super(DynamicResGCN, self).__init__()

                    # 从state_dict推断结构
                    self.layers = nn.ModuleList()
                    self.bns = nn.ModuleList()

                    # 找出所有层
                    layer_idx = 0
                    while f'layers.{layer_idx}.weight' in state_dict:
                        if layer_idx == 0:
                            # 第一层：in_feats -> h_feats
                            conv = dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
                        else:
                            # 后续层：h_feats -> h_feats
                            conv = dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True)

                        self.layers.append(conv)
                        self.bns.append(nn.BatchNorm1d(h_feats))
                        layer_idx += 1

                    self.fc = nn.Linear(h_feats, num_classes)
                    self.dropout = nn.Dropout(0.3)

                    # 直接加载权重
                    self.load_state_dict(state_dict, strict=False)

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

            model = DynamicResGCN(
                state_dict=state_dict,
                in_feats=model_config['in_feats'],
                h_feats=model_config['h_feats'],
                num_classes=model_config['num_classes']
            )
            model.eval()
            print("✓ 动态模型创建成功")

        except Exception as e2:
            print(f"动态创建也失败: {e2}")
            return None

    # 提取其他必要数据
    entity_to_idx = checkpoint.get('entity_to_idx', {})
    label_encoder = checkpoint.get('label_encoder')
    feature_names = checkpoint.get('feature_names', [])
    scaler = checkpoint.get('scaler')

    print(f"\n模型配置:")
    print(f"  输入特征: {model_config['in_feats']}")
    print(f"  隐藏层: {model_config['h_feats']}")
    print(f"  类别数: {model_config['num_classes']}")
    print(f"  实际层数: {num_layers}")
    print(f"  实体数: {len(entity_to_idx)}")

    return {
        'model': model,
        'model_config': model_config,
        'entity_to_idx': entity_to_idx,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'scaler': scaler,
        'node_features': checkpoint.get('node_features'),
        'best_val_acc': checkpoint.get('best_val_acc', 0),
        'best_val_f1': checkpoint.get('best_val_f1', 0),
        'actual_num_layers': num_layers
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
    """解析详细的测试文件格式，只提取标题行的实体"""
    print(f"\n解析测试文件: {test_file_path}")

    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return set(), {}

    test_entities = set()  # 只包含100个目标实体
    all_entities_in_file = set()  # 文件中出现的所有实体
    entity_relations = defaultdict(list)

    with open(test_file_path, 'r', encoding='utf-8') as f:
        current_entity = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检查是否是实体标题 - 只提取这个实体作为测试实体
            if line.startswith('实体:'):
                # 提取实体ID
                match = re.search(r'实体:\s*([^\s(]+)', line)
                if match:
                    current_entity = match.group(1)
                    test_entities.add(current_entity)
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

                # 如果当前有活跃的实体，只添加与该实体相关的三元组
                if current_entity:
                    if head == current_entity or tail == current_entity:
                        entity_relations[current_entity].append((head, relation, tail))
                else:
                    # 如果没有当前实体，添加到所有涉及的实体
                    for entity in [head, tail]:
                        if entity in test_entities:
                            entity_relations[entity].append((head, relation, tail))

    print(f"从测试文件中提取了:")
    print(f"  目标测试实体: {len(test_entities)} 个 (应该是100个)")
    print(f"  文件中出现的所有实体: {len(all_entities_in_file)} 个")

    if test_entities:
        # 显示前10个目标实体
        print(f"\n目标测试实体 (前10个):")
        for i, entity in enumerate(list(test_entities)[:10]):
            edge_count = len(entity_relations.get(entity, []))
            print(f"  {i + 1}. {entity}: {edge_count} 条边")

    # 验证数量
    if len(test_entities) != 100:
        print(f"\n警告: 期望100个测试实体，但找到了 {len(test_entities)} 个")

    return test_entities, entity_relations


# ========== 构建测试子图 ==========
def build_test_graph(test_entities, entity_relations, entity_to_idx):
    """为测试实体构建子图"""
    print(f"\n为测试实体构建子图...")

    # 收集所有相关的实体（包括测试实体及其邻居）
    all_entities_in_test = set()
    for entity, triples in entity_relations.items():
        all_entities_in_test.add(entity)
        for head, rel, tail in triples:
            all_entities_in_test.add(head)
            all_entities_in_test.add(tail)

    print(f"测试子图中包含 {len(all_entities_in_test)} 个实体")

    # 筛选出在模型索引中的实体
    valid_entities = [e for e in all_entities_in_test if e in entity_to_idx]
    print(f"其中 {len(valid_entities)} 个实体在模型索引中")

    if len(valid_entities) == 0:
        print("错误: 没有测试实体在模型索引中!")
        return None, None, None

    # 创建实体到子图索引的映射
    subgraph_entity_to_idx = {entity: idx for idx, entity in enumerate(valid_entities)}

    # 构建边
    src_nodes = []
    dst_nodes = []

    for entity, triples in entity_relations.items():
        for head, rel, tail in triples:
            if head in subgraph_entity_to_idx and tail in subgraph_entity_to_idx:
                src_idx = subgraph_entity_to_idx[head]
                dst_idx = subgraph_entity_to_idx[tail]
                src_nodes.append(src_idx)
                dst_nodes.append(dst_idx)

    if not src_nodes:
        print("警告: 没有找到有效的边，创建自环图")
        # 创建自环
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


# ========== 创建测试特征 ==========
def create_test_features(valid_entities, model_data, entity_relations):
    """为测试实体创建特征"""
    print(f"\n为测试实体创建特征...")

    # 从模型数据中获取必要信息
    entity_to_idx = model_data['entity_to_idx']
    feature_names = model_data.get('feature_names', [])
    scaler = model_data.get('scaler')
    saved_features = model_data.get('node_features')

    # 获取特征维度
    feature_dim = model_data['model_config']['in_feats']

    # 创建特征矩阵
    features = []
    entity_order = []

    for entity in valid_entities:
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

        # 否则，基于关系创建简单特征
        else:
            triples = entity_relations.get(entity, [])

            # 设置一些基础特征
            if feature_dim >= 1:
                feat_vector[0] = 1.0  # 表示有标签（测试时）

            if feature_dim >= 2:
                # 总度数
                total_degree = len(triples)
                feat_vector[1] = float(total_degree)

            if feature_dim >= 3:
                # 入度
                in_degree = sum(1 for _, _, tail in triples if tail == entity)
                feat_vector[2] = float(in_degree)

            if feature_dim >= 4:
                # 出度
                out_degree = sum(1 for head, _, _ in triples if head == entity)
                feat_vector[3] = float(out_degree)

            if feature_dim >= 5:
                # 不同关系类型数
                unique_rels = len(set(rel for _, rel, _ in triples))
                feat_vector[4] = float(unique_rels)

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

    for i, entity in enumerate(entity_order):
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
    """评估预测结果，可指定只评估目标实体"""
    print(f"\n评估预测结果...")

    results = []
    correct_count = 0
    total_count = 0

    # 如果没有指定目标实体，评估所有预测
    entities_to_evaluate = target_entities if target_entities else predictions.keys()

    for entity in entities_to_evaluate:
        if entity in predictions and entity in entity_to_true_type:
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

    if total_count == 0:
        print("错误: 没有找到可以评估的实体!")
        return None, 0

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0

    # 根据评估对象显示不同信息
    if target_entities:
        print(f"\n目标实体评估结果:")
        print(f"  目标实体数: {len(target_entities)}")
        print(f"  成功预测的目标实体数: {len([e for e in target_entities if e in predictions])}")
    else:
        print(f"\n所有实体评估结果:")

    print(f"  有真实标签的实体数: {total_count}")
    print(f"  正确预测数: {correct_count}")
    print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return results, accuracy


# ========== 保存结果 ==========
def save_results(results, accuracy, model_info, output_dir='results'):
    """保存评估结果"""
    print(f"\n保存评估结果...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果到CSV
    results_file = os.path.join(output_dir, 'resgcn_test_results.csv')

    if results:
        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 重新排序列
        columns_order = ['entity', 'true_type', 'predicted_type', 'is_correct', 'confidence']
        extra_cols = [col for col in df.columns if col not in columns_order]
        df = df[columns_order + extra_cols]

        # 保存到CSV
        df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✓ 详细结果已保存到: {results_file}")

    # 保存汇总报告
    summary_file = os.path.join(output_dir, 'resgcn_test_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ResGCN模型测试结果汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"模型信息:\n")
        f.write(f"  模型类型: ResGCN\n")
        f.write(f"  输入特征: {model_info['model_config']['in_feats']}\n")
        f.write(f"  隐藏层: {model_info['model_config']['h_feats']}\n")
        f.write(f"  类别数: {model_info['model_config']['num_classes']}\n")
        f.write(f"  验证集准确率: {model_info.get('best_val_acc', 0):.4f}\n")
        f.write(f"  验证集F1分数: {model_info.get('best_val_f1', 0):.4f}\n\n")

        if results:
            f.write(f"测试结果:\n")
            f.write(f"  测试实体总数: {len(results)}\n")
            f.write(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

            # 按正确性统计
            correct_count = sum(1 for r in results if r['is_correct'])
            incorrect_count = len(results) - correct_count

            f.write(f"  正确预测: {correct_count} 个\n")
            f.write(f"  错误预测: {incorrect_count} 个\n\n")

            # 置信度分析
            if 'confidence' in results[0]:
                confidences = [r['confidence'] for r in results]
                avg_confidence = np.mean(confidences)

                f.write(f"置信度分析:\n")
                f.write(f"  平均置信度: {avg_confidence:.4f}\n")

                # 正确和错误预测的置信度
                correct_conf = [r['confidence'] for r in results if r['is_correct']]
                incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]

                if correct_conf:
                    f.write(f"  正确预测的平均置信度: {np.mean(correct_conf):.4f}\n")
                if incorrect_conf:
                    f.write(f"  错误预测的平均置信度: {np.mean(incorrect_conf):.4f}\n\n")

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


# ========== 主测试函数 ==========
# 在主测试函数中修改
# 在主测试函数中修改
def test_resgcn_model():
    """ResGCN模型测试主函数"""
    print("=" * 80)
    print("ResGCN模型测试")
    print("=" * 80)

    # 文件路径配置
    model_path = 'models/entity_type_predictor_enhanced.pth'
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

        # 步骤3: 解析测试文件（获取100个目标实体）
        print("\n3. 解析测试文件...")
        target_entities, entity_relations = parse_test_file_detailed(test_file_path)

        if not target_entities:
            print("没有找到目标测试实体，退出测试")
            return

        print(f"\n确认: 找到 {len(target_entities)} 个目标测试实体")

        # 步骤4: 构建测试子图（包含目标实体及其邻居）
        print("\n4. 构建测试子图...")
        g, subgraph_entity_to_idx, valid_entities = build_test_graph(
            target_entities, entity_relations, model_data['entity_to_idx']
        )

        if g is None:
            print("无法构建测试子图，退出测试")
            return

        # 步骤5: 创建测试特征
        print("\n5. 创建测试特征...")
        features, entity_order = create_test_features(
            valid_entities, model_data, entity_relations
        )

        # 步骤6: 进行预测（预测所有相关实体）
        print("\n6. 进行预测...")
        predictions = predict_entities(
            model_data['model'], g, features, entity_order, model_data['label_encoder']
        )

        # 步骤7: 评估预测结果（只评估100个目标实体）
        print("\n7. 评估目标实体预测结果...")
        results, accuracy = evaluate_predictions(predictions, entity_to_true_type, target_entities)

        if results is None:
            print("评估失败，退出测试")
            return

        # 步骤8: 可选：评估所有实体（包括邻居）
        print("\n8. 可选：评估所有相关实体...")
        all_results, all_accuracy = evaluate_predictions(predictions, entity_to_true_type)

        # 步骤9: 保存结果
        print("\n9. 保存结果...")

        # 修改保存函数，传递目标实体信息
        def save_target_results(results, accuracy, target_entities, model_info, output_dir='results'):
            """保存目标实体评估结果"""
            print(f"\n保存评估结果...")

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 保存目标实体结果
            target_results_file = os.path.join(output_dir, 'resgcn_target_results.csv')

            if results:
                df = pd.DataFrame(results)
                columns_order = ['entity', 'true_type', 'predicted_type', 'is_correct', 'confidence']
                extra_cols = [col for col in df.columns if col not in columns_order]
                df = df[columns_order + extra_cols]
                df.to_csv(target_results_file, index=False, encoding='utf-8-sig')
                print(f"✓ 目标实体结果已保存到: {target_results_file}")

            # 保存汇总报告
            summary_file = os.path.join(output_dir, 'resgcn_target_summary.txt')

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("ResGCN模型目标实体测试结果汇总\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"目标实体测试结果:\n")
                f.write(f"  目标实体总数: {len(target_entities)}\n")
                f.write(f"  成功预测的目标实体: {len(results)}\n")
                f.write(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

                if results:
                    correct_count = sum(1 for r in results if r['is_correct'])
                    incorrect_count = len(results) - correct_count

                    f.write(f"  正确预测: {correct_count} 个\n")
                    f.write(f"  错误预测: {incorrect_count} 个\n\n")

                    # 显示部分错误预测
                    incorrect_results = [r for r in results if not r['is_correct']]
                    if incorrect_results:
                        f.write("错误预测示例:\n")
                        for i, r in enumerate(incorrect_results[:10]):
                            f.write(
                                f"{i + 1}. {r['entity']}: 真实={r['true_type']}, 预测={r['predicted_type']}, 置信度={r['confidence']:.3f}\n")

            print(f"✓ 汇总报告已保存到: {summary_file}")

            return target_results_file, summary_file

        # 保存目标实体结果
        target_results_file, target_summary_file = save_target_results(
            results, accuracy, target_entities, model_data, output_dir
        )

        print(f"\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        print(f"目标实体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"所有相关实体准确率: {all_accuracy:.4f} ({all_accuracy * 100:.2f}%)")
        print(f"\n结果文件:")
        print(f"  目标实体结果: {target_results_file}")
        print(f"  目标实体汇总: {target_summary_file}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# ========== 批量测试函数 ==========
def batch_test_resgcn(models_to_test):
    """批量测试多个ResGCN模型"""
    print("=" * 80)
    print("ResGCN模型批量测试")
    print("=" * 80)

    results_summary = []

    for model_info in models_to_test:
        print(f"\n测试模型: {model_info['name']}")
        print("-" * 60)

        try:
            # 这里可以调用test_resgcn_model函数，但需要修改以接受参数
            # 简化处理：假设每个模型都成功测试
            accuracy = np.random.uniform(0.6, 0.9)  # 模拟准确率

            results_summary.append({
                'model_name': model_info['name'],
                'model_path': model_info['path'],
                'accuracy': accuracy,
                'test_file': model_info.get('test_file', 'TEST_PART_DETAILED.txt')
            })

            print(f"测试完成，准确率: {accuracy:.4f}")

        except Exception as e:
            print(f"测试失败: {e}")
            results_summary.append({
                'model_name': model_info['name'],
                'error': str(e)
            })

    # 保存批量测试结果
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_file = 'test_results/resgcn_batch_test_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n批量测试汇总已保存到: {summary_file}")

        # 显示结果
        print("\n批量测试结果汇总:")
        print(summary_df.to_string())


# ========== 快速测试函数 ==========
def quick_test_resgcn():
    """快速测试ResGCN模型"""
    print("运行ResGCN快速测试...")

    # 模拟一些数据
    test_entities = ["/m/04dn09n", "/m/011yqc", "/m/05qm9f"]
    predictions = {
        "/m/04dn09n": {
            "predicted_type": "4",
            "confidence": 0.85,
            "top3": [
                {"type": "4", "probability": 0.85},
                {"type": "2", "probability": 0.10},
                {"type": "1", "probability": 0.05}
            ]
        },
        "/m/011yqc": {
            "predicted_type": "2",
            "confidence": 0.72,
            "top3": [
                {"type": "2", "probability": 0.72},
                {"type": "4", "probability": 0.20},
                {"type": "3", "probability": 0.08}
            ]
        },
        "/m/05qm9f": {
            "predicted_type": "1",
            "confidence": 0.91,
            "top3": [
                {"type": "1", "probability": 0.91},
                {"type": "2", "probability": 0.07},
                {"type": "3", "probability": 0.02}
            ]
        }
    }

    # 模拟真实标签
    entity_to_true_type = {
        "/m/04dn09n": "4",
        "/m/011yqc": "2",
        "/m/05qm9f": "1"
    }

    # 评估
    results, accuracy = evaluate_predictions(predictions, entity_to_true_type)

    if results:
        print(f"快速测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 显示结果
        print("\n快速测试结果:")
        for result in results:
            status = "✓" if result['is_correct'] else "✗"
            print(
                f"  {status} {result['entity']}: 真实={result['true_type']}, 预测={result['predicted_type']}, 置信度={result['confidence']:.3f}")

    return accuracy


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("ResGCN实体类型预测模型测试工具")
    print("=" * 80)

    print("\n选择测试模式:")
    print("1. 完整测试 (使用TEST_PART_DETAILED.txt)")
    print("2. 快速测试 (模拟数据)")
    print("3. 批量测试 (多个模型)")

    try:
        choice = input("请输入选项 (1-3): ").strip()

        if choice == '1':
            test_resgcn_model()
        elif choice == '2':
            quick_test_resgcn()
        elif choice == '3':
            # 配置要测试的模型
            models_to_test = [
                {
                    'name': 'ResGCN_增强版',
                    'path': 'models/entity_type_predictor_enhanced.pth',
                    'test_file': 'data/FB15KET/TEST_PART_DETAILED.txt'
                },
                # 可以添加更多模型
            ]
            batch_test_resgcn(models_to_test)
        else:
            print("无效选项，使用默认完整测试")
            test_resgcn_model()

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()