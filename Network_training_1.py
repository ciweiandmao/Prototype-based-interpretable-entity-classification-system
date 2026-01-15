import math
import pandas as pd
import numpy as np
import torch
import dgl
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class EnhancedFeatureExtractor:
    """增强特征提取器"""

    def extract_structural_features(self, entity_id, triplets_by_entity):
        """提取结构特征"""
        if entity_id not in triplets_by_entity:
            return [0.0] * 5

        relations = triplets_by_entity[entity_id]

        # 1. 度中心性特征
        out_degree = sum(1 for d, _, _ in relations if d == 'out')
        in_degree = sum(1 for d, _, _ in relations if d == 'in')
        total_degree = len(relations)

        # 2. 关系多样性
        rel_types = set(rel for _, rel, _ in relations)
        rel_diversity = len(rel_types) / (total_degree + 1e-8)

        # 3. 邻居类别分布（简化）
        # 这里可以添加邻居的类别统计

        return [
            out_degree / 100.0,  # 归一化
            in_degree / 100.0,
            total_degree / 200.0,
            rel_diversity,
            len(rel_types) / 50.0  # 关系类型数量归一化
        ]

    def extract_text_features(self, entity_id, entity_info):
        """提取文本特征（如果有）"""
        # 如果有实体描述文本，可以使用BERT提取特征
        # 这里先返回空特征
        return [0.0] * 5  # 占位符


class DataAugmenter:
    """数据增强器"""

    def __init__(self, triplets_by_entity):
        self.triplets_by_entity = triplets_by_entity

    def augment_by_relation(self, entity_id, max_neighbors=5):
        """通过关系进行数据增强"""
        if entity_id not in self.triplets_by_entity:
            return []

        relations = self.triplets_by_entity[entity_id]

        # 找到与该实体有相同关系的其他实体
        augmented_samples = []

        for direction, rel, neighbor in relations:
            # 找到有相同关系的其他实体
            similar_entities = self.find_similar_entities(entity_id, rel, direction)

            for sim_entity in similar_entities[:max_neighbors]:
                augmented_samples.append({
                    'source': entity_id,
                    'target': sim_entity,
                    'relation': rel,
                    'direction': direction,
                    'type': 'relation_augment'
                })

        return augmented_samples

    def find_similar_entities(self, entity_id, relation, direction, max_results=10):
        """找到有相同关系的实体"""
        similar = []

        # 遍历所有实体，找到有相同关系模式的
        for other_entity, relations in self.triplets_by_entity.items():
            if other_entity == entity_id:
                continue

            # 检查是否有相同的关系
            has_same_rel = any(
                r[1] == relation and r[0] == direction
                for r in relations[:20]  # 只检查前20个关系
            )

            if has_same_rel:
                similar.append(other_entity)
                if len(similar) >= max_results:
                    break

        return similar


class EnsembleModel:
    """集成模型"""

    def __init__(self, model_configs, device='cuda'):
        self.models = []
        self.device = device

        for config in model_configs:
            model = EnhancedFB15KETXGradNet(**config)
            model = model.to(device)
            self.models.append(model)

    def train_ensemble(self, g, n_epochs=50):
        """训练集成模型"""
        for i, model in enumerate(self.models):
            print(f"\n训练第 {i + 1}/{len(self.models)} 个模型...")
            trainer = EnhancedTrainer(model, g, self.device)
            trainer.train(epochs=n_epochs)

            # 保存模型
            torch.save(model.state_dict(), f'ensemble_model_{i}.pth')

    def predict(self, features):
        """集成预测"""
        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                embeddings = model(features)
                probs, _, _ = model.classify(embeddings, torch.arange(features.size(0)).to(self.device))
                if probs is not None:
                    all_probs.append(probs)

        if all_probs:
            # 平均概率
            avg_probs = torch.stack(all_probs).mean(dim=0)
            _, predicted = torch.max(avg_probs, dim=1)
            return avg_probs, predicted
        else:
            return None, None


def improved_training_pipeline(g):
    """改进的训练流程"""
    print("=" * 80)
    print("FB15KET实体分类系统 - 改进训练流程")
    print("=" * 80)

    # 1. 构建增强特征
    print("\n[1] 构建增强特征...")
    enhanced_features = build_enhanced_features()

    # 2. 重新构建图（使用增强特征）
    print("\n[2] 重新构建图（增强特征）...")
    # 这里需要修改build_heterogeneous_graph函数以使用enhanced_features

    # 3. 训练增强模型
    print("\n[3] 训练增强模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 新特征维度 = 9（原始得分）+ 5（结构特征）= 14
    feature_dim = 14
    hidden_dim = 256
    out_dim = 128

    model = EnhancedFB15KETXGradNet(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_classes=9,
        num_prototypes_per_class=3,
        dropout_rate=0.5
    )

    trainer = EnhancedTrainer(model, g, device=device)

    # 训练更多epoch
    train_losses, val_accuracies = trainer.train(
        epochs=150,  # 更多epoch
        lr=0.001,
        weight_decay=1e-4,
        warmup_epochs=15,
        patience=40
    )

    # 4. 测试
    print("\n[4] 测试改进模型...")
    results = trainer.test(save_results=True)

    # 5. 分析
    print("\n[5] 性能分析...")
    analyze_results(results, train_losses, val_accuracies)

    return results


def analyze_results(results, train_losses, val_accuracies):
    """分析训练结果"""
    print("\n性能分析报告:")
    print("-" * 60)

    train_acc = results.get('train_acc', 0)
    val_acc = results.get('valid_acc', 0)
    test_acc = results.get('test_acc', 0)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 分析过拟合/欠拟合
    if train_acc > val_acc + 0.05:
        print("⚠️  可能存在过拟合 (训练集 >> 验证集)")
        print("   建议: 增加dropout, 数据增强, 早停")
    elif train_acc < val_acc:
        print("⚠️  可能存在欠拟合 (训练集 < 验证集)")
        print("   建议: 增加模型复杂度, 训练更多epoch, 数据增强")
    else:
        print("✓ 训练集和验证集性能平衡")

    # 泛化能力
    if abs(val_acc - test_acc) < 0.02:
        print("✓ 泛化能力良好 (验证集 ≈ 测试集)")
    else:
        print("⚠️  泛化能力有待提升")

    # 绝对性能
    if test_acc > 0.7:
        print("🎉 性能优秀 (>70%)")
    elif test_acc > 0.6:
        print("👍 性能良好 (60-70%)")
    elif test_acc > 0.5:
        print("👌 性能一般 (50-60%)")
    else:
        print("🔧 需要大幅改进 (<50%)")

def build_enhanced_features():
    """构建增强特征"""
    # 加载三元组数据
    triplets_by_entity = defaultdict(list)
    for file_name in ['train.txt', 'valid.txt', 'test.txt']:
        file_path = f'data/FB15KET/{file_name}'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        h, r, t = parts
                        triplets_by_entity[h].append(('out', r, t))
                        triplets_by_entity[t].append(('in', r, h))

    # 加载实体数据
    entity_df = pd.read_csv('data/FB15KET/Entity_All_typed.csv')

    # 创建特征提取器
    extractor = EnhancedFeatureExtractor()

    enhanced_features = {}
    for _, row in entity_df.iterrows():
        eid = row['entity_id']

        # 基本特征：9个类别得分
        base_features = [row[f'category_{i}_score'] for i in range(1, 10)]

        # 结构特征
        structural_features = extractor.extract_structural_features(eid, triplets_by_entity)

        # 组合特征
        all_features = base_features + structural_features

        enhanced_features[eid] = all_features

    return enhanced_features
class FB15KETDataLoader:
    def __init__(self, data_dir='data/FB15KET'):
        self.data_dir = data_dir
        self.entity_type_path = os.path.join(data_dir, 'Entity_All_typed.csv')
        self.train_path = os.path.join(data_dir, 'xunlian.txt')
        #self.valid_path = os.path.join(data_dir, 'valid.txt')
        #self.test_path = os.path.join(data_dir, 'test.txt')

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

    def analyze_data_quality(self):
        """分析数据质量和分布"""
        print("=" * 60)
        print("FB15KET 数据集质量分析")
        print("=" * 60)

        # 1. 分析实体类型文件
        print("\n1. 实体类型文件分析:")
        entity_df = pd.read_csv(self.entity_type_path)
        print(f"  实体总数: {len(entity_df)}")
        print(f"  列信息: {entity_df.columns.tolist()}")

        # 类别分布
        if 'predicted_category' in entity_df.columns:
            category_counts = entity_df['predicted_category'].value_counts()
            print(f"\n  类别分布:")
            for cat_id, count in category_counts.items():
                cat_name = self.category_names.get(int(cat_id), f"未知类别{cat_id}")
                print(f"    {cat_id}: {cat_name} - {count}个实体 ({count / len(entity_df) * 100:.2f}%)")

        # 2. 分析三元组文件
        print("\n2. 三元组文件分析:")
        file_info = []
        for file_name in ['xunlian.txt']:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                triplets = [line.strip().split('\t') for line in lines]
                unique_entities = set()
                unique_relations = set()
                for h, r, t in triplets:
                    unique_entities.add(h)
                    unique_entities.add(t)
                    unique_relations.add(r)

                file_info.append({
                    'file': file_name,
                    'triplets': len(triplets),
                    'entities': len(unique_entities),
                    'relations': len(unique_relations)
                })

        # 打印统计信息
        for info in file_info:
            print(f"  {info['file']}:")
            print(f"    三元组数量: {info['triplets']:,}")
            print(f"    唯一实体数: {info['entities']:,}")
            print(f"    唯一关系数: {info['relations']:,}")

        # 3. 合并分析
        print("\n3. 合并分析:")
        all_entities = set()
        all_relations = set()
        total_triplets = 0

        for file_name in ['xunlian.txt']:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = line.strip().split('\t')
                    all_entities.add(h)
                    all_entities.add(t)
                    all_relations.add(r)
                    total_triplets += 1

        print(f"  总三元组数: {total_triplets:,}")
        print(f"  总唯一实体数: {len(all_entities):,}")
        print(f"  总唯一关系数: {len(all_relations):,}")

        # 4. 检查实体类型覆盖
        print("\n4. 实体类型覆盖分析:")
        typed_entities = set(entity_df['entity_id'].unique())
        all_entity_ids = all_entities
        typed_count = len(typed_entities & all_entity_ids)
        untyped_count = len(all_entity_ids - typed_entities)

        print(f"  有类型标注的实体: {typed_count:,} ({typed_count / len(all_entity_ids) * 100:.2f}%)")
        print(f"  无类型标注的实体: {untyped_count:,} ({untyped_count / len(all_entity_ids) * 100:.2f}%)")

        # 5. 关系频率分析
        print("\n5. 关系频率分析 (Top 20):")
        relation_counts = Counter()
        for file_name in ['xunlian.txt']:  # 只分析训练集的关系分布
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    _, r, _ = line.strip().split('\t')
                    relation_counts[r] += 1

        print("  最常见的关系:")
        for i, (rel, count) in enumerate(relation_counts.most_common(20)):
            print(f"    {i + 1:2d}. {rel}: {count:,}")

        # 6. 可视化类别分布
        if 'predicted_category' in entity_df.columns:
            plt.figure(figsize=(12, 6))
            category_dist = entity_df['predicted_category'].value_counts().sort_index()
            categories = [f"{idx}\n{self.category_names.get(idx, '')[:10]}..." for idx in category_dist.index]

            plt.bar(range(len(categories)), category_dist.values)
            plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
            plt.title('实体类别分布')
            plt.xlabel('类别')
            plt.ylabel('实体数量')
            plt.tight_layout()
            plt.savefig('processed_data/category_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n  类别分布图已保存到: processed_data/category_distribution.png")

        # 7. 检查数据完整性
        print("\n6. 数据完整性检查:")
        missing_files = []
        for file_path in [self.entity_type_path, self.train_path]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f"  警告: 以下文件不存在: {missing_files}")
        else:
            print("  所有必需文件都存在")

        print("\n" + "=" * 60)

        # 返回统计信息
        return {
            'entity_count': len(all_entities),
            'relation_count': len(all_relations),
            'triplet_count': total_triplets,
            'typed_entity_count': typed_count,
            'untyped_entity_count': untyped_count,
            'category_distribution': category_counts if 'predicted_category' in entity_df.columns else None
        }


class FB15KETGraphBuilder:
    def __init__(self, data_dir='data/FB15KET'):
        self.data_dir = data_dir
        self.data_loader = FB15KETDataLoader(data_dir)

    def load_all_data(self):
        """加载所有数据（修复索引越界问题）"""
        print("正在加载数据...")

        try:
            # 1. 加载实体类型信息
            entity_df = pd.read_csv(os.path.join(self.data_dir, 'Entity_All_typed.csv'))

            # 提取类别得分作为特征，并确定主要类别
            score_cols = [f'category_{i}_score' for i in range(1, 10)]

            # 创建实体到特征的映射
            entity_features = {}
            entity_labels = {}

            for _, row in entity_df.iterrows():
                eid = row['entity_id']
                # 特征：9个类别的得分
                features = [float(row[col]) for col in score_cols]
                entity_features[eid] = features

                # 标签：得分最高的类别（1-9）
                if 'predicted_category' in row and not pd.isna(row['predicted_category']):
                    entity_labels[eid] = int(row['predicted_category'])
                else:
                    # 如果没有预测类别，使用得分最高的
                    scores = [float(row[col]) for col in score_cols]
                    entity_labels[eid] = np.argmax(scores) + 1

            # 2. 加载所有三元组
            all_triplets = []
            entity_set = set()
            relation_set = set()

            for file_name in ['xunlian.txt']:
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 3:
                                h, r, t = parts
                                all_triplets.append((h, r, t))
                                entity_set.add(h)
                                entity_set.add(t)
                                relation_set.add(r)

            # 检查实体数量
            print(f"加载完成: {len(entity_set)} 个实体, {len(relation_set)} 种关系, {len(all_triplets)} 个三元组")

            # 3. 检查哪些实体有特征
            entities_with_features = set(entity_features.keys())
            entities_without_features = entity_set - entities_with_features

            print(f"  有特征的实体: {len(entities_with_features)}")
            print(f"  无特征的实体: {len(entities_without_features)}")

            if entities_without_features:
                print(f"  前10个无特征实体: {list(entities_without_features)[:10]}")
                # 为无特征实体创建默认特征（全零）
                for eid in entities_without_features:
                    entity_features[eid] = [0.0] * 9

            # 4. 划分数据集
            # 读取每个文件中的实体，用于数据集划分
            train_entities = set()
            valid_entities = set()
            test_entities = set()

            file_mapping = {
                'xunlian.txt': train_entities,
               # 'valid.txt': valid_entities,
               # 'test.txt': test_entities
            }

            for file_name, entity_set_ref in file_mapping.items():
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 3:
                                h, r, t = parts
                                entity_set_ref.add(h)
                                entity_set_ref.add(t)

            # 5. 确保所有实体都有标签（如果没有，使用默认标签1）
            for eid in entity_set:
                if eid not in entity_labels:
                    entity_labels[eid] = 1  # 默认类别为1（人物和生命）

            return {
                'entity_features': entity_features,
                'entity_labels': entity_labels,
                'all_triplets': all_triplets,
                'all_entities': list(entity_set),
                'all_relations': list(relation_set),
                'train_entities': train_entities,
                'valid_entities': valid_entities,
                'test_entities': test_entities,
                'entities_without_features': list(entities_without_features)
            }

        except Exception as e:
            print(f"数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_heterogeneous_graph(self, use_relation_types=True, max_relations=30):
        """构建异构图（修复索引越界问题）"""
        print("\n正在构建异构图...")

        data = self.load_all_data()

        if not data:
            print("数据加载失败")
            return None, None, None

        # 1. 创建ID映射
        print("创建实体和关系ID映射...")
        entity_id_map = {eid: idx for idx, eid in enumerate(data['all_entities'])}
        relation_id_map = {rid: idx for idx, rid in enumerate(data['all_relations'][:max_relations])}

        print(f"  实体映射: {len(entity_id_map)} 个实体")
        print(f"  关系映射: {len(relation_id_map)} 种关系")

        # 2. 构建边数据
        print("构建边数据...")

        if use_relation_types and len(data['all_relations']) > 0:
            # 使用异构边
            edge_dict = {}

            # 统计关系频率
            relation_counts = Counter()
            for h, r, t in data['all_triplets']:
                if r in relation_id_map:  # 只使用前max_relations种关系
                    relation_counts[r] += 1

            print(f"使用 {len(relation_counts)} 种关系类型")

            # 为每种关系类型创建边
            for h, r, t in data['all_triplets']:
                if r in relation_id_map:
                    rel_key = f'rel_{relation_id_map[r]}'
                    if rel_key not in edge_dict:
                        edge_dict[rel_key] = ([], [])
                    # 确保实体在映射中
                    if h in entity_id_map and t in entity_id_map:
                        edge_dict[rel_key][0].append(entity_id_map[h])
                        edge_dict[rel_key][1].append(entity_id_map[t])

            # 转换为DGL图格式
            hetero_edges = {}
            for rel_key, (src, dst) in edge_dict.items():
                if src and dst:  # 确保边不为空
                    hetero_edges[('entity', rel_key, 'entity')] = (torch.tensor(src), torch.tensor(dst))

            if hetero_edges:
                g = dgl.heterograph(hetero_edges)
                print(f"构建异构图完成: {g}")
            else:
                print("警告: 没有有效的边，创建空图")
                # 创建空图
                g = dgl.heterograph({
                    ('entity', 'rel_0', 'entity'): ([0], [0])  # 至少一条边
                })

        else:
            # 简化：所有关系视为同一种类型
            print("使用同构图...")
            src_nodes, dst_nodes = [], []
            edge_count = 0

            for h, r, t in data['all_triplets']:
                # 确保实体在映射中
                if h in entity_id_map and t in entity_id_map:
                    src_nodes.append(entity_id_map[h])
                    dst_nodes.append(entity_id_map[t])
                    edge_count += 1

            print(f"  有效边数: {edge_count}/{len(data['all_triplets'])}")

            if src_nodes and dst_nodes:
                g = dgl.graph((src_nodes, dst_nodes))
                print(f"构建同构图完成: {g}")
            else:
                print("警告: 没有有效的边，创建空图")
                g = dgl.graph(([0], [0]))

        # 3. 检查图节点数量并调整
        num_entities = len(data['all_entities'])
        print(f"  实体总数: {num_entities}")
        print(f"  图节点数: {g.num_nodes()}")

        # 如果图节点数小于实体数，添加孤立节点
        if g.num_nodes() < num_entities:
            print(f"  添加 {num_entities - g.num_nodes()} 个孤立节点")
            g = dgl.add_nodes(g, num_entities - g.num_nodes())

        # 4. 添加节点特征
        print("添加节点特征...")
        node_feat_dim = 10
        node_features = np.zeros((num_entities, node_feat_dim))

        # 标签
        node_labels = np.full(num_entities, -1, dtype=int)  # -1表示无标签

        feature_count = 0
        labeled_count = 0

        for eid, idx in entity_id_map.items():
            # 调试：检查索引是否在有效范围内
            if idx >= num_entities:
                print(f"  警告: 索引越界: eid={eid}, idx={idx}, num_entities={num_entities}")
                continue

            if eid in data['entity_features']:
                # 前9维：类别得分
                scores = data['entity_features'][eid]
                if len(scores) == 9:
                    node_features[idx, :9] = scores
                else:
                    print(f"  警告: 实体 {eid} 的特征长度错误: {len(scores)}")
                    node_features[idx, :min(9, len(scores))] = scores[:9]

                # 第10维：特征存在标志
                node_features[idx, 9] = 1.0
                feature_count += 1

                if eid in data['entity_labels']:
                    # 标签：0-8，对应类别1-9
                    label_val = data['entity_labels'][eid]
                    if 1 <= label_val <= 9:
                        node_labels[idx] = label_val - 1
                        labeled_count += 1
                    else:
                        print(f"  警告: 实体 {eid} 的标签值无效: {label_val}")
                        node_labels[idx] = 0  # 默认类别

        print(f"  有特征的实体: {feature_count}/{num_entities}")
        print(f"  有标签的实体: {labeled_count}/{num_entities}")

        # 检查是否有未分配的标签
        unlabeled_count = (node_labels == -1).sum()
        if unlabeled_count > 0:
            print(f"  警告: {unlabeled_count} 个实体没有标签，分配默认标签")
            node_labels[node_labels == -1] = 0  # 默认类别

        # 5. 添加数据集划分掩码
        print("添加数据集划分掩码...")
        train_mask = np.zeros(num_entities, dtype=bool)
        valid_mask = np.zeros(num_entities, dtype=bool)
        test_mask = np.zeros(num_entities, dtype=bool)

        train_entity_count = 0
        valid_entity_count = 0
        test_entity_count = 0

        for eid, idx in entity_id_map.items():
            if idx >= num_entities:
                continue

            if eid in data['train_entities']:
                train_mask[idx] = True
                train_entity_count += 1
            if eid in data['valid_entities']:
                valid_mask[idx] = True
                valid_entity_count += 1
            if eid in data['test_entities']:
                test_mask[idx] = True
                test_entity_count += 1

        # 训练集中有标签的节点
        labeled_train_mask = train_mask & (node_labels != -1)

        # 6. 添加到图数据中
        print(f"  添加特征: 矩阵形状={node_features.shape}, 图节点={g.num_nodes()}")

        # 再次检查维度
        if node_features.shape[0] != g.num_nodes():
            print(f"  错误: 特征矩阵行数({node_features.shape[0]}) != 图节点数({g.num_nodes()})")
            # 调整图节点数
            if g.num_nodes() < node_features.shape[0]:
                g = dgl.add_nodes(g, node_features.shape[0] - g.num_nodes())
            else:
                # 这不应该发生，但如果发生就截断
                node_features = node_features[:g.num_nodes()]
                node_labels = node_labels[:g.num_nodes()]
                train_mask = train_mask[:g.num_nodes()]
                valid_mask = valid_mask[:g.num_nodes()]
                test_mask = test_mask[:g.num_nodes()]
                labeled_train_mask = labeled_train_mask[:g.num_nodes()]

        # 验证索引范围
        print(f"  验证: 特征矩阵索引范围 0-{node_features.shape[0] - 1}")
        print(f"  验证: 标签矩阵索引范围 0-{len(node_labels) - 1}")

        # 添加自循环处理零入度节点
        try:
            if hasattr(g, 'etypes') and len(g.etypes) > 1:
                # 异构图中添加自循环比较麻烦，跳过
                pass
            else:
                # 同构图中添加自循环
                g = dgl.add_self_loop(g)
                print("  已添加自循环")
        except Exception as e:
            print(f"  添加自循环失败: {e}")

        g.ndata['feat'] = torch.FloatTensor(node_features)
        g.ndata['label'] = torch.LongTensor(node_labels)
        g.ndata['train_mask'] = torch.BoolTensor(train_mask)
        g.ndata['valid_mask'] = torch.BoolTensor(valid_mask)
        g.ndata['test_mask'] = torch.BoolTensor(test_mask)
        g.ndata['labeled_mask'] = torch.BoolTensor(labeled_train_mask)

        # 7. 统计信息
        print("\n图构建统计信息:")
        print(f"  节点数: {g.num_nodes()}")
        print(f"  边数: {g.num_edges()}")
        print(f"  关系类型数: {len(g.etypes) if hasattr(g, 'etypes') else 1}")
        print(f"  特征维度: {node_feat_dim}")
        print(f"  训练节点: {train_entity_count}")
        print(f"  有标签的训练节点: {labeled_train_mask.sum()}")
        print(f"  验证节点: {valid_entity_count}")
        print(f"  测试节点: {test_entity_count}")

        # 检查索引问题
        labeled_indices = torch.where(g.ndata['labeled_mask'])[0]
        if len(labeled_indices) > 0:
            max_idx = labeled_indices.max().item()
            print(f"  有标签节点最大索引: {max_idx}, 节点总数: {g.num_nodes()}")
            if max_idx >= g.num_nodes():
                print(f"  警告: 索引越界! max_idx={max_idx} >= num_nodes={g.num_nodes()}")

        # 8. 保存图数据
        os.makedirs('processed_data', exist_ok=True)
        dgl.save_graphs('processed_data/fb15ket_graph.bin', [g])

        # 保存映射关系
        mapping_data = {
            'entity_id_map': entity_id_map,
            'relation_id_map': relation_id_map,
            'category_names': self.data_loader.category_names
        }
        torch.save(mapping_data, 'processed_data/fb15ket_mappings.pt')

        print(f"\n图数据已保存到: processed_data/fb15ket_graph.bin")
        print(f"映射数据已保存到: processed_data/fb15ket_mappings.pt")

        return g, entity_id_map, relation_id_map


class FB15KETHetGNN(nn.Module):
    """针对FB15KET的异构图神经网络"""

    def __init__(self, in_feats, hid_feats, out_feats, num_relations):
        super().__init__()
        self.num_relations = num_relations

        # 关系嵌入
        self.relation_emb = nn.Embedding(num_relations, hid_feats)

        # 为每种关系类型创建图卷积层，允许零入度节点
        self.conv_layers = nn.ModuleDict()
        for i in range(num_relations):
            self.conv_layers[f'rel_{i}'] = dglnn.GraphConv(in_feats, hid_feats, allow_zero_in_degree=True)

        # 融合层
        self.fusion = nn.Linear(hid_feats * num_relations, out_feats)

        # BiLSTM用于信息融合
        self.bilstm = nn.LSTM(
            input_size=out_feats,
            hidden_size=out_feats // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 自循环层，用于处理孤立节点
        self.self_loop_proj = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # inputs: [num_nodes, in_feats]
        h = inputs

        # 对每种关系类型分别进行卷积
        relation_outputs = []
        for i, etype in enumerate(g.etypes):
            rel_emb = self.relation_emb(torch.tensor([i], device=inputs.device))

            # 获取该关系类型的子图
            subgraph = g[etype]
            if subgraph.num_edges() > 0:
                try:
                    # 应用图卷积
                    conv_out = self.conv_layers[f'rel_{i}'](subgraph, h)
                    # 加入关系嵌入信息
                    conv_out = conv_out + rel_emb.expand_as(conv_out)
                    relation_outputs.append(conv_out)
                except Exception as e:
                    # 如果卷积失败，使用输入特征
                    print(f"关系 {etype} 卷积失败: {e}")
                    relation_outputs.append(h)
            else:
                # 如果没有边，使用输入特征
                relation_outputs.append(h)

        if relation_outputs:
            # 融合所有关系类型的输出
            combined = torch.cat(relation_outputs, dim=1)
            h = F.relu(self.fusion(combined))
        else:
            h = F.relu(self.fusion(h.repeat(1, self.num_relations)))

        # BiLSTM处理
        if h.dim() == 2:
            h = h.unsqueeze(1)  # 增加batch维度
            h, _ = self.bilstm(h)
            h = h.squeeze(1)

        return h


class FB15KETSubgraphBuilder:
    """针对FB15KET的子图构建器"""

    def __init__(self, hetero_graph):
        self.g = hetero_graph

    def get_relation_aware_neighbors(self, node_id, max_neighbors=50):
        """获取关系感知的邻居"""
        neighbor_info = defaultdict(list)

        # 检查节点是否在图中
        if node_id >= self.g.num_nodes():
            return neighbor_info

        # 遍历所有关系类型
        for etype in self.g.etypes:
            try:
                # 获取出边邻居
                successors = self.g.successors(node_id, etype=etype)
                if len(successors) > 0:
                    # 限制邻居数量
                    if len(successors) > max_neighbors:
                        indices = torch.randperm(len(successors))[:max_neighbors]
                        successors = successors[indices]

                    neighbor_info[etype].extend([(n.item(), etype, 'out') for n in successors])

                # 获取入边邻居
                predecessors = self.g.predecessors(node_id, etype=etype)
                if len(predecessors) > 0:
                    if len(predecessors) > max_neighbors:
                        indices = torch.randperm(len(predecessors))[:max_neighbors]
                        predecessors = predecessors[indices]

                    neighbor_info[etype].extend([(n.item(), etype, 'in') for n in predecessors])
            except Exception as e:
                # 如果获取邻居失败，继续下一个关系类型
                continue

        return neighbor_info

    def build_subgraph_embedding(self, node_id, node_feats, relation_embeddings=None):
        """构建子图嵌入"""
        # 检查节点ID是否有效
        if node_id >= len(node_feats):
            # 返回零向量
            zero_emb = torch.zeros_like(node_feats[0])
            return zero_emb, {'self': 1.0, 'neighbor': 0.0, 'error': 'invalid_node_id'}

        # 获取邻居信息
        neighbor_info = self.get_relation_aware_neighbors(node_id)

        if not neighbor_info:
            # 无邻居，返回节点自身特征
            return node_feats[node_id], {'self': 1.0, 'neighbor': 0.0}

        # 中心节点特征
        center_feat = node_feats[node_id]

        # 聚合邻居特征
        neighbor_embs = []
        relation_weights = []

        for etype, neighbors in neighbor_info.items():
            for neighbor_id, rel_type, direction in neighbors:
                # 检查邻居ID是否有效
                if neighbor_id >= len(node_feats):
                    continue

                neighbor_feat = node_feats[neighbor_id]

                # 加入关系信息
                if relation_embeddings is not None:
                    try:
                        rel_idx = int(rel_type.split('_')[1])  # 从'rel_X'中提取X
                        if rel_idx < len(relation_embeddings):
                            rel_emb = relation_embeddings[rel_idx]
                            combined = torch.cat([neighbor_feat, rel_emb])
                        else:
                            combined = neighbor_feat
                    except:
                        combined = neighbor_feat
                else:
                    combined = neighbor_feat

                neighbor_embs.append(combined)

                # 关系权重（可学习或简单分配）
                if direction == 'out':
                    weight = 1.0  # 出边
                else:
                    weight = 0.8  # 入边，权重稍低

                relation_weights.append(weight)

        # 加权聚合
        if neighbor_embs and len(neighbor_embs) > 0:
            weights = torch.softmax(torch.tensor(relation_weights), dim=0)
            weights = weights.to(node_feats.device)

            neighbor_tensor = torch.stack(neighbor_embs)
            aggregated = torch.sum(weights.unsqueeze(1) * neighbor_tensor, dim=0)

            # 与中心节点特征结合
            self_weight = 0.7
            neighbor_weight = 0.3

            final_emb = self_weight * center_feat + neighbor_weight * aggregated
        else:
            final_emb = center_feat
            self_weight = 1.0
            neighbor_weight = 0.0

        # 构建解释信息
        components = {
            'self_contribution': self_weight,
            'neighbor_contribution': neighbor_weight,
            'neighbor_count': len(neighbor_embs)
        }

        return final_emb, components


class FB15KETPrototypeNetwork(nn.Module):
    """针对FB15KET的原型网络"""

    def __init__(self, feature_dim, num_classes, num_prototypes_per_class=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes_per_class * num_classes

        # 可学习的原型
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, feature_dim))

        # 原型到类别的映射
        self.prototype_to_class = torch.repeat_interleave(
            torch.arange(num_classes), num_prototypes_per_class
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.1))

        # 初始化原型
        self._init_prototypes()

    def _init_prototypes(self):
        """初始化原型"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        # 检查特征维度
        if features.size(1) != self.feature_dim:
            # 如果维度不匹配，尝试调整
            if features.size(1) > self.feature_dim:
                features = features[:, :self.feature_dim]
            else:
                # 填充零
                padding = torch.zeros(features.size(0), self.feature_dim - features.size(1),
                                      device=features.device)
                features = torch.cat([features, padding], dim=1)

        # 计算与所有原型的距离
        distances = torch.cdist(features, self.prototypes, p=2)  # 欧氏距离

        # 转换为相似度（距离越小，相似度越高）
        similarities = torch.exp(-distances / (self.temperature.abs() + 1e-8))

        # 按类别聚合相似度
        class_similarities = torch.zeros(features.size(0), self.num_classes, device=features.device)

        for c in range(self.num_classes):
            # 获取该类别的原型索引
            proto_indices = (self.prototype_to_class == c).nonzero(as_tuple=True)[0]
            if len(proto_indices) > 0:
                # 取该类原型相似度的最大值
                if len(proto_indices) == 1:
                    class_similarities[:, c] = similarities[:, proto_indices[0]]
                else:
                    class_similarities[:, c] = similarities[:, proto_indices].max(dim=1).values

        # 计算分类概率
        probs = F.softmax(class_similarities, dim=1)

        return probs, similarities, class_similarities

    def update_prototypes(self, features, labels, learning_rate=0.01):
        """更新原型向量"""
        with torch.no_grad():
            for c in range(self.num_classes):
                # 获取属于类别c的样本
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask]

                    # 获取该类别的原型索引
                    proto_indices = (self.prototype_to_class == c).nonzero(as_tuple=True)[0]

                    if len(class_features) > 0 and len(proto_indices) > 0:
                        # 使用K-means思想更新原型
                        centroids = []

                        # 如果样本数少于原型数，复制样本
                        if len(class_features) < len(proto_indices):
                            for i in range(len(proto_indices)):
                                idx = i % len(class_features)
                                centroids.append(class_features[idx])
                        else:
                            # 简单聚类：均匀选择样本
                            step = len(class_features) // len(proto_indices)
                            for i in range(len(proto_indices)):
                                idx = min(i * step, len(class_features) - 1)
                                centroids.append(class_features[idx])

                        if centroids:
                            new_prototypes = torch.stack(centroids)
                            # 平滑更新
                            self.prototypes.data[proto_indices] = (
                                                                          1 - learning_rate) * self.prototypes.data[
                                                                      proto_indices] + learning_rate * new_prototypes


class FB15KETXGradNet(nn.Module):
    """完整的FB15KET XGradNet模型"""

    def __init__(self, hetero_graph, feature_dim, hidden_dim=128, out_dim=64,
                 num_classes=9, num_prototypes_per_class=3):
        super().__init__()
        self.g = hetero_graph

        # 检查图是否为空
        if hetero_graph.num_edges() == 0:
            print("警告: 图没有边，模型可能无法学习关系信息")

        # 1. 异构图神经网络
        num_relations = len(hetero_graph.etypes) if hasattr(hetero_graph, 'etypes') else 1
        self.hetgnn = FB15KETHetGNN(feature_dim, hidden_dim, out_dim, num_relations)

        # 2. 子图构建器
        self.subgraph_builder = FB15KETSubgraphBuilder(hetero_graph)

        # 3. 原型网络
        self.prototype_net = FB15KETPrototypeNetwork(
            out_dim, num_classes, num_prototypes_per_class
        )

        # 4. 结构贡献权重
        self.structure_weights = nn.Parameter(torch.tensor([0.8, 0.2]))  # [self, neighbor]

        # 5. 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim)
        )

        # 6. 输出层（备用，如果原型网络效果不好）
        self.fc = nn.Linear(out_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features):
        """前向传播"""
        # 特征转换
        transformed_features = self.feature_transform(node_features)

        # 异构图卷积
        try:
            node_embeddings = self.hetgnn(self.g, transformed_features)
        except Exception as e:
            print(f"异构图卷积失败: {e}")
            # 如果失败，使用转换后的特征
            node_embeddings = transformed_features

        return node_embeddings

    def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
        """分类预测"""
        if node_embeddings is None or len(node_ids) == 0:
            return None, None, None

        subgraph_embeddings = []
        components_list = []
        valid_node_ids = []

        # 为每个节点构建子图嵌入
        for node_id in node_ids:
            # 检查节点ID是否有效
            if node_id < len(node_embeddings):
                # 使用当前的结构权重
                self_weight = torch.sigmoid(self.structure_weights[0])
                neighbor_weight = torch.sigmoid(self.structure_weights[1])

                subgraph_emb, components = self.subgraph_builder.build_subgraph_embedding(
                    node_id, node_embeddings
                )
                subgraph_embeddings.append(subgraph_emb)
                components_list.append(components)
                valid_node_ids.append(node_id)

        if not subgraph_embeddings:
            return None, None, None

        subgraph_embeddings = torch.stack(subgraph_embeddings)

        # 原型网络分类
        probs, similarities, class_similarities = self.prototype_net(subgraph_embeddings)

        # 更新原型
        if update_prototypes and labels is not None:
            # 只使用有效节点的标签
            valid_labels = labels[valid_node_ids]
            self.prototype_net.update_prototypes(subgraph_embeddings, valid_labels)

        # 预测类别
        _, predicted_classes = torch.max(probs, dim=1)

        # 构建解释信息
        explanations = {
            'similarities': similarities,
            'class_similarities': class_similarities,
            'components': components_list,
            'structure_weights': self.structure_weights,
            'prototypes': self.prototype_net.prototypes.data,
            'valid_node_ids': valid_node_ids
        }

        return probs, predicted_classes, explanations

    def simple_classify(self, node_embeddings, node_ids):
        """简化分类（不使用子图和原型）"""
        if node_embeddings is None or len(node_ids) == 0:
            return None, None

        # 直接使用全连接层分类
        embeddings = node_embeddings[node_ids]
        logits = self.fc(embeddings)
        probs = F.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)

        return probs, predicted_classes

    def get_interpretation(self, node_id, node_embeddings, class_names):
        """获取单个节点的解释"""
        probs, pred_class, explanations = self.classify(
            node_embeddings, [node_id], update_prototypes=False
        )

        if probs is None:
            # 如果原型分类失败，使用简化分类
            probs, pred_class = self.simple_classify(node_embeddings, [node_id])
            if probs is None:
                return None

            pred_class_idx = pred_class.item()
            pred_prob = probs[0, pred_class_idx].item()

            interpretation = {
                'predicted_class': {
                    'index': pred_class_idx,
                    'name': class_names.get(pred_class_idx + 1, f"类别{pred_class_idx + 1}"),
                    'probability': pred_prob
                },
                'method': 'simple_classification'
            }
            return interpretation

        # 提取解释数据
        pred_class_idx = pred_class.item()
        pred_prob = probs[0, pred_class_idx].item()

        # 获取与各类别的相似度
        class_sims = explanations['class_similarities'][0].cpu().detach().numpy()

        # 获取结构贡献
        components = explanations['components'][0]

        # 构建解释结果
        interpretation = {
            'predicted_class': {
                'index': pred_class_idx,
                'name': class_names.get(pred_class_idx + 1, f"类别{pred_class_idx + 1}"),
                'probability': pred_prob
            },
            'class_similarities': {
                class_names.get(i + 1, f"类别{i + 1}"): float(sim)
                for i, sim in enumerate(class_sims)
            },
            'structure_contributions': components,
            'top_prototype_similarities': explanations['similarities'][0].cpu().detach().numpy(),
            'method': 'prototype_classification'
        }

        return interpretation


class EnhancedFB15KETXGradNet(nn.Module):
    """增强的模型架构"""

    def __init__(self, feature_dim, hidden_dim=256, out_dim=128, num_classes=9,
                 num_prototypes_per_class=3, dropout_rate=0.5):
        super().__init__()

        # 1. 更深的特征转换网络
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),

            nn.Linear(hidden_dim // 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6)
        )

        # 2. 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 3. 原型网络
        self.prototype_net = FB15KETPrototypeNetwork(
            out_dim, num_classes, num_prototypes_per_class
        )

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 5. 残差连接
        self.residual = nn.Linear(feature_dim, out_dim) if feature_dim != out_dim else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """更好的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, node_features):
        """前向传播"""
        # 残差连接
        residual = self.residual(node_features)

        # 特征转换
        transformed = self.feature_transform(node_features)

        # 注意力机制（将特征视为序列）
        batch_size = transformed.size(0)
        if batch_size > 1:
            # 重塑为序列形式 [batch_size, 1, feature_dim]
            seq_features = transformed.unsqueeze(1)
            attn_output, _ = self.attention(seq_features, seq_features, seq_features)
            attn_features = attn_output.squeeze(1)
        else:
            attn_features = transformed

        # 残差连接
        combined = attn_features + residual

        return combined

    def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
        """分类预测"""
        if node_embeddings is None or len(node_ids) == 0:
            return None, None, None

        # 获取指定节点的嵌入
        embeddings = node_embeddings[node_ids]

        # 原型网络分类
        prototype_probs, similarities, class_similarities = self.prototype_net(embeddings)

        # 分类头分类
        classifier_logits = self.classifier(
            torch.cat([embeddings, class_similarities], dim=1)
        )
        classifier_probs = F.softmax(classifier_logits, dim=1)

        # 融合两种分类结果
        alpha = 0.7  # 原型网络权重
        combined_probs = alpha * prototype_probs + (1 - alpha) * classifier_probs

        # 更新原型
        if update_prototypes and labels is not None:
            self.prototype_net.update_prototypes(embeddings, labels[node_ids])

        # 预测类别
        _, predicted_classes = torch.max(combined_probs, dim=1)

        return combined_probs, predicted_classes, {
            'prototype_probs': prototype_probs,
            'classifier_probs': classifier_probs,
            'similarities': similarities,
            'class_similarities': class_similarities
        }

class SimpleFB15KETXGradNet(nn.Module):
    """简化的FB15KET XGradNet模型，避免异构图的复杂操作"""

    def __init__(self, feature_dim, hidden_dim=128, out_dim=64, num_classes=9):
        super().__init__()

        # 1. 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

        # 2. 原型网络
        self.prototype_net = FB15KETPrototypeNetwork(
            out_dim, num_classes, num_prototypes_per_class=2
        )

        # 3. 备用分类层
        self.fc = nn.Linear(out_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features):
        """前向传播 - 仅特征转换"""
        return self.feature_transform(node_features)

    def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
        """分类预测"""
        if node_embeddings is None or len(node_ids) == 0:
            return None, None, None

        # 获取指定节点的嵌入
        embeddings = node_embeddings[node_ids]

        # 原型网络分类
        probs, similarities, class_similarities = self.prototype_net(embeddings)

        # 更新原型
        if update_prototypes and labels is not None:
            self.prototype_net.update_prototypes(embeddings, labels[node_ids])

        # 预测类别
        _, predicted_classes = torch.max(probs, dim=1)

        # 构建解释信息
        explanations = {
            'similarities': similarities,
            'class_similarities': class_similarities,
            'prototypes': self.prototype_net.prototypes.data
        }

        return probs, predicted_classes, explanations

    def simple_classify(self, node_embeddings, node_ids):
        """简化分类（不使用原型）"""
        if node_embeddings is None or len(node_ids) == 0:
            return None, None

        # 直接使用全连接层分类
        embeddings = node_embeddings[node_ids]
        logits = self.fc(embeddings)
        probs = F.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)

        return probs, predicted_classes

    def get_interpretation(self, node_id, node_embeddings, class_names):
        """获取单个节点的解释"""
        # 直接分类
        probs, pred_class = self.simple_classify(node_embeddings, [node_id])
        if probs is None:
            return None

        pred_class_idx = pred_class.item()
        pred_prob = probs[0, pred_class_idx].item()

        # 尝试获取原型相似度
        try:
            embeddings = node_embeddings[[node_id]]
            _, similarities, class_similarities = self.prototype_net(embeddings)

            class_sims = class_similarities[0].cpu().detach().numpy()

            interpretation = {
                'predicted_class': {
                    'index': pred_class_idx,
                    'name': class_names.get(pred_class_idx + 1, f"类别{pred_class_idx + 1}"),
                    'probability': pred_prob
                },
                'class_similarities': {
                    class_names.get(i + 1, f"类别{i + 1}"): float(sim)
                    for i, sim in enumerate(class_sims)
                },
                'method': 'prototype_classification'
            }
        except:
            interpretation = {
                'predicted_class': {
                    'index': pred_class_idx,
                    'name': class_names.get(pred_class_idx + 1, f"类别{pred_class_idx + 1}"),
                    'probability': pred_prob
                },
                'method': 'simple_classification'
            }

        return interpretation



class SemiSupervisedTrainer:
    """半监督训练器"""

    def __init__(self, model, graph, device='cuda'):
        self.model = model
        self.g = graph
        self.device = device

        # 将模型和图移动到设备
        self.model = self.model.to(device)

        try:
            self.g = self.g.to(device)
        except:
            print("警告: 无法将图移动到设备，将使用CPU")
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)

        # 获取掩码
        self.labeled_mask = self.g.ndata['labeled_mask'].to(self.device)
        self.train_mask = self.g.ndata['train_mask'].to(self.device)
        self.valid_mask = self.g.ndata['valid_mask'].to(self.device)
        self.test_mask = self.g.ndata['test_mask'].to(self.device)

        # 标签
        self.labels = self.g.ndata['label'].to(self.device)

        # 特征
        self.features = self.g.ndata['feat'].to(self.device)

        # 获取有标签的节点索引
        self.labeled_indices = torch.where(self.labeled_mask)[0]

        print(f"训练节点: {self.train_mask.sum().item()}")
        print(f"有标签的训练节点: {len(self.labeled_indices)}")
        print(f"验证节点: {self.valid_mask.sum().item()}")
        print(f"测试节点: {self.test_mask.sum().item()}")

        # 检查数据
        if len(self.labeled_indices) == 0:
            print("警告: 没有有标签的训练节点！")

    def train(self, epochs=100, lr=0.001, weight_decay=1e-4,
              contrastive_weight=0.1, prototype_weight=0.05):
        """训练模型"""
        if len(self.labeled_indices) == 0:
            print("错误: 没有有标签的训练节点，无法训练！")
            return [], []

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_acc = 0
        best_epoch = 0
        patience = 30

        train_losses = []
        val_accuracies = []

        print("\n开始训练...")
        print("-" * 80)

        for epoch in range(epochs):
            self.model.train()

            try:
                # 前向传播
                node_embeddings = self.model(self.features)

                # 分类（只对有标签的训练节点）
                probs, preds, explanations = self.model.classify(
                    node_embeddings, self.labeled_indices,
                    update_prototypes=True,
                    labels=self.labels[self.labeled_indices]
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    print(f"Epoch {epoch + 1}: 原型分类失败，使用简化分类")
                    probs, preds = self.model.simple_classify(
                        node_embeddings, self.labeled_indices
                    )

                    if probs is None:
                        print(f"Epoch {epoch + 1}: 简化分类也失败，跳过本轮")
                        continue

                    # 只计算分类损失
                    cls_loss = F.cross_entropy(
                        probs, self.labels[self.labeled_indices]
                    )
                    total_loss = cls_loss

                else:
                    # 计算损失
                    # 1. 分类损失
                    cls_loss = F.cross_entropy(
                        probs, self.labels[self.labeled_indices]
                    )

                    # 2. 对比学习损失（鼓励相似节点的嵌入接近）
                    contrastive_loss = self.compute_contrastive_loss(
                        node_embeddings, self.labeled_indices
                    )

                    # 3. 原型多样性损失
                    prototype_loss = self.compute_prototype_diversity_loss()

                    # 总损失
                    total_loss = cls_loss + contrastive_weight * contrastive_loss + prototype_weight * prototype_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # 记录训练损失
                train_losses.append(total_loss.item())

            except Exception as e:
                print(f"Epoch {epoch + 1} 训练出错: {e}")
                # 跳过这个epoch
                continue

            # 验证
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                try:
                    val_acc = self.evaluate(mode='valid')
                    val_accuracies.append(val_acc)

                    print(f"Epoch {epoch + 1:3d}/{epochs} | "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

                    # 检查是否是最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch

                        # 保存最佳模型
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_acc,
                            'loss': total_loss.item()
                        }, 'models/best_model.pth')

                        print(f"  保存最佳模型 (Val Acc: {val_acc:.4f})")

                except Exception as e:
                    print(f"Epoch {epoch + 1} 验证出错: {e}")
                    val_accuracies.append(0.0)

            ''''# 早停
            if epoch - best_epoch > patience:
                print(f"\n早停在 epoch {epoch + 1}，最佳验证准确率: {best_val_acc:.4f}")
                break
            '''
        print("-" * 80)
        print(f"训练完成，最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch + 1})")

        # 加载最佳模型
        try:
            checkpoint = torch.load('models/best_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载最佳模型")
        except Exception as e:
            print(f"加载最佳模型失败: {e}")

        return train_losses, val_accuracies

    def compute_contrastive_loss(self, embeddings, indices, temperature=0.1):
        """计算对比学习损失"""
        if len(indices) < 2:
            return torch.tensor(0.0, device=self.device)

        # 选择一部分节点
        if len(indices) > 100:
            selected = indices[torch.randperm(len(indices))[:100]]
        else:
            selected = indices

        # 获取嵌入
        selected_embeddings = embeddings[selected]

        # 检查嵌入是否有效
        if torch.isnan(selected_embeddings).any() or torch.isinf(selected_embeddings).any():
            return torch.tensor(0.0, device=self.device)

        # 计算相似度矩阵
        similarity = torch.matmul(selected_embeddings, selected_embeddings.T) / (temperature + 1e-8)

        # 对角线设置为负无穷（排除自身）
        mask = torch.eye(len(selected), device=self.device).bool()
        similarity.masked_fill_(mask, -1e9)

        # 计算对比损失
        labels = self.labels[selected]

        # 创建正样本对（相同类别的节点）
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_matrix.masked_fill_(mask, False)  # 排除自身

        # 计算损失
        pos_indices = label_matrix.nonzero()
        if len(pos_indices) == 0:
            pos_loss = torch.tensor(0.0, device=self.device)
        else:
            pos_similarity = similarity[label_matrix]
            pos_loss = -torch.mean(pos_similarity)

        # 负样本损失
        neg_indices = (~label_matrix).nonzero()
        if len(neg_indices) == 0:
            neg_loss = torch.tensor(0.0, device=self.device)
        else:
            neg_similarity = similarity[~label_matrix]
            neg_loss = torch.mean(torch.exp(neg_similarity))

        return pos_loss + torch.log(neg_loss + 1e-8)

    def compute_prototype_diversity_loss(self):
        """计算原型多样性损失"""
        try:
            prototypes = self.model.prototype_net.prototypes

            if prototypes.size(0) < 2:
                return torch.tensor(0.0, device=self.device)

            # 计算原型间的距离
            distances = torch.cdist(prototypes, prototypes, p=2)

            # 排除对角线
            mask = torch.eye(prototypes.size(0), device=self.device).bool()
            distances = distances[~mask].view(prototypes.size(0), prototypes.size(0) - 1)

            # 鼓励原型间保持一定距离
            min_distances = distances.min(dim=1).values
            diversity_loss = -torch.mean(min_distances)  # 最小距离越大越好

            return diversity_loss
        except Exception as e:
            print(f"计算原型多样性损失失败: {e}")
            return torch.tensor(0.0, device=self.device)

    def evaluate(self, mode='valid'):
        """评估模型"""
        self.model.eval()

        if mode == 'valid':
            mask = self.valid_mask
        elif mode == 'test':
            mask = self.test_mask
        elif mode == 'train':
            mask = self.labeled_mask
        else:
            raise ValueError(f"未知评估模式: {mode}")

        indices = torch.where(mask)[0]

        if len(indices) == 0:
            return 0.0

        with torch.no_grad():
            try:
                node_embeddings = self.model(self.features)

                # 尝试使用原型分类
                probs, preds, _ = self.model.classify(
                    node_embeddings, indices, update_prototypes=False
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    probs, preds = self.model.simple_classify(node_embeddings, indices)

                    if probs is None:
                        return 0.0

                # 计算准确率
                acc = (preds == self.labels[indices]).float().mean().item()

                return acc

            except Exception as e:
                print(f"评估失败 ({mode}): {e}")
                return 0.0

    def test(self, save_results=True):
        """测试模型"""
        print("\n" + "=" * 60)
        print("模型测试")
        print("=" * 60)

        try:
            train_acc = self.evaluate(mode='train')
            valid_acc = self.evaluate(mode='valid')
            test_acc = self.evaluate(mode='test')

            print(f"训练集准确率: {train_acc:.4f}")
            print(f"验证集准确率: {valid_acc:.4f}")
            print(f"测试集准确率: {test_acc:.4f}")

            # 生成详细预测结果
            if save_results:
                self.save_predictions()

            return {
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'test_acc': test_acc
            }

        except Exception as e:
            print(f"测试失败: {e}")
            return {
                'train_acc': 0.0,
                'valid_acc': 0.0,
                'test_acc': 0.0
            }

    def save_predictions(self):
        """保存预测结果"""
        self.model.eval()

        all_indices = torch.arange(self.g.num_nodes(), device=self.device)

        with torch.no_grad():
            try:
                node_embeddings = self.model(self.features)

                # 尝试使用原型分类
                probs, preds, explanations = self.model.classify(
                    node_embeddings, all_indices, update_prototypes=False
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    probs, preds = self.model.simple_classify(node_embeddings, all_indices)

                    if probs is None:
                        print("无法生成预测结果")
                        return

                # 转换预测结果
                predictions = preds.cpu().numpy() + 1  # 转换回1-9的类别
                probabilities = probs.cpu().numpy()

                # 创建结果DataFrame
                results = []
                for idx in range(len(all_indices)):
                    pred_class = predictions[idx]
                    true_class = self.labels[idx].item() + 1 if self.labels[idx] != -1 else -1

                    results.append({
                        'node_id': idx,
                        'predicted_class': pred_class,
                        'true_class': true_class,
                        'prediction_prob': probabilities[idx, pred_class - 1] if pred_class <= probabilities.shape[
                            1] else 0.0,
                        'in_train': self.train_mask[idx].item(),
                        'in_valid': self.valid_mask[idx].item(),
                        'in_test': self.test_mask[idx].item(),
                        'has_label': self.labels[idx] != -1
                    })

                results_df = pd.DataFrame(results)

                # 保存结果
                results_df.to_csv('predictions/fb15ket_predictions.csv', index=False)

                print(f"\n预测结果已保存到: predictions/fb15ket_predictions.csv")

                # 保存类别级别的性能
                self.save_class_level_performance(predictions)

            except Exception as e:
                print(f"保存预测结果失败: {e}")

    def save_class_level_performance(self, predictions):
        """保存类别级别的性能分析"""
        try:
            from sklearn.metrics import classification_report, confusion_matrix

            # 只分析有真实标签的测试集节点
            test_indices = torch.where(self.test_mask & (self.labels != -1))[0]

            if len(test_indices) == 0:
                print("没有测试集标签，无法生成分类报告")
                return

            y_true = self.labels[test_indices].cpu().numpy() + 1
            y_pred = predictions[test_indices.cpu().numpy()]

            # 生成分类报告
            category_names = FB15KETDataLoader().category_names
            class_names = [f"{i}: {name}" for i, name in category_names.items()]

            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True
            )

            # 保存报告
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('predictions/classification_report.csv')

            print(f"分类报告已保存到: predictions/classification_report.csv")

            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=class_names,
                columns=class_names
            )
            cm_df.to_csv('predictions/confusion_matrix.csv')

            print(f"混淆矩阵已保存到: predictions/confusion_matrix.csv")

        except Exception as e:
            print(f"保存类别级别性能失败: {e}")
class SimpleTrainer:
    """简化训练器，不依赖复杂的图结构"""

    def __init__(self, model, graph, device='cuda'):
        self.model = model
        self.g = graph
        self.device = device

        # 将模型移动到设备
        self.model = self.model.to(device)

        # 获取数据
        self.features = self.g.ndata['feat'].to(device)
        self.labels = self.g.ndata['label'].to(device)

        # 获取掩码
        self.train_mask = self.g.ndata['train_mask'].to(device)
        self.valid_mask = self.g.ndata['valid_mask'].to(device)
        self.test_mask = self.g.ndata['test_mask'].to(device)
        self.labeled_mask = self.g.ndata['labeled_mask'].to(device)

        # 获取有标签的节点索引
        self.labeled_indices = torch.where(self.labeled_mask)[0]

        print(f"训练节点: {self.train_mask.sum().item()}")
        print(f"有标签的训练节点: {len(self.labeled_indices)}")
        print(f"验证节点: {self.valid_mask.sum().item()}")
        print(f"测试节点: {self.test_mask.sum().item()}")

        # 检查数据
        if len(self.labeled_indices) == 0:
            print("警告: 没有有标签的训练节点！")

    def train(self, epochs=50, lr=0.001, weight_decay=1e-4):
        """训练模型"""
        if len(self.labeled_indices) == 0:
            print("错误: 没有有标签的训练节点，无法训练！")
            return [], []

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_val_acc = 0
        best_epoch = 0
        patience = 20

        train_losses = []
        val_accuracies = []

        print("\n开始训练...")
        print("-" * 80)

        for epoch in range(epochs):
            self.model.train()

            try:
                # 前向传播
                node_embeddings = self.model(self.features)

                # 分类（只对有标签的训练节点）
                probs, preds, explanations = self.model.classify(
                    node_embeddings, self.labeled_indices,
                    update_prototypes=True,
                    labels=self.labels[self.labeled_indices]
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    probs, preds = self.model.simple_classify(
                        node_embeddings, self.labeled_indices
                    )

                    if probs is None:
                        print(f"Epoch {epoch + 1}: 分类失败，跳过本轮")
                        continue

                # 计算分类损失
                cls_loss = F.cross_entropy(
                    probs, self.labels[self.labeled_indices]
                )

                total_loss = cls_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # 记录训练损失
                train_losses.append(total_loss.item())

            except Exception as e:
                print(f"Epoch {epoch + 1} 训练出错: {e}")
                continue

            # 验证
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                try:
                    val_acc = self.evaluate(mode='valid')
                    val_accuracies.append(val_acc)

                    print(f"Epoch {epoch + 1:3d}/{epochs} | "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

                    # 检查是否是最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch

                        # 保存最佳模型
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_acc,
                            'loss': total_loss.item()
                        }, 'models/best_model.pth')

                        print(f"  保存最佳模型 (Val Acc: {val_acc:.4f})")

                except Exception as e:
                    print(f"Epoch {epoch + 1} 验证出错: {e}")
                    val_accuracies.append(0.0)
            '''
            # 早停
            if epoch - best_epoch > patience:
                print(f"\n早停在 epoch {epoch + 1}，最佳验证准确率: {best_val_acc:.4f}")
                break
            '''
        print("-" * 80)
        print(f"训练完成，最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch + 1})")

        # 加载最佳模型
        try:
            checkpoint = torch.load('models/best_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载最佳模型")
        except Exception as e:
            print(f"加载最佳模型失败: {e}")

        return train_losses, val_accuracies

    def evaluate(self, mode='valid'):
        """评估模型"""
        self.model.eval()

        if mode == 'valid':
            mask = self.valid_mask
        elif mode == 'test':
            mask = self.test_mask
        elif mode == 'train':
            mask = self.labeled_mask
        else:
            raise ValueError(f"未知评估模式: {mode}")

        indices = torch.where(mask & (self.labels != -1))[0]  # 只评估有标签的节点

        if len(indices) == 0:
            return 0.0

        with torch.no_grad():
            try:
                node_embeddings = self.model(self.features)

                # 使用原型分类
                probs, preds, _ = self.model.classify(
                    node_embeddings, indices, update_prototypes=False
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    probs, preds = self.model.simple_classify(node_embeddings, indices)

                    if probs is None:
                        return 0.0

                # 计算准确率
                acc = (preds == self.labels[indices]).float().mean().item()

                return acc

            except Exception as e:
                print(f"评估失败 ({mode}): {e}")
                return 0.0

    def test(self, save_results=True):
        """测试模型"""
        print("\n" + "=" * 60)
        print("模型测试")
        print("=" * 60)

        try:
            train_acc = self.evaluate(mode='train')
            valid_acc = self.evaluate(mode='valid')
            test_acc = self.evaluate(mode='test')

            print(f"训练集准确率: {train_acc:.4f}")
            print(f"验证集准确率: {valid_acc:.4f}")
            print(f"测试集准确率: {test_acc:.4f}")

            # 生成详细预测结果
            if save_results:
                self.save_predictions()

            return {
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'test_acc': test_acc
            }

        except Exception as e:
            print(f"测试失败: {e}")
            return {
                'train_acc': 0.0,
                'valid_acc': 0.0,
                'test_acc': 0.0
            }

    def save_predictions(self):
        """保存预测结果（修复numpy问题）"""
        self.model.eval()

        all_indices = torch.arange(self.g.num_nodes(), device=self.device)
        labeled_indices = all_indices[self.labels != -1]

        with torch.no_grad():
            try:
                node_embeddings = self.model(self.features)

                # 预测所有有标签的节点
                probs, preds, _ = self.model.classify(
                    node_embeddings, labeled_indices, update_prototypes=False
                )

                if probs is None:
                    # 如果原型分类失败，使用简化分类
                    probs, preds = self.model.simple_classify(node_embeddings, labeled_indices)

                    if probs is None:
                        print("无法生成预测结果")
                        return

                # 转换预测结果 - 确保在CPU上操作
                predictions = preds.cpu().numpy() + 1  # 转换回1-9的类别
                probabilities = probs.cpu().numpy()

                # 创建结果DataFrame
                results = []
                for i, idx in enumerate(labeled_indices.cpu().numpy()):
                    pred_class = predictions[i]
                    true_class = self.labels[idx].item() + 1

                    results.append({
                        'node_id': int(idx),
                        'predicted_class': int(pred_class),
                        'true_class': int(true_class),
                        'prediction_prob': float(
                            probabilities[i, pred_class - 1] if pred_class <= probabilities.shape[1] else 0.0),
                        'in_train': bool(self.train_mask[idx].item()),
                        'in_valid': bool(self.valid_mask[idx].item()),
                        'in_test': bool(self.test_mask[idx].item())
                    })

                results_df = pd.DataFrame(results)

                # 保存结果
                os.makedirs('predictions', exist_ok=True)
                results_df.to_csv('predictions/fb15ket_predictions.csv', index=False)

                print(f"\n预测结果已保存到: predictions/fb15ket_predictions.csv")

                # 保存类别级别的性能
                self.save_class_level_performance(predictions, labeled_indices.cpu().numpy())

            except Exception as e:
                print(f"保存预测结果失败: {e}")
                import traceback
                traceback.print_exc()

    def save_class_level_performance(self, predictions, labeled_indices):
        """保存类别级别的性能分析"""
        try:
            from sklearn.metrics import classification_report, confusion_matrix

            # 获取测试集的预测结果
            test_indices = []
            test_preds = []
            test_labels = []

            for i, idx in enumerate(labeled_indices):
                if self.test_mask[idx] and self.labels[idx] != -1:
                    test_indices.append(i)
                    test_preds.append(predictions[i])
                    test_labels.append(self.labels[idx].item() + 1)

            if len(test_indices) == 0:
                print("没有测试集标签，无法生成分类报告")
                return

            y_true = np.array(test_labels)
            y_pred = np.array(test_preds)

            # 生成分类报告
            category_names = FB15KETDataLoader().category_names
            class_names = [f"{i}: {name[:15]}..." for i, name in category_names.items()]

            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True
            )

            # 保存报告
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('predictions/classification_report.csv')

            print(f"分类报告已保存到: predictions/classification_report.csv")

            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=class_names,
                columns=class_names
            )
            cm_df.to_csv('predictions/confusion_matrix.csv')

            print(f"混淆矩阵已保存到: predictions/confusion_matrix.csv")

        except Exception as e:
            print(f"保存类别级别性能失败: {e}")
class EnhancedTrainer(SimpleTrainer):
    """增强的训练器"""

    def train(self, epochs=100, lr=0.001, weight_decay=1e-4,
              warmup_epochs=10, patience=30):
        """改进的训练策略"""

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 带warmup的学习率调度
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # warmup阶段：线性增加学习率
                return (epoch + 1) / warmup_epochs
            else:
                # cosine衰减
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 混合精度训练（如果可用）
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        best_val_acc = 0
        best_epoch = 0

        train_losses = []
        val_accuracies = []

        print("\n开始增强训练...")
        print("-" * 80)

        for epoch in range(epochs):
            self.model.train()

            try:
                # 混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        node_embeddings = self.model(self.features)

                        # 分类
                        probs, preds, explanations = self.model.classify(
                            node_embeddings, self.labeled_indices,
                            update_prototypes=True,
                            labels=self.labels[self.labeled_indices]
                        )

                        if probs is None:
                            continue

                        # 计算损失
                        cls_loss = F.cross_entropy(
                            probs, self.labels[self.labeled_indices]
                        )

                        # 添加正则化损失
                        reg_loss = 0.0
                        for param in self.model.parameters():
                            if param.requires_grad:
                                reg_loss += torch.norm(param, p=2)

                        total_loss = cls_loss + 1e-4 * reg_loss

                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()

                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 普通训练
                    node_embeddings = self.model(self.features)

                    # 分类
                    probs, preds, explanations = self.model.classify(
                        node_embeddings, self.labeled_indices,
                        update_prototypes=True,
                        labels=self.labels[self.labeled_indices]
                    )

                    if probs is None:
                        continue

                    # 计算损失
                    cls_loss = F.cross_entropy(
                        probs, self.labels[self.labeled_indices]
                    )

                    # 添加标签平滑
                    smooth_labels = self.label_smoothing(
                        self.labels[self.labeled_indices],
                        num_classes=9,
                        smoothing=0.1
                    )
                    smooth_loss = F.kl_div(
                        F.log_softmax(probs, dim=1),
                        smooth_labels,
                        reduction='batchmean'
                    )

                    total_loss = 0.7 * cls_loss + 0.3 * smooth_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                # 更新学习率
                scheduler.step()

                # 记录训练损失
                train_losses.append(total_loss.item())

            except Exception as e:
                print(f"Epoch {epoch + 1} 训练出错: {e}")
                continue

            # 验证
            if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                try:
                    val_acc = self.evaluate(mode='valid')
                    val_accuracies.append(val_acc)

                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch + 1:3d}/{epochs} | "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {current_lr:.6f}")

                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'val_acc': val_acc,
                            'loss': total_loss.item(),
                            'config': {
                                'hidden_dim': self.model.feature_transform[0].out_features,
                                'dropout_rate': 0.5
                            }
                        }, 'best_enhanced_model.pth')

                        print(f"  💾 保存最佳模型 (Val Acc: {val_acc:.4f})")

                except Exception as e:
                    print(f"Epoch {epoch + 1} 验证出错: {e}")
                    val_accuracies.append(0.0)

            # 早停
            if epoch - best_epoch > patience:
                print(f"\n⏹️  早停在 epoch {epoch + 1}，最佳验证准确率: {best_val_acc:.4f}")
                break

        print("-" * 80)
        print(f"✅ 训练完成，最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch + 1})")

        return train_losses, val_accuracies

    def label_smoothing(self, labels, num_classes, smoothing=0.1):
        """标签平滑"""
        confidence = 1.0 - smoothing
        smooth_labels = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), confidence)
        return smooth_labels

def main():
    """主函数：完整的训练评估流程（最终版本）"""
    print("=" * 80)
    print("基于原型的可解释性FB15KET实体分类系统")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 创建必要的目录
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)

    try:
        # 1. 检查数据质量
        print("\n[步骤1] 数据质量检查...")
        data_loader = FB15KETDataLoader()
        stats = data_loader.analyze_data_quality()

        # 2. 构建图
        print("\n[步骤2] 构建异构图...")
        graph_builder = FB15KETGraphBuilder()

        # 简化：使用同构图，避免异构图的复杂操作
        print("注意：使用同构图简化处理")
        g, entity_map, relation_map = graph_builder.build_heterogeneous_graph(
            use_relation_types=False  # 不使用异构边
        )

        if g is None:
            print("图构建失败，退出程序")
            return

        # 3. 创建模型
        print("\n[步骤3] 创建模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        feature_dim = g.ndata['feat'].shape[1]
        hidden_dim = 128
        out_dim = 64
        num_classes = 9

        # 使用简化模型
        model = SimpleFB15KETXGradNet(
            feature_dim, hidden_dim, out_dim, num_classes
        )

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        # 4. 训练模型
        print("\n[步骤4] 训练模型...")
        trainer = SimpleTrainer(model, g, device=device)

        # 训练
        train_losses, val_accuracies = trainer.train(
            epochs=50,
            lr=0.001,
            weight_decay=1e-4
        )

        # 5. 评估模型
        print("\n[步骤5] 评估模型...")
        results = trainer.test(save_results=True)

        # 6. 生成可解释性分析
        print("\n[步骤6] 生成可解释性分析...")
        try:
            # 加载映射数据
            mapping_data = torch.load('processed_data/fb15ket_mappings.pt')
            category_names = mapping_data['category_names']

            # 选择一些示例节点进行分析
            test_indices = torch.where(g.ndata['test_mask'] & (g.ndata['label'] != -1))[0]

            if len(test_indices) > 0:
                # 随机选择5个节点
                if len(test_indices) > 5:
                    sample_indices = test_indices[torch.randperm(len(test_indices))[:5]]
                else:
                    sample_indices = test_indices

                interpretations = []
                for idx in sample_indices:
                    interpretation = model.get_interpretation(
                        idx.item(), model(g.ndata['feat'].to(device)), category_names
                    )

                    if interpretation is not None:
                        interpretations.append({
                            'node_id': idx.item(),
                            'interpretation': interpretation
                        })

                # 保存解释结果
                if interpretations:
                    import json

                    # 转换为可序列化的格式
                    serializable_interpretations = []
                    for item in interpretations:
                        node_id = item['node_id']
                        interp = item['interpretation']

                        serializable = {
                            'node_id': node_id,
                            'predicted_class': interp['predicted_class'],
                            'method': interp.get('method', 'unknown')
                        }

                        if 'class_similarities' in interp:
                            serializable['class_similarities'] = interp['class_similarities']

                        serializable_interpretations.append(serializable)

                    # 保存为JSON
                    with open('predictions/interpretations.json', 'w', encoding='utf-8') as f:
                        json.dump(serializable_interpretations, f, ensure_ascii=False, indent=2)

                    print(f"可解释性分析已保存到: predictions/interpretations.json")

                    # 打印一个示例
                    print("\n示例解释分析:")
                    print("-" * 60)
                    example = serializable_interpretations[0]
                    print(f"节点ID: {example['node_id']}")
                    print(f"预测类别: {example['predicted_class']['name']} "
                          f"(概率: {example['predicted_class']['probability']:.3f})")

                    if 'class_similarities' in example:
                        print("\n与各类别的相似度:")
                        for class_name, similarity in example['class_similarities'].items():
                            print(f"  {class_name}: {similarity:.4f}")

        except Exception as e:
            print(f"生成可解释性分析失败: {e}")

        # 7. 生成可视化
        print("\n[步骤7] 生成可视化...")
        try:
            from visualization import VisualizationTool
            predictions_df = pd.read_csv('predictions/fb15ket_predictions.csv')
            viz_tool = VisualizationTool(model, g, category_names)
            viz_tool.plot_class_distribution(predictions_df)

            # 生成报告
            results = {
                'train_acc': predictions_df[predictions_df['in_train']]
                .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean(),
                'valid_acc': predictions_df[predictions_df['in_valid']]
                .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean(),
                'test_acc': predictions_df[predictions_df['in_test']]
                .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean()
            }

            viz_tool.generate_report(results, predictions_df)

        except Exception as e:
            print(f"生成可视化失败: {e}")

        print("\n" + "=" * 80)
        print("所有步骤完成！")
        print("=" * 80)

        return model, g, results

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def fix_137_dimension_issue():
    """专门修复137维特征问题"""
    print("=" * 80)
    print("FB15KET实体分类系统 - 137维特征修复版")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建必要的目录
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    try:
        # 1. 加载图数据
        print("\n[1] 加载图数据...")
        g_list, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
        g = g_list[0]

        # 检查特征维度
        feature_dim = g.ndata['feat'].shape[1]
        print(f"检测到的特征维度: {feature_dim}")

        # 2. 创建专门处理137维的模型
        print("\n[2] 创建137维专用模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 专门为137维特征设计的模型
        class Model137D(nn.Module):
            def __init__(self, input_dim=137, hidden_dim=256, out_dim=128, num_classes=9):
                super().__init__()

                print(f"创建137维专用模型: {input_dim} -> {hidden_dim} -> {out_dim}")

                # 输入层（137维专用）
                self.input_layer = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )

                # 隐藏层
                self.hidden_layers = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )

                # 输出层
                self.classifier = nn.Linear(64, num_classes)

                # 初始化
                self._init_weights()

            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            def forward(self, x):
                x = self.input_layer(x)
                x = self.hidden_layers(x)
                return x

            def classify(self, embeddings, indices, update_prototypes=False, labels=None):
                if embeddings is None or len(indices) == 0:
                    return None, None, None

                node_embeddings = embeddings[indices]
                logits = self.classifier(node_embeddings)
                probs = F.softmax(logits, dim=1)
                _, predicted_classes = torch.max(probs, dim=1)

                return probs, predicted_classes, {}

            def simple_classify(self, embeddings, indices):
                return self.classify(embeddings, indices)

        model = Model137D(input_dim=feature_dim)
        model = model.to(device)

        # 3. 创建训练器
        print("\n[3] 创建训练器...")
        trainer = SimpleTrainer(model, g, device=device)

        # 4. 测试前向传播
        print("\n[4] 测试前向传播...")
        test_features = g.ndata['feat'].to(device)
        test_output = model(test_features)
        print(f"✓ 前向传播测试成功")
        print(f"  输入: {test_features.shape}")
        print(f"  输出: {test_output.shape}")

        # 5. 训练模型
        print("\n[5] 训练模型...")
        train_losses, val_accuracies = trainer.train(
            epochs=50,  # 先训练50个epoch
            lr=0.001,
            weight_decay=1e-4
        )

        # 6. 评估模型
        print("\n[6] 评估模型...")
        results = trainer.test(save_results=True)

        # 7. 保存模型
        print("\n[7] 保存模型...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dim': feature_dim,
            'results': results
        }, 'models/model_137d.pth')

        print(f"\n模型已保存到: models/model_137d.pth")

        return model, g, results

    except Exception as e:
        print(f"修复过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
def improved_main():
    """改进的主函数（修复维度问题）"""
    print("=" * 80)
    print("FB15KET实体分类系统 - 改进版本")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 创建必要的目录
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('enhanced_data', exist_ok=True)

    try:
        # ============================================
        # 1. 数据质量检查
        # ============================================
        print("\n[1/7] 数据质量检查...")
        data_loader = FB15KETDataLoader()
        stats = data_loader.analyze_data_quality()

        # ============================================
        # 2. 加载或构建图数据
        # ============================================
        print("\n[2/7] 加载图数据...")

        # 首先检查是否有原始的图数据
        if os.path.exists('processed_data/fb15ket_graph.bin'):
            print("加载原始图数据...")
            g_list, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
            g = g_list[0]

            # 获取特征维度
            feature_dim = g.ndata['feat'].shape[1]
            print(f"原始特征维度: {feature_dim}")

            # 检查数据集划分问题
            print("\n检查数据集划分...")
            train_mask = g.ndata['train_mask'].sum().item()
            valid_mask = g.ndata['valid_mask'].sum().item()
            test_mask = g.ndata['test_mask'].sum().item()
            labeled_mask = g.ndata['labeled_mask'].sum().item()

            print(f"训练节点: {train_mask}")
            print(f"验证节点: {valid_mask}")
            print(f"测试节点: {test_mask}")
            print(f"有标签的训练节点: {labeled_mask}")

            # 如果验证集和测试集太小，重新划分
            if valid_mask == 0 or test_mask == 0:
                print("警告: 验证集或测试集为空，重新划分数据集...")
                g = re_split_dataset(g)
        else:
            print("原始图数据不存在，重新构建...")
            graph_builder = FB15KETGraphBuilder()
            g, entity_id_map, relation_id_map = graph_builder.build_heterogeneous_graph(
                use_relation_types=False
            )

            # 保存映射
            mapping_data = {
                'entity_id_map': entity_id_map,
                'relation_id_map': relation_id_map,
                'category_names': data_loader.category_names
            }
            torch.save(mapping_data, 'processed_data/fb15ket_mappings.pt')

        # ============================================
        # 3. 创建适合维度的模型
        # ============================================
        print("\n[3/7] 创建增强模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 获取实际的输入维度
        feature_dim = g.ndata['feat'].shape[1]
        print(f"实际特征维度: {feature_dim}")

        model = SimpleImprovedModel(
            feature_dim=feature_dim,  # 使用实际的137维
            hidden_dim=256,
            out_dim=128,
            num_classes=9
        )
        '''
        # 创建适合维度的模型
        if feature_dim == 137:
            print("检测到137维特征，使用适配模型...")
            model = DimensionAdaptiveModel(feature_dim)
        else:
            print(f"使用标准增强模型，特征维度: {feature_dim}")
            model = EnhancedFB15KETXGradNet(
                feature_dim=feature_dim,
                hidden_dim=256,
                out_dim=128,
                num_classes=9,
                num_prototypes_per_class=3,
                dropout_rate=0.5
            )
        '''
        # 打印模型信息
        print("\n模型架构:")
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        # ============================================
        # 4. 修复数据集划分问题
        # ============================================
        print("\n[4/7] 准备训练数据...")

        # 检查当前的数据集划分
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        test_mask = g.ndata['test_mask']
        labeled_mask = g.ndata['labeled_mask']

        print(f"数据集划分统计:")
        print(f"  训练节点: {train_mask.sum().item()}")
        print(f"  验证节点: {valid_mask.sum().item()}")
        print(f"  测试节点: {test_mask.sum().item()}")
        print(f"  有标签的训练节点: {labeled_mask.sum().item()}")

        # 如果验证集和测试集太小，重新划分
        if valid_mask.sum().item() < 100 or test_mask.sum().item() < 100:
            print("数据集划分不合理，重新划分...")
            g = re_split_dataset(g)

        # 检查有标签的节点
        labeled_indices = torch.where(labeled_mask)[0]
        print(f"有标签的节点索引数量: {len(labeled_indices)}")

        if len(labeled_indices) == 0:
            print("错误: 没有有标签的训练节点!")
            return None, None, None

        # ============================================
        # 5. 训练增强模型（简化版，避免复杂错误）
        # ============================================
        print("\n[5/7] 训练增强模型（简化版）...")

        # 使用简化训练器
        trainer = SimpleTrainer(model, g, device=device)

        # 先测试一次前向传播
        print("测试前向传播...")
        try:
            test_features = g.ndata['feat'].to(device)
            test_output = model(test_features)
            print(f"前向传播测试成功!")
            print(f"输入维度: {test_features.shape}")
            print(f"输出维度: {test_output.shape}")
        except Exception as e:
            print(f"前向传播测试失败: {e}")
            print("调试模型架构...")
            return None, None, None

        # 训练
        train_losses, val_accuracies = trainer.train(
            epochs=100,  # 先训练100个epoch
            lr=0.001,
            weight_decay=1e-4
        )

        # ============================================
        # 6. 评估模型
        # ============================================
        print("\n[6/7] 评估模型...")
        results = trainer.test(save_results=True)

        # ============================================
        # 7. 尝试进一步优化（如果第一次训练成功）
        # ============================================
        if results and results.get('train_acc', 0) > 0.5:
            print("\n[7/7] 进一步优化...")

            # 保存第一次训练结果
            first_train_acc = results.get('train_acc', 0)
            first_test_acc = results.get('test_acc', 0)

            print(f"第一次训练结果: 训练集={first_train_acc:.4f}, 测试集={first_test_acc:.4f}")

            # 尝试继续训练或使用更复杂的模型
            if first_train_acc < 0.65:
                print("尝试继续训练...")
                # 可以继续训练或调整超参数

        print("\n" + "=" * 80)
        print("改进训练流程完成！")
        print("=" * 80)

        return model, g, results

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def re_split_dataset(g, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """重新划分数据集"""
    print("重新划分数据集...")

    num_nodes = g.num_nodes()
    labels = g.ndata['label']

    # 找到有标签的节点
    labeled_indices = torch.where(labels != -1)[0]
    num_labeled = len(labeled_indices)

    print(f"有标签的节点数: {num_labeled}")

    if num_labeled == 0:
        print("错误: 没有有标签的节点!")
        return g

    # 随机打乱
    shuffled_indices = labeled_indices[torch.randperm(num_labeled)]

    # 计算划分点
    train_end = int(num_labeled * train_ratio)
    val_end = train_end + int(num_labeled * val_ratio)

    # 创建掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[shuffled_indices[:train_end]] = True
    valid_mask[shuffled_indices[train_end:val_end]] = True
    test_mask[shuffled_indices[val_end:]] = True
    labeled_mask[shuffled_indices[:train_end]] = True

    # 更新图的掩码
    g.ndata['train_mask'] = train_mask
    g.ndata['valid_mask'] = valid_mask
    g.ndata['test_mask'] = test_mask
    g.ndata['labeled_mask'] = labeled_mask

    print(f"重新划分结果:")
    print(f"  训练集: {train_mask.sum().item()} 节点")
    print(f"  验证集: {valid_mask.sum().item()} 节点")
    print(f"  测试集: {test_mask.sum().item()} 节点")
    print(f"  有标签的训练节点: {labeled_mask.sum().item()} 节点")

    return g


class DimensionAdaptiveModel(nn.Module):
    """维度自适应模型（专门处理137维特征）"""

    def __init__(self, input_dim=137, hidden_dim=256, out_dim=128, num_classes=9):
        super().__init__()

        print(f"创建维度自适应模型: 输入={input_dim}, 隐藏={hidden_dim}, 输出={out_dim}")

        # 1. 自适应输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 2. 中间层
        self.hidden_layers = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim // 2, num_classes)
        )

        # 4. 原型网络（简化）
        self.prototype_net = SimplePrototypeNetwork(out_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 调试信息
        if hasattr(self, 'debug') and self.debug:
            print(f"输入维度: {x.shape}")
            print(f"输入层权重维度: {self.input_layer.weight.shape}")

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return x

    def classify(self, embeddings, indices, update_prototypes=False, labels=None):
        """分类"""
        if embeddings is None or len(indices) == 0:
            return None, None, None

        # 获取指定节点的嵌入
        node_embeddings = embeddings[indices]

        # 原型分类
        proto_probs, similarities, class_similarities = self.prototype_net(node_embeddings)

        # 分类器分类
        logits = self.classifier(node_embeddings)
        classifier_probs = F.softmax(logits, dim=1)

        # 融合（原型网络权重0.6，分类器权重0.4）
        alpha = 0.6
        combined_probs = alpha * proto_probs + (1 - alpha) * classifier_probs

        # 更新原型
        if update_prototypes and labels is not None:
            self.prototype_net.update_prototypes(node_embeddings, labels[indices])

        # 预测类别
        _, predicted_classes = torch.max(combined_probs, dim=1)

        return combined_probs, predicted_classes, {
            'proto_probs': proto_probs,
            'classifier_probs': classifier_probs,
            'similarities': similarities
        }

    def simple_classify(self, embeddings, indices):
        """简化分类"""
        if embeddings is None or len(indices) == 0:
            return None, None

        node_embeddings = embeddings[indices]
        logits = self.classifier(node_embeddings)
        probs = F.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)

        return probs, predicted_classes


class SimplePrototypeNetwork(nn.Module):
    """简化原型网络"""

    def __init__(self, feature_dim, num_classes, num_prototypes_per_class=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes = num_classes * num_prototypes_per_class

        # 可学习的原型
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, feature_dim))

        # 原型到类别的映射
        self.prototype_to_class = torch.repeat_interleave(
            torch.arange(num_classes), num_prototypes_per_class
        )

        # 初始化
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        # 计算距离
        distances = torch.cdist(features, self.prototypes, p=2)

        # 转换为相似度
        similarities = torch.exp(-distances)

        # 按类别聚合
        class_similarities = torch.zeros(features.size(0), self.num_classes, device=features.device)

        for c in range(self.num_classes):
            proto_indices = (self.prototype_to_class == c).nonzero(as_tuple=True)[0]
            if len(proto_indices) > 0:
                if len(proto_indices) == 1:
                    class_similarities[:, c] = similarities[:, proto_indices[0]]
                else:
                    class_similarities[:, c] = similarities[:, proto_indices].max(dim=1).values

        # 计算概率
        probs = F.softmax(class_similarities, dim=1)

        return probs, similarities, class_similarities

    def update_prototypes(self, features, labels):
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask]
                    proto_indices = (self.prototype_to_class == c).nonzero(as_tuple=True)[0]

                    if len(class_features) >= len(proto_indices):
                        # 均匀选择样本作为原型
                        step = len(class_features) // len(proto_indices)
                        for i, idx in enumerate(proto_indices):
                            sample_idx = min(i * step, len(class_features) - 1)
                            self.prototypes.data[idx] = 0.9 * self.prototypes.data[idx] + 0.1 * class_features[
                                sample_idx]


# 简化版改进函数
def simple_improvement():
    """简化版改进（修复版）"""
    print("=" * 80)
    print("FB15KET实体分类系统 - 简化改进版")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. 加载图数据
        print("\n[1] 加载图数据...")
        if not os.path.exists('processed_data/fb15ket_graph.bin'):
            print("图数据不存在，请先运行原始版本构建图")
            return None, None, None

        g_list, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
        g = g_list[0]

        # 检查数据集划分
        train_mask = g.ndata['train_mask'].sum().item()
        valid_mask = g.ndata['valid_mask'].sum().item()
        test_mask = g.ndata['test_mask'].sum().item()

        print(f"数据集划分:")
        print(f"  训练节点: {train_mask}")
        print(f"  验证节点: {valid_mask}")
        print(f"  测试节点: {test_mask}")

        # 2. 创建合适的模型
        print("\n[2] 创建改进模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_dim = g.ndata['feat'].shape[1]

        print(f"特征维度: {feature_dim}")

        # 根据特征维度选择合适的模型
        if feature_dim == 10:
            # 原始特征维度
            hidden_dim = 128
            out_dim = 64
        elif feature_dim == 137:
            # 增强特征维度
            hidden_dim = 256
            out_dim = 128
        else:
            # 自适应
            hidden_dim = min(256, feature_dim * 2)
            out_dim = min(128, feature_dim)

        print(f"使用模型配置: hidden_dim={hidden_dim}, out_dim={out_dim}")

        model = SimpleImprovedModel(feature_dim, hidden_dim, out_dim, 9)
        model = model.to(device)

        # 3. 训练
        print("\n[3] 训练改进模型...")
        trainer = SimpleTrainer(model, g, device=device)

        # 先测试前向传播
        print("测试前向传播...")
        try:
            test_features = g.ndata['feat'].to(device)
            test_output = model(test_features)
            print(f"✓ 前向传播测试成功")
            print(f"  输入: {test_features.shape}")
            print(f"  输出: {test_output.shape}")
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            return None, None, None

        # 开始训练
        print("\n开始训练...")
        train_losses, val_accuracies = trainer.train(
            epochs=100,
            lr=0.001,
            weight_decay=1e-4
        )

        # 4. 评估
        print("\n[4] 评估模型...")
        results = trainer.test(save_results=True)

        return model, g, results

    except Exception as e:
        print(f"简化改进失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


class SimpleImprovedModel(nn.Module):
    """简单改进模型（修复137维问题）"""

    def __init__(self, feature_dim, hidden_dim=128, out_dim=64, num_classes=9):
        super().__init__()

        print(f"创建简单改进模型: input={feature_dim}, hidden={hidden_dim}, output={out_dim}")

        # 自适应特征转换
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim // 2, num_classes)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.feature_transform(x)

    def classify(self, embeddings, indices, update_prototypes=False, labels=None):
        if embeddings is None or len(indices) == 0:
            return None, None, None

        node_embeddings = embeddings[indices]
        logits = self.classifier(node_embeddings)
        probs = F.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)

        return probs, predicted_classes, {}

    def simple_classify(self, embeddings, indices):
        return self.classify(embeddings, indices)





# ============================================
# 辅助函数
# ============================================

def build_enhanced_features(triplets_by_entity):
    """构建增强特征"""
    print("构建增强特征...")

    # 加载实体数据
    entity_df = pd.read_csv('data/FB15KET/Entity_All_typed.csv')

    enhanced_features = {}

    for _, row in entity_df.iterrows():
        eid = row['entity_id']

        # 1. 基本特征：9个类别得分
        base_features = []
        for i in range(1, 10):
            col_name = f'category_{i}_score'
            if col_name in row and not pd.isna(row[col_name]):
                base_features.append(float(row[col_name]))
            else:
                base_features.append(0.0)

        # 2. 结构特征
        structural_features = extract_structural_features(eid, triplets_by_entity)

        # 3. 组合所有特征
        all_features = base_features + structural_features

        enhanced_features[eid] = all_features

    print(f"构建了 {len(enhanced_features)} 个实体的增强特征")
    print(f"特征维度: {len(next(iter(enhanced_features.values())))}")

    return enhanced_features


def extract_structural_features(entity_id, triplets_by_entity):
    """提取结构特征"""
    if entity_id not in triplets_by_entity:
        return [0.0] * 5  # 返回默认特征

    relations = triplets_by_entity[entity_id]

    # 计算各种结构特征
    out_degree = sum(1 for d, _, _ in relations if d == 'out')
    in_degree = sum(1 for d, _, _ in relations if d == 'in')
    total_degree = len(relations)

    # 关系类型多样性
    rel_types = set(rel for _, rel, _ in relations)
    rel_diversity = len(rel_types) / (total_degree + 1e-8)

    # 归一化特征
    features = [
        min(out_degree / 100.0, 1.0),  # 归一化到[0,1]
        min(in_degree / 100.0, 1.0),
        min(total_degree / 200.0, 1.0),
        rel_diversity,
        min(len(rel_types) / 50.0, 1.0)
    ]

    return features


def analyze_results(results, train_losses, val_accuracies):
    """分析训练结果"""
    print("\n性能分析报告:")
    print("-" * 60)

    train_acc = results.get('train_acc', 0)
    val_acc = results.get('valid_acc', 0)
    test_acc = results.get('test_acc', 0)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 分析过拟合/欠拟合
    if train_acc > val_acc + 0.05:
        print("⚠️  可能存在过拟合 (训练集 >> 验证集)")
        print("   建议: 增加dropout, 数据增强, 早停")
    elif train_acc < val_acc:
        print("⚠️  可能存在欠拟合 (训练集 < 验证集)")
        print("   建议: 增加模型复杂度, 训练更多epoch, 数据增强")
    else:
        print("✓ 训练集和验证集性能平衡")

    # 泛化能力
    if abs(val_acc - test_acc) < 0.02:
        print("✓ 泛化能力良好 (验证集 ≈ 测试集)")
    else:
        print("⚠️  泛化能力有待提升")

    # 绝对性能
    if test_acc > 0.7:
        print("🎉 性能优秀 (>70%)")
    elif test_acc > 0.6:
        print("👍 性能良好 (60-70%)")
    elif test_acc > 0.5:
        print("👌 性能一般 (50-60%)")
    else:
        print("🔧 需要大幅改进 (<50%)")

    # 训练过程分析
    if train_losses:
        final_loss = train_losses[-1]
        print(f"\n训练损失分析:")
        print(f"  最终训练损失: {final_loss:.4f}")
        if len(train_losses) > 1:
            loss_decrease = train_losses[0] - final_loss
            print(f"  总损失下降: {loss_decrease:.4f}")
            if loss_decrease < 0.1:
                print("  警告: 损失下降不足，可能学习率太小或模型太简单")


# ============================================
# 简化版改进（逐步实施）
# ============================================

def simple_improvement():
    """简化版改进：只修改最容易实现的"""
    print("=" * 80)
    print("FB15KET实体分类系统 - 简化改进版")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建必要的目录
    os.makedirs('predictions', exist_ok=True)

    try:
        # 1. 加载现有图数据
        print("\n[1] 加载图数据...")
        g_list, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
        g = g_list[0]

        # 2. 创建增强模型（简化版）
        print("\n[2] 创建增强模型（简化版）...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_dim = g.ndata['feat'].shape[1]

        # 简化版增强模型
        class SimplifiedEnhancedModel(SimpleFB15KETXGradNet):
            def __init__(self, feature_dim, hidden_dim=256, out_dim=128, num_classes=9):
                super().__init__(feature_dim, hidden_dim, out_dim, num_classes)

                # 增强特征转换层
                self.enhanced_transform = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )

                # 增强分类头
                self.enhanced_classifier = nn.Sequential(
                    nn.Linear(out_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, num_classes)
                )

                # 更好的初始化
                self._init_enhanced_weights()

            def _init_enhanced_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            def forward(self, node_features):
                # 使用增强的特征转换
                return self.enhanced_transform(node_features)

            def classify(self, node_embeddings, node_ids, update_prototypes=False, labels=None):
                # 使用增强的分类器
                embeddings = node_embeddings[node_ids]
                logits = self.enhanced_classifier(embeddings)
                probs = F.softmax(logits, dim=1)
                _, predicted_classes = torch.max(probs, dim=1)

                return probs, predicted_classes, {}

        model = SimplifiedEnhancedModel(feature_dim)

        # 3. 改进训练器
        print("\n[3] 改进训练...")

        class ImprovedSimpleTrainer(SimpleTrainer):
            def train(self, epochs=100, lr=0.001):
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=1e-4,
                    betas=(0.9, 0.999)
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                best_val_acc = 0
                train_losses = []
                val_accuracies = []

                print("\n开始改进训练...")
                print("-" * 80)

                for epoch in range(epochs):
                    self.model.train()

                    try:
                        node_embeddings = self.model(self.features)
                        probs, preds, _ = self.model.classify(
                            node_embeddings, self.labeled_indices,
                            update_prototypes=False,
                            labels=self.labels[self.labeled_indices]
                        )

                        if probs is None:
                            continue

                        # 计算损失
                        cls_loss = F.cross_entropy(probs, self.labels[self.labeled_indices])

                        # 添加标签平滑
                        smooth_labels = torch.full_like(probs, 0.1 / 8.0)
                        smooth_labels.scatter_(1, self.labels[self.labeled_indices].unsqueeze(1), 0.9)
                        smooth_loss = F.kl_div(F.log_softmax(probs, dim=1), smooth_labels, reduction='batchmean')

                        total_loss = 0.8 * cls_loss + 0.2 * smooth_loss

                        # 反向传播
                        optimizer.zero_grad()
                        total_loss.backward()

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        optimizer.step()
                        scheduler.step()

                        train_losses.append(total_loss.item())

                    except Exception as e:
                        print(f"Epoch {epoch + 1} 训练出错: {e}")
                        continue

                    # 验证
                    if (epoch + 1) % 5 == 0:
                        val_acc = self.evaluate(mode='valid')
                        val_accuracies.append(val_acc)

                        print(f"Epoch {epoch + 1:3d}/{epochs} | "
                              f"Loss: {total_loss.item():.4f} | "
                              f"Val Acc: {val_acc:.4f}")

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'val_acc': val_acc
                            }, 'improved_model.pth')

                print(f"\n最佳验证准确率: {best_val_acc:.4f}")
                return train_losses, val_accuracies

        trainer = ImprovedSimpleTrainer(model, g, device=device)

        # 训练更多epochs
        train_losses, val_accuracies = trainer.train(epochs=100, lr=0.001)

        # 4. 评估
        print("\n[4] 评估改进模型...")
        results = trainer.test(save_results=True)

        # 5. 比较改进效果
        print("\n[5] 改进效果对比:")
        print("-" * 60)
        print("原始模型: 训练集准确率 ~0.5731")
        print(f"改进模型: 训练集准确率 {results.get('train_acc', 0):.4f}")
        print(f"提升幅度: {results.get('train_acc', 0) - 0.5731:.4f}")

        return model, g, results

    except Exception as e:
        print(f"简化改进失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ============================================
# 执行函数
# ============================================



def generate_interpretations(model, graph, category_names, device):
    """生成可解释性分析"""
    print("生成可解释性分析...")

    # 选择一些示例节点进行分析
    test_indices = torch.where(graph.ndata['test_mask'] & (graph.ndata['label'] != -1))[0]

    if len(test_indices) == 0:
        print("没有找到测试节点进行分析")
        return

    # 随机选择10个节点
    if len(test_indices) > 10:
        sample_indices = test_indices[torch.randperm(len(test_indices))[:10]]
    else:
        sample_indices = test_indices

    # 获取特征
    features = graph.ndata['feat'].to(device)

    interpretations = []

    for idx in sample_indices:
        interpretation = model.get_interpretation(
            idx.item(), model(features), category_names
        )

        if interpretation is not None:
            interpretations.append({
                'node_id': idx.item(),
                'interpretation': interpretation
            })

    # 保存解释结果
    if interpretations:
        import json

        # 转换为可序列化的格式
        serializable_interpretations = []
        for item in interpretations:
            node_id = item['node_id']
            interp = item['interpretation']

            serializable = {
                'node_id': node_id,
                'predicted_class': interp['predicted_class'],
                'class_similarities': interp['class_similarities'],
                'structure_contributions': {
                    k: (float(v) if isinstance(v, torch.Tensor) else v)
                    for k, v in interp['structure_contributions'].items()
                }
            }
            serializable_interpretations.append(serializable)

        # 保存为JSON
        with open('predictions/interpretations.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_interpretations, f, ensure_ascii=False, indent=2)

        print(f"可解释性分析已保存到: predictions/interpretations.json")

        # 打印一个示例
        print("\n示例解释分析:")
        print("-" * 60)
        example = serializable_interpretations[0]
        print(f"节点ID: {example['node_id']}")
        print(f"预测类别: {example['predicted_class']['name']} "
              f"(概率: {example['predicted_class']['probability']:.3f})")

        print("\n与各类别的相似度:")
        for class_name, similarity in example['class_similarities'].items():
            print(f"  {class_name}: {similarity:.4f}")

        print("\n结构贡献:")
        for component, value in example['structure_contributions'].items():
            print(f"  {component}: {value:.4f}")


class VisualizationTool:
    """可视化工具类"""

    def __init__(self, model, graph, category_names):
        self.model = model
        self.graph = graph
        self.category_names = category_names

    def plot_training_history(self, train_losses, val_accuracies):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))

        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        # 验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch (每5轮)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_prototype_similarity(self):
        """可视化原型相似度"""
        self.model.eval()

        # 获取原型
        prototypes = self.model.prototype_net.prototypes.detach().cpu().numpy()

        # 计算原型间的相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(prototypes)

        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='相似度')
        plt.title('原型间相似度矩阵')
        plt.xlabel('原型索引')
        plt.ylabel('原型索引')

        # 添加原型类别标签
        num_prototypes_per_class = self.model.prototype_net.num_prototypes // 9
        for i in range(9):
            start_idx = i * num_prototypes_per_class
            plt.axhline(y=start_idx - 0.5, color='white', linestyle='--', alpha=0.5)
            plt.axvline(x=start_idx - 0.5, color='white', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('visualizations/prototype_similarity.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_class_distribution(self, predictions_df):
        """绘制类别分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 真实类别分布
        true_classes = predictions_df[predictions_df['true_class'] != -1]['true_class']
        true_counts = true_classes.value_counts().sort_index()

        axes[0].bar(range(len(true_counts)), true_counts.values)
        axes[0].set_title('真实类别分布')
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('实体数量')
        axes[0].set_xticks(range(len(true_counts)))
        axes[0].set_xticklabels([self.category_names.get(i, i) for i in true_counts.index], rotation=45, ha='right')

        # 预测类别分布
        pred_counts = predictions_df['predicted_class'].value_counts().sort_index()

        axes[1].bar(range(len(pred_counts)), pred_counts.values)
        axes[1].set_title('预测类别分布')
        axes[1].set_xlabel('类别')
        axes[1].set_ylabel('实体数量')
        axes[1].set_xticks(range(len(pred_counts)))
        axes[1].set_xticklabels([self.category_names.get(i, i) for i in pred_counts.index], rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('visualizations/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, cm_path='predictions/confusion_matrix.csv'):
        """绘制混淆矩阵"""
        if not os.path.exists(cm_path):
            return

        cm_df = pd.read_csv(cm_path, index_col=0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

    def generate_report(self, results, predictions_df):
        """生成完整报告"""
        report = f"""
# FB15KET实体分类系统报告

## 1. 模型性能
- 训练集准确率: {results.get('train_acc', 0):.4f}
- 验证集准确率: {results.get('valid_acc', 0):.4f}
- 测试集准确率: {results.get('test_acc', 0):.4f}

## 2. 数据统计
- 总实体数: {len(predictions_df)}
- 有标签的实体: {len(predictions_df[predictions_df['has_label']])}
- 无标签的实体: {len(predictions_df[~predictions_df['has_label']])}

## 3. 类别分布
"""

        # 添加类别统计
        for class_id in range(1, 10):
            class_name = self.category_names.get(class_id, f"类别{class_id}")
            true_count = len(predictions_df[(predictions_df['true_class'] == class_id)])
            pred_count = len(predictions_df[(predictions_df['predicted_class'] == class_id)])

            report += f"- {class_name}:\n"
            report += f"  真实数量: {true_count}\n"
            report += f"  预测数量: {pred_count}\n"

        # 保存报告
        with open('predictions/system_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"系统报告已保存到: predictions/system_report.md")

        return report


# 使用示例
def run_visualization():
    """运行可视化分析"""
    # 加载数据
    g, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
    graph = g[0]

    mapping_data = torch.load('processed_data/fb15ket_mappings.pt')
    category_names = mapping_data['category_names']

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = graph.ndata['feat'].shape[1]

    model = FB15KETXGradNet(graph, feature_dim)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 加载预测结果
    predictions_df = pd.read_csv('predictions/fb15ket_predictions.csv')

    # 创建可视化工具
    viz_tool = VisualizationTool(model, graph, category_names)

    # 生成可视化
    print("生成可视化图表...")
    viz_tool.plot_class_distribution(predictions_df)
    viz_tool.plot_confusion_matrix()
    viz_tool.plot_prototype_similarity()

    # 生成报告
    results = {
        'train_acc': predictions_df[predictions_df['in_train'] & predictions_df['has_label']]
        .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean(),
        'valid_acc': predictions_df[predictions_df['in_valid'] & predictions_df['has_label']]
        .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean(),
        'test_acc': predictions_df[predictions_df['in_test'] & predictions_df['has_label']]
        .apply(lambda row: row['predicted_class'] == row['true_class'], axis=1).mean()
    }

    viz_tool.generate_report(results, predictions_df)

    print("可视化分析完成！")


if __name__ == "__main__":
    print("FB15KET实体分类改进系统")
    print("=" * 80)
    print("您的特征维度为137，需要特殊处理")
    print("=" * 80)

    print("\n修复方案选择:")
    print("1. 执行137维专用修复 (推荐)")
    print("2. 运行简化改进版")
    print("3. 运行完整改进版")
    print("4. 退出")

    choice = input("\n请输入选择 (1-4): ").strip()

    if choice == '1':
        print("\n执行137维专用修复...")
        model, graph, results = fix_137_dimension_issue()
    elif choice == '2':
        print("\n执行简化改进版...")
        model, graph, results = simple_improvement()
    elif choice == '3':
        print("\n执行完整改进版...")
        model, graph, results = improved_main()
    elif choice == '4':
        print("退出程序")
    else:
        print("无效选择")



'''
if __name__ == "__main__":
    model, graph, results = main()

'''

# 运行图构建
'''if __name__ == "__main__":
    print("=" * 60)
    print("第2步：构建FB15KET异构图")
    print("=" * 60)

    graph_builder = FB15KETGraphBuilder()
    g, entity_map, relation_map = graph_builder.build_heterogeneous_graph(use_relation_types=True)
'''