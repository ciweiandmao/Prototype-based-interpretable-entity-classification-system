from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import os
import random


# ==================== 模型定义 ====================
class SimpleTestModel(nn.Module):
    """用于测试的简单模型（适配137维特征）"""

    def __init__(self, input_dim=137, hidden_dim=256, out_dim=128, num_classes=9):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(out_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
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

    def classify(self, embeddings, indices):
        if embeddings is None or len(indices) == 0:
            return None, None

        node_embeddings = embeddings[indices]
        logits = self.classifier(node_embeddings)
        probs = F.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)
        return probs, predicted_classes


# ==================== 辅助函数 ====================
def load_all_data():
    """加载所有必要的数据"""
    print("加载所有数据...")

    try:
        # 1. 加载图数据和映射
        g_list, _ = dgl.load_graphs('processed_data/fb15ket_graph.bin')
        g = g_list[0]

        mapping_data = torch.load('processed_data/fb15ket_mappings.pt')
        entity_id_map = mapping_data['entity_id_map']
        reverse_entity_map = {v: k for k, v in entity_id_map.items()}
        category_names = mapping_data['category_names']

        # 2. 加载Entity_All_typed.csv获取真实类型
        entity_df = pd.read_csv('data/FB15KET/Entity_All_typed.csv')
        entity_info_dict = {}
        for _, row in entity_df.iterrows():
            eid = row['entity_id']
            if 'predicted_category' in row and not pd.isna(row['predicted_category']):
                entity_info_dict[eid] = {
                    'category': int(row['predicted_category']),
                    'name': row['predicted_category_name'] if 'predicted_category_name' in row and not pd.isna(
                        row['predicted_category_name']) else f"类别{int(row['predicted_category'])}",
                    'confidence': row['confidence_score'] if 'confidence_score' in row and not pd.isna(
                        row['confidence_score']) else None
                }

        # 3. 加载三元组数据
        triplets_by_entity = defaultdict(list)
        file_path = 'data/FB15KET/xunlian.txt'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        h, r, t = parts
                        triplets_by_entity[h].append(('out', r, t))
                        triplets_by_entity[t].append(('in', r, h))

        print(f"数据加载完成:")
        print(f"  图节点数: {g.num_nodes()}")
        print(f"  实体映射数: {len(entity_id_map)}")
        print(f"  有类型标注的实体: {len(entity_info_dict)}")
        print(f"  加载三元组的实体数: {len(triplets_by_entity)}")

        return {
            'graph': g,
            'entity_id_map': entity_id_map,
            'reverse_entity_map': reverse_entity_map,
            'category_names': category_names,
            'entity_info_dict': entity_info_dict,
            'triplets_by_entity': triplets_by_entity
        }

    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_and_load_model(g, device):
    """创建并加载模型"""
    print("创建和加载模型...")

    try:
        feature_dim = g.ndata['feat'].shape[1]
        print(f"特征维度: {feature_dim}")

        model = SimpleTestModel(
            input_dim=feature_dim,
            hidden_dim=256,
            out_dim=128,
            num_classes=9
        )

        # 加载训练好的模型权重
        checkpoint_path = 'models/model_137d.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']

                # 只加载匹配的参数
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in model_state_dict.items()
                                   if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"  成功加载 {len(pretrained_dict)} 个参数")
            else:
                print("  检查点中没有模型状态字典，使用随机初始化")
        else:
            print(f"  模型文件不存在: {checkpoint_path}")
            print("  使用随机初始化的模型")

        model = model.to(device)
        model.eval()
        print(f"  模型已移动到设备: {device}")

        return model

    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 测试函数 ====================
def test_single_entity():
    """测试单个实体的预测和关系信息"""
    print("=" * 80)
    print("FB15KET实体类型预测测试工具")
    print("=" * 80)

    # 加载所有数据
    data = load_all_data()
    if data is None:
        print("数据加载失败，退出测试")
        return

    g = data['graph']
    entity_id_map = data['entity_id_map']
    reverse_entity_map = data['reverse_entity_map']
    category_names = data['category_names']
    entity_info_dict = data['entity_info_dict']
    triplets_by_entity = data['triplets_by_entity']

    # 创建和加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_and_load_model(g, device)
    if model is None:
        print("模型加载失败，退出测试")
        return

    print(f"\n{'=' * 80}")
    print("数据准备完成！")
    print(f"  设备: {device}")
    print(f"{'=' * 80}")

    # 交互式测试循环
    while True:
        print("\n" + "=" * 60)
        print("交互式测试模式")
        print("输入实体ID（如 /m/0124ld），或输入 'q' 退出")
        print("输入 's' 查看示例实体ID")
        print("=" * 60)

        entity_id = input("\n请输入实体ID: ").strip()

        if entity_id.lower() == 'q':
            print("退出测试工具")
            break

        if entity_id.lower() == 's':
            print("\n示例实体ID (来自数据集的前10个):")
            sample_entities = list(entity_id_map.keys())[:10]
            for eid in sample_entities:
                print(f"  - {eid}")
            continue

        # 检查实体是否存在
        if entity_id not in entity_id_map:
            print(f"错误: 实体ID '{entity_id}' 不存在于图中！")
            print("是否要在三元组中查找包含该字符串的实体？(y/n)")
            choice = input().strip().lower()

            if choice == 'y':
                # 查找包含该字符串的实体
                found_entities = []
                for eid in entity_id_map.keys():
                    if entity_id.lower() in eid.lower():
                        found_entities.append(eid)

                if found_entities:
                    print(f"\n找到 {len(found_entities)} 个包含 '{entity_id}' 的实体:")
                    for i, eid in enumerate(found_entities[:5]):  # 最多显示5个
                        print(f"  {i + 1}. {eid}")
                    if len(found_entities) > 5:
                        print(f"  ... 还有 {len(found_entities) - 5} 个")
                else:
                    print(f"  未找到包含 '{entity_id}' 的实体")
            continue

        node_idx = entity_id_map[entity_id]

        print(f"\n{'=' * 80}")
        print(f"实体: {entity_id}")
        print(f"节点索引: {node_idx}")
        print(f"{'=' * 80}")

        # A. 显示真实类型信息
        print("\n[A] 真实类型信息:")
        if entity_id in entity_info_dict:
            info = entity_info_dict[entity_id]
            cat_id = info['category']
            cat_name = info['name']
            confidence = info['confidence']

            print(f"  类别ID: {cat_id}")
            print(f"  类别名称: {cat_name}")
            if confidence is not None:
                print(f"  置信度: {confidence:.4f}")
        else:
            print(f"  未在Entity_All_typed.csv中找到该实体的类型信息")
            print(f"  注: 只有部分实体有标注的类型信息")

        # B. 显示关系网络
        print(f"\n[B] 关系网络分析:")
        if entity_id in triplets_by_entity:
            relations = triplets_by_entity[entity_id]

            # 统计信息
            relation_counts = Counter([rel for _, rel, _ in relations])
            neighbor_entities = set()
            for _, _, neighbor in relations:
                neighbor_entities.add(neighbor)

            print(f"  总关系数: {len(relations)}")
            print(f"  唯一关系类型: {len(relation_counts)}")
            print(f"  邻居实体数: {len(neighbor_entities)}")

            # 显示最常见的关系
            if relation_counts:
                print(f"\n  最常见的关系类型 (前5):")
                for rel, count in relation_counts.most_common(5):
                    print(f"    {rel}: {count}次")

            # 显示邻居的类别分布
            neighbor_categories = Counter()
            for _, _, neighbor in relations:
                if neighbor in entity_info_dict:
                    neighbor_categories[entity_info_dict[neighbor]['category']] += 1

            if neighbor_categories:
                print(f"\n  邻居实体类别分布:")
                for cat_id, count in neighbor_categories.most_common(5):
                    cat_name = category_names.get(cat_id, f"类别{cat_id}")
                    print(f"    {cat_id}({cat_name}): {count}个")

            # 显示示例关系
            print(f"\n  示例关系 (前10个):")
            for i, (direction, rel, neighbor) in enumerate(relations[:10]):
                direction_symbol = "→" if direction == 'out' else "←"

                # 获取邻居类型
                neighbor_type = "未知"
                if neighbor in entity_info_dict:
                    neighbor_info = entity_info_dict[neighbor]
                    neighbor_type = f"{neighbor_info['category']}({neighbor_info['name']})"

                print(f"    {direction_symbol} [{rel}] {neighbor} [{neighbor_type}]")
        else:
            print(f"  该实体没有任何关系记录")

        # C. 模型预测
        print(f"\n[C] 模型预测结果:")

        with torch.no_grad():
            features = g.ndata['feat'].to(device)
            node_embeddings = model(features)

            # 获取预测
            probs, pred = model.classify(node_embeddings, torch.tensor([node_idx], device=device))

            if probs is not None:
                pred_class = pred.item() + 1  # 转换为1-9的类别编号
                pred_prob = probs[0, pred.item()].item()
                pred_name = category_names.get(pred_class, f"类别{pred_class}")

                print(f"  预测类别: {pred_class} ({pred_name})")
                print(f"  预测概率: {pred_prob:.4f}")

                # 显示所有类别概率
                print(f"\n  所有类别概率详细:")
                sorted_probs = []
                for i in range(9):
                    cat_id = i + 1
                    prob = probs[0, i].item()
                    cat_name = category_names.get(cat_id, f"类别{cat_id}")
                    sorted_probs.append((cat_id, prob, cat_name))

                # 按概率排序
                sorted_probs.sort(key=lambda x: x[1], reverse=True)

                for cat_id, prob, cat_name in sorted_probs:
                    marker = "✓" if cat_id == pred_class else " "
                    star = "*" if (entity_id in entity_info_dict and
                                   cat_id == entity_info_dict[entity_id]['category']) else " "
                    print(f"    {marker}{star} {cat_id:2d}. {cat_name:<25} {prob:.4f}")

                # 比较预测与真实
                if entity_id in entity_info_dict:
                    true_category = entity_info_dict[entity_id]['category']
                    if true_category == pred_class:
                        print(f"\n  ✅ 预测正确！")
                    else:
                        true_name = entity_info_dict[entity_id]['name']
                        print(f"\n  ❌ 预测错误！")
                        print(f"     真实类别: {true_category} ({true_name})")
                        print(f"     预测类别: {pred_class} ({pred_name})")

                        # 计算预测概率中的真实类别概率
                        true_prob = probs[0, true_category - 1].item()

                        # 找到真实类别的排名
                        rank = 1
                        for cat_id, prob, _ in sorted_probs:
                            if cat_id == true_category:
                                break
                            rank += 1

                        print(f"     真实类别的预测概率: {true_prob:.4f} (排名第{rank})")
            else:
                print(f"  预测失败")

        # D. 特征信息
        print(f"\n[D] 特征向量分析:")
        features_vector = g.ndata['feat'][node_idx]

        # 显示9个类别得分
        print(f"  类别得分 (来自Entity_All_typed.csv):")
        for i in range(9):
            cat_id = i + 1
            score = features_vector[i].item()
            cat_name = category_names.get(cat_id, f"类别{cat_id}")

            # 标记最高得分
            if i == torch.argmax(features_vector[:9]).item():
                marker = "★"
            else:
                marker = " "

            print(f"    {marker} {cat_id:2d}. {cat_name:<25} {score:.4f}")

        # 结构特征（如果存在）
        if features_vector.shape[0] > 10:
            print(f"\n  结构特征:")
            for i in range(9, min(14, features_vector.shape[0])):
                feature_value = features_vector[i].item()
                feature_names = ['出度归一化', '入度归一化', '总度归一化', '关系多样性', '关系类型数归一化']
                if i - 9 < len(feature_names):
                    print(f"    {feature_names[i - 9]:<15} {feature_value:.4f}")

        # E. 数据集信息
        print(f"\n[E] 数据集信息:")
        train_mask = g.ndata['train_mask'][node_idx].item()
        test_mask = g.ndata['test_mask'][node_idx].item() if 'test_mask' in g.ndata else False
        label = g.ndata['label'][node_idx].item()

        print(f"  训练集: {'是' if train_mask else '否'}")
        print(f"  测试集: {'是' if test_mask else '否'}")
        print(f"  标签: {label if label != -1 else '无标签'}")

        print(f"\n{'=' * 80}")


def batch_test_entities():
    """批量测试多个实体"""
    print("=" * 80)
    print("FB15KET批量实体测试")
    print("=" * 80)

    # 加载数据
    data = load_all_data()
    if data is None:
        print("数据加载失败，退出测试")
        return

    g = data['graph']
    entity_id_map = data['entity_id_map']
    reverse_entity_map = data['reverse_entity_map']
    category_names = data['category_names']
    entity_info_dict = data['entity_info_dict']

    # 创建和加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_and_load_model(g, device)
    if model is None:
        print("模型加载失败，退出测试")
        return

    print("\n选择测试模式:")
    print("1. 随机测试10个实体")
    print("2. 测试特定类别的实体")
    print("3. 测试预测概率最高的实体")
    print("4. 测试预测概率最低的实体")

    choice = input("请输入选择 (1-4): ").strip()

    # 获取所有有标签的实体
    labeled_indices = []
    labeled_entity_ids = []

    for eid, idx in entity_id_map.items():
        if eid in entity_info_dict:
            labeled_indices.append(idx)
            labeled_entity_ids.append(eid)

    print(f"有标签的实体数: {len(labeled_entity_ids)}")

    test_entities = []

    if choice == '1':
        # 随机选择10个实体
        if len(labeled_entity_ids) < 10:
            test_entities = labeled_entity_ids
        else:
            test_entities = random.sample(labeled_entity_ids, 10)
        print(f"随机选择 {len(test_entities)} 个实体进行测试")

    elif choice == '2':
        # 测试特定类别
        print("\n可用类别:")
        for cat_id, cat_name in category_names.items():
            print(f"  {cat_id}: {cat_name}")

        try:
            selected_cat = int(input("请输入类别编号 (1-9): "))
            if 1 <= selected_cat <= 9:
                # 找到属于该类别的实体
                for eid in labeled_entity_ids:
                    if entity_info_dict[eid]['category'] == selected_cat:
                        test_entities.append(eid)
                        if len(test_entities) >= 10:
                            break
                print(f"找到 {len(test_entities)} 个类别 {selected_cat} 的实体")
            else:
                print("无效的类别编号")
                return
        except:
            print("输入错误")
            return

    elif choice == '3' or choice == '4':
        # 需要先进行预测计算概率
        print("正在计算实体的预测概率...")

        with torch.no_grad():
            features = g.ndata['feat'].to(device)
            node_embeddings = model(features)

            entity_probs = []

            # 限制计算数量
            max_entities = min(1000, len(labeled_indices))
            sample_indices = random.sample(labeled_indices, max_entities)

            for idx in sample_indices:
                probs, pred = model.classify(node_embeddings, torch.tensor([idx], device=device))
                if probs is not None:
                    prob = probs[0, pred.item()].item()
                    eid = reverse_entity_map.get(idx)
                    if eid:
                        entity_probs.append((eid, prob))

            # 按概率排序
            if choice == '3':  # 最高概率
                entity_probs.sort(key=lambda x: x[1], reverse=True)
                test_entities = [eid for eid, _ in entity_probs[:10]]
                print(f"选择预测概率最高的10个实体")
            else:  # 最低概率
                entity_probs.sort(key=lambda x: x[1])
                test_entities = [eid for eid, _ in entity_probs[:10]]
                print(f"选择预测概率最低的10个实体")

    else:
        print("无效选择")
        return

    if not test_entities:
        print("没有找到符合条件的实体")
        return

    print(f"\n测试 {len(test_entities)} 个实体:")
    print("-" * 80)

    # 批量测试
    results = []
    with torch.no_grad():
        features = g.ndata['feat'].to(device)
        node_embeddings = model(features)

        for eid in test_entities:
            if eid in entity_id_map:
                node_idx = entity_id_map[eid]

                # 预测
                probs, pred = model.classify(node_embeddings, torch.tensor([node_idx], device=device))

                if probs is not None:
                    pred_class = pred.item() + 1
                    pred_prob = probs[0, pred.item()].item()
                    pred_name = category_names.get(pred_class, f"类别{pred_class}")

                    # 真实信息
                    true_info = entity_info_dict.get(eid, {})
                    true_category = true_info.get('category', '未知')
                    true_name = true_info.get('name', '未知')

                    if true_category != '未知':
                        if true_category == pred_class:
                            result = "✓"
                        else:
                            result = "✗"
                    else:
                        result = "?"

                    results.append({
                        '实体': eid,
                        '预测类别': f"{pred_class}({pred_name[:15]}...)",
                        '预测概率': f"{pred_prob:.3f}",
                        '真实类别': f"{true_category}({true_name[:15]}...)" if true_category != '未知' else "未知",
                        '结果': result
                    })

    # 显示结果表格
    if results:
        print("\n测试结果:")
        print("-" * 100)
        print(f"{'实体ID':<30} {'预测类别':<25} {'概率':<8} {'真实类别':<25} {'结果':<3}")
        print("-" * 100)

        for res in results:
            print(
                f"{res['实体']:<30} {res['预测类别']:<25} {res['预测概率']:<8} {res['真实类别']:<25} {res['结果']:<3}")

        # 统计准确率（仅对有真实标签的）
        if choice == '1' or choice == '2':
            correct = sum(1 for res in results if res['结果'] == '✓')
            total = len([res for res in results if res['真实类别'] != '未知'])
            if total > 0:
                accuracy = correct / total
                print(f"\n准确率: {correct}/{total} = {accuracy:.2%}")

        # 保存结果
        save_choice = input("\n是否保存结果到文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.csv"

            results_df = pd.DataFrame(results)
            results_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"结果已保存到: {filename}")


def analyze_prediction_patterns():
    """分析预测模式"""
    print("=" * 80)
    print("预测模式分析工具")
    print("=" * 80)

    # 加载数据
    data = load_all_data()
    if data is None:
        print("数据加载失败，退出分析")
        return

    g = data['graph']
    entity_id_map = data['entity_id_map']
    reverse_entity_map = data['reverse_entity_map']
    category_names = data['category_names']
    entity_info_dict = data['entity_info_dict']

    # 创建和加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_and_load_model(g, device)
    if model is None:
        print("模型加载失败，退出分析")
        return

    print("\n选择分析模式:")
    print("1. 查看每个类别的预测准确率")
    print("2. 查看预测置信度分布")
    print("3. 查看最容易混淆的类别对")
    print("4. 返回主菜单")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == '1':
        # 按类别统计准确率
        category_correct = [0] * 9
        category_total = [0] * 9

        print("\n正在计算各类别准确率...")

        with torch.no_grad():
            features = g.ndata['feat'].to(device)
            node_embeddings = model(features)

            # 分批处理
            batch_size = 1000
            all_indices = list(entity_id_map.values())

            for i in range(0, len(all_indices), batch_size):
                batch_indices = all_indices[i:i + batch_size]
                probs, preds = model.classify(node_embeddings, torch.tensor(batch_indices, device=device))

                if probs is not None:
                    pred_classes = preds.cpu().numpy() + 1

                    for j, idx in enumerate(batch_indices):
                        eid = reverse_entity_map.get(idx)
                        if eid and eid in entity_info_dict:
                            true_class = entity_info_dict[eid]['category']
                            pred_class = pred_classes[j]

                            category_total[true_class - 1] += 1
                            if true_class == pred_class:
                                category_correct[true_class - 1] += 1

        print("\n各类别准确率统计:")
        print("-" * 80)
        print(f"{'类别ID':<8} {'类别名称':<25} {'准确率':<10} {'样本数':<10}")
        print("-" * 80)

        total_correct = 0
        total_samples = 0

        for i in range(9):
            cat_id = i + 1
            cat_name = category_names.get(cat_id, f"类别{cat_id}")
            if category_total[i] > 0:
                accuracy = category_correct[i] / category_total[i]
                total_correct += category_correct[i]
                total_samples += category_total[i]
                print(f"{cat_id:<8} {cat_name:<25} {accuracy:.4f}     {category_total[i]:<10}")

        if total_samples > 0:
            total_accuracy = total_correct / total_samples
            print(f"\n总体准确率: {total_accuracy:.4f} ({total_correct}/{total_samples})")

    elif choice == '2':
        # 预测置信度分布
        print("\n正在计算预测置信度...")

        with torch.no_grad():
            features = g.ndata['feat'].to(device)
            node_embeddings = model(features)

            # 采样一些实体
            sample_size = min(1000, len(entity_id_map))
            sample_indices = random.sample(list(entity_id_map.values()), sample_size)
            probs, preds = model.classify(node_embeddings, torch.tensor(sample_indices, device=device))

            if probs is not None:
                # 计算最大概率
                max_probs = probs.max(dim=1).values.cpu().numpy()

                # 统计分布
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                hist, _ = np.histogram(max_probs, bins=bins)

                print("\n预测置信度分布:")
                print("-" * 50)
                for i in range(len(hist)):
                    lower = bins[i]
                    upper = bins[i + 1]
                    count = hist[i]
                    percentage = count / sample_size * 100
                    print(f"  {lower:.1f} - {upper:.1f}: {count} 个 ({percentage:.1f}%)")

                # 平均置信度
                avg_confidence = max_probs.mean()
                print(f"\n平均置信度: {avg_confidence:.4f}")

    elif choice == '3':
        print("此功能正在开发中...")

    elif choice == '4':
        return

    else:
        print("无效选择")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("FB15KET实体类型预测测试工具")
    print("=" * 80)

    while True:
        print("\n选择测试模式:")
        print("1. 单个实体交互测试")
        print("2. 批量实体测试")
        print("3. 预测模式分析")
        print("4. 退出程序")

        choice = input("请输入选择 (1-4): ").strip()

        if choice == '1':
            test_single_entity()
        elif choice == '2':
            batch_test_entities()
        elif choice == '3':
            analyze_prediction_patterns()
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效选择")