# predict_entity_classification_fixed.py
import os
import csv
import math
from collections import defaultdict, Counter


def load_tag_classifications(classification_file):
    """加载tag分类数据"""
    tag_data = {}  # tag -> {category, weight, category_name}

    with open(classification_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row['Tag'].strip()

            # 检查类别列（尝试不同可能的列名）
            category = None
            category_name = None

            # 可能的类别列名
            category_cols = ['类别', 'category', 'Category']
            name_cols = ['类别名称', 'category_name', 'Category_Name']

            for col in category_cols:
                if col in row and row[col] and row[col].strip():
                    try:
                        category = int(float(row[col]))
                        break
                    except:
                        continue

            for col in name_cols:
                if col in row and row[col] and row[col].strip():
                    category_name = row[col].strip()
                    break

            # 检查权重列
            weight = 0.0
            weight_cols = ['权重', 'weight', 'Weight']
            for col in weight_cols:
                if col in row and row[col] and row[col].strip():
                    try:
                        weight = float(row[col])
                        break
                    except:
                        continue

            if category and category_name:
                tag_data[tag] = {
                    'category': category,
                    'category_name': category_name,
                    'weight': weight if weight > 0 else 1.0  # 默认权重为1
                }

    print(f"加载了 {len(tag_data)} 个已分类的tag")
    return tag_data


def predict_entity_category(entity_tags, tag_data):
    """根据实体包含的tag预测实体类别"""
    if not entity_tags:
        return None, None, 0.0, {}

    category_scores = defaultdict(float)
    all_basic_tags = set()
    matched_tags = []

    for composite_tag in entity_tags:
        if not composite_tag:
            continue

        if composite_tag.startswith('/'):
            composite_tag = composite_tag[1:]

        segments = composite_tag.split('/')
        for segment in segments:
            if segment:
                all_basic_tags.add(segment)

    print(f"从实体提取的基本tag: {list(all_basic_tags)[:10]}...")  # 显示前10个
    print(f"tag_data中的已知tag: {list(tag_data.keys())[:10]}...")  # 显示前10个

    for tag in all_basic_tags:
        if tag in tag_data:
            matched_tags.append(tag)
            category = tag_data[tag]['category']
            weight = tag_data[tag]['weight']
            score = math.log(weight + 1) if weight > 0 else 1.0
            category_scores[category] += score

    print(f"成功匹配的tag: {matched_tags}")
    print(f"匹配到的类别得分: {dict(category_scores)}")

    # ... 其余代码不变 ...

    if not category_scores:
        return None, None, 0.0, {}

    # 选择得分最高的类别
    best_category = max(category_scores.items(), key=lambda x: x[1])
    category_num = best_category[0]
    total_score = best_category[1]

    # 获取类别名称
    category_name = None
    for tag in all_basic_tags:
        if tag in tag_data and tag_data[tag]['category'] == category_num:
            category_name = tag_data[tag]['category_name']
            break

    # 计算置信度（归一化）
    total_all_scores = sum(category_scores.values())
    confidence = total_score / total_all_scores if total_all_scores > 0 else 0.0

    return category_num, category_name, confidence, category_scores


def process_entities(entity_file, tag_data, output_file):
    """处理所有实体，预测类别"""
    results = []
    classification_stats = Counter()

    with open(entity_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                entity_id = parts[0].strip()
                tags_str = parts[1].strip()

                tags = tags_str.split('|') if tags_str else []

                # 预测类别
                category, category_name, confidence, all_scores = predict_entity_category(tags, tag_data)

                # 准备结果行
                result = {
                    'entity_id': entity_id,
                    'tags': tags_str,
                    'predicted_category': category if category else '',
                    'predicted_category_name': category_name if category_name else '',
                    'confidence_score': round(confidence, 4) if confidence else 0.0
                }

                # 添加所有类别的得分
                if all_scores:
                    for cat, cat_score in all_scores.items():
                        result[f'category_{cat}_score'] = round(cat_score, 4)

                results.append(result)

                # 统计
                if category:
                    classification_stats[category] += 1

                # 显示进度
                if line_num % 1000 == 0:
                    print(f"已处理 {line_num}/{total_lines} 个实体 ({line_num / total_lines * 100:.1f}%)...")

    # 保存到CSV
    if results:
        # 提取所有可能的列名
        all_columns = set()
        for result in results:
            all_columns.update(result.keys())

        # 固定列顺序
        base_columns = [
            'entity_id',
            'predicted_category',
            'predicted_category_name',
            'confidence_score',
            'tags'
        ]

        # 其他列按类别排序
        other_columns = sorted([col for col in all_columns if col not in base_columns])
        fieldnames = base_columns + other_columns

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # 确保所有列都存在
                row = {}
                for col in fieldnames:
                    row[col] = result.get(col, 0.0 if 'score' in col else '')
                writer.writerow(row)

    print(f"\n预测完成!")
    print(f"总实体数: {len(results)}")
    print(f"已分类实体: {sum(classification_stats.values())}")
    print(f"未分类实体: {len(results) - sum(classification_stats.values())}")

    return results, classification_stats


def main():
    entity_file = r"data\FB15KET\Entity_All_en_single.txt"
    classification_file = r"data\FB15KET\Tag_Classifications.csv"
    output_file = r"data\FB15KET\Entity_All_typed.csv"

    if not os.path.exists(entity_file):
        print(f"错误: 实体文件不存在: {entity_file}")
        return

    if not os.path.exists(classification_file):
        print(f"错误: 分类文件不存在: {classification_file}")
        return

    print("开始预测实体分类...")
    print(f"输入实体文件: {entity_file}")
    print(f"输入分类文件: {classification_file}")
    print("=" * 60)

    # 加载tag分类数据
    tag_data = load_tag_classifications(classification_file)

    if not tag_data:
        print("警告: 未找到已分类的tag数据!")
        return

    # 处理实体
    results, stats = process_entities(entity_file, tag_data, output_file)

    print(f"\n结果已保存到: {output_file}")

    # 显示统计信息
    if stats:
        print(f"\n分类统计:")
        print(f"{'类别':<30} {'数量':<10} {'占比':<10}")
        print("-" * 50)

        total_entities = len(results)
        total_classified = sum(stats.values())

        category_names = {
            1: "人物（Person）",
            2: "组织与机构（Organization）",
            3: "地点与地理（Location）",
            4: "创作与娱乐作品（Creative Work）",
            5: "事件与活动（Event）",
            6: "学科与概念（Concept & Subject）",
            7: "物品与产品（Product & Object）",
            8: "属性与度量（Attribute & Measurement）",
            9: "其他（Others）"
        }

        for category in sorted(stats.keys()):
            count = stats[category]
            percentage = count / total_entities * 100
            category_name = category_names.get(category, f"类别{category}")
            print(f"{category_name:<30} {count:<10} {percentage:.1f}%")

        print(f"\n分类覆盖率: {total_classified / total_entities * 100:.1f}%")
        print(
            f"平均置信度: {sum(r['confidence_score'] for r in results if r['predicted_category']) / (total_classified if total_classified > 0 else 1):.3f}")

    # 显示示例
    #if results:
    if 0:
        print(f"\n前5个实体预测示例:")
        displayed = 0
        for result in results:
            if result['predicted_category']:
                category_str = f"{result['predicted_category']}: {result['predicted_category_name']}"
                print(f"{displayed + 1}. {result['entity_id']}")
                print(f"   预测类别: {category_str}")
                print(f"   置信度: {result['confidence_score']:.4f}")

                # 显示前2个类别的得分
                category_scores = []
                for key, value in result.items():
                    if key.startswith('category_') and key.endswith('_score') and value > 0:
                        cat_num = int(key.replace('category_', '').replace('_score', ''))
                        category_scores.append((cat_num, value))

                if category_scores:
                    category_scores.sort(key=lambda x: x[1], reverse=True)
                    scores_str = ', '.join([f"类别{cat}:{score:.2f}" for cat, score in category_scores[:3]])
                    print(f"   得分: {scores_str}")

                tags_preview = result['tags'][:100] + "..." if len(result['tags']) > 100 else result['tags']
                print(f"   Tags预览: {tags_preview[:80]}...")
                print()

                displayed += 1
                if displayed >= 5:
                    break


if __name__ == "__main__":
    main()