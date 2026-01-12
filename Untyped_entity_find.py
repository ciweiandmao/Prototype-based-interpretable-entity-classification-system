# find_true_unclassified_entities.py
import os
import csv


def find_true_unclassified_entities(entity_file, classification_file):
    """找出真正未分类的实体（所有tag都未分类）"""

    # 读取分类文件，构建已分类的tag集合
    classified_tags = set()
    with open(classification_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row['Tag']
            # 检查tag是否有分类（类别列不为空）
            has_classification = False

            # 检查可能的列名
            for col in ['类别', 'category', 'Category']:
                if col in row and row[col] and row[col].strip():
                    has_classification = True
                    break

            if has_classification:
                # 记录tag，去除开头的斜杠
                if tag.startswith('/'):
                    classified_tags.add(tag[1:])  # 去除开头的斜杠
                else:
                    classified_tags.add(tag)

    print(f"已分类的tag数量: {len(classified_tags)}")

    # 读取实体文件，找出所有tag都未分类的实体
    unclassified_entities = []
    classified_entities = []

    with open(entity_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                entity_id = parts[0].strip()
                tags_str = parts[1].strip()

                if not tags_str:
                    # 没有tag的实体视为未分类
                    unclassified_entities.append({
                        'entity_id': entity_id,
                        'tags': [],
                        'line_num': line_num,
                        'reason': 'no_tags'
                    })
                    continue

                # 分割复合tag（用|分隔）
                composite_tags = tags_str.split('|')
                all_tags = set()

                # 提取所有基本tag
                for composite in composite_tags:
                    if composite.startswith('/'):
                        composite = composite[1:]  # 去除开头的斜杠

                    # 分割路径的各个部分
                    segments = composite.split('/')
                    for segment in segments:
                        if segment:  # 跳过空段
                            all_tags.add(segment)

                # 检查是否有任何一个tag被分类
                entity_classified = False
                classified_tag_list = []

                for tag in all_tags:
                    if tag in classified_tags:
                        entity_classified = True
                        classified_tag_list.append(tag)

                if entity_classified:
                    classified_entities.append({
                        'entity_id': entity_id,
                        'classified_tags': classified_tag_list,
                        'all_tags': list(all_tags),
                        'line_num': line_num
                    })
                else:
                    # 所有tag都未分类
                    unclassified_entities.append({
                        'entity_id': entity_id,
                        'tags': list(all_tags),
                        'line_num': line_num,
                        'reason': 'no_tag_classified'
                    })

    # 输出结果
    output_file = "true_unclassified_entities.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== 实体分类统计 ===\n")
        f.write(f"总实体数: {len(unclassified_entities) + len(classified_entities)}\n")
        f.write(f"已分类实体: {len(classified_entities)}\n")
        f.write(f"未分类实体: {len(unclassified_entities)}\n")
        f.write(
            f"分类覆盖率: {len(classified_entities) / (len(unclassified_entities) + len(classified_entities)) * 100:.1f}%\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"=== 未分类实体详情 ===")

        # 按tag数量排序
        unclassified_entities.sort(key=lambda x: len(x['tags']), reverse=True)

        for entity in unclassified_entities:
            f.write(f"\n实体ID: {entity['entity_id']}\n")
            f.write(f"行号: {entity['line_num']}\n")
            f.write(f"Tag数量: {len(entity['tags'])}\n")
            f.write(f"Tag列表: {' | '.join(sorted(entity['tags']))}\n")
            f.write(f"原因: {entity['reason']}\n")
            f.write("-" * 40 + "\n")

    # 输出统计信息
    print(f"\n{'=' * 60}")
    print(f"分析完成!")
    print(f"总实体数: {len(unclassified_entities) + len(classified_entities)}")
    print(f"已分类实体: {len(classified_entities)}")
    print(f"未分类实体: {len(unclassified_entities)}")
    print(
        f"分类覆盖率: {len(classified_entities) / (len(unclassified_entities) + len(classified_entities)) * 100:.1f}%")
    print(f"结果已保存到: {output_file}")

    # 显示前10个未分类实体示例
    if unclassified_entities:
        print("\n前10个未分类实体示例:")
        for i, entity in enumerate(unclassified_entities[:10], 1):
            print(f"{i}. {entity['entity_id']}")
            print(f"   Tag数量: {len(entity['tags'])}")
            print(f"   Tags: {', '.join(sorted(entity['tags'])[:5])}")
            if len(entity['tags']) > 5:
                print(f"   ... +{len(entity['tags']) - 5}个更多")
            print()

    return unclassified_entities, classified_entities


def analyze_tag_distribution(unclassified_entities):
    """分析未分类实体中的tag分布"""
    tag_frequency = {}

    for entity in unclassified_entities:
        for tag in entity['tags']:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

    # 按频率排序
    sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)

    print("\n未分类实体中最常见的tag:")
    print(f"{'Tag':<40} {'出现次数':<10} {'实体占比':<10}")
    print("-" * 60)

    total_entities = len(unclassified_entities)
    for tag, freq in sorted_tags[:20]:
        percentage = freq / total_entities * 100
        print(f"{tag:<40} {freq:<10} {percentage:.1f}%")

    # 输出到文件
    tag_file = "unclassified_tags_frequency.txt"
    with open(tag_file, 'w', encoding='utf-8') as f:
        f.write("未分类实体中最常见的tag:\n")
        f.write("=" * 60 + "\n")
        for tag, freq in sorted_tags:
            percentage = freq / total_entities * 100
            f.write(f"{tag:<40} {freq:<10} {percentage:.1f}%\n")

    print(f"\n完整tag频率列表已保存到: {tag_file}")

    return tag_frequency


def main():
    entity_file = r"data\FB15KET\Entity_All_en_single.txt"
    classification_file = r"data\FB15KET\Tag_Classifications.csv"

    if not os.path.exists(entity_file):
        print(f"错误: 实体文件不存在: {entity_file}")
        return

    if not os.path.exists(classification_file):
        print(f"错误: 分类文件不存在: {classification_file}")
        return

    print("开始分析真正未分类的实体...")
    print("(只要实体有一个tag被分类，该实体就算已分类)")
    print("=" * 60)

    # 找出未分类实体
    unclassified, classified = find_true_unclassified_entities(entity_file, classification_file)

    # 分析tag分布
    if unclassified:
        analyze_tag_distribution(unclassified)

        # 找出需要优先分类的关键tag
        print(f"\n{'=' * 60}")
        print("建议优先分类以下tag（覆盖最多未分类实体）:")

        # 计算每个tag能覆盖多少未分类实体
        tag_coverage = {}
        tag_entities = {}

        for tag in set(tag for entity in unclassified for tag in entity['tags']):
            covered_entities = []
            for entity in unclassified:
                if tag in entity['tags']:
                    covered_entities.append(entity['entity_id'])

            tag_coverage[tag] = len(covered_entities)
            tag_entities[tag] = covered_entities

        # 按覆盖实体数排序
        sorted_coverage = sorted(tag_coverage.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Tag':<40} {'覆盖实体数':<12} {'覆盖率':<10}")
        print("-" * 60)

        total_unclassified = len(unclassified)
        for i, (tag, coverage) in enumerate(sorted_coverage[:15], 1):
            coverage_pct = coverage / total_unclassified * 100
            print(f"{i:2}. {tag:<36} {coverage:<12} {coverage_pct:.1f}%")

        # 保存优先分类列表
        priority_file = "priority_tags_to_classify.txt"
        with open(priority_file, 'w', encoding='utf-8') as f:
            f.write("建议优先分类的tag（按覆盖实体数排序）:\n")
            f.write("=" * 60 + "\n")
            for tag, coverage in sorted_coverage:
                coverage_pct = coverage / total_unclassified * 100
                f.write(f"{tag:<40} {coverage:<10} {coverage_pct:.1f}%\n")

        print(f"\n优先分类列表已保存到: {priority_file}")


if __name__ == "__main__":
    main()