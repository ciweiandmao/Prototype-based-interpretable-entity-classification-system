# manual_classify_bilingual.py
import os
import json

categories = {
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


def load_data(zh_file, en_file):
    """加载中英文数据并确保对齐"""
    zh_data = {}
    en_data = {}

    # 读取中文文件
    with open(zh_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    zh_data[parts[0]] = parts[1]

    # 读取英文文件
    with open(en_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    en_data[parts[0]] = parts[1]

    # 检查两个文件实体是否一致
    zh_entities = set(zh_data.keys())
    en_entities = set(en_data.keys())

    if zh_entities != en_entities:
        print(f"警告: 中英文文件实体不一致!")
        print(f"中文特有实体: {zh_entities - en_entities}")
        print(f"英文特有实体: {en_entities - zh_entities}")
        # 只保留共有的实体
        common_entities = zh_entities & en_entities
        zh_data = {e: zh_data[e] for e in common_entities}
        en_data = {e: en_data[e] for e in common_entities}

    # 转换为列表并按实体ID排序
    entities = sorted(zh_data.keys())
    data_pairs = [(e, zh_data[e], en_data[e]) for e in entities]

    return data_pairs


def load_progress():
    """加载进度"""
    progress_file = "classify_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": [], "labels": {}}


def save_progress(processed, labels):
    """保存进度"""
    with open("classify_progress.json", 'w', encoding='utf-8') as f:
        json.dump({"processed": processed, "labels": labels}, f)


def main():
    zh_file = r"data\FB15KET\Entity_All_zh_single.txt"
    en_file = r"data\FB15KET\Entity_All_en_single.txt"
    output_file = r"data\FB15KET\Entity_All_typed.txt"

    # 加载数据
    print("加载数据...")
    data = load_data(zh_file, en_file)
    total = len(data)

    if total == 0:
        print("没有找到数据!")
        return

    print(f"总实体数: {total}")

    # 加载进度
    progress = load_progress()
    processed_ids = set(progress["processed"])
    labels = progress["labels"]

    print(f"已处理: {len(processed_ids)}")
    print(f"剩余: {total - len(processed_ids)}")
    print("=" * 60)

    try:
        for i, (entity_id, zh_desc, en_desc) in enumerate(data):
            if entity_id in processed_ids:
                continue

            # 显示进度
            current = len(processed_ids) + 1
            progress_pct = current / total * 100

            print(f"\n{'=' * 60}")
            print(f"[{current}/{total}] 进度: {progress_pct:.1f}%")
            print(f"实体ID: {entity_id}")
            print("-" * 40)
            print(f"中文描述: {zh_desc}")
            print(f"英文描述: {en_desc}")
            print("-" * 40)

            # 显示类别选项
            for num, name in categories.items():
                print(f"  {num}: {name}")

            while True:
                choice = input("\n请选择类别 (1-9, s跳过, q退出): ").strip().lower()

                if choice == 'q':
                    print("退出并保存进度...")
                    save_progress(list(processed_ids), labels)
                    return
                elif choice == 's':
                    print("跳过此实体")
                    processed_ids.add(entity_id)
                    break
                elif choice.isdigit() and 1 <= int(choice) <= 9:
                    cat_num = int(choice)
                    labels[entity_id] = f"{cat_num}: {categories[cat_num]}"
                    processed_ids.add(entity_id)
                    print(f"已标注: {categories[cat_num]}")
                    break
                else:
                    print("无效输入，请重试")

    except KeyboardInterrupt:
        print("\n\n用户中断，保存进度...")

    finally:
        # 保存进度和最终结果
        save_progress(list(processed_ids), labels)

        # 写入最终结果文件
        print(f"\n写入结果文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entity_id, zh_desc, en_desc in data:
                if entity_id in labels:
                    # 保存标注结果
                    f.write(f"{entity_id}\t{labels[entity_id]}\n")
                else:
                    # 未标注的保持原样（中文描述）
                    f.write(f"{entity_id}\t{zh_desc}\n")

        print(f"完成! 已处理 {len(processed_ids)}/{total} 个实体")
        print(f"结果保存到: {output_file}")

        # 显示各类别统计
        print("\n类别统计:")
        category_counts = {}
        for label in labels.values():
            cat_num = label.split(':')[0]
            category_counts[cat_num] = category_counts.get(cat_num, 0) + 1

        for num in range(1, 10):
            count = category_counts.get(str(num), 0)
            if count > 0:
                print(f"  {categories[num]}: {count} 个")


if __name__ == "__main__":
    main()