# tag_classification_v2.py
import os
import json
import csv

categories = {
    1: "人物与生命（Person & Life）",
    2: "组织与机构（Organization）",
    3: "地点与地理（Location）",
    4: "创作与娱乐作品（Creative Work）",
    5: "事件与活动（Event）",
    6: "学科与概念（Concept & Subject）",
    7: "物品与产品（Product & Object）",
    8: "属性与度量（Attribute & Measurement）",
    9: "其他（Others）"
}


def load_tag_data(csv_file):
    """从CSV加载tag数据"""
    tag_data = []
    total_count = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 确保有需要的字段
            if 'Tag' not in row or '出现次数' not in row:
                print(f"警告: CSV缺少必要字段，请检查文件格式")
                continue

            tag = row['Tag'].strip()
            count = int(row['出现次数'])
            chinese_desc = row.get('中文描述', '').strip()

            # 初始数据结构
            tag_data.append({
                'tag': tag,
                'chinese_description': chinese_desc,
                'count': count,
                'category': 0,  # 未分类
                'category_name': ''
            })
            total_count += count

    # 计算权重（基于频率）
    for tag_info in tag_data:
        if total_count > 0:
            weight = (tag_info['count'] / total_count) * 100
            tag_info['weight'] = round(weight, 4)
        else:
            tag_info['weight'] = 0.0

    # 按出现次数降序排序
    tag_data.sort(key=lambda x: x['count'], reverse=True)

    print(f"加载了 {len(tag_data)} 个tag，总出现次数: {total_count}")
    return tag_data


def load_progress():
    """加载分类进度"""
    progress_file = "tag_classify_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_tags": [], "classifications": {}}


def save_progress(processed_tags, classifications):
    """保存进度"""
    with open("tag_classify_progress.json", 'w', encoding='utf-8') as f:
        json.dump({
            "processed_tags": processed_tags,
            "classifications": classifications
        }, f)


def save_to_csv(tag_data, output_file):
    """保存带分类的结果到CSV"""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Tag', '出现次数', '中文描述', '权重', '类别', '类别名称'
        ])
        writer.writeheader()

        for tag_info in tag_data:
            writer.writerow({
                'Tag': tag_info['tag'],
                '出现次数': tag_info['count'],
                '中文描述': tag_info['chinese_description'],
                '权重': tag_info['weight'],
                '类别': tag_info['category'] if tag_info['category'] > 0 else '',
                '类别名称': tag_info['category_name'] if tag_info['category'] > 0 else ''
            })


def main():
    input_csv = r"data\FB15KET\Tag_Statistics.csv"
    output_csv = r"data\FB15KET\Tag_Classifications.csv"

    if not os.path.exists(input_csv):
        print(f"错误: 输入文件不存在: {input_csv}")
        return

    # 加载数据
    print("从CSV加载tag数据...")
    tag_data = load_tag_data(input_csv)
    total = len(tag_data)

    if total == 0:
        print("没有找到tag数据!")
        return

    # 加载进度
    progress = load_progress()
    processed_tags = set(progress["processed_tags"])
    classifications = progress["classifications"]

    # 更新已分类的tag
    for tag_info in tag_data:
        tag = tag_info['tag']
        if tag in classifications:
            tag_info['category'] = int(classifications[tag]['category'])
            tag_info['category_name'] = classifications[tag]['category_name']

    print(f"总tag数: {total}")
    print(f"已分类: {len(classifications)}")
    print(f"剩余: {total - len(classifications)}")
    print("=" * 60)

    try:
        for i, tag_info in enumerate(tag_data):
            tag = tag_info['tag']

            if tag in processed_tags:
                continue

            # 显示进度
            current = len(processed_tags) + 1
            progress_pct = current / total * 100

            print(f"\n{'=' * 60}")
            print(f"[{current}/{total}] 进度: {progress_pct:.1f}%")
            print(f"Tag: {tag}")
            print(f"中文描述: {tag_info['chinese_description']}")
            print(f"出现次数: {tag_info['count']:,}")
            print(f"权重: {tag_info['weight']:.4f}")
            print("-" * 40)

            # 如果已有分类，显示当前分类
            if tag in classifications:
                current_cat = classifications[tag]['category']
                current_name = classifications[tag]['category_name']
                print(f"当前分类: {current_cat}: {current_name}")

            # 显示类别选项
            for num, name in categories.items():
                print(f"  {num}: {name}")

            while True:
                choice = input("\n请选择类别 (1-9, s跳过, b返回上一个, q退出): ").strip().lower()

                if choice == 'q':
                    print("退出并保存进度...")
                    save_progress(list(processed_tags), classifications)
                    save_to_csv(tag_data, output_csv)
                    return
                elif choice == 's':
                    print("跳过此tag")
                    processed_tags.add(tag)
                    break
                elif choice == 'b' and i > 0:
                    print("返回上一个tag")
                    # 移除上一个tag的标记
                    prev_tag = tag_data[i - 1]['tag']
                    if prev_tag in processed_tags:
                        processed_tags.remove(prev_tag)
                    if prev_tag in classifications:
                        del classifications[prev_tag]
                    # 回退到上一个tag
                    continue_outer = True
                    break
                elif choice.isdigit() and 1 <= int(choice) <= 9:
                    cat_num = int(choice)
                    cat_name = categories[cat_num]

                    # 保存分类
                    classifications[tag] = {
                        'category': cat_num,
                        'category_name': cat_name
                    }

                    # 更新tag数据
                    tag_info['category'] = cat_num
                    tag_info['category_name'] = cat_name

                    processed_tags.add(tag)
                    print(f"已分类: {cat_num}: {cat_name}")
                    break
                else:
                    print("无效输入，请重试")

            # 每处理10个tag自动保存一次
            if current % 10 == 0:
                print("自动保存进度...")
                save_progress(list(processed_tags), classifications)
                save_to_csv(tag_data, output_csv)

    except KeyboardInterrupt:
        print("\n\n用户中断，保存进度...")

    finally:
        # 最终保存
        save_progress(list(processed_tags), classifications)
        save_to_csv(tag_data, output_csv)

        print(f"\n完成! 已分类 {len(classifications)}/{total} 个tag")
        print(f"结果保存到: {output_csv}")

        # 显示各类别统计
        print("\n类别统计:")
        from collections import Counter
        category_counts = Counter([info['category'] for info in classifications.values()])

        total_classified = sum(category_counts.values())
        for num in range(1, 10):
            count = category_counts.get(num, 0)
            if count > 0:
                percentage = count / total_classified * 100
                print(f"  {categories[num]}: {count} 个 ({percentage:.1f}%)")


if __name__ == "__main__":
    main()