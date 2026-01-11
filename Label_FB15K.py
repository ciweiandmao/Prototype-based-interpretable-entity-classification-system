# label_fb15k.py
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict


class FB15KLabeler:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = self._read_mapping('entity2id.txt')
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = self._read_mapping('relation2id.txt')

        # 8个类别定义
        self.categories = {
            1: "人物（Person）",
            2: "组织与机构（Organization）",
            3: "地点与地理（Location）",
            4: "创作与娱乐作品（Creative Work）",
            5: "事件与活动（Event）",
            6: "学科与概念（Concept & Subject）",
            7: "物品与产品（Product & Object）",
            8: "属性与度量（Attribute & Measurement）"
        }

        # 加载数据
        self.train_triples = self._read_triples('train.txt')
        self.valid_triples = self._read_triples('valid.txt')
        self.test_triples = self._read_triples('test.txt')

        # 统计实体出现次数
        self.entity_counts = self._count_entity_appearances()

    def _read_mapping(self, filename):
        """读取映射文件"""
        mapping = {}
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    def _read_triples(self, filename):
        """读取三元组文件"""
        triples = []
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h in self.entity2id and r in self.relation2id and t in self.entity2id:
                        triples.append((h, r, t))
        return triples

    def _count_entity_appearances(self):
        """统计实体在所有三元组中出现的次数"""
        entity_counts = defaultdict(int)

        # 统计训练集
        for h, r, t in self.train_triples:
            entity_counts[h] += 1
            entity_counts[t] += 1

        # 统计验证集
        for h, r, t in self.valid_triples:
            entity_counts[h] += 1
            entity_counts[t] += 1

        # 统计测试集
        for h, r, t in self.test_triples:
            entity_counts[h] += 1
            entity_counts[t] += 1

        return dict(entity_counts)

    def display_entity_info(self, entity_name):
        """显示实体的详细信息"""
        print(f"\n{'=' * 60}")
        print(f"实体名称: {entity_name}")
        print(f"实体ID: {self.entity2id[entity_name]}")
        print(f"出现次数: {self.entity_counts.get(entity_name, 0)}")

        # 显示相关三元组
        print(f"\n相关三元组:")
        count = 0
        for h, r, t in self.train_triples:
            if h == entity_name or t == entity_name:
                print(f"  {h} -- {r} --> {t}")
                count += 1
                if count >= 5:  # 最多显示5个
                    break

        # 显示类别选项
        print(f"\n请选择类别 (1-8):")
        for cat_id, cat_name in self.categories.items():
            print(f"  {cat_id}: {cat_name}")

    def manual_labeling(self, sample_size=1000):
        """手动标注实体"""
        # 获取所有实体
        all_entities = list(self.entity2id.keys())

        # 按出现频率排序，优先标注高频实体
        sorted_entities = sorted(
            all_entities,
            key=lambda x: self.entity_counts.get(x, 0),
            reverse=True
        )

        # 如果sample_size大于实体总数，使用全部实体
        if sample_size > len(sorted_entities):
            sample_size = len(sorted_entities)

        # 选择前sample_size个实体进行标注
        entities_to_label = sorted_entities[:sample_size]

        print(f"需要标注 {len(entities_to_label)} 个实体")
        print("输入 's' 跳过，'q' 退出，'b' 返回上一个")

        entity_labels = {}
        current_idx = 0
        skip_count = 0

        while current_idx < len(entities_to_label):
            entity = entities_to_label[current_idx]

            # 如果实体已经有标签，跳过
            if entity in entity_labels:
                current_idx += 1
                continue

            # 显示实体信息
            self.display_entity_info(entity)

            # 获取用户输入
            user_input = input("\n请输入类别编号 (1-8): ").strip().lower()

            if user_input == 'q':
                print("退出标注")
                break
            elif user_input == 's':
                print(f"跳过实体: {entity}")
                skip_count += 1
                current_idx += 1
                continue
            elif user_input == 'b':
                if current_idx > 0:
                    current_idx -= 1
                    print("返回上一个实体")
                    # 移除上一个实体的标签以便重新标注
                    if current_idx >= 0:
                        prev_entity = entities_to_label[current_idx]
                        if prev_entity in entity_labels:
                            del entity_labels[prev_entity]
                continue
            elif user_input.isdigit():
                cat_id = int(user_input)
                if 1 <= cat_id <= 8:
                    entity_labels[entity] = cat_id
                    print(f"已标注: {entity} -> {self.categories[cat_id]}")
                    current_idx += 1

                    # 每标注10个实体保存一次进度
                    if len(entity_labels) % 10 == 0:
                        self.save_labels(entity_labels, "labels_partial.json")
                        print(f"已保存进度，已标注 {len(entity_labels)} 个实体")
                else:
                    print("错误：请输入1-8之间的数字")
            else:
                print("错误：无效输入，请输入1-8之间的数字或命令")

        # 保存最终标签
        self.save_labels(entity_labels, "labels_final.json")

        print(f"\n标注完成！")
        print(f"标注实体数: {len(entity_labels)}")
        print(f"跳过实体数: {skip_count}")

        return entity_labels

    def auto_labeling_by_rules(self, entity_labels=None):
        """基于规则的自动标注（辅助手动标注）"""
        if entity_labels is None:
            entity_labels = {}

        rules = [
            # 人物相关
            (['/people/person', '/sports/pro_athlete', '/olympics/olympic_athlete'], 1),
            # 组织相关
            (['/organization/', '/business/', '/education/educational_institution'], 2),
            # 地点相关
            (['/location/', '/travel/travel_destination', '/geography/'], 3),
            # 作品相关
            (['/film/film', '/tv/tv_program', '/book/', '/music/album'], 4),
            # 事件相关
            (['/olympics/olympic_games', '/award/award_ceremony', '/sports/sports_event'], 5),
            # 概念相关
            (['/education/field_of_study', '/medicine/disease', '/language/human_language'], 6),
            # 产品相关
            (['/food/food', '/computer/software', '/cvg/computer_videogame'], 7),
            # 属性相关
            (['/measurement_unit/', '/time/time_zone'], 8),
        ]

        auto_labeled = 0
        for entity in self.entity2id.keys():
            if entity not in entity_labels:
                for keywords, cat_id in rules:
                    for keyword in keywords:
                        if keyword in entity:
                            entity_labels[entity] = cat_id
                            auto_labeled += 1
                            break
                    if entity in entity_labels:
                        break

        print(f"自动标注了 {auto_labeled} 个实体")
        return entity_labels

    def save_labels(self, entity_labels, filename):
        """保存标签到文件"""
        # 保存为JSON格式
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(entity_labels, f, ensure_ascii=False, indent=2)

        # 同时保存为CSV格式
        df = pd.DataFrame([
            {'entity': entity, 'category_id': cat_id, 'category_name': self.categories[cat_id]}
            for entity, cat_id in entity_labels.items()
        ])
        df.to_csv(filename.replace('.json', '.csv'), index=False, encoding='utf-8')

        print(f"标签已保存到 {filename}")

    def generate_labeled_datasets(self, entity_labels, output_dir='data/FB15K_labeled'):
        """基于标签生成新的数据集文件"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存entity2label文件
        with open(os.path.join(output_dir, 'entity2label.txt'), 'w', encoding='utf-8') as f:
            for entity, label in entity_labels.items():
                if entity in self.entity2id:
                    f.write(f"{entity}\t{label}\n")

        # 2. 为每个三元组文件添加标签
        datasets = {
            'train.txt': self.train_triples,
            'valid.txt': self.valid_triples,
            'test.txt': self.test_triples
        }

        for filename, triples in datasets.items():
            labeled_triples = []
            unlabeled_count = 0

            for h, r, t in triples:
                # 获取头实体和尾实体的标签
                h_label = entity_labels.get(h, 0)  # 0表示未标注
                t_label = entity_labels.get(t, 0)

                if h_label > 0 and t_label > 0:
                    labeled_triples.append(f"{h}\t{r}\t{t}\t{h_label}\t{t_label}\n")
                else:
                    unlabeled_count += 1

            # 保存标注后的文件
            output_file = filename.replace('.txt', '_labeled.txt')
            with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
                f.writelines(labeled_triples)

            print(f"{filename}: 已标注三元组 {len(labeled_triples)}, 未标注三元组 {unlabeled_count}")

        # 3. 复制原始映射文件
        import shutil
        for filename in ['entity2id.txt', 'relation2id.txt']:
            src = os.path.join(self.data_dir, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)

        print(f"\n所有文件已生成到 {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='FB15K实体标注工具')
    parser.add_argument('--data_dir', type=str, default='data/FB15K', help='FB15K数据目录')
    parser.add_argument('--mode', type=str, choices=['manual', 'auto', 'generate'], default='manual',
                        help='标注模式: manual(手动), auto(自动规则), generate(生成数据集)')
    parser.add_argument('--sample_size', type=int, default=1000, help='手动标注的样本数量')
    parser.add_argument('--labels_file', type=str, default='labels_final.json', help='标签文件路径')

    args = parser.parse_args()

    # 创建标注器
    labeler = FB15KLabeler(args.data_dir)

    if args.mode == 'manual':
        # 手动标注模式
        print("=" * 60)
        print("FB15K实体标注工具 - 手动模式")
        print("=" * 60)

        # 先自动标注一部分
        entity_labels = labeler.auto_labeling_by_rules()
        print(f"自动标注后，已标注 {len(entity_labels)} 个实体")

        # 手动标注
        new_labels = labeler.manual_labeling(sample_size=args.sample_size)
        entity_labels.update(new_labels)

        # 保存标签
        labeler.save_labels(entity_labels, args.labels_file)

    elif args.mode == 'auto':
        # 自动标注模式（基于规则）
        print("=" * 60)
        print("FB15K实体标注工具 - 自动规则模式")
        print("=" * 60)

        entity_labels = labeler.auto_labeling_by_rules()
        print(f"自动标注完成，共标注 {len(entity_labels)} 个实体")

        # 保存标签
        labeler.save_labels(entity_labels, args.labels_file)

    elif args.mode == 'generate':
        # 生成标注数据集模式
        print("=" * 60)
        print("FB15K实体标注工具 - 生成数据集模式")
        print("=" * 60)

        # 加载已有的标签
        if os.path.exists(args.labels_file):
            with open(args.labels_file, 'r', encoding='utf-8') as f:
                entity_labels = json.load(f)
            print(f"已加载 {len(entity_labels)} 个实体标签")

            # 生成标注后的数据集
            labeler.generate_labeled_datasets(entity_labels)
        else:
            print(f"错误：标签文件 {args.labels_file} 不存在")
            print("请先运行手动或自动标注模式生成标签文件")


if __name__ == "__main__":
    main()