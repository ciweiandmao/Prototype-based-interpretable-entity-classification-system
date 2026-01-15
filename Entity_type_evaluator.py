# entity_classification_evaluator.py
import os
import csv
import random
import pandas as pd
import time
from collections import defaultdict, Counter


class EntityClassificationEvaluator:
    def __init__(self, typed_csv_path, zh_entity_path, random_seed=None):
        """
        初始化评估器

        Args:
            typed_csv_path: 预测结果的CSV文件路径
            zh_entity_path: 包含中文描述的实体文件路径
            random_seed: 随机种子，None则使用当前时间
        """
        self.typed_csv_path = typed_csv_path
        self.zh_entity_path = zh_entity_path
        self.random_seed = random_seed if random_seed is not None else int(time.time())
        self.sample_data = []
        self.zh_descriptions = {}
        self.category_names = {
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
        print(f"随机种子: {self.random_seed}")

    def load_zh_descriptions(self):
        """加载中文描述"""
        print(f"正在加载中文描述文件: {self.zh_entity_path}")

        if not os.path.exists(self.zh_entity_path):
            print(f"错误: 中文描述文件不存在: {self.zh_entity_path}")
            return False

        with open(self.zh_entity_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0].strip()
                    zh_desc = parts[1].strip()
                    self.zh_descriptions[entity_id] = zh_desc

        print(f"加载了 {len(self.zh_descriptions)} 个中文描述")
        return True

    def load_predicted_data(self):
        """加载预测结果"""
        print(f"正在加载预测结果文件: {self.typed_csv_path}")

        if not os.path.exists(self.typed_csv_path):
            print(f"错误: 预测结果文件不存在: {self.typed_csv_path}")
            return False

        # 读取CSV文件
        df = pd.read_csv(self.typed_csv_path, encoding='utf-8')
        print(f"加载了 {len(df)} 条预测记录")

        # 只选择有预测类别的记录
        df_with_predictions = df[df['predicted_category'].notna() & (df['predicted_category'] != '')]
        print(f"其中有 {len(df_with_predictions)} 条记录有预测类别")

        # 随机抽取100条记录 - 使用动态随机种子
        if len(df_with_predictions) > 100:
            # 设置随机种子
            random.seed(self.random_seed)
            # 随机抽取索引
            sampled_indices = random.sample(range(len(df_with_predictions)), 100)
            sampled_df = df_with_predictions.iloc[sampled_indices]
        else:
            sampled_df = df_with_predictions
            print(f"警告: 有预测类别的记录不足100条，只抽取 {len(sampled_df)} 条")

        # 转换为列表格式
        for _, row in sampled_df.iterrows():
            record = {
                'entity_id': str(row['entity_id']),
                'tags': row['tags'] if 'tags' in row else '',
                'predicted_category': int(float(row['predicted_category'])) if pd.notna(
                    row['predicted_category']) else None,
                'predicted_category_name': row['predicted_category_name'] if pd.notna(
                    row['predicted_category_name']) else '',
                'confidence_score': float(row['confidence_score']) if pd.notna(row['confidence_score']) else 0.0
            }

            # 提取所有类别的得分
            category_scores = {}
            for col in row.index:
                if col.startswith('category_') and col.endswith('_score'):
                    try:
                        cat_num = int(col.replace('category_', '').replace('_score', ''))
                        category_scores[cat_num] = float(row[col]) if pd.notna(row[col]) else 0.0
                    except:
                        continue

            record['category_scores'] = category_scores
            self.sample_data.append(record)

        print(f"成功抽取 {len(self.sample_data)} 条记录用于评估")
        return True

    def interactive_evaluation(self):
        """交互式评估"""
        if not self.sample_data:
            print("没有可用于评估的数据")
            return

        print("\n" + "=" * 80)
        print("实体分类准确性评估")
        print("=" * 80)
        print("请为每个实体选择正确的人工类别（1-9）：")
        print("1: 人物（Person）")
        print("2: 组织与机构（Organization）")
        print("3: 地点与地理（Location）")
        print("4: 创作与娱乐作品（Creative Work）")
        print("5: 事件与活动（Event）")
        print("6: 学科与概念（Concept & Subject）")
        print("7: 物品与产品（Product & Object）")
        print("8: 属性与度量（Attribute & Measurement）")
        print("9: 其他（Others）")
        print("=" * 80)

        correct_count = 0
        manual_judgments = []

        for i, record in enumerate(self.sample_data, 1):
            entity_id = record['entity_id']
            zh_desc = self.zh_descriptions.get(entity_id, "无中文描述")
            predicted_cat = record['predicted_category']
            predicted_name = record['predicted_category_name']
            confidence = record['confidence_score']
            scores = record['category_scores']

            print(f"\n[{i}/{len(self.sample_data)}] 实体ID: {entity_id}")
            print(f"中文描述: {zh_desc}")
            print(f"Tags: {record['tags'][:200]}{'...' if len(record['tags']) > 200 else ''}")

            # 显示预测结果
            print(f"\n预测结果:")
            print(f"  最可能类别: {predicted_cat} - {predicted_name}")
            print(f"  置信度: {confidence:.4f}")

            # 显示所有类别的得分（按得分排序）
            print(f"\n所有类别得分:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for cat_num, score in sorted_scores[:5]:  # 只显示前5个
                cat_name = self.category_names.get(cat_num, f"类别{cat_num}")
                print(f"  {cat_num}. {cat_name}: {score:.4f}")

            # 获取人工判断
            while True:
                try:
                    manual_input = input("\n请选择正确的人工类别 (1-9, 输入'skip'跳过, 输入'quit'退出): ").strip()

                    if manual_input.lower() == 'skip':
                        manual_choice = None
                        break
                    elif manual_input.lower() == 'quit':
                        print("\n评估提前结束")
                        return manual_judgments
                    else:
                        manual_choice = int(manual_input)
                        if 1 <= manual_choice <= 9:
                            break
                        else:
                            print("错误: 请输入1-9之间的数字")
                except ValueError:
                    print("错误: 请输入有效数字")

            if manual_choice is not None:
                is_correct = (manual_choice == predicted_cat)
                if is_correct:
                    correct_count += 1

                judgment = {
                    'entity_id': entity_id,
                    'zh_description': zh_desc,
                    'predicted_category': predicted_cat,
                    'predicted_category_name': predicted_name,
                    'manual_choice': manual_choice,
                    'manual_choice_name': self.category_names.get(manual_choice, f"类别{manual_choice}"),
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'scores': scores
                }
                manual_judgments.append(judgment)

                print(f"人工选择: {manual_choice} - {self.category_names.get(manual_choice, '未知类别')}")
                print(f"判断结果: {'✓ 正确' if is_correct else '✗ 错误'}")

        # 显示统计结果
        self.display_statistics(manual_judgments, correct_count)
        return manual_judgments

    def display_statistics(self, judgments, correct_count):
        """显示统计结果"""
        print("\n" + "=" * 80)
        print("评估统计结果")
        print("=" * 80)

        total_judged = len(judgments)
        if total_judged > 0:
            accuracy = correct_count / total_judged * 100

            print(f"总评估记录数: {total_judged}")
            print(f"正确预测数: {correct_count}")
            print(f"错误预测数: {total_judged - correct_count}")
            print(f"准确率: {accuracy:.2f}%")

            # 按类别统计
            category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
            for judgment in judgments:
                predicted = judgment['predicted_category']
                manual = judgment['manual_choice']
                is_correct = judgment['is_correct']

                category_stats[predicted]['total'] += 1
                if is_correct:
                    category_stats[predicted]['correct'] += 1

            print("\n按预测类别统计:")
            print(f"{'预测类别':<35} {'数量':<8} {'正确数':<8} {'准确率':<8}")
            print("-" * 60)

            for cat_num in sorted(category_stats.keys()):
                stats = category_stats[cat_num]
                if stats['total'] > 0:
                    cat_accuracy = stats['correct'] / stats['total'] * 100
                    cat_name = self.category_names.get(cat_num, f"类别{cat_num}")
                    print(f"{cat_name:<35} {stats['total']:<8} {stats['correct']:<8} {cat_accuracy:.1f}%")

            # 置信度分析
            conf_correct = []
            conf_incorrect = []
            for judgment in judgments:
                if judgment['is_correct']:
                    conf_correct.append(judgment['confidence'])
                else:
                    conf_incorrect.append(judgment['confidence'])

            if conf_correct:
                avg_conf_correct = sum(conf_correct) / len(conf_correct)
                print(f"\n正确预测的平均置信度: {avg_conf_correct:.4f}")

            if conf_incorrect:
                avg_conf_incorrect = sum(conf_incorrect) / len(conf_incorrect)
                print(f"错误预测的平均置信度: {avg_conf_incorrect:.4f}")

            # 保存评估结果到文件
            self.save_evaluation_results(judgments, accuracy)

    """    def save_evaluation_results(self, judgments, accuracy):
        #保存评估结果到CSV文件
        output_file = "entity_classification_evaluation_results.csv"

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                'entity_id',
                'zh_description',
                'predicted_category',
                'predicted_category_name',
                'manual_choice',
                'manual_choice_name',
                'is_correct',
                'confidence',
                'score_1', 'score_2', 'score_3', 'score_4', 'score_5',
                'score_6', 'score_7', 'score_8', 'score_9'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for judgment in judgments:
                row = {
                    'entity_id': judgment['entity_id'],
                    'zh_description': judgment['zh_description'],
                    'predicted_category': judgment['predicted_category'],
                    'predicted_category_name': judgment['predicted_category_name'],
                    'manual_choice': judgment['manual_choice'],
                    'manual_choice_name': judgment['manual_choice_name'],
                    'is_correct': '是' if judgment['is_correct'] else '否',
                    'confidence': judgment['confidence']
                }

                # 添加所有类别的得分
                scores = judgment['scores']
                for cat_num in range(1, 10):
                    row[f'score_{cat_num}'] = scores.get(cat_num, 0.0)

                writer.writerow(row)

        print(f"\n详细评估结果已保存到: {output_file}")

        # 保存统计摘要
        summary_file = "evaluation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("实体分类评估结果摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {pd.Timestamp.now()}\n")
            f.write(f"总评估记录数: {len(judgments)}\n")
            f.write(f"正确预测数: {sum(1 for j in judgments if j['is_correct'])}\n")
            f.write(f"准确率: {accuracy:.2f}%\n\n")

            # 按类别统计
            category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
            for judgment in judgments:
                predicted = judgment['predicted_category']
                if judgment['is_correct']:
                    category_stats[predicted]['correct'] += 1
                category_stats[predicted]['total'] += 1

            f.write("按类别统计:\n")
            f.write(f"{'类别':<25} {'总数':<6} {'正确数':<6} {'准确率':<8}\n")
            f.write("-" * 45 + "\n")

            for cat_num in sorted(category_stats.keys()):
                stats = category_stats[cat_num]
                if stats['total'] > 0:
                    cat_accuracy = stats['correct'] / stats['total'] * 100
                    cat_name = self.category_names.get(cat_num, f"类别{cat_num}")
                    f.write(f"{cat_name:<25} {stats['total']:<6} {stats['correct']:<6} {cat_accuracy:.1f}%\n")

        print(f"评估摘要已保存到: {summary_file}")
"""
    def run(self):
        """运行评估"""
        print("开始实体分类准确性评估...")

        # 加载数据
        if not self.load_zh_descriptions():
            return

        if not self.load_predicted_data():
            return

        # 交互式评估
        judgments = self.interactive_evaluation()

        if judgments:
            print("\n评估完成!")
        else:
            print("\n评估未完成或没有有效数据")


def main():
    # 配置文件路径
    typed_csv_path = r"data\FB15KET\Entity_All_typed.csv"
    zh_entity_path = r"data\FB15KET\Entity_All_zh_single.txt"

    # 创建评估器并运行
    evaluator = EntityClassificationEvaluator(typed_csv_path, zh_entity_path)
    evaluator.run()


if __name__ == "__main__":
    main()