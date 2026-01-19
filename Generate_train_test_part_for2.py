import os
import shutil
import time          # 导入的是整个 time 模块
import random

from trimesh.util import now

seed_now = int(time.time())

def extract_test_entities(input_file, output_test_file, output_train_file, num_test_entities=100, seed=seed_now):
    """
    从数据集中随机抽取指定数量的测试实体，并分离数据集

    Args:
        input_file: 原始数据文件
        output_test_file: 测试集输出文件
        output_train_file: 训练集输出文件
        num_test_entities: 要抽取的测试实体数量
        seed: 随机种子
    """

    # 设置随机种子确保可重复性
    random.seed(seed_now)

    print(f"当前随机种子: {seed_now}")

    print(f"开始处理文件: {input_file}")

    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"原始数据行数: {len(lines)}")

    # 提取所有不重复的实体
    all_entities = set()
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, _, tail = parts
            all_entities.add(head)
            all_entities.add(tail)

    print(f"所有不重复实体数量: {len(all_entities)}")

    # 随机选择测试实体
    all_entities_list = list(all_entities)
    if len(all_entities_list) < num_test_entities:
        print(f"警告: 实体数量不足，将选择所有 {len(all_entities_list)} 个实体作为测试集")
        test_entities = set(all_entities_list)
    else:
        test_entities = set(random.sample(all_entities_list, num_test_entities))

    print(f"随机选择的测试实体数量: {len(test_entities)}")
    '''
    # 显示部分测试实体
    print("前10个测试实体示例:")
    for i, entity in enumerate(list(test_entities)[:10]):
        print(f"  {i + 1}. {entity}")
    '''
    # 分离数据
    test_lines = []
    train_lines = []

    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, relation, tail = parts

            # 检查是否包含测试实体
            if head in test_entities or tail in test_entities:
                test_lines.append(line)
            else:
                train_lines.append(line)

    print(f"\n分离结果:")
    print(f"测试集行数: {len(test_lines)}")
    print(f"训练集行数: {len(train_lines)}")

    # 保存测试集
    with open(output_test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    print(f"测试集已保存到: {output_test_file}")

    # 保存训练集
    with open(output_train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    print(f"训练集已保存到: {output_train_file}")

    # 统计每个测试实体的边数
    entity_edge_counts = {}
    for line in test_lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, relation, tail = parts
            for entity in [head, tail]:
                if entity in test_entities:
                    entity_edge_counts[entity] = entity_edge_counts.get(entity, 0) + 1

    # 保存测试实体列表
    '''
    test_entities_file = 'data/FB15KET/test_entities_list.txt'
    with open(test_entities_file, 'w', encoding='utf-8') as f:
        f.write("# 测试实体列表\n")
        for entity in sorted(test_entities):
            edge_count = entity_edge_counts.get(entity, 0)
            f.write(f"{entity}\t# 边数: {edge_count}\n")

    print(f"测试实体列表已保存到: {test_entities_file}")
    '''
    # 显示统计信息
    '''
    print("\n测试实体边数统计:")
    sorted_entities = sorted(entity_edge_counts.items(), key=lambda x: x[1], reverse=True)
    for entity, count in sorted_entities[:10]:  # 显示前10个
        print(f"  {entity}: {count} 条边")
    '''
    sorted_entities = sorted(entity_edge_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_entities:
        avg_edges = sum(entity_edge_counts.values()) / len(entity_edge_counts)
        print(f"平均每个测试实体有 {avg_edges:.1f} 条边")

    return test_entities, test_lines, train_lines


def analyze_test_data(test_file, detail_file, test_entities):
    """分析测试集数据"""
    print(f"\n分析测试集文件: {test_file}")
    mathnum = 100 * 2
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"测试集总行数: {len(lines)}")

    # 按测试实体分组
    entity_lines = {}
    for entity in test_entities:
        entity_lines[entity] = []

    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, relation, tail = parts
            if head in test_entities:
                if head not in entity_lines:
                    entity_lines[head] = []
                entity_lines[head].append(line)
            if tail in test_entities:
                if tail not in entity_lines:
                    entity_lines[tail] = []
                entity_lines[tail].append(line)

    # 显示每个测试实体的记录
    print(f"\n每个测试实体的记录数:")
    sorted_items = sorted(entity_lines.items(), key=lambda x: len(x[1]), reverse=True)

    # 创建详细的测试集文件（按实体分组）
    detailed_test_file = 'data/FB15KET/TEST_PART_DETAILED.txt'
    with open(detailed_test_file, 'w', encoding='utf-8') as f:
        f.write("# 测试集详细数据（按实体分组）\n")
        f.write("=" * 80 + "\n")


        for entity, entity_lines_list in sorted_items:

            if entity_lines_list:
                # 去重（有些边可能被统计两次，因为是双向的）
                unique_lines = set(entity_lines_list)

                f.write(f"\n实体: {entity} (共 {len(unique_lines)} 条记录)\n")
                f.write("-" * 60 + "\n")

                for line in unique_lines:
                    f.write(line + "\n")
            mathnum-=10
            if(mathnum<=0):break
                #print(f"  {entity}: {len(unique_lines)} 条记录")


    print(f"\n详细测试集已保存到: {detailed_test_file}")
    with open(detailed_test_file, 'r', encoding='utf-8') as fa, \
            open(detail_file, 'r', encoding='utf-8') as fb:
        content_a = fa.read()
        combined_content = content_a + fb.read()
    with open(detailed_test_file, 'w', encoding='utf-8') as f_out:
        f_out.write(combined_content)

    return entity_lines

'''
def verify_separation(original_file, test_file, train_file, test_entities):
    """验证数据分离是否正确"""
    print(f"\n验证数据分离结果...")

    # 读取原始数据
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = set([line.strip() for line in f if line.strip()])

    # 读取测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_lines = set([line.strip() for line in f if line.strip()])

    # 读取训练数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = set([line.strip() for line in f if line.strip()])

    # 验证1：测试集和训练集没有交集
    intersection = test_lines.intersection(train_lines)
    if len(intersection) == 0:
        print("✓ 测试集和训练集没有重叠数据")
    else:
        print(f"✗ 测试集和训练集有 {len(intersection)} 条重叠数据")
        for line in list(intersection)[:5]:  # 显示前5条重叠数据
            print(f"  重叠: {line}")

    # 验证2：测试集+训练集 = 原始数据集
    union = test_lines.union(train_lines)
    if union == original_lines:
        print("✓ 测试集和训练集的并集等于原始数据集")
    else:
        missing_in_union = original_lines - union
        extra_in_union = union - original_lines
        print(f"✗ 数据不匹配:")
        print(f"  原始有但并集中没有: {len(missing_in_union)} 条")
        print(f"  并集中有但原始没有: {len(extra_in_union)} 条")

    # 验证3：测试集只包含测试实体的边
    test_entity_edges = 0
    non_test_entity_edges = 0

    for line in test_lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, relation, tail = parts
            if head in test_entities or tail in test_entities:
                test_entity_edges += 1
            else:
                non_test_entity_edges += 1
                #print(f"  警告: 测试集中包含非测试实体的边: {line}")

    #print(f"测试集中包含测试实体的边: {test_entity_edges} 条")
    #print(f"测试集中包含非测试实体的边: {non_test_entity_edges} 条")

    # 验证4：训练集不包含测试实体的边
    train_contains_test = 0
    for line in train_lines:
        parts = line.split('\t')
        if len(parts) == 3:
            head, relation, tail = parts
            if head in test_entities or tail in test_entities:
                train_contains_test += 1
                print(f"  错误: 训练集中包含测试实体 {head if head in test_entities else tail} 的边: {line}")

    if train_contains_test == 0:
        print("✓ 训练集不包含任何测试实体的边")
    else:
        print(f"✗ 训练集中包含 {train_contains_test} 条测试实体的边")
'''

def create_sample_files():
    """创建示例文件，展示数据格式"""

    # 创建示例数据文件
    sample_data = """# FB15KET 数据集示例格式
# 每行格式: 实体ID-关系tag数组-实体ID
/m/027rn\t/location/country/form_of_government\t/m/06cx9
/m/017dcd\t/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor\t/m/06v8s0
/m/07s9rl0\t/media_common/netflix_genre/titles\t/m/0170z3
/m/011k_j\t/music/instrument\t/m/01234
/m/011k_j\t/music/performance_role\t/m/05678
/m/01234\t/media_common/netflix_genre/titles\t/m/0170z3
/m/05678\t/location/country/form_of_government\t/m/027rn
"""

    sample_file = 'data/FB15KET/sample_format.txt'
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)

    print(f"\n示例格式文件已创建: {sample_file}")


def main():
    """主函数"""

    # 文件路径
    input_file = 'data/FB15KET/xunlian.txt'
    output_test_file = 'data/FB15KET/TEST_PART.txt'
    output_train_file = 'data/FB15KET/TRAIN_PART.txt'
    detail_file='data/FB15KET/TEST_PART_DETAILEED.txt'

    # 确保目录存在
    os.makedirs('data/FB15KET', exist_ok=True)

    print("=" * 80)
    print("FB15KET 数据集分割工具")
    print("=" * 80)

    # 步骤1: 创建示例文件（可选）
    create_sample_files()

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print(f"请确保文件路径正确，或手动创建示例数据")
        return

    try:
        # 步骤2: 提取测试实体并分割数据
        test_entities, test_lines, train_lines = extract_test_entities(
            input_file=input_file,
            output_test_file=output_test_file,
            output_train_file=output_train_file,
            num_test_entities=100,
            seed=seed_now  # 固定随机种子确保可重复性
        )

        # 步骤3: 分析测试数据
        entity_lines = analyze_test_data(output_test_file, detail_file, test_entities)

        # 步骤4: 验证分离结果
        #verify_separation(input_file, output_test_file, output_train_file, test_entities)

        # 步骤5: 生成统计报告
        print(f"\n" + "=" * 80)
        print("分割结果统计报告")
        print("=" * 80)

        # 读取原始文件统计
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = len([line for line in f if line.strip()])

        test_percentage = (len(test_lines) / total_lines * 100) if total_lines > 0 else 0
        train_percentage = (len(train_lines) / total_lines * 100) if total_lines > 0 else 0

        print(f"原始数据集: {total_lines} 条边")
        print(f"测试集: {len(test_lines)} 条边 ({test_percentage:.1f}%)")
        print(f"训练集: {len(train_lines)} 条边 ({train_percentage:.1f}%)")
        print(f"测试实体数: {len(test_entities)} 个")

        # 计算平均度数
        total_test_edges = sum(len(lines) for lines in entity_lines.values())
        avg_degree = total_test_edges / len(test_entities) if test_entities else 0
        print(f"测试实体平均度数: {avg_degree:.1f}")

        # 保存统计报告
        report_file = 'data/FB15KET/split_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FB15KET 数据集分割报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"原始文件: {input_file}\n")
            f.write(f"总边数: {total_lines}\n\n")
            f.write(f"测试集文件: {output_test_file}\n")
            f.write(f"测试集边数: {len(test_lines)} ({test_percentage:.1f}%)\n")
            f.write(f"训练集文件: {output_train_file}\n")
            f.write(f"训练集边数: {len(train_lines)} ({train_percentage:.1f}%)\n\n")
            f.write(f"测试实体数量: {len(test_entities)}\n")
            f.write(f"测试实体平均度数: {avg_degree:.1f}\n\n")
            f.write("测试实体列表:\n")
            for entity in sorted(test_entities):
                edge_count = len(entity_lines.get(entity, []))
                f.write(f"  {entity}: {edge_count} 条边\n")

        #print(f"\n统计报告已保存到: {report_file}")

        #print(f"\n" + "=" * 80)
        #print("分割完成！")
        #print("=" * 80)
        #print(f"测试集: {output_test_file}")
        #print(f"训练集: {output_train_file}")
        #print(f"\n下一步:")
        #print(f"1. 使用 TRAIN_PART.txt 训练模型")
        #print(f"2. 使用 TEST_PART.txt 中的实体进行测试")
        #print(f"3. 注意: 测试时模型不应见过 TEST_PART.txt 中的任何边")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()