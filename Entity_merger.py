# merge_entities.py
import sys
import os
from collections import defaultdict


def merge_entity_descriptions(input_file, output_file):
    """合并同一实体的多个描述到一行"""
    entity_descriptions = defaultdict(set)  # 使用set自动去重

    # 读取并合并
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                entity_id = parts[0]
                description = parts[1]
                entity_descriptions[entity_id].add(description)

    # 写入合并结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity_id in sorted(entity_descriptions.keys()):
            descriptions = '|'.join(sorted(entity_descriptions[entity_id]))
            f.write(f"{entity_id}\t{descriptions}\n")

    print(f"完成! 合并前: {sum(len(v) for v in entity_descriptions.values())} 行")
    print(f"合并后: {len(entity_descriptions)} 行")
    print(f"输出: {output_file}")


if __name__ == "__main__":
    input_file = r"data\FB15KET\Entity_All_en.txt"
    output_file = r"data\FB15KET\Entity_All_en_single.txt"

    merge_entity_descriptions(input_file, output_file)