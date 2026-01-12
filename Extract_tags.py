# extract_tags_correct.py
import os


def extract_all_tags(input_file, output_file):
    """正确提取所有tag并去重"""
    tags_set = set()

    # 读取文件并提取tag
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                path = parts[1].strip()
                if path:
                    # 分割路径的每一级作为单独tag
                    segments = path.strip('/').split('/')
                    for segment in segments:
                        if segment:  # 跳过空段
                            # 每个段作为一个tag
                            tags_set.add(f"/{segment}")

    # 按字母顺序排序
    sorted_tags = sorted(tags_set)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for tag in sorted_tags:
            f.write(f"{tag}\n")

    print(f"完成! 提取了 {len(sorted_tags)} 个唯一tag")
    print(f"输出文件: {output_file}")


def test_with_sample():
    """用示例测试"""
    test_input = [
        "/m/042kbj	/tv/tv_actor",
        "/m/01lf293	/music/artist",
        "/m/01d0fp	/music/artist",
        "/m/056rgc	/film/actor",
        "/m/02wr6r	/award/award_nominee"
    ]

    expected = {
        "/tv", "/tv_actor", "/music", "/artist",
        "/film", "/actor", "/award", "/award_nominee"
    }

    tags_set = set()
    for line in test_input:
        parts = line.split('\t')
        if len(parts) >= 2:
            path = parts[1].strip()
            if path:
                segments = path.strip('/').split('/')
                for segment in segments:
                    if segment:
                        tags_set.add(f"/{segment}")

    print("测试结果:")
    print(f"提取的tag: {sorted(tags_set)}")
    print(f"预期的tag: {sorted(expected)}")
    print(f"匹配: {tags_set == expected}")


if __name__ == "__main__":
    # 先测试
    test_with_sample()

    print("\n" + "=" * 50 + "\n")

    # 处理实际文件
    input_file = r"data\FB15KET\Entity_All_en.txt"
    output_file = r"data\FB15KET\All_Tags.txt"

    extract_all_tags(input_file, output_file)