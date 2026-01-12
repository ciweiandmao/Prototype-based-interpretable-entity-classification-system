# tag_statistics_translate_fixed.py
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


def load_and_count_tags(input_file):
    """加载文件并统计tag出现次数"""
    tag_counter = Counter()

    print("统计tag出现次数...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                path = parts[1].strip()
                if path:
                    # 提取每个段作为tag（去掉开头的斜杠）
                    segments = path.strip('/').split('/')
                    for segment in segments:
                        if segment:
                            tag_counter[segment] += 1

            if line_num % 10000 == 0:
                print(f"  已处理 {line_num} 行...")

    return tag_counter


def translate_tag(tag, translator):
    """翻译单个tag"""
    try:
        import argostranslate.translate
        return argostranslate.translate.translate(tag, "en", "zh")
    except Exception as e:
        print(f"翻译失败 '{tag}': {e}")
        return tag


def translate_tags_batch(tags_list):
    """批量翻译tags（多线程）"""
    print("初始化翻译器...")

    # 初始化翻译器
    try:
        import argostranslate.package
        import argostranslate.translate

        # 检查并安装语言包
        installed = argostranslate.package.get_installed_packages()
        has_en_zh = False

        for pkg in installed:
            if hasattr(pkg, 'from_code') and pkg.from_code == 'en' and pkg.to_code == 'zh':
                has_en_zh = True
                print("✓ 已安装英文->中文语言包")
                break

        if not has_en_zh:
            print("安装英文->中文语言包...")
            available = argostranslate.package.get_available_packages()
            for pkg in available:
                if hasattr(pkg, 'from_code') and pkg.from_code == 'en' and pkg.to_code == 'zh':
                    download_path = pkg.download()
                    argostranslate.package.install_from_path(download_path)
                    print("语言包安装完成")
                    break

        translator = argostranslate.translate

    except Exception as e:
        print(f"初始化翻译器失败: {e}")
        return {tag: tag for tag in tags_list}

    # 多线程翻译
    print(f"开始翻译 {len(tags_list)} 个tags (16线程)...")
    start_time = time.time()

    translations = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交翻译任务
        future_to_tag = {
            executor.submit(translate_tag, tag, translator): tag
            for tag in tags_list
        }

        # 收集结果
        for future in as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                translated = future.result()
                translations[tag] = translated
                completed += 1

                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  进度: {completed}/{len(tags_list)} ({completed / len(tags_list) * 100:.1f}%) - {speed:.1f}个/秒")

            except Exception as e:
                print(f"翻译 '{tag}' 失败: {e}")
                translations[tag] = tag

    elapsed = time.time() - start_time
    print(f"翻译完成! 耗时: {elapsed:.1f}秒, 平均速度: {len(tags_list) / elapsed:.1f}个/秒")

    return translations


def escape_csv_value(value):
    """转义CSV特殊字符"""
    if ',' in value or '"' in value or '\n' in value:
        # 将双引号替换为两个双引号，然后整个值用双引号包围
        value = value.replace('"', '""')
        return f'"{value}"'
    return value


def save_to_csv(tag_counter, translations, output_file):
    """保存到CSV文件（按tag长度排序）"""
    print("保存到CSV文件...")

    # 按tag长度排序，然后按字母顺序
    sorted_tags = sorted(tag_counter.keys(), key=lambda x: (len(x), x))

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入CSV头部
        f.write("tag,count,translation\n")

        # 写入数据
        for tag in sorted_tags:
            count = tag_counter[tag]
            translation = translations.get(tag, tag)
            # 转义CSV特殊字符
            escaped_translation = escape_csv_value(translation)
            f.write(f"{tag},{count},{escaped_translation}\n")

    print(f"CSV文件已保存: {output_file}")


def main():
    input_file = r"data\FB15KET\Entity_All_en.txt"
    output_file = r"data\FB15KET\Tag_Statistics.csv"

    # 1. 统计tag出现次数
    tag_counter = load_and_count_tags(input_file)
    print(f"找到 {len(tag_counter)} 个唯一tag")

    # 显示前20个最常见的tag
    print("\n最常见的20个tag:")
    for tag, count in tag_counter.most_common(20):
        print(f"  {tag}: {count}次")

    # 2. 翻译所有tags
    tags_list = list(tag_counter.keys())
    translations = translate_tags_batch(tags_list)

    # 3. 保存到CSV
    save_to_csv(tag_counter, translations, output_file)

    # 4. 显示一些翻译示例
    print("\n翻译示例:")
    sample_tags = list(tag_counter.keys())[:10]
    for tag in sample_tags:
        print(f"  {tag} -> {translations.get(tag, tag)}")


if __name__ == "__main__":
    main()