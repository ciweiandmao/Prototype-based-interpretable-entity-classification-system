# fb15k_translate_multithread.py
# !/usr/bin/env python3
"""
多线程Freebase翻译工具 - 提高效率
"""

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import argparse


# 全局翻译器实例（线程安全）
class ArgosTranslator:
    """Argos翻译器单例（线程安全）"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def initialize(self):
        """初始化翻译器"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

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
                try:
                    # 不更新索引以加速
                    # argostranslate.package.update_package_index()

                    available = argostranslate.package.get_available_packages()
                    for pkg in available:
                        if hasattr(pkg, 'from_code') and pkg.from_code == 'en' and pkg.to_code == 'zh':
                            print(f"找到语言包，下载中...")
                            download_path = pkg.download()
                            argostranslate.package.install_from_path(download_path)
                            print("语言包安装完成")
                            break
                except Exception as e:
                    print(f"语言包安装失败: {e}")

            self._initialized = True

    def translate(self, text, from_code="en", to_code="zh"):
        """翻译文本（线程安全）"""
        try:
            import argostranslate.translate
            return argostranslate.translate.translate(text, from_code, to_code)
        except Exception as e:
            print(f"翻译错误: {e}")
            return text


def preprocess_freebase_path(path):
    """预处理Freebase路径为可翻译文本"""
    if not path or path.strip() == '':
        return ''

    if path.startswith('/'):
        text = path[1:]
    else:
        text = path

    # 将斜杠和下划线替换为空格
    text = text.replace('/', ' ').replace('_', ' ')
    # 清理多余空格
    text = ' '.join(text.split())

    return text.strip()


def postprocess_translation(original_path, translated_text):
    """后处理：将翻译结果转换回路径格式"""
    if not translated_text or not original_path:
        return original_path

    # 将空格替换为斜杠
    translated_path = '/' + translated_text.replace(' ', '/')
    return translated_path


def process_line(line, translator, line_num, show_examples=False):
    """处理单行数据（工作线程函数）"""
    if not line:
        return line_num, line

    parts = line.split('\t')
    if len(parts) >= 2:
        entity_id = parts[0].strip()
        freebase_path = parts[1].strip()

        # 显示前几个示例
        if show_examples and line_num <= 3:
            print(f"[示例 {line_num}] 处理: {entity_id}\t{freebase_path}")

        # 预处理
        text_to_translate = preprocess_freebase_path(freebase_path)

        if text_to_translate:
            # 翻译
            translated_text = translator.translate(text_to_translate)

            # 后处理
            translated_path = postprocess_translation(freebase_path, translated_text)
        else:
            translated_path = freebase_path

        result = f"{entity_id}\t{translated_path}"
    else:
        result = line

    return line_num, result


def translate_file_multithread(input_file, output_file, num_threads=8, batch_size=100):
    """多线程翻译文件"""
    print(f"翻译 {input_file} -> {output_file}")
    print(f"使用 {num_threads} 个线程，批处理大小: {batch_size}")

    # 初始化翻译器
    translator = ArgosTranslator()
    translator.initialize()

    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]  # 保持换行符由我们控制

    total_lines = len(lines)
    print(f"读取 {total_lines} 行")

    # 创建进度队列
    progress_queue = Queue()

    def progress_monitor(total):
        """进度监控线程"""
        start_time = time.time()
        processed = 0

        while processed < total:
            # 从队列获取进度更新
            count = progress_queue.get()
            if count == -1:  # 结束信号
                break

            processed += count
            elapsed = time.time() - start_time
            progress = processed / total * 100

            if elapsed > 0:
                speed = processed / elapsed
                eta = (total - processed) / speed if speed > 0 else 0
                print(f"\r进度: {processed}/{total} ({progress:.1f}%) - "
                      f"速度: {speed:.1f}行/秒 - ETA: {eta:.0f}秒", end='', flush=True)

        elapsed = time.time() - start_time
        print(f"\n完成! 总耗时: {elapsed:.1f}秒, 平均速度: {total / elapsed:.1f}行/秒")

    # 启动进度监控线程
    import threading
    monitor_thread = threading.Thread(target=progress_monitor, args=(total_lines,))
    monitor_thread.start()

    start_time = time.time()
    results = [None] * total_lines  # 预分配结果列表

    # 分批处理
    for batch_start in range(0, total_lines, batch_size):
        batch_end = min(batch_start + batch_size, total_lines)
        batch_lines = lines[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交批处理任务
            future_to_line = {
                executor.submit(process_line, line, translator, i + 1, batch_start == 0): i
                for i, line in enumerate(batch_lines, batch_start)
            }

            # 收集结果
            batch_processed = 0
            for future in as_completed(future_to_line):
                try:
                    line_num, result = future.result()
                    results[line_num] = result
                    batch_processed += 1
                except Exception as e:
                    line_idx = future_to_line[future]
                    print(f"\n处理第 {line_idx + 1} 行时出错: {e}")
                    results[line_idx] = lines[line_idx]  # 出错时保留原行

        # 更新进度
        progress_queue.put(batch_processed)

    # 发送结束信号
    progress_queue.put(-1)
    monitor_thread.join()

    # 写入输出文件
    print(f"\n写入输出文件...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for result in results:
            if result is not None:
                f_out.write(result + '\n')

    elapsed = time.time() - start_time
    print(f"文件写入完成!")
    print(f"总耗时: {elapsed:.1f}秒")
    print(f"平均速度: {total_lines / elapsed:.1f}行/秒")
    print(f"输出文件: {output_file}")

    # 显示前几个结果示例
    print("\n结果示例:")
    for i in range(min(3, len(results))):
        if results[i] is not None:
            print(f"  {results[i]}")


def main():
    parser = argparse.ArgumentParser(description='多线程Freebase翻译工具')
    parser.add_argument('input', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（可选）')
    parser.add_argument('-t', '--threads', type=int, default=8,
                        help='线程数（默认: 8）')
    parser.add_argument('-b', '--batch', type=int, default=100,
                        help='批处理大小（默认: 100）')
    parser.add_argument('--test', action='store_true',
                        help='测试模式（只处理前100行）')

    args = parser.parse_args()

    input_file = args.input

    if args.output:
        output_file = args.output
    else:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_zh{ext}"

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)

    # 测试模式：只处理前100行
    if args.test:
        print("测试模式：只处理前100行")
        # 创建测试文件副本
        with open(input_file, 'r', encoding='utf-8') as f:
            test_lines = [next(f).rstrip('\n') for _ in range(100)]

        test_input = "test_input.txt"
        test_output = "test_output.txt"

        with open(test_input, 'w', encoding='utf-8') as f:
            for line in test_lines:
                f.write(line + '\n')

        translate_file_multithread(test_input, test_output,
                                   args.threads, args.batch)

        # 显示测试结果
        print("\n测试结果对比:")
        with open(test_input, 'r', encoding='utf-8') as f_in, \
                open(test_output, 'r', encoding='utf-8') as f_out:

            for i in range(min(5, len(test_lines))):
                original = f_in.readline().strip()
                translated = f_out.readline().strip()
                print(f"{i + 1}. 原始: {original}")
                print(f"   翻译: {translated}")
                print()

        # 清理测试文件
        os.remove(test_input)
        os.remove(test_output)
    else:
        # 正常模式
        translate_file_multithread(input_file, output_file,
                                   args.threads, args.batch)


if __name__ == "__main__":
    main()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    #Tips python quick_translate.py data\FB15KET\Entity_Type_test.txt -t 16 -b 200
