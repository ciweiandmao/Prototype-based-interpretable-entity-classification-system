import sys
import torch
import numpy as np
import dgl


def test_compatibility():
    """测试PyTorch、NumPy和DGL的版本兼容性"""

    print("=" * 60)
    print("PyTorch, NumPy, DGL 版本兼容性测试")
    print("=" * 60)

    # 获取版本信息
    python_version = sys.version_info
    torch_version = torch.__version__
    numpy_version = np.__version__
    dgl_version = dgl.__version__

    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"PyTorch 版本: {torch_version}")
    print(f"NumPy 版本: {numpy_version}")
    print(f"DGL 版本: {dgl_version}")

    # 检查CUDA可用性
    print("\n" + "=" * 60)
    print("CUDA 支持检查")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        print(f"当前GPU设备: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

        # 测试CUDA张量
        try:
            cuda_tensor = torch.tensor([1, 2, 3]).cuda()
            print("CUDA 张量创建: ✓ 成功")
            print(f"CUDA 张量设备: {cuda_tensor.device}")
        except Exception as e:
            print(f"CUDA 张量创建: ✗ 失败 - {e}")

    # ========== 基础功能测试 ==========
    print("\n" + "=" * 60)
    print("基础功能测试")
    print("=" * 60)

    # 1. NumPy与PyTorch互操作性测试
    print("\n1. NumPy <-> PyTorch 互操作性测试:")

    try:
        # NumPy到PyTorch
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        torch_tensor = torch.from_numpy(np_array)
        print(f"  NumPy -> PyTorch: ✓ 成功")
        print(f"    数据类型: {torch_tensor.dtype}, 形状: {torch_tensor.shape}")

        # PyTorch到NumPy
        torch_tensor2 = torch.tensor([4.0, 5.0, 6.0])
        np_array2 = torch_tensor2.numpy()
        print(f"  PyTorch -> NumPy: ✓ 成功")
        print(f"    数据类型: {np_array2.dtype}, 形状: {np_array2.shape}")

        # 原地操作测试
        torch_tensor.add_(1)
        print(f"  原地操作后NumPy数组: {np_array}")
        print(f"  说明: 共享内存" if np.array_equal(np_array, torch_tensor.numpy()) else "  说明: 不共享内存")

    except Exception as e:
        print(f"  ✗ 失败 - {e}")

    # 2. DGL基础功能测试
    print("\n2. DGL 基础功能测试:")

    try:
        # 创建简单图
        src = torch.tensor([0, 0, 1, 2])
        dst = torch.tensor([1, 2, 3, 3])
        g = dgl.graph((src, dst), num_nodes=4)

        print(f"  创建DGL图: ✓ 成功")
        print(f"    节点数: {g.num_nodes()}, 边数: {g.num_edges()}")

        # 添加节点特征
        g.ndata['feat'] = torch.randn(4, 5)
        print(f"  添加节点特征: ✓ 成功")
        print(f"    特征形状: {g.ndata['feat'].shape}")

        # 添加自环
        g_with_loop = dgl.add_self_loop(g)
        print(f"  添加自环: ✓ 成功")
        print(f"    新边数: {g_with_loop.num_edges()}")

    except Exception as e:
        print(f"  ✗ 失败 - {e}")

    # 3. 模型兼容性测试
    print("\n3. GNN模型兼容性测试:")

    try:
        import dgl.nn as dglnn

        class TestGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = dglnn.GraphConv(5, 10)

            def forward(self, g, features):
                return self.conv(g, features)

        model = TestGNN()
        print(f"  创建GNN模型: ✓ 成功")
        print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        if 'g' in locals():
            feat = g.ndata['feat']
            output = model(g, feat)
            print(f"  前向传播: ✓ 成功")
            print(f"    输出形状: {output.shape}")

    except Exception as e:
        print(f"  ✗ 失败 - {e}")

    # 4. 内存和性能测试
    print("\n" + "=" * 60)
    print("内存和性能测试")
    print("=" * 60)

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"当前内存使用: {memory_mb:.2f} MB")

        # 创建大张量测试
        big_tensor = torch.randn(10000, 1000)
        memory_after = process.memory_info().rss / 1024 / 1024
        print(f"创建大张量后内存: {memory_after:.2f} MB")
        print(f"内存增加: {memory_after - memory_mb:.2f} MB")

        # 清理
        del big_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except ImportError:
        print("psutil未安装，跳过内存测试")
        print("安装: pip install psutil")

    # 5. 版本兼容性建议
    print("\n" + "=" * 60)
    print("版本兼容性建议")
    print("=" * 60)

    # 解析版本号
    def parse_version(version_str):
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2].split('+')[0]) if len(parts) > 2 else 0
        return major, minor, patch

    torch_major, torch_minor, _ = parse_version(torch_version)
    numpy_major, numpy_minor, _ = parse_version(numpy_version)
    dgl_major, dgl_minor, _ = parse_version(dgl_version)

    # PyTorch版本检查
    print("PyTorch版本检查:")
    if torch_major >= 2:
        print("  ✓ PyTorch 2.x 版本，支持编译优化")
    elif torch_major == 1 and torch_minor >= 8:
        print("  ✓ PyTorch 1.8+ 版本，兼容性良好")
    else:
        print("  ⚠ PyTorch版本较旧，建议升级到1.8+")

    # NumPy版本检查
    print("\nNumPy版本检查:")
    if numpy_major >= 1 and numpy_minor >= 19:
        print("  ✓ NumPy 1.19+ 版本，与PyTorch兼容性良好")
    else:
        print("  ⚠ NumPy版本较旧，建议升级到1.19+")

    # DGL版本检查
    print("\nDGL版本检查:")
    if dgl_major >= 1 and dgl_minor >= 0:
        print("  ✓ DGL 1.0+ 版本，稳定性良好")
    else:
        print("  ⚠ DGL版本较旧，建议升级到1.0+")

    # 已知兼容性问题
    print("\n已知兼容性问题检查:")
    issues = []

    # 检查PyTorch 2.0+与旧版DGL
    if torch_major >= 2 and dgl_major == 0:
        issues.append("PyTorch 2.0+ 需要 DGL 1.0+")

    # 检查CUDA版本匹配
    if cuda_available:
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"  CUDA版本: {cuda_version}")
        except:
            pass

    if not issues:
        print("  ✓ 未发现已知兼容性问题")
    else:
        print("  ⚠ 发现潜在问题:")
        for issue in issues:
            print(f"    - {issue}")

    # 6. 推荐配置
    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)

    print("当前环境评估:")
    if all([
        torch_major >= 1 and torch_minor >= 8,
        numpy_major >= 1 and numpy_minor >= 19,
        dgl_major >= 0 and dgl_minor >= 8,
        cuda_available
    ]):
        print("  ✓ 环境配置良好，适合GNN训练")
    elif cuda_available:
        print("  ⚠ 环境基本可用，建议升级库版本")
    else:
        print("  ⚠ 仅CPU环境，训练速度较慢")

    print("\n推荐版本组合:")
    print("  1. PyTorch 2.0+ + DGL 1.0+ + NumPy 1.23+ (最新稳定版)")
    print("  2. PyTorch 1.13 + DGL 0.9 + NumPy 1.21 (稳定兼容版)")

    print("\n安装命令参考:")
    print("  升级所有库:")
    print("    pip install --upgrade torch numpy dgl")

    if not cuda_available:
        print("\n  安装CPU版本:")
        print("    pip install torch numpy dgl")

    print("\n  安装特定版本（示例）:")
    print("    pip install torch==2.0.0 numpy==1.24.0 dgl==1.0.0")

    return True


def test_specific_issue():
    """测试特定问题：PyTorch张量转NumPy"""
    print("\n" + "=" * 60)
    print("特定问题测试：PyTorch张量转NumPy")
    print("=" * 60)

    test_cases = [
        ("FloatTensor CPU", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
        ("FloatTensor GPU",
         torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda() if torch.cuda.is_available() else None),
        ("IntTensor", torch.tensor([1, 2, 3], dtype=torch.int32)),
        ("BoolTensor", torch.tensor([True, False, True])),
    ]

    for name, tensor in test_cases:
        print(f"\n测试 {name}:")
        if tensor is None:
            print("  GPU不可用，跳过")
            continue

        try:
            print(f"  张量设备: {tensor.device}")
            print(f"  张量dtype: {tensor.dtype}")
            print(f"  张量requires_grad: {tensor.requires_grad}")

            # 测试转换
            numpy_array = tensor.cpu().numpy()
            print(f"  转换到NumPy: ✓ 成功")
            print(f"  NumPy dtype: {numpy_array.dtype}")
            print(f"  形状: {numpy_array.shape}")

            # 测试转换回来
            tensor_back = torch.from_numpy(numpy_array)
            print(f"  转换回PyTorch: ✓ 成功")

            # 检查值是否一致
            if torch.allclose(tensor.cpu(), tensor_back):
                print(f"  值一致性: ✓ 一致")
            else:
                print(f"  值一致性: ✗ 不一致")

        except Exception as e:
            print(f"  ✗ 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # 检查必要的导入
    try:
        import torch.nn as nn

        print("所有必要库已成功导入")

        # 运行兼容性测试
        test_compatibility()

        # 运行特定问题测试
        test_specific_issue()

    except ImportError as e:
        print(f"导入失败: {e}")
        print("\n请确保已安装所有必要的库:")
        print("pip install torch numpy dgl scikit-learn pandas")

        # 检查各个库
        libraries = ['torch', 'numpy', 'dgl']
        missing = []

        for lib in libraries:
            try:
                __import__(lib)
                print(f"  ✓ {lib} 已安装")
            except ImportError:
                print(f"  ✗ {lib} 未安装")
                missing.append(lib)

        if missing:
            print(f"\n缺少库: {missing}")
            print("安装命令: pip install " + " ".join(missing))