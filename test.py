import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


# ==================== GPU负载测试模型 ====================
class GPUStressTestModel(nn.Module):
    """专门用于GPU满载测试的模型"""

    def __init__(self, input_dim=10000, hidden_dim=5000, output_dim=1000, num_layers=10):
        super().__init__()

        # 创建多层大型线性层（消耗大量GPU内存）
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # 激活函数和归一化
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # 损失函数
        self.criterion = nn.MSELoss()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.batch_norm(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x

    def train_step(self, x, y, optimizer):
        """单步训练"""
        optimizer.zero_grad()
        output = self(x)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()


# ==================== GPU测试函数 ====================
def gpu_stress_test(batch_size=512, num_iterations=100, device='cuda'):
    """
    运行GPU满载测试
    Args:
        batch_size: 批大小（越大越耗显存）
        num_iterations: 迭代次数
        device: 设备（'cuda' 或 'cpu'）
    """
    print(f"{'=' * 60}")
    print("GPU满载压力测试")
    print(f"{'=' * 60}")

    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA不可用，使用CPU模式")
        device = 'cpu'

    device = torch.device(device)
    print(f"测试设备: {device}")

    if device.type == 'cuda':
        # 显示GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU型号: {gpu_name}")
        print(f"GPU总显存: {total_memory:.2f} GB")

    # 创建模型（根据显存大小自动调整）
    if device.type == 'cuda':
        if total_memory >= 16:  # 16GB以上显存
            input_dim, hidden_dim = 15000, 8000
        elif total_memory >= 8:  # 8GB显存
            input_dim, hidden_dim = 10000, 5000
        else:  # 小于8GB显存
            input_dim, hidden_dim = 5000, 2500
    else:
        input_dim, hidden_dim = 1000, 500  # CPU模式用小模型

    print(f"模型配置: input_dim={input_dim}, hidden_dim={hidden_dim}")

    # 创建模型并移动到设备
    model = GPUStressTestModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1000,
        num_layers=8
    ).to(device)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 预热GPU
    print("正在预热GPU...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # 执行预热操作
        warmup_tensor = torch.randn(1000, 1000).to(device)
        _ = torch.mm(warmup_tensor, warmup_tensor)
        torch.cuda.synchronize()

    # 记录性能指标
    losses = []
    times = []
    gpu_usage = []

    print(f"\n开始压力测试 ({num_iterations}次迭代)...")
    print("-" * 60)

    try:
        for iteration in range(num_iterations):
            start_time = time.time()

            # 生成随机数据（在设备上创建以减少传输开销）
            if device.type == 'cuda':
                # 直接在GPU上创建数据
                x = torch.randn(batch_size, input_dim, device=device)
                y = torch.randn(batch_size, 1000, device=device)
            else:
                x = torch.randn(batch_size, input_dim).to(device)
                y = torch.randn(batch_size, 1000).to(device)

            # 训练步骤
            loss = model.train_step(x, y, optimizer)

            # 同步GPU（确保时间测量准确）
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # 计算迭代时间
            iter_time = time.time() - start_time

            # 记录指标
            losses.append(loss)
            times.append(iter_time)

            # 记录GPU使用情况
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                gpu_usage.append((memory_allocated, memory_reserved))

            # 每10次迭代显示进度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                avg_time = np.mean(times[-10:]) if len(times) >= 10 else iter_time
                fps = batch_size / avg_time  # 样本/秒

                print(f"迭代 {iteration + 1:3d}/{num_iterations} | "
                      f"损失: {loss:.4f} | "
                      f"时间: {iter_time * 1000:.1f}ms | "
                      f"速度: {fps:.1f} samples/sec", end="")

                if device.type == 'cuda' and len(gpu_usage) > 0:
                    mem_alloc, mem_reserved = gpu_usage[-1]
                    print(f" | GPU内存: {mem_alloc:.2f}/{mem_reserved:.2f} GB")
                else:
                    print()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n⚠️ GPU内存不足！当前batch_size={batch_size}")
            print("尝试减小batch_size...")
            return False
        else:
            raise e

    # 测试完成，打印总结
    print("-" * 60)
    print("测试完成！性能总结:")
    print(f"平均损失: {np.mean(losses[-50:]):.6f}")
    print(f"平均迭代时间: {np.mean(times) * 1000:.1f}ms")
    print(f"平均吞吐量: {batch_size / np.mean(times):.1f} samples/sec")

    if device.type == 'cuda' and gpu_usage:
        max_memory = max([mem[0] for mem in gpu_usage])
        avg_memory = np.mean([mem[0] for mem in gpu_usage])
        print(f"峰值GPU内存: {max_memory:.2f} GB")
        print(f"平均GPU内存: {avg_memory:.2f} GB")

        # 评估GPU利用率
        if max_memory > total_memory * 0.8:
            print("✅ GPU利用率: 优秀 (>80%)")
        elif max_memory > total_memory * 0.6:
            print("✅ GPU利用率: 良好 (60-80%)")
        elif max_memory > total_memory * 0.4:
            print("⚠️ GPU利用率: 中等 (40-60%)")
        else:
            print("❌ GPU利用率: 较低 (<40%)")

    return True


# ==================== 自动调整批大小 ====================
def adaptive_gpu_test(max_batch_size=4096, min_batch_size=64):
    """
    自动调整批大小以找到GPU极限
    """
    print(f"{'=' * 60}")
    print("自适应GPU压力测试")
    print(f"{'=' * 60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        print("没有可用的GPU，退出测试")
        return

    # 从较小的batch_size开始测试
    batch_size = min_batch_size
    successful_runs = []

    while batch_size <= max_batch_size:
        print(f"\n尝试 batch_size = {batch_size}...")

        try:
            # 清理GPU缓存
            torch.cuda.empty_cache()

            # 运行测试
            success = gpu_stress_test(
                batch_size=batch_size,
                num_iterations=20,  # 短测试
                device='cuda'
            )

            if success:
                successful_runs.append(batch_size)
                # 倍增batch_size
                batch_size *= 2
            else:
                # 如果失败，尝试更小的增幅
                batch_size = int(batch_size * 1.2)

        except Exception as e:
            print(f"batch_size={batch_size} 时出错: {str(e)[:100]}")
            break

    if successful_runs:
        optimal_batch_size = max(successful_runs)
        print(f"\n{'=' * 60}")
        print(f"最优批大小: {optimal_batch_size}")
        print(f"测试通过的批大小: {successful_runs}")
        print(f"{'=' * 60}")

        # 用最优批大小运行最终测试
        print(f"\n使用最优批大小 {optimal_batch_size} 进行最终测试...")
        gpu_stress_test(
            batch_size=optimal_batch_size,
            num_iterations=100,
            device='cuda'
        )


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 运行简单的压力测试
    print("选择测试模式:")
    print("1. 快速测试 (默认)")
    print("2. 自适应测试 (寻找最优批大小)")
    print("3. 极限压力测试")

    choice = input("请输入选择 (1-3, 默认1): ").strip()

    if choice == "2":
        # 自适应测试
        adaptive_gpu_test(max_batch_size=8192, min_batch_size=128)
    elif choice == "3":
        # 极限测试
        success = gpu_stress_test(
            batch_size=1024,
            num_iterations=200,
            device='cuda'
        )
    else:
        # 快速测试（默认）
        success = gpu_stress_test(
            batch_size=512,
            num_iterations=50,
            device='cuda'
        )

    print(f"\n{'=' * 60}")
    print("GPU测试完成！")
    print(f"{'=' * 60}")
