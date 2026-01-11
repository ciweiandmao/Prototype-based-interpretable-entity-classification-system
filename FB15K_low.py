import pandas as pd
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import os
import gc
import time
from datetime import datetime, timedelta
from collections import defaultdict


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 检查是否有可用的图形设备
def setup_device():
    """
    自动选择最佳计算设备
    优先级：CUDA > MPS > CPU
    """

    # 1. 检查CUDA（NVIDIA显卡）
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ Using NVIDIA GPU: {gpu_name}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        return device

    # 2. 检查MPS（Apple Silicon）- 修复版本检查
    if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                print("✓ Using Apple MPS (Metal Performance Shaders)")
                return device
        except:
            pass

    # 3. 检查ROCm（AMD显卡）- 简化检查
    try:
        if hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
            device = torch.device('cuda')
            print("✓ Using AMD GPU via ROCm")
            return device
    except:
        pass

    # 4. 检查DirectML（Windows上的AMD/Intel显卡）
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✓ Using DirectML device (Windows GPU acceleration)")
        return device
    except ImportError:
        pass

    # 5. 检查Intel GPU
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'is_available') and ipex.is_available():
            device = torch.device('xpu')
            print("✓ Using Intel GPU via IPEX")
            return device
    except ImportError:
        pass

    # 6. 使用CPU
    device = torch.device('cpu')

    # 检查CPU信息
    import platform
    import multiprocessing

    cpu_info = platform.processor()
    cpu_count = multiprocessing.cpu_count()

    print(f"ⓘ Using CPU: {cpu_info}")
    print(f"  Cores: {cpu_count}")
    print("  Note: For better performance, consider:")
    print("  - Installing DirectML: pip install torch-directml")
    print("  - Or using CPU with OpenMP optimization")

    return device

# 内存友好的异构图神经网络（分批处理）
class MemoryFriendlyHetGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_relations):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_relations = num_relations

        # 为每种关系类型创建独立的卷积层
        self.conv1_layers = nn.ModuleList([
            dglnn.GraphConv(in_feats, hid_feats) for _ in range(num_relations)
        ])
        self.conv2_layers = nn.ModuleList([
            dglnn.GraphConv(hid_feats, out_feats) for _ in range(num_relations)
        ])

        self.bilstm = nn.LSTM(
            input_size=out_feats,
            hidden_size=out_feats // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward_batch(self, entity_emb, rel_adj_matrices):
        """分批处理图卷积"""
        batch_size = entity_emb.size(0)

        # 第一层卷积
        h1_list = []
        for rel_idx in range(self.num_relations):
            if rel_idx < len(rel_adj_matrices):
                adj_matrix = rel_adj_matrices[rel_idx]
                if adj_matrix is not None and adj_matrix.size(0) > 0:
                    # 创建同构图
                    src, dst = adj_matrix.nonzero(as_tuple=True)
                    if src.numel() > 0:
                        g = dgl.graph((src, dst))
                        g = g.to(entity_emb.device)
                        # 应用卷积
                        h_rel = self.conv1_layers[rel_idx](g, entity_emb)
                        h1_list.append(h_rel)

        if h1_list:
            h1 = torch.stack(h1_list).mean(dim=0)
            h1 = F.relu(h1)
        else:
            h1 = F.relu(entity_emb)

        # 第二层卷积
        h2_list = []
        for rel_idx in range(self.num_relations):
            if rel_idx < len(rel_adj_matrices):
                adj_matrix = rel_adj_matrices[rel_idx]
                if adj_matrix is not None and adj_matrix.size(0) > 0:
                    src, dst = adj_matrix.nonzero(as_tuple=True)
                    if src.numel() > 0:
                        g = dgl.graph((src, dst))
                        g = g.to(entity_emb.device)
                        h_rel = self.conv2_layers[rel_idx](g, h1)
                        h2_list.append(h_rel)

        if h2_list:
            h2 = torch.stack(h2_list).mean(dim=0)
        else:
            h2 = h1

        # BiLSTM处理
        h2 = h2.unsqueeze(1)
        h2, _ = self.bilstm(h2)
        h2 = h2.squeeze(1)

        return h2


# 链接预测模块（轻量版）
class LightweightLinkPredictor(nn.Module):
    def __init__(self, in_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations

        # 简化版双线性变换
        self.rel_emb = nn.Embedding(num_relations, in_dim)
        self.bias = nn.Parameter(torch.zeros(num_relations))

        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, head_emb, tail_emb, rel_ids):
        # 获取关系嵌入
        rel_emb = self.rel_emb(rel_ids)  # (batch_size, in_dim)

        # 简化评分: s = sum((h + r) * t) + b
        scores = torch.sum((head_emb + rel_emb) * tail_emb, dim=1)
        scores = scores + self.bias[rel_ids]

        return scores


# 分块处理的FB15K模型
class BlockFB15K_XGradNet(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim=64, hidden_dim=64, out_dim=32):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.out_dim = out_dim

        # 实体嵌入
        self.entity_emb = nn.Embedding(num_entities, feature_dim)

        # 图神经网络
        self.gnn = MemoryFriendlyHetGNN(feature_dim, hidden_dim, out_dim, num_relations)

        # 链接预测器
        self.link_predictor = LightweightLinkPredictor(out_dim, num_relations)

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_emb.weight)

    def forward_batch(self, entity_ids, rel_adj_matrices):
        """分批处理前向传播"""
        entity_features = self.entity_emb(entity_ids)
        embeddings = self.gnn.forward_batch(entity_features, rel_adj_matrices)
        return embeddings

    def predict_link_batch(self, head_ids, tail_ids, rel_ids, rel_adj_matrices):
        """分批进行链接预测"""
        # 获取所有涉及的实体ID
        all_entity_ids = torch.unique(torch.cat([head_ids, tail_ids]))

        # 获取实体嵌入
        entity_embeddings = self.forward_batch(all_entity_ids, rel_adj_matrices)

        # 创建ID到索引的映射
        id_to_idx = {id.item(): idx for idx, id in enumerate(all_entity_ids)}

        # 获取对应的嵌入
        head_indices = torch.tensor([id_to_idx[id.item()] for id in head_ids], device=head_ids.device)
        tail_indices = torch.tensor([id_to_idx[id.item()] for id in tail_ids], device=tail_ids.device)

        head_emb = entity_embeddings[head_indices]
        tail_emb = entity_embeddings[tail_indices]

        # 计算链接预测得分
        scores = self.link_predictor(head_emb, tail_emb, rel_ids)

        return scores


# 内存优化的数据加载
class MemoryEfficientFB15KLoader:
    def __init__(self, data_dir='FB15K', max_relations=100, batch_size=10000):
        self.data_dir = data_dir
        self.max_relations = max_relations  # 限制关系数量以减少内存
        self.batch_size = batch_size

        # 加载数据
        self.load_data()

    def load_data(self):
        """内存友好的数据加载"""
        print("Loading FB15K data with memory optimization...")

        # 统计实体和关系
        self.entity2id = {}
        self.relation2id = {}
        self.train_triples = []

        # 只加载部分数据以减少内存使用
        train_file = os.path.join(self.data_dir, 'train.txt')
        if not os.path.exists(train_file):
            # 查找其他可能的文件
            for f in os.listdir(self.data_dir):
                if 'train' in f.lower():
                    train_file = os.path.join(self.data_dir, f)
                    break

        # 读取训练数据（分批读取）
        line_count = 0
        max_lines = 100000  # 限制加载的行数

        with open(train_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line_count >= max_lines:
                    break

                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[:3]

                    # 添加实体
                    if h not in self.entity2id:
                        self.entity2id[h] = len(self.entity2id)
                    if t not in self.entity2id:
                        self.entity2id[t] = len(self.entity2id)

                    # 限制关系数量
                    if r not in self.relation2id:
                        if len(self.relation2id) < self.max_relations:
                            self.relation2id[r] = len(self.relation2id)

                    # 只保留现有关系
                    if r in self.relation2id:
                        self.train_triples.append((
                            self.entity2id[h],
                            self.relation2id[r],
                            self.entity2id[t]
                        ))

                line_count += 1

        # 采样测试数据
        self.test_triples = []
        if len(self.train_triples) > 1000:
            indices = np.random.choice(len(self.train_triples), 1000, replace=False)
            for idx in indices:
                self.test_triples.append(self.train_triples[idx])
        else:
            self.test_triples = self.train_triples[:100]

        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        print(f"Loaded {self.num_entities} entities, {self.num_relations} relations")
        print(f"Training triples: {len(self.train_triples)}")
        print(f"Test triples: {len(self.test_triples)}")

    def get_relation_adjacency(self, rel_id, entity_subset=None):
        """获取特定关系的邻接矩阵（稀疏格式）"""
        if entity_subset is None:
            entity_subset = set(range(self.num_entities))

        # 收集该关系的边
        edges = []
        for h, r, t in self.train_triples:
            if r == rel_id and h in entity_subset and t in entity_subset:
                edges.append((h, t))

        if not edges:
            return None

        # 转换为稀疏张量
        heads, tails = zip(*edges)
        heads = torch.tensor(heads, dtype=torch.long)
        tails = torch.tensor(tails, dtype=torch.long)

        # 创建稀疏邻接矩阵
        n = len(entity_subset)
        indices = torch.stack([heads, tails])
        values = torch.ones(len(edges))

        return torch.sparse_coo_tensor(indices, values, (n, n))

    def get_batch_data(self, batch_size=None):
        """获取批量数据"""
        if batch_size is None:
            batch_size = self.batch_size

        # 随机打乱
        indices = np.random.permutation(len(self.train_triples))

        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            batch_triples = [self.train_triples[i] for i in batch_indices]

            # 提取批量的头、关系、尾实体
            heads = torch.tensor([t[0] for t in batch_triples], dtype=torch.long)
            rels = torch.tensor([t[1] for t in batch_triples], dtype=torch.long)
            tails = torch.tensor([t[2] for t in batch_triples], dtype=torch.long)

            # 获取该批次涉及的实体
            entity_subset = torch.unique(torch.cat([heads, tails])).tolist()

            # 获取相关关系的邻接矩阵
            rel_adj_matrices = []
            for rel_id in range(self.num_relations):
                # 只包含该批次中出现的实体
                adj_matrix = self.get_relation_adjacency(rel_id, set(entity_subset))
                rel_adj_matrices.append(adj_matrix)

            yield heads, rels, tails, entity_subset, rel_adj_matrices


# 内存优化的训练函数
def train_memory_efficient(model, data_loader, num_epochs=20, lr=0.001, device='cpu'):
    import time
    from datetime import datetime, timedelta

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []

    # ETA相关变量
    total_start_time = time.time()
    epoch_times = []

    print(f"\n{'=' * 60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        # 获取批量数据
        batches = list(data_loader.get_batch_data(batch_size=512))
        total_batches = len(batches)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Total batches: {total_batches}")

        for batch_idx, (heads, rels, tails, entity_subset, rel_adj_matrices) in enumerate(batches):
            batch_start_time = time.time()

            # 移到设备
            heads = heads.to(device)
            rels = rels.to(device)
            tails = tails.to(device)
            entity_subset_tensor = torch.tensor(entity_subset, dtype=torch.long).to(device)

            # 正样本得分
            pos_scores = model.predict_link_batch(heads, tails, rels, rel_adj_matrices)

            # 生成负样本（破坏尾部实体）
            neg_tails = torch.randint(0, model.num_entities, heads.shape).to(device)
            neg_scores = model.predict_link_batch(heads, neg_tails, rels, rel_adj_matrices)

            # 损失函数
            loss = F.relu(1.0 + neg_scores - pos_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 计算批次时间
            batch_time = time.time() - batch_start_time

            # 显示进度
            if batch_idx % max(1, total_batches // 20) == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100

                # 计算剩余时间
                elapsed_time = time.time() - epoch_start_time
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                epoch_eta_seconds = remaining_batches * avg_time_per_batch

                # 格式化为可读时间
                epoch_eta = str(timedelta(seconds=int(epoch_eta_seconds)))

                # 计算总ETA
                if epoch_times:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                else:
                    avg_epoch_time = elapsed_time * (
                            total_batches / (batch_idx + 1)) if batch_idx > 0 else elapsed_time

                remaining_epochs = num_epochs - (epoch + 1)
                total_eta_seconds = epoch_eta_seconds + avg_epoch_time * remaining_epochs
                total_eta = str(timedelta(seconds=int(total_eta_seconds)))

                # 显示进度条
                bar_length = 30
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)

                print(f"\r  [{bar}] {progress:.1f}% | "
                      f"Batch {batch_idx + 1}/{total_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Epoch ETA: {epoch_eta} | "
                      f"Total ETA: {total_eta}", end='')

        # 完成一个epoch
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)

        print(f"\n  ✓ Epoch {epoch + 1} completed in {epoch_time:.1f}s | Loss: {avg_loss:.4f}")

        # 清理内存
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 训练完成
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"Training completed in {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.1f}s")
    print(f"{'=' * 60}")

    return model, train_losses


# 评估函数
def evaluate_memory_efficient(model, test_triples, device='cpu'):
    model.eval()

    with torch.no_grad():
        # 将测试数据分批处理
        batch_size = 256
        total_correct = 0
        total_samples = 0

        for start_idx in range(0, len(test_triples), batch_size):
            end_idx = min(start_idx + batch_size, len(test_triples))
            batch = test_triples[start_idx:end_idx]

            heads = torch.tensor([t[0] for t in batch], dtype=torch.long).to(device)
            rels = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
            tails = torch.tensor([t[2] for t in batch], dtype=torch.long).to(device)

            # 获取相关关系的邻接矩阵
            entity_subset = torch.unique(torch.cat([heads, tails])).tolist()
            rel_adj_matrices = []
            for rel_id in range(model.num_relations):
                adj_matrix = None  # 简化评估，不使用邻接矩阵
                rel_adj_matrices.append(adj_matrix)

            # 预测得分
            scores = model.predict_link_batch(heads, tails, rels, rel_adj_matrices)

            # 生成负样本进行比较
            neg_tails = torch.randint(0, model.num_entities, heads.shape).to(device)
            neg_scores = model.predict_link_batch(heads, neg_tails, rels, rel_adj_matrices)

            # 正样本得分应该高于负样本
            correct = (scores > neg_scores).sum().item()
            total_correct += correct
            total_samples += len(batch)

            # 清理内存
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f'Test Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})')

        return accuracy


# 主函数（内存友好版）
def main_memory_friendly():
    # 设置设备
    device = setup_device()  # 自动选择最佳设备
    #device = torch.device('cpu')  # 强制使用CPU以减少内存
    print(f'Using device: {device}')
    print('Warning: Using CPU only to reduce memory usage')

    # 设置随机种子
    set_seed(42)

    # 加载数据（内存优化版）
    print('Loading data with memory optimization...')
    try:
        data_loader = MemoryEfficientFB15KLoader(
            data_dir='FB15K',
            max_relations=50,  # 限制关系数量
            batch_size=5000
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for testing...")
        data_loader = create_synthetic_loader()

    num_entities = data_loader.num_entities
    num_relations = data_loader.num_relations

    print(f'\nModel Configuration:')
    print(f'Number of entities: {num_entities}')
    print(f'Number of relations: {num_relations}')
    print(f'Training triples: {len(data_loader.train_triples)}')
    print(f'Test triples: {len(data_loader.test_triples)}')

    # 创建模型（更小的维度）
    print('\nCreating model with reduced dimensions...')
    model = BlockFB15K_XGradNet(
        num_entities=num_entities,
        num_relations=num_relations,
        feature_dim=32,  # 减小特征维度
        hidden_dim=32,  # 减小隐藏维度
        out_dim=16  # 减小输出维度
    )

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # 训练模型
    print('\nStarting training...')
    model, train_losses = train_memory_efficient(
        model=model,
        data_loader=data_loader,
        num_epochs=10,  # 减少训练轮数
        lr=0.001,
        device=device
    )

    # 评估模型
    print('\nEvaluating model...')
    accuracy = evaluate_memory_efficient(
        model=model,
        test_triples=data_loader.test_triples,
        device=device
    )

    # 保存最终模型
    torch.save(model.state_dict(), 'memory_efficient_fb15k_model.pth')
    print('Model saved to memory_efficient_fb15k_model.pth')

    # 绘制训练损失
    if train_losses:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs (Memory Efficient)')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_loss_memory_efficient.png', dpi=100)
            plt.show()
        except:
            print("Training losses:", train_losses)

    return model, accuracy, train_losses


def create_synthetic_loader():
    """创建合成数据加载器"""

    class SyntheticLoader:
        def __init__(self):
            self.num_entities = 500
            self.num_relations = 20
            self.train_triples = []
            self.test_triples = []

            # 生成训练数据
            for _ in range(10000):
                h = np.random.randint(0, self.num_entities)
                t = np.random.randint(0, self.num_entities)
                r = np.random.randint(0, self.num_relations)
                self.train_triples.append((h, r, t))

            # 生成测试数据
            for _ in range(1000):
                h = np.random.randint(0, self.num_entities)
                t = np.random.randint(0, self.num_entities)
                r = np.random.randint(0, self.num_relations)
                self.test_triples.append((h, r, t))

        def get_relation_adjacency(self, rel_id, entity_subset=None):
            return None

        def get_batch_data(self, batch_size=512):
            # 随机打乱
            indices = np.random.permutation(len(self.train_triples))

            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                batch_triples = [self.train_triples[i] for i in batch_indices]

                heads = torch.tensor([t[0] for t in batch_triples], dtype=torch.long)
                rels = torch.tensor([t[1] for t in batch_triples], dtype=torch.long)
                tails = torch.tensor([t[2] for t in batch_triples], dtype=torch.long)

                entity_subset = torch.unique(torch.cat([heads, tails])).tolist()
                rel_adj_matrices = [None] * self.num_relations

                yield heads, rels, tails, entity_subset, rel_adj_matrices

    loader = SyntheticLoader()
    print(f"Created synthetic data with {loader.num_entities} entities, {loader.num_relations} relations")
    return loader


if __name__ == '__main__':
    print("=" * 60)
    print("Memory-Efficient FB15K Training")
    print("=" * 60)

    # 检查内存
    import psutil

    memory_info = psutil.virtual_memory()
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Memory usage: {memory_info.percent}%")

    # 运行内存友好版
    model, accuracy, train_losses = main_memory_friendly()

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print("=" * 60)