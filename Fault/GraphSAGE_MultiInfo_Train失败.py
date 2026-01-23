import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import dgl
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import defaultdict, Counter
import warnings
import random

from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms
from tqdm import tqdm
import gc
import os
import time
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import pandas as pd
import numpy as np

from GraphSAGE_Train import TypeAwareGraphSAGE

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

class MultiModalEncoder:
    """修复的多模态编码器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"使用设备: {device}")

        # 1. 图像特征提取器
        self.image_extractor = self._init_image_extractor()

        # 2. 数值特征编码器
        self.numeric_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        ).to(device)

        # 3. 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 256, 512),  # 2048维图像 + 256维数值
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(device)

        # 特征维度
        self.feature_dim = 512

    def _init_image_extractor(self):
        """初始化图像特征提取器"""
        print("初始化ResNet50图像特征提取器...")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])  # 去掉分类层
        model = model.to(self.device)
        model.eval()

        # 冻结参数
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _image_transform(self):
        """图像预处理"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_single_image_features(self, image_path):
        """提取单张图片特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            transform = self._image_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.image_extractor(image_tensor)

            return features.squeeze().cpu()
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return torch.zeros(2048)  # ResNet50特征维度

    def extract_entity_image_features(self, entity_id, image_dir="D:/Z-Downloader/download"):
        """提取实体所有图片特征"""
        # 构建文件夹路径
        folder_name = entity_id.replace('/', '.').strip('.')
        entity_image_dir = os.path.join(image_dir, folder_name)

        if not os.path.exists(entity_image_dir):
            print(f"警告: 实体 {entity_id} 的图片文件夹不存在: {entity_image_dir}")
            return torch.zeros(2048)

        # 查找图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
        image_files = [f for f in os.listdir(entity_image_dir)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"警告: 实体 {entity_id} 的图片文件夹中没有图片")
            return torch.zeros(2048)

        # 提取特征
        all_features = []
        max_images = min(10, len(image_files))  # 最多10张

        for i, img_file in enumerate(image_files[:max_images]):
            img_path = os.path.join(entity_image_dir, img_file)
            feat = self.extract_single_image_features(img_path)
            if feat is not None:
                all_features.append(feat)

        if not all_features:
            return torch.zeros(2048)

        # 平均池化
        all_features = torch.stack(all_features)
        aggregated = torch.mean(all_features, dim=0)

        return aggregated

    def extract_numeric_features_fixed(self, entity_id, entity_types_df):
        """修复的数值特征提取函数"""
        try:
            # 查找实体
            matches = entity_types_df[entity_types_df['entity_id'] == entity_id]

            if matches.empty:
                print(f"警告: 实体 {entity_id} 在CSV中未找到")
                return torch.zeros(9, dtype=torch.float32, device=self.device)

            row = matches.iloc[0]
            numeric_features = []

            # 处理每个数值列
            for col_idx in range(1, 10):
                col_name = f'category_{col_idx}_score'

                if col_name not in row.index:
                    numeric_features.append(0.0)
                    continue

                value = row[col_name]

                # 处理不同类型的数据
                if pd.isna(value):
                    numeric_features.append(0.0)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_features.append(float(value))
                elif isinstance(value, str):
                    try:
                        numeric_features.append(float(value.strip()))
                    except:
                        numeric_features.append(0.0)
                else:
                    numeric_features.append(0.0)

            # 转换为张量
            features_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=self.device)
            return features_tensor

        except Exception as e:
            print(f"提取实体 {entity_id} 数值特征时出错: {e}")
            return torch.zeros(9, dtype=torch.float32, device=self.device)

    def encode_entity(self, entity_id, entity_types_df, image_dir):
        """编码单个实体的多模态特征"""
        try:
            # 1. 图像特征
            image_feat = self.extract_entity_image_features(entity_id, image_dir)
            image_feat = image_feat.to(self.device)

            # 2. 数值特征
            numeric_raw = self.extract_numeric_features_fixed(entity_id, entity_types_df)
            numeric_feat = self.numeric_encoder(numeric_raw.unsqueeze(0)).squeeze()

            # 3. 检查维度
            if image_feat.dim() == 0:
                image_feat = image_feat.unsqueeze(0)
            if numeric_feat.dim() == 0:
                numeric_feat = numeric_feat.unsqueeze(0)

            # 4. 融合特征
            combined = torch.cat([image_feat, numeric_feat], dim=-1)

            if combined.dim() == 1:
                combined = combined.unsqueeze(0)

            fused_feature = self.fusion(combined)

            # 5. 确保输出维度正确
            if fused_feature.dim() == 2:
                fused_feature = fused_feature.squeeze(0)

            return fused_feature

        except Exception as e:
            print(f"编码实体 {entity_id} 时出错: {e}")
            return torch.zeros(self.feature_dim, device=self.device)


# 在你的GraphSAGE_Train.py中添加
class MultiModalGraphSAGE(nn.Module):
    """支持多模态输入的GraphSAGE"""

    def __init__(self, structural_dim, multimodal_dim, h_feats, num_classes, num_layers=2, dropout=0.3):
        super().__init__()

        # 总输入维度 = 结构特征维度 + 多模态特征维度
        total_in_feats = structural_dim + multimodal_dim

        # 输入编码层 - 使用总维度
        self.input_encoder = nn.Sequential(
            nn.Linear(total_in_feats, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.num_layers = num_layers
        self.dropout = dropout

        # GraphSAGE层
        self.sage_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层
        self.sage_layers.append(dglnn.SAGEConv(
            in_feats=h_feats,
            out_feats=h_feats * 2,
            aggregator_type='mean',
            feat_drop=dropout
        ))
        self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 中间层
        for i in range(1, num_layers - 1):
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats * 2,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 输出层
        if num_layers > 1:
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 关系类型编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 类型模式聚合层
        self.type_pattern_aggregator = nn.Sequential(
            nn.Linear(h_feats * 3, h_feats * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_feats // 2, num_classes)
        )

        # 打印维度信息（调试用）
        print(f"[MultiModalGraphSAGE] 结构维度: {structural_dim}")
        print(f"[MultiModalGraphSAGE] 多模态维度: {multimodal_dim}")
        print(f"[MultiModalGraphSAGE] 总输入维度: {total_in_feats}")
        print(f"[MultiModalGraphSAGE] 隐藏层维度: {h_feats}")

    def forward(self, g, structural_features, multimodal_features):
        # 检查维度
        num_nodes = g.num_nodes()
        if structural_features.shape[0] != num_nodes:
            raise ValueError(f"结构特征行数 ({structural_features.shape[0]}) ≠ 节点数 ({num_nodes})")
        if multimodal_features.shape[0] != num_nodes:
            raise ValueError(f"多模态特征行数 ({multimodal_features.shape[0]}) ≠ 节点数 ({num_nodes})")

        # 打印维度信息（调试用）
        # print(f"[forward] 结构特征: {structural_features.shape}")
        # print(f"[forward] 多模态特征: {multimodal_features.shape}")

        # 融合特征
        combined_features = torch.cat([structural_features, multimodal_features], dim=-1)
        # print(f"[forward] 融合后特征: {combined_features.shape}")

        # 输入编码
        h = self.input_encoder(combined_features)
        # print(f"[forward] 编码后: {h.shape}")

        # 保存每层的输出用于特征融合
        layer_outputs = [h]

        # GraphSAGE传播
        for i in range(self.num_layers):
            h = self.sage_layers[i](g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)
            # print(f"[forward] GraphSAGE层{i+1}后: {h.shape}")

        # 多层特征融合
        if len(layer_outputs) > 1:
            # 使用最后一层和第一层的特征
            h_final = torch.cat([layer_outputs[0], layer_outputs[-1]], dim=1)
            h_final = self.relation_encoder(h_final)
        else:
            h_final = layer_outputs[0]

        # print(f"[forward] 融合后: {h_final.shape}")

        # 最终分类
        out = self.classifier(h_final)
        # print(f"[forward] 最终输出: {out.shape}")

        return out
class TypeAwareGraphSAGE(nn.Module):
    """类型感知的GraphSAGE模型，专门用于实体类型预测"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.3):
        super(TypeAwareGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入编码层
        self.input_encoder = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GraphSAGE层
        self.sage_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层
        self.sage_layers.append(dglnn.SAGEConv(
            in_feats=h_feats,
            out_feats=h_feats * 2,
            aggregator_type='mean',
            feat_drop=dropout
        ))
        self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 中间层
        for i in range(1, num_layers - 1):
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats * 2,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats * 2))

        # 输出层
        if num_layers > 1:
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=h_feats * 2,
                out_feats=h_feats,
                aggregator_type='mean',
                feat_drop=dropout
            ))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 关系类型编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(h_feats * 2, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 类型模式聚合层
        self.type_pattern_aggregator = nn.Sequential(
            nn.Linear(h_feats * 3, h_feats * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_feats // 2, num_classes)
        )

    def forward(self, g, features):
        h = self.input_encoder(features)

        # 保存每层的输出用于特征融合
        layer_outputs = [h]

        # GraphSAGE传播
        for i in range(self.num_layers):
            h = self.sage_layers[i](g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # 多层特征融合
        if len(layer_outputs) > 1:
            # 使用最后一层和第一层的特征
            h_final = torch.cat([layer_outputs[0], layer_outputs[-1]], dim=1)
            h_final = self.relation_encoder(h_final)
        else:
            h_final = layer_outputs[0]

        # 最终分类
        out = self.classifier(h_final)

        return out


class TypePatternGraphSAGE(nn.Module):
    """更强调类型模式的GraphSAGE变体"""

    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.3):
        super(TypePatternGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_feats, h_feats * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 关系类型编码器
        self.relation_type_encoder = nn.Embedding(100, h_feats // 2)  # 假设最多100种关系

        # 注意力机制层
        self.attention = nn.MultiheadAttention(
            embed_dim=h_feats * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # GraphSAGE层
        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = h_feats * 2 if i == 0 else h_feats * 2
            out_dim = h_feats * 2 if i < num_layers - 1 else h_feats
            self.sage_layers.append(dglnn.SAGEConv(
                in_feats=in_dim,
                out_feats=out_dim,
                aggregator_type='mean',
                feat_drop=dropout
            ))

        # 批归一化层
        self.bns = nn.ModuleList([nn.BatchNorm1d(h_feats * 2) for _ in range(num_layers - 1)])
        self.bns.append(nn.BatchNorm1d(h_feats))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(h_feats * 3, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_feats, num_classes)
        )

    def forward(self, g, features, edge_types=None):
        # 编码节点特征
        h = self.feature_encoder(features)

        # 如果有边类型信息，增强特征
        if edge_types is not None:
            edge_embeddings = self.relation_type_encoder(edge_types)
            # 这里可以添加边类型信息到节点特征的处理

        # 多层GraphSAGE传播
        layer_outputs = []

        for i in range(self.num_layers):
            # GraphSAGE聚合
            h = self.sage_layers[i](g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # 特征融合：结合初始特征、中间层特征和最终层特征
        if len(layer_outputs) > 1:
            # 使用初始特征、中间层特征和最终层特征
            initial_features = self.feature_encoder(features)
            final_features = torch.cat([initial_features, layer_outputs[-1]], dim=1)
        else:
            final_features = layer_outputs[0]

        # 最终分类
        out = self.output_layer(final_features)

        return out


def preprocess_all_multimodal_features(entity_to_idx, entity_types_df, image_dir,
                                       save_path='data/multimodal_features.pt'):
    """预提取所有实体的多模态特征并保存"""

    if os.path.exists(save_path):
        print(f"加载已保存的多模态特征: {save_path}")
        return torch.load(save_path)

    print("开始预提取多模态特征...")
    encoder = MultiModalEncoder()

    # 预加载CSV数据到内存
    entity_data = {}
    for _, row in entity_types_df.iterrows():
        entity_data[row['entity_id']] = row

    multimodal_features = []
    total_entities = len(entity_to_idx)

    for i, entity_id in enumerate(tqdm(entity_to_idx.keys(), desc="提取多模态特征")):
        # 提取特征
        mm_feat = encoder.encode_entity(entity_id, entity_types_df, image_dir)
        multimodal_features.append(mm_feat.cpu())

        # 每1000个实体保存一次进度
        if (i + 1) % 1000 == 0:
            temp_features = torch.stack(multimodal_features)
            torch.save(temp_features, f'{save_path}_temp_{i + 1}.pt')
            print(f"已处理 {i + 1}/{total_entities} 个实体")

    # 最终保存
    all_features = torch.stack(multimodal_features)
    torch.save(all_features, save_path)
    print(f"多模态特征已保存到: {save_path}")

    return all_features

def load_training_data():
    """加载训练数据"""
    print("=" * 60)
    print("步骤1: 加载训练数据...")
    print("=" * 60)

    # 加载关系三元组
    triples = []
    relation_counter = Counter()

    try:
        with open('data/FB15KET/xunlian.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="加载关系数据"):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    triples.append((head, relation, tail))
                    relation_counter[relation] += 1
    except FileNotFoundError:
        print("错误: xunlian.txt 文件不存在")
        return None, None, None

    print(f"加载了 {len(triples)} 个关系三元组")
    print(f"发现 {len(relation_counter)} 种不同关系类型")

    # 加载实体类型数据
    try:
        entity_types = pd.read_csv('data/FB15KET/Entity_All_typed.csv', encoding='utf-8')

        # ✅ 新增：预处理数值列，确保是数字类型
        numeric_columns = ['category_1_score', 'category_2_score', 'category_3_score',
                           'category_4_score', 'category_5_score', 'category_6_score',
                           'category_7_score', 'category_8_score', 'category_9_score']

        for col in numeric_columns:
            # 将列转换为数值类型，非数字转为NaN，然后填充0
            entity_types[col] = pd.to_numeric(entity_types[col], errors='coerce').fillna(0.0)

        print(f"已预处理CSV中的数值列")

    except FileNotFoundError:
        print("错误: Entity_All_typed.csv 文件不存在")
        return None, None, None

    print(f"加载了 {len(entity_types)} 个实体类型记录")



    # 统计类型分布
    if 'predicted_category' in entity_types.columns:
        type_counts = entity_types['predicted_category'].value_counts()
        print(f"发现 {len(type_counts)} 种不同实体类型")
        print(f"最常见的5种类型:")
        for i, (type_id, count) in enumerate(type_counts.head().items()):
            print(f"  类型 {type_id}: {count} 个实体")

    return triples, entity_types, relation_counter


def build_entity_graph(triples,max_entities=15000):
    """构建实体关系图"""
    print("\n构建实体关系图...")

    # 收集所有实体
    all_entities = set()
    for h, r, t in triples:
        all_entities.update([h, t])

    print(f"总实体数: {len(all_entities)}")

    # 如果实体太多，随机采样
    if len(all_entities) > max_entities:
        import random
        sampled_entities = random.sample(list(all_entities), max_entities)
        entity_set = set(sampled_entities)

        # 筛选只包含采样实体的三元组
        filtered_triples = []
        for h, r, t in triples:
            if h in entity_set and t in entity_set:
                filtered_triples.append((h, r, t))

        print(f"采样后: {len(sampled_entities)} 个实体, {len(filtered_triples)} 个三元组")
        triples = filtered_triples
        all_entities = sampled_entities


    # 创建实体到索引的映射
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}

    # 构建边
    src_nodes = []
    dst_nodes = []

    for h, r, t in tqdm(triples, desc="构建图边"):
        src_nodes.append(entity_to_idx[h])
        dst_nodes.append(entity_to_idx[t])

    # 创建DGL图
    num_nodes = len(all_entities)
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                  num_nodes=num_nodes)

    # 添加自环
    g = dgl.add_self_loop(g)

    print(f"图构建完成: 节点数={g.num_nodes()}, 边数={g.num_edges()}")

    return g, entity_to_idx, idx_to_entity


def extract_entity_features_for_sage(triples, entity_types, entity_to_idx, relation_counter,cache_dir='models', force_recompute=False):
    """为GraphSAGE模型提取实体特征"""
    """为GraphSAGE模型提取实体特征，支持缓存"""
    global StandardScaler
    print("\n为GraphSAGE提取实体特征...")

    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)

    # 生成缓存文件名
    import hashlib
    import pickle

    # 根据输入数据生成唯一的缓存key
    data_info = {
        'triples_count': len(triples),
        'entities_count': len(entity_to_idx),
        'relations_count': len(relation_counter),
        'entity_types_count': len(entity_types)
    }

    # 创建哈希值用于标识数据
    data_str = f"{len(triples)}_{len(entity_to_idx)}_{len(relation_counter)}_{len(entity_types)}"
    data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]

    cache_file = os.path.join(cache_dir, f'graph_features_{data_hash}.pkl')
    print(f"缓存文件: {cache_file}")

    # 检查缓存是否存在且不需要强制重新计算
    if os.path.exists(cache_file) and not force_recompute:
        try:
            print("加载缓存的图特征...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # 验证缓存数据是否匹配当前数据
            cached_info = cached_data.get('data_info', {})
            if (cached_info.get('triples_count') == data_info['triples_count'] and
                    cached_info.get('entities_count') == data_info['entities_count']):

                print("✅ 缓存命中，使用已计算的特征")

                # 恢复scaler状态（如果有）
                if 'scaler_state' in cached_data:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.mean_ = cached_data['scaler_state']['mean']
                    scaler.scale_ = cached_data['scaler_state']['scale']
                    scaler.var_ = cached_data['scaler_state']['var']
                    scaler.n_features_in_ = cached_data['scaler_state']['n_features_in']
                    scaler.n_samples_seen_ = cached_data['scaler_state']['n_samples_seen']
                else:
                    scaler = cached_data.get('scaler', None)

                return {
                    'node_features': cached_data['node_features'],
                    'entity_to_type': cached_data['entity_to_type'],
                    'feature_names': cached_data['feature_names'],
                    'scaler': scaler,
                    'top_relations': cached_data['top_relations'],
                    'type_to_idx': cached_data['type_to_idx'],
                    'all_types': cached_data['all_types'],
                    'from_cache': True  # 添加标志表示来自缓存
                }
            else:
                print("⚠️ 缓存数据不匹配，重新计算...")
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}，重新计算...")

    print("开始提取图特征...")

    # 获取实体类型映射
    entity_to_type = dict(zip(entity_types['entity_id'],
                              entity_types['predicted_category']))

    # 统计实体关系信息
    entity_relations = defaultdict(list)
    entity_in_degree = Counter()
    entity_out_degree = Counter()
    entity_neighbors = defaultdict(set)

    for h, r, t in tqdm(triples, desc="统计实体关系"):
        entity_relations[h].append(r)
        entity_relations[t].append(r)
        entity_out_degree[h] += 1
        entity_in_degree[t] += 1
        entity_neighbors[h].add(t)
        entity_neighbors[t].add(h)

    # 选择最常见的关系类型
    top_relations = [rel for rel, _ in relation_counter.most_common(50)]
    relation_to_feat_idx = {rel: i for i, rel in enumerate(top_relations)}

    # ========== 为GraphSAGE提取特征 ==========
    print("提取GraphSAGE特征...")

    # 1. 基础特征
    base_features = []

    # 2. 关系模式特征（重点）
    relation_pattern_features = []

    # 3. 邻居类型特征
    neighbor_type_features = []

    # 收集所有类型
    all_types = set(entity_to_type.values())
    type_to_idx = {t: i for i, t in enumerate(sorted(all_types))}

    for entity in tqdm(entity_to_idx.keys(), desc="提取特征"):
        idx = entity_to_idx[entity]

        # 1. 基础特征
        base_feat = [
            1.0 if entity in entity_to_type else 0.0,  # has_label
            float(entity_in_degree.get(entity, 0)),  # in_degree
            float(entity_out_degree.get(entity, 0)),  # out_degree
            float(entity_in_degree.get(entity, 0) + entity_out_degree.get(entity, 0)),  # total_degree
            float(len(set(entity_relations.get(entity, [])))),  # unique_relations
        ]

        # 邻居信息
        neighbors = entity_neighbors.get(entity, [])
        neighbor_count = len(neighbors)
        base_feat.append(float(neighbor_count))

        # 计算邻居类型分布
        neighbor_types = []
        labeled_neighbors = 0
        for neighbor in neighbors:
            if neighbor in entity_to_type:
                labeled_neighbors += 1
                neighbor_types.append(entity_to_type[neighbor])

        base_feat.append(float(labeled_neighbors))
        base_feat.append(float(len(set(neighbor_types)) if neighbor_types else 0))

        # 如果有邻居类型，计算最常见的类型
        if neighbor_types:
            type_counts = Counter(neighbor_types)
            most_common = type_counts.most_common(1)[0][0]
            base_feat.append(float(type_to_idx.get(most_common, 0)))
        else:
            base_feat.append(0.0)

        base_features.append(base_feat)

        # 2. 关系模式特征（重点改进）
        rel_pattern_feat = np.zeros(len(top_relations) * 2, dtype=np.float32)

        # 统计作为头实体和尾实体的关系分布
        head_relations = Counter()
        tail_relations = Counter()

        for h, r, t in triples:
            if h == entity:
                head_relations[r] += 1
            if t == entity:
                tail_relations[r] += 1

        total_head = sum(head_relations.values())
        total_tail = sum(tail_relations.values())

        for rel_idx, rel in enumerate(top_relations):
            # 作为头实体的关系频率
            if total_head > 0:
                rel_pattern_feat[rel_idx] = head_relations.get(rel, 0) / total_head

            # 作为尾实体的关系频率
            if total_tail > 0:
                rel_pattern_feat[rel_idx + len(top_relations)] = tail_relations.get(rel, 0) / total_tail

        relation_pattern_features.append(rel_pattern_feat)

        # 3. 邻居类型特征
        neighbor_type_feat = np.zeros(len(all_types), dtype=np.float32)
        if neighbor_types:
            total_neighbors = len(neighbor_types)
            for ntype in neighbor_types:
                if ntype in type_to_idx:
                    type_idx = type_to_idx[ntype]
                    neighbor_type_feat[type_idx] += 1.0 / total_neighbors

        neighbor_type_features.append(neighbor_type_feat)

    # 组合所有特征
    base_features_np = np.array(base_features, dtype=np.float32)
    relation_pattern_np = np.array(relation_pattern_features, dtype=np.float32)
    neighbor_type_np = np.array(neighbor_type_features, dtype=np.float32)

    print(f"特征维度统计:")
    print(f"  基础特征: {base_features_np.shape[1]} 维")
    print(f"  关系模式特征: {relation_pattern_np.shape[1]} 维")
    print(f"  邻居类型特征: {neighbor_type_np.shape[1]} 维")

    # 合并特征
    all_features = np.concatenate([base_features_np, relation_pattern_np, neighbor_type_np], axis=1)

    # 标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    node_features = torch.tensor(all_features_scaled, dtype=torch.float32)

    print(f"总特征维度: {node_features.shape[1]}")

    # 特征名称
    feature_names = [
        'has_label', 'in_degree', 'out_degree', 'total_degree', 'unique_relations',
        'neighbor_count', 'labeled_neighbors', 'unique_neighbor_types', 'most_common_neighbor_type'
    ]

    # 添加关系模式特征名称
    for i in range(len(top_relations)):
        feature_names.append(f'head_rel_{i}')
    for i in range(len(top_relations)):
        feature_names.append(f'tail_rel_{i}')

    # 添加邻居类型特征名称
    for i in range(len(all_types)):
        feature_names.append(f'neighbor_type_{i}')

    return {
        'node_features': node_features,
        'entity_to_type': entity_to_type,
        'feature_names': feature_names,
        'scaler': scaler,
        'top_relations': top_relations,
        'type_to_idx': type_to_idx,
        'all_types': list(all_types)
    }


def prepare_labels(entity_to_idx, entity_to_type, label_encoder=None):
    """准备标签数据"""
    print("\n准备标签数据...")

    # 获取有标签的实体
    labeled_entities = [e for e in entity_to_idx.keys() if e in entity_to_type]
    labeled_indices = [entity_to_idx[e] for e in labeled_entities]
    labels = [entity_to_type[e] for e in labeled_entities]

    print(f"有标签的实体: {len(labeled_entities)} / {len(entity_to_idx)}")

    # 编码标签
    if label_encoder is None:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
    else:
        encoded_labels = label_encoder.transform(labels)

    # 创建完整的标签张量
    full_labels = torch.full((len(entity_to_idx),), -1, dtype=torch.long)
    for i, idx in enumerate(labeled_indices):
        full_labels[idx] = encoded_labels[i]

    num_classes = len(label_encoder.classes_)

    # 计算类别权重（用于处理类别不平衡）
    class_counts = np.bincount(encoded_labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f"类别数量: {num_classes}")
    print(f"类别分布范围: {class_counts.min()} ~ {class_counts.max()}")

    return {
        'labels': full_labels,
        'labeled_indices': labeled_indices,
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'class_weights': class_weights
    }


def train_sage_model(model, g, structural_features, multimodal_features, labels, train_mask, val_mask,class_weights):
    """鲁棒的训练函数，彻底解决梯度问题"""
    print("\n" + "=" * 60)
    print("鲁棒训练多模态GraphSAGE模型")
    print("=" * 60)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 移动数据到设备（每次前向传播都会用到的数据）
    model = model.to(device)
    g = g.to(device)
    structural_features = structural_features.to(device)
    multimodal_features = multimodal_features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 记录器
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []

    best_val_f1 = 0
    best_model_state = None
    patience = 30
    patience_counter = 0

    print("\n开始训练...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
    print("-" * 80)

    for epoch in range(200):
        # ========== 训练阶段 ==========
        model.train()

        # 关键：在每个epoch开始时完全清除计算图
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 方法1：使用detach和clone创建新的计算图
        with torch.no_grad():
            # 创建数据的副本，确保新的计算图
            structural_features_epoch = structural_features.clone().requires_grad_(False)
            multimodal_features_epoch = multimodal_features.clone().requires_grad_(False)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)

        # 前向传播 - 使用副本数据
        logits = model(g, structural_features_epoch, multimodal_features_epoch)

        # 计算损失
        loss = criterion(logits[train_mask], labels[train_mask])

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 计算训练准确率
        with torch.no_grad():
            preds = logits[train_mask].argmax(dim=1)
            train_acc = (preds == labels[train_mask]).float().mean()

        train_losses.append(loss.item())
        train_accs.append(train_acc.item())

        # ========== 验证阶段 ==========
        model.eval()
        with torch.no_grad():
            # 验证时也使用副本数据
            val_logits = model(g, structural_features_epoch.detach(), multimodal_features_epoch.detach())
            val_loss = criterion(val_logits[val_mask], labels[val_mask])
            val_preds = val_logits[val_mask].argmax(dim=1)
            val_acc = (val_preds == labels[val_mask]).float().mean()

            # 计算F1
            val_preds_cpu = val_preds.cpu().numpy()
            val_true_cpu = labels[val_mask].cpu().numpy()
            val_f1 = f1_score(val_true_cpu, val_preds_cpu, average='weighted', zero_division=0)

        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())
        val_f1s.append(val_f1)

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch + 1:<6} {loss.item():<12.4f} {train_acc.item():<12.4f} "
                  f"{val_loss.item():<12.4f} {val_acc.item():<12.4f} {val_f1:<12.4f}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch + 1} 轮停止训练")
            break

        # 学习率衰减
        if epoch > 0 and epoch % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            print(f"学习率衰减至 {optimizer.param_groups[0]['lr']:.6f}")

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型，验证集F1分数: {best_val_f1:.4f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        final_logits = model(g, structural_features.detach(), multimodal_features.detach())
        final_preds = final_logits[val_mask].argmax(dim=1)
        final_acc = (final_preds == labels[val_mask]).float().mean()

        final_preds_cpu = final_preds.cpu().numpy()
        final_true_cpu = labels[val_mask].cpu().numpy()
        final_f1 = f1_score(final_true_cpu, final_preds_cpu, average='weighted', zero_division=0)

    print(f"\n训练完成:")
    print(f"  最终验证集准确率: {final_acc.item():.4f}")
    print(f"  最终验证集F1分数: {final_f1:.4f}")

    return model, final_acc.item(), final_f1


def save_sage_model(model, model_config, data_dict, best_val_acc, best_val_f1):
    """保存GraphSAGE模型和相关数据"""
    print("\n" + "=" * 60)
    print("步骤4: 保存GraphSAGE模型和数据...")
    print("=" * 60)

    # 确保目录存在
    os.makedirs('models', exist_ok=True)

    # 准备保存数据
    save_dict = {
        'model_state_dict': model.cpu().state_dict(),
        'model_config': model_config,
        'entity_to_idx': data_dict['entity_to_idx'],
        'idx_to_entity': data_dict['idx_to_entity'],
        'label_encoder': data_dict['label_encoder'],
        'node_features': data_dict['node_features'].cpu(),
        'feature_names': data_dict['feature_names'],
        'scaler': data_dict['scaler'],
        'top_relations': data_dict['top_relations'],
        'type_to_idx': data_dict.get('type_to_idx', {}),
        'all_types': data_dict.get('all_types', []),
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1
    }

    # 保存模型
    model_path = 'models/entity_type_predictor_multi_sage.pth'
    torch.save(save_dict, model_path)
    print(f"✓ 模型已保存到: {model_path}")

    # 保存配置信息
    config_info = f"""
GraphSAGE实体类型预测模型训练配置
==========================================
模型信息:
  模型类型: {model_config.get('model_type', 'TypeAwareGraphSAGE')}
  隐藏层维度: {model_config['h_feats']}
  类别数量: {model_config['num_classes']}
  模型层数: {model_config['num_layers']}
  Dropout率: {model_config['dropout']}

数据信息:
  实体总数: {len(data_dict['entity_to_idx'])}
  有标签实体: {len(data_dict['labeled_indices'])}
  特征数量: {len(data_dict['feature_names'])}
  关系类型数: {len(data_dict['top_relations'])}
  实体类型数: {len(data_dict.get('all_types', []))}

训练结果:
  验证集准确率: {best_val_acc:.4f}
  验证集F1分数: {best_val_f1:.4f}
  训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

模型文件: {model_path}
"""

    with open('models/training_config_sage.txt', 'w', encoding='utf-8') as f:
        f.write(config_info)

    print(config_info)

    return model_path


def main():
    """主训练函数"""
    print("=" * 80)
    print("GraphSAGE实体类型预测模型训练")
    print("基于关系网络和类型模式的TypeAwareGraphSAGE模型")
    print("=" * 80)

    start_time = time.time()

    try:
        # 步骤1: 加载数据
        print("\n1. 加载训练数据...")
        triples, entity_types, relation_counter = load_training_data()
        if triples is None:
            return

        # 步骤2: 构建图
        print("\n2. 构建实体关系图...")
        g, entity_to_idx, idx_to_entity = build_entity_graph(triples)

        # 步骤3: 为GraphSAGE提取特征
        print("\n3. 提取实体特征...")
        feature_dict = extract_entity_features_for_sage(triples, entity_types, entity_to_idx, relation_counter)

        # 修改这部分：
        print("\n3.5 预处理多模态特征...")

        multimodal_encoder = MultiModalEncoder()
        '''
        # 方法1：直接提取（慢）
        encoder = MultiModalEncoder()
        multimodal_features_list = []
        for entity_id in tqdm(entity_to_idx.keys(), desc="提取多模态特征"):
            mm_feat = encoder.encode_entity(entity_id, entity_types, "D:/Z-Downloader/download")
            multimodal_features_list.append(mm_feat)
        multimodal_features = torch.stack(multimodal_features_list)
        '''

        # 方法2：预提取并保存（推荐）
        multimodal_features = preprocess_all_multimodal_features(
            entity_to_idx,
            entity_types,
            image_dir="D:/Z-Downloader/download",
            save_path='data/multimodal_features.pt'
        )

        print(f"多模态特征维度: {multimodal_features.shape}")


        # 新增：提取所有实体的多模态特征
        print("\n3.6 提取多模态特征...")
        multimodal_features_list = []

        for entity_id in tqdm(entity_to_idx.keys(), desc="提取多模态特征"):
            # 提取该实体的多模态特征
            mm_feat = multimodal_encoder.encode_entity(
                entity_id,
                entity_types,
                image_dir="D:/Z-Downloader/download"
            )
            multimodal_features_list.append(mm_feat)

        multimodal_features = torch.stack(multimodal_features_list)  # [num_entities, 512]

        # 修改：现在有两个特征矩阵
        # 1. structural_features: 原有的结构特征
        # 2. multimodal_features: 新的多模态特征

        data_dict = {
            'entity_to_idx': entity_to_idx,
            'idx_to_entity': idx_to_entity,
            'node_features': feature_dict['node_features'],
            'feature_names': feature_dict['feature_names'],
            'scaler': feature_dict['scaler'],
            'top_relations': feature_dict['top_relations'],
            'type_to_idx': feature_dict.get('type_to_idx', {}),
            'all_types': feature_dict.get('all_types', [])
        }

        # 步骤4: 准备标签
        print("\n4. 准备标签数据...")
        label_dict = prepare_labels(entity_to_idx, feature_dict['entity_to_type'])

        # 合并数据字典
        data_dict.update(label_dict)

        # 步骤5: 划分数据集
        print("\n5. 划分训练集和验证集...")
        labeled_indices = np.array(data_dict['labeled_indices'])

        # 获取标签值
        label_values = []
        for idx in labeled_indices:
            entity = idx_to_entity[idx]
            label = feature_dict['entity_to_type'][entity]
            label_values.append(label)

        # 编码标签
        encoded_labels = label_dict['label_encoder'].transform(label_values)

        # 检查每个类别的样本数
        from collections import Counter
        class_counts = Counter(encoded_labels)
        print(f"类别分布:")
        for cls, count in class_counts.items():
            cls_name = label_dict['label_encoder'].inverse_transform([cls])[0]
            print(f"  类别 {cls_name}: {count} 个样本")

        # 检查是否有类别样本过少
        rare_classes = [cls for cls, count in class_counts.items() if count < 2]
        if rare_classes:
            print(f"警告: {len(rare_classes)} 个类别样本数少于2")
            print("将不使用分层抽样 (stratify=False)")
            use_stratify = False
        else:
            use_stratify = True
            print("所有类别都有足够样本，使用分层抽样")

        # 划分数据集
        try:
            if use_stratify and len(set(encoded_labels)) > 1:
                train_idx, val_idx = train_test_split(
                    labeled_indices,
                    test_size=0.2,
                    stratify=encoded_labels,
                    random_state=42
                )
            else:
                print("使用随机划分（非分层）")
                train_idx, val_idx = train_test_split(
                    labeled_indices,
                    test_size=0.2,
                    stratify=None,  # 不使用分层
                    random_state=42
                )

        except ValueError as e:
            print(f"分层抽样失败: {e}")
            print("使用简单随机划分...")
            # 简单随机划分
            np.random.shuffle(labeled_indices)
            split_idx = int(len(labeled_indices) * 0.8)
            train_idx = labeled_indices[:split_idx]
            val_idx = labeled_indices[split_idx:]

        train_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        val_mask = torch.zeros(len(entity_to_idx), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        print(f"训练集: {train_mask.sum().item()} 个节点")
        print(f"验证集: {val_mask.sum().item()} 个节点")

        # 步骤6: 创建多模态GraphSAGE模型
        print("\n6. 创建MultiModalGraphSAGE模型...")
        # 获取正确的维度
        structural_dim = data_dict['node_features'].shape[1]
        multimodal_dim = multimodal_features.shape[1]

        print(f"结构特征维度: {structural_dim}")
        print(f"多模态特征维度: {multimodal_dim}")
        print(f"总输入维度: {structural_dim + multimodal_dim}")
        print(f"类别数量: {data_dict['num_classes']}")

        # 创建模型时传入两个维度
        model = MultiModalGraphSAGE(
            structural_dim=structural_dim,  # 结构特征维度
            multimodal_dim=multimodal_dim,  # 多模态特征维度
            h_feats=256,
            num_classes=data_dict['num_classes'],
            num_layers=2,
            dropout=0.3
        )

        model_config = {
            'structural_dim': structural_dim,
            'multimodal_dim': multimodal_dim,
            'h_feats': 256,
            'num_classes': data_dict['num_classes'],
            'num_layers': 2,
            'dropout': 0.3,
            'model_type': 'MultiModalGraphSAGE'
        }

        print(f"模型配置:")
        print(f"  结构特征维度: {structural_dim}")
        print(f"  多模态特征维度: {multimodal_dim}")
        print(f"  总输入特征: {structural_dim + multimodal_dim}")
        print(f"  隐藏层: 256")
        print(f"  类别数: {data_dict['num_classes']}")
        print(f"  层数: 2")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 步骤7: 训练模型
        print("\n7. 开始训练...")
        model, best_val_acc, best_val_f1 = train_sage_model(
            model, g,
            structural_features=data_dict['node_features'],
            multimodal_features=multimodal_features,  # 添加这个参数！
            labels=data_dict['labels'],
            train_mask=train_mask,
            val_mask=val_mask,
            class_weights=data_dict['class_weights']
        )
        # 步骤8: 保存模型
        print("\n8. 保存模型...")
        model_path = save_sage_model(model, model_config, data_dict, best_val_acc, best_val_f1)

        # 训练完成
        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        print(f"总训练时间: {training_time:.2f} 秒")
        print(f"模型文件: {model_path}")
        print(f"验证集准确率: {best_val_acc:.4f}")
        print(f"验证集F1分数: {best_val_f1:.4f}")

        # 显示模型特点
        print(f"\nGraphSAGE模型特点:")
        print(f"  1. 邻居采样聚合，适合大规模图")
        print(f"  2. 显式的关系模式特征提取")
        print(f"  3. 类型感知的特征编码")
        print(f"  4. 多层特征融合机制")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()