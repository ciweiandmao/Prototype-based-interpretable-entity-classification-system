#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量清洗图片
python clean_images.py  <图片根目录>
"""
import os, shutil, argparse, logging, math
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import pandas as pd
import numpy as np

MIN_SIDE = 20              # 最小边长
MAX_COLORS = 64            # 颜色种类上限，超过认为花屏
MIN_VAR   = 5              # 方差下限，低于认为单调

warnings.filterwarnings('ignore')
def delete_path(root, src):
    """返回 DELETE 目录下的目标路径，保持原目录结构"""
    rel = os.path.relpath(src, root)
    return os.path.join(root, 'DELETE', rel)

def is_bad(img_path: str) -> bool:
    """True -> 坏图（截断/过小）"""
    try:
        with Image.open(img_path) as im:
            im.load()          # 强制读完整，截断会在这里抛异常
    except (OSError, UnidentifiedImageError, ValueError):
        return True             # 无法打开/截断都算坏图

    if im.width < 20 or im.height < 20:
        return True

    return False

def process_one(root, file_path):
    """单文件检查，坏图则移动到 DELETE"""
    try:
        """单文件检查，坏图则移动到 DELETE"""
        if is_bad(file_path):  # ✅ 传路径
            dst = delete_path(root, file_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(file_path, dst)
            return 'bad'
        return 'ok'
    except (UnidentifiedImageError, OSError, ValueError):
        # 无法解析也算坏图
        dst = delete_path(root, file_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(file_path, dst)
        return 'broken'

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='图片根目录')
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    delete_root = os.path.join(root, 'DELETE')

    # 1. 收集所有图片路径
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        if dirpath.startswith(delete_root):
            continue
        for f in filenames:
            if os.path.splitext(f.lower())[1] in img_ext:
                all_files.append(os.path.join(dirpath, f))

    print(f'共扫描到 {len(all_files)} 张图片，开始清洗...')
    bad_cnt = 0
    with Pool(cpu_count()) as pool:
        func = partial(process_one, root)
        for res in tqdm(pool.imap_unordered(func, all_files), total=len(all_files)):
            if res in ('bad', 'broken'):
                bad_cnt += 1
    print(f'清洗完成，坏图/无效图共计 {bad_cnt} 张，已移动到 {delete_root}')

    # 2. 统计空文件夹
    print('正在扫描空文件夹...')
    empty_dirs = []
    for dirpath, dirs, files in os.walk(root, topdown=False):  # 自底向上
        if dirpath == delete_root:
            continue
        # 目录里既无文件也无子目录才算空
        if not dirs and not files:
            empty_dirs.append(os.path.relpath(dirpath, root))

    empty_txt = os.path.join(root, 'empty.txt')
    with open(empty_txt, 'w', encoding='utf-8') as f:
        for d in empty_dirs:
            f.write(d + '\n')
    print(f'空文件夹共计 {len(empty_dirs)} 个，列表已写入 {empty_txt}')



if __name__ == '__main__':
    main()