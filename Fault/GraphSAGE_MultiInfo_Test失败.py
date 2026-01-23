import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict, Counter
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class MultiModalGraphSAGE(nn.Module):
    """多模态GraphSAGE模型（与训练代码完全一致）"""

    def __init__(self, structural_dim, multimodal_dim, h_feats, num_classes, num_layers=2, dropout=0.3):
        super().__init__()

        total_in_feats = structural_dim + multimodal_dim

        # 输入编码层
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

    def forward(self, g, structural_features, multimodal_features):
        # 融合特征
        combined_features = torch.cat([structural_features, multimodal_features], dim=-1)

        # 输入编码
        h = self.input_encoder(combined_features)

        # GraphSAGE传播
        layer_outputs = [h]
        for i in range(self.num_layers):
            h = self.sage_layers[i](g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # 多层特征融合
        if len(layer_outputs) > 1:
            h_final = torch.cat([layer_outputs[0], layer_outputs[-1]], dim=1)
            h_final = self.relation_encoder(h_final)
        else:
            h_final = layer_outputs[0]

        # 最终分类
        out = self.classifier(h_final)

        return out


class MultiModalEncoder:
    """多模态编码器（与训练代码一致）"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数值特征编码器（与训练时保持一致）
        self.numeric_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        ).to(self.device)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 256, 512),  # 2048维图像 + 256维数值
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(self.device)

    def encode_entity(self, entity_id, entity_types_df):
        """编码实体的多模态特征"""
        try:
            # 1. 提取数值特征
            row = entity_types_df[entity_types_df['entity_id'] == entity_id]
            if row.empty:
                numeric_feat = torch.zeros(9, dtype=torch.float32, device=self.device)
            else:
                numeric_values = []
                for i in range(1, 10):
                    col_name = f'category_{i}_score'
                    if col_name in row.columns:
                        val = row[col_name].values[0]
                        numeric_values.append(float(val) if not pd.isna(val) else 0.0)
                    else:
                        numeric_values.append(0.0)
                numeric_feat = torch.tensor(numeric_values, dtype=torch.float32, device=self.device)

            # 2. 数值编码
            numeric_encoded = self.numeric_encoder(numeric_feat.unsqueeze(0)).squeeze(0)

            # 3. 使用零图像特征
            image_feat = torch.zeros(2048, device=self.device)

            # 4. 融合特征
            combined = torch.cat([image_feat, numeric_encoded], dim=-1)
            fused_feature = self.fusion(combined.unsqueeze(0)).squeeze(0)

            return fused_feature
        except Exception as e:
            print(f"编码实体 {entity_id} 时出错: {e}")
            return torch.zeros(512, device=self.device)


class EntityTypePredictor:
    """实体类型预测器"""

    def __init__(self, model_path='models/entity_type_predictor_multi_sage.pth'):
        print("=" * 60)
        print("初始化实体类型预测器...")
        print("=" * 60)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载模型
        print("1. 加载训练好的模型...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取配置
        self.model_config = checkpoint['model_config']
        self.entity_to_idx = checkpoint['entity_to_idx']
        self.idx_to_entity = checkpoint['idx_to_entity']
        self.label_encoder = checkpoint['label_encoder']
        self.top_relations = checkpoint['top_relations']
        self.type_to_idx = checkpoint['type_to_idx']
        self.scaler = checkpoint['scaler']

        print(f"模型配置:")
        print(f"  结构特征维度: {self.model_config['structural_dim']}")
        print(f"  多模态特征维度: {self.model_config['multimodal_dim']}")
        print(f"  隐藏层维度: {self.model_config['h_feats']}")
        print(f"  类别数量: {self.model_config['num_classes']}")

        # 创建模型（必须与训练时完全一致）
        self.model = MultiModalGraphSAGE(
            structural_dim=self.model_config['structural_dim'],
            multimodal_dim=self.model_config['multimodal_dim'],
            h_feats=self.model_config['h_feats'],
            num_classes=self.model_config['num_classes'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        )

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print("✅ 模型加载成功")

        # 2. 加载实体类型数据
        print("\n2. 加载实体类型数据...")
        self.entity_types_df = pd.read_csv('data/FB15KET/Entity_All_typed.csv', encoding='utf-8')

        # 预处理数值列
        numeric_cols = [f'category_{i}_score' for i in range(1, 10)]
        for col in numeric_cols:
            if col in self.entity_types_df.columns:
                self.entity_types_df[col] = pd.to_numeric(self.entity_types_df[col], errors='coerce').fillna(0.0)

        print(f"加载了 {len(self.entity_types_df)} 个实体类型记录")

        # 3. 初始化多模态编码器
        print("\n3. 初始化多模态编码器...")
        self.multimodal_encoder = MultiModalEncoder()

        print("✅ 预测器初始化完成")

    def parse_test_file(self, test_file_path):
        """解析测试文件"""
        print(f"\n解析测试文件: {test_file_path}")

        test_cases = {}
        current_entity = None
        current_triples = []

        with open(test_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # 检测新实体
            if line.startswith("实体:") and "(" in line:
                # 保存上一个实体
                if current_entity and current_triples:
                    test_cases[current_entity] = current_triples

                # 提取新实体ID
                parts = line.split()
                if len(parts) >= 2:
                    entity_id = parts[1].rstrip('(').rstrip(')')
                    current_entity = entity_id
                    current_triples = []

            # 解析三元组
            elif line and current_entity and '\t' in line:
                parts = line.split('\t')
                if len(parts) == 3:
                    current_triples.append((parts[0], parts[1], parts[2]))

        # 保存最后一个实体
        if current_entity and current_triples:
            test_cases[current_entity] = current_triples

        print(f"解析完成: 找到 {len(test_cases)} 个测试实体")

        # 显示前几个实体作为示例
        print("前5个实体示例:")
        for i, (entity_id, triples) in enumerate(list(test_cases.items())[:5]):
            print(f"  {i + 1}. {entity_id}: {len(triples)} 个关系")

        return test_cases

    def extract_structural_features(self, target_entity, neighbor_triples):
        """提取结构特征"""
        # 统计关系信息
        entity_relations = []
        entity_in_degree = 0
        entity_out_degree = 0

        for h, r, t in neighbor_triples:
            if h == target_entity:
                entity_out_degree += 1
                entity_relations.append(r)
            if t == target_entity:
                entity_in_degree += 1
                entity_relations.append(r)

        # 1. 基础特征
        has_label = 1.0 if target_entity in self.entity_types_df['entity_id'].values else 0.0
        base_features = np.array([
            has_label,
            float(entity_in_degree),
            float(entity_out_degree),
            float(entity_in_degree + entity_out_degree),
            float(len(set(entity_relations))),
            0.0, 0.0, 0.0, 0.0  # 占位符，与训练时一致
        ], dtype=np.float32)

        # 2. 关系模式特征
        rel_pattern_feat = np.zeros(len(self.top_relations) * 2, dtype=np.float32)

        # 统计作为头实体和尾实体的关系分布
        head_relations = Counter()
        tail_relations = Counter()

        for h, r, t in neighbor_triples:
            if h == target_entity:
                head_relations[r] += 1
            if t == target_entity:
                tail_relations[r] += 1

        total_head = sum(head_relations.values())
        total_tail = sum(tail_relations.values())

        for rel_idx, rel in enumerate(self.top_relations):
            # 作为头实体的关系频率
            if total_head > 0 and rel in head_relations:
                rel_pattern_feat[rel_idx] = head_relations[rel] / total_head

            # 作为尾实体的关系频率
            if total_tail > 0 and rel in tail_relations:
                rel_pattern_feat[rel_idx + len(self.top_relations)] = tail_relations[rel] / total_tail

        # 3. 邻居类型特征（预测时用零向量）
        neighbor_type_feat = np.zeros(len(self.type_to_idx), dtype=np.float32)

        # 4. 组合所有特征
        all_features = np.concatenate([base_features, rel_pattern_feat, neighbor_type_feat])

        # 5. 标准化（使用训练时的scaler）
        features_scaled = self.scaler.transform(all_features.reshape(1, -1))

        return torch.tensor(features_scaled, dtype=torch.float32).squeeze(0)

    def build_test_graph(self, target_entity, neighbor_triples):
        """为测试实体构建图"""
        # 收集所有相关实体
        all_entities = set([target_entity])
        for h, r, t in neighbor_triples:
            all_entities.update([h, t])

        # 创建实体到索引的映射
        entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}

        # 构建边
        src_nodes = []
        dst_nodes = []
        for h, r, t in neighbor_triples:
            src_nodes.append(entity_to_idx[h])
            dst_nodes.append(entity_to_idx[t])

        # 创建DGL图
        num_nodes = len(all_entities)
        g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)),
                      num_nodes=num_nodes)

        # 添加自环
        g = dgl.add_self_loop(g)

        return g, entity_to_idx

    def predict_entity_type(self, target_entity, neighbor_triples):
        """预测单个实体的类型"""
        try:
            # 1. 构建测试图
            g, entity_to_idx = self.build_test_graph(target_entity, neighbor_triples)
            g = g.to(self.device)

            target_idx = entity_to_idx[target_entity]

            # 2. 提取结构特征
            structural_feat = self.extract_structural_features(target_entity, neighbor_triples)

            # 为所有节点创建特征矩阵
            num_nodes = g.num_nodes()
            structural_dim = structural_feat.shape[0]
            all_structural_features = torch.zeros(num_nodes, structural_dim)
            all_structural_features[target_idx] = structural_feat

            # 3. 提取多模态特征
            multimodal_feat = self.multimodal_encoder.encode_entity(target_entity, self.entity_types_df)

            # 为所有节点创建多模态特征矩阵
            multimodal_dim = multimodal_feat.shape[0]
            all_multimodal_features = torch.zeros(num_nodes, multimodal_dim)
            all_multimodal_features[target_idx] = multimodal_feat

            # 4. 移动到设备
            all_structural_features = all_structural_features.to(self.device)
            all_multimodal_features = all_multimodal_features.to(self.device)

            # 5. 预测
            with torch.no_grad():
                logits = self.model(g, all_structural_features, all_multimodal_features)

                # 只获取目标节点的预测
                target_logits = logits[target_idx:target_idx + 1]
                probabilities = F.softmax(target_logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()

                # 解码类别
                predicted_type = self.label_encoder.inverse_transform([predicted_class])[0]
                confidence = probabilities[0, predicted_class].item()

                # 获取top-3预测
                top3_probs, top3_indices = torch.topk(probabilities[0], k=min(3, len(probabilities[0])))
                top3_types = self.label_encoder.inverse_transform(top3_indices.cpu().numpy())
                top3_confidences = top3_probs.cpu().numpy()

            # 准备结果
            result = {
                'entity_id': target_entity,
                'predicted_type': predicted_type,
                'confidence': confidence,
                'top_predictions': [
                    {'type': t, 'confidence': float(c)}
                    for t, c in zip(top3_types, top3_confidences)
                ]
            }

            return result

        except Exception as e:
            print(f"预测实体 {target_entity} 时出错: {e}")
            return None

    def batch_predict(self, test_cases):
        """批量预测"""
        print(f"\n开始批量预测 {len(test_cases)} 个实体...")

        results = []
        success_count = 0

        for entity_id, triples in tqdm(test_cases.items(), desc="预测进度"):
            result = self.predict_entity_type(entity_id, triples)
            if result:
                results.append(result)
                success_count += 1

        print(f"✅ 批量预测完成: 成功 {success_count}/{len(test_cases)}")
        return results

    def save_predictions(self, predictions, output_file='predictions_fixed.csv'):
        """保存预测结果"""
        if not predictions:
            print("没有预测结果可保存")
            return None

        # 准备数据
        data = []
        for pred in predictions:
            row = {
                'entity_id': pred['entity_id'],
                'predicted_type': pred['predicted_type'],
                'confidence': pred['confidence'],
                'top1_type': pred['top_predictions'][0]['type'] if len(pred['top_predictions']) > 0 else '',
                'top1_confidence': pred['top_predictions'][0]['confidence'] if len(pred['top_predictions']) > 0 else 0,
                'top2_type': pred['top_predictions'][1]['type'] if len(pred['top_predictions']) > 1 else '',
                'top2_confidence': pred['top_predictions'][1]['confidence'] if len(pred['top_predictions']) > 1 else 0,
                'top3_type': pred['top_predictions'][2]['type'] if len(pred['top_predictions']) > 2 else '',
                'top3_confidence': pred['top_predictions'][2]['confidence'] if len(pred['top_predictions']) > 2 else 0,
            }
            data.append(row)

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 保存到CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 预测结果已保存到: {output_file}")

        # 显示前5条结果
        print("\n前5条预测结果:")
        print(df.head().to_string())

        return df

    def evaluate_predictions(self, predictions):
        """评估预测结果"""
        print("\n" + "=" * 60)
        print("评估预测结果")
        print("=" * 60)

        correct = 0
        total = 0
        evaluation_results = []

        for pred in predictions:
            entity_id = pred['entity_id']

            # 查找真实标签
            true_row = self.entity_types_df[self.entity_types_df['entity_id'] == entity_id]
            if not true_row.empty:
                true_type = true_row.iloc[0]['predicted_category']
                predicted_type = pred['predicted_type']

                total += 1
                is_correct = (true_type == predicted_type)

                if is_correct:
                    correct += 1

                evaluation_results.append({
                    'entity_id': entity_id,
                    'true_type': true_type,
                    'predicted_type': predicted_type,
                    'confidence': pred['confidence'],
                    'is_correct': is_correct
                })

        if total > 0:
            accuracy = correct / total
            print(f"\n📊 评估结果:")
            print(f"  正确预测: {correct}/{total}")
            print(f"  准确率: {accuracy:.4f}")

            # 保存评估结果
            eval_df = pd.DataFrame(evaluation_results)
            eval_df.to_csv('evaluation_results_fixed.csv', index=False, encoding='utf-8')
            print(f"✅ 评估结果已保存到: evaluation_results_fixed.csv")

            return accuracy
        else:
            print("⚠️ 没有找到真实标签，无法评估")
            return None


def main():
    """主函数"""
    print("=" * 80)
    print("实体类型预测系统")
    print("=" * 80)

    try:
        # 1. 初始化预测器
        predictor = EntityTypePredictor('models/entity_type_predictor_multi_sage.pth')

        # 2. 解析测试文件
        test_file = 'data/FB15KET/TEST_PART_DETAILED.txt'
        test_cases = predictor.parse_test_file(test_file)

        if not test_cases:
            print("使用示例数据...")
            test_cases = {
                '/m/027rn': [
                    ('/m/027rn', '/location/country/form_of_government', '/m/06cx9'),
                    ('/m/01wy61y', '/people/person/nationality', '/m/027rn'),
                ]
            }

        # 3. 批量预测
        results = predictor.batch_predict(test_cases)

        if results:
            # 4. 保存结果
            predictor.save_predictions(results)

            # 5. 评估
            accuracy = predictor.evaluate_predictions(results)

            if accuracy is not None:
                print(f"\n🎯 最终准确率: {accuracy:.4f}")

                # 显示置信度统计
                confidences = [r['confidence'] for r in results]
                print(f"\n📈 置信度统计:")
                print(f"  平均置信度: {np.mean(confidences):.4f}")
                print(f"  中位数置信度: {np.median(confidences):.4f}")
                print(f"  最高置信度: {np.max(confidences):.4f}")
                print(f"  最低置信度: {np.min(confidences):.4f}")
        else:
            print("❌ 没有成功的预测结果")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()