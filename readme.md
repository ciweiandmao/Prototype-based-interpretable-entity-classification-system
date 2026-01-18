1.根据requirements安装运行环境<br><br>
2.data/FB15KET中含有初始的6个文件，包含训练，测试，验证集以及对应的实体ID转实体TAG文件，由于这些文件不是为从关系网预测实体所设计，我们要预处理。<br><br>
3.将所有实体ID转实体TAG文件合并为Entity_All_en.txt<br><br>
4.运行Entity_merger.py合并有相同实体ID的TAG列为Entity_All_en_single.txt"<br><br>
5.使用argostranslate组件，运行Extract_Count_Translate_Tags.py提取Entity_All_en.txt中所有tag并翻译为中文,保存至Tag_Statistics.csv<br><br>
6.运行Tag_Typer.py根据中英文描述和tag的出现次数，确定tag的类型以及其权重，保存至Tag_Classifications.csv<br><br>
7.运行Entity_typer_by_tags.py根据实体含有的tag以及tag信息，直接为实体进行分类，结果保存至Entity_All_typed.csv<br><br>
8.运行Entity_type_evaluator.py对结果进行人工评估，若出现和事实不符的实体返回对tag信息进行人工修正，直到准确率95%以上<br><br>
9.将train.txt ,valid.txt, test.txt和并为xunlian.txt<br><br>
10.运行Generate_train_test_part.py随机将xunlian.txt分割为TEST_PART_DETAIL.txt和TRAIN_PART.txt,其中被选为测试实体的任意一条边都不存在于训练集<br><br>
11.运行Torch_Train_2.py使用IGardNet和ResGCN结合的办法训练模型，得到entity_type_predictor_resgcn.pth<br><br>
12.运行测试Torch_Test.py输入TEST_PART_DETAIL.txt,对选中的实体进行预测并和Entity_All_typed.csv中真实结果对比，准确率80%-90%<br><br>
