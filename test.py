import dgl
print(f"DGL版本: {dgl.__version__}")
# 尝试创建一个简单的图，如果不报错，就说明核心库加载成功了
g = dgl.graph(([0, 1], [1, 2]))
print(g)

