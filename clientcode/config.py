import os
import yaml

# 读取YAML配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 数据集配置
DATA_CONFIG = config['data']

# 模型配置
MODEL_CONFIG = config['model']
