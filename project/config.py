import os
import yaml

# 读取YAML配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
# 联邦学习配置
FEDERATED_CONFIG = config['federated']
