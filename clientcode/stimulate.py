import multiprocessing
from flwr.client import start_client
import torch
from torch.utils.data import DataLoader,random_split
from Dataset import CancerDataset, train_transform
from Model import CancerModel
import Client
from config import DATA_CONFIG
import logging

def split_dataset(dataset, num_clients):
    """将数据集划分为多个子集，每个子集分配给一个客户端"""
    total_size = len(dataset)
    subset_sizes = [total_size // num_clients] * num_clients
    # 处理余数
    remainder = total_size % num_clients
    for i in range(remainder):
        subset_sizes[i] += 1
    subsets = random_split(dataset, subset_sizes)
    return subsets

def start_client_process(client_id, train_subset, val_subset):
    """启动一个客户端进程"""
    # 设置日志记录
    logger = logging.getLogger(f'client_{client_id}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'client_{client_id}.log', mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"客户端 {client_id} 启动")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化模型
    logger.info("正在初始化模型...")
    model = CancerModel().to(device)
    logger.info("模型初始化完毕")

    # 准备数据集
    logger.info("开始准备数据集")
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    logger.info("数据集准备完毕")

    logger.info("正在初始化客户端")
    # 初始化客户端
    client = Client.FlowerClient(model, train_loader, val_loader)
    logger.info("客户端初始化完毕")
    # 连接服务器
    logger.info("正在连接服务器...")
    try:
        start_client(server_address="127.0.0.1:8078", client=client.to_client())
        logger.info("成功连接到服务器")
    except Exception as e:
        logger.error(f"连接服务器失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置根日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("开始加载数据集")
    csv_file_path = DATA_CONFIG['csv_file_path']
    image_folder_path = DATA_CONFIG['image_folder_path']
    dataset = CancerDataset(csv_file=csv_file_path, root_dir=image_folder_path, transform=train_transform)
    logging.info("数据集加载完毕")

    # 先全局划分训练集和验证集（80%训练，20%验证）
    train_size = int(0.008 * len(dataset))
    val_size = int(0.002*len(dataset))
    test_size = len(dataset)-train_size-val_size
    train_dataset, val_dataset ,test_dataset = random_split(dataset, [train_size, val_size,test_size])

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset)-train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 再分配给各客户端
    num_clients = 1
    train_subsets = split_dataset(train_dataset, num_clients)
    val_subsets = split_dataset(val_dataset, num_clients)


    # 启动多个客户端
    processes = []
    for client_id in range(num_clients):
        p = multiprocessing.Process(target=start_client_process, args=(client_id, train_subsets[client_id], val_subsets[client_id]))
        p.start()
        processes.append(p)

    # 等待所有客户端进程结束
    for p in processes:
        p.join()