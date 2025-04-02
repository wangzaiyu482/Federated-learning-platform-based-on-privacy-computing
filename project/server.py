import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import torch
import logging
import json
from config import FEDERATED_CONFIG


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_logs.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 定义加权平均函数
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 定义训练配置函数
def fit_config(rnd: int):
    return {"num_epochs": FEDERATED_CONFIG['num_epochs']}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # 初始化列表以存储准确值和损失
        rounds = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        confusionMatrices = []
        # 输出每一个客户端的参数结果和 num_rounds
        for client_index, (client_proxy, fit_res) in enumerate(results):
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            acc = metrics.get("accuracy", "N/A")
            loss = metrics.get("loss", "N/A")
            pre = metrics.get("precision", "N/A")
            rec = metrics.get("recall", "N/A")
            f = metrics.get("f1", "N/A")
            logging.info(
                f"Round {server_round}, Client {client_index}: "
                f"Num examples = {num_examples}, Accuracy = {acc}, Loss = {loss}"
            )
            # 存储准确值
            if isinstance(acc, (int, float)):
                rounds.append(server_round)
                accuracy.append(acc)
                precision.append(pre)
                recall.append(rec)
                f1.append(f)
                confusionMatrices.append([[0, 0], [0, 0]])
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # 保存模型
            model_path = f"fed_avg_model_round_{server_round}.pth"
            torch.save(aggregated_parameters, model_path)
            print(f"Model saved at round {server_round} as {model_path}")
            # 将数据写入 JSON 文件
            data = {
                "rounds": rounds,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusionMatrices": confusionMatrices
            }
            with open(f'fit_results_round_{server_round}.json', 'w') as f:
                json.dump(data, f, indent=4)
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        # 初始化列表以存储准确值和损失
        rounds = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        confusionMatrices = []
        # 输出每一个客户端的验证结果
        for client_index, (client_proxy, evaluate_res) in enumerate(results):
            num_examples = evaluate_res.num_examples
            metrics = evaluate_res.metrics
            acc = metrics.get("accuracy", "N/A")
            loss = metrics.get("loss", "N/A")
            pre = metrics.get("precision", "N/A")
            rec = metrics.get("recall", "N/A")
            f = metrics.get("f1","N/A")
            logging.info(
                f"Round {server_round}, Client {client_index} (Validation): "
                f"Num examples = {num_examples}, Accuracy = {acc}, Loss = {loss}, Precision = {pre}"
            )
            # 存储准确值
            if isinstance(acc, (int, float)):
                rounds.append(server_round)
                accuracy.append(acc)
                # 这里假设 precision, recall, f1 和 confusionMatrices 暂时没有值，可根据实际情况修改
                precision.append(pre)
                recall.append(rec)
                f1.append(f)
                confusionMatrices.append([[0, 0], [0, 0]])
        # 将数据写入 JSON 文件
        data = {
            "rounds": rounds,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusionMatrices": confusionMatrices
        }
        with open(f'evaluate_results_round_{server_round}.json', 'w') as f:
            json.dump(data, f, indent=4)
        return super().aggregate_evaluate(server_round, results, failures)

# 启动服务器
strategy = SaveModelStrategy(
    fraction_fit=FEDERATED_CONFIG['fraction_fit'],
    fraction_evaluate=FEDERATED_CONFIG['fraction_evaluate'],
    min_fit_clients=FEDERATED_CONFIG['min_fit_clients'],
    min_evaluate_clients=FEDERATED_CONFIG['min_evaluate_clients'],
    min_available_clients=FEDERATED_CONFIG['min_available_clients'],
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
)

fl.server.start_server(
    server_address=FEDERATED_CONFIG['server_address'],
    config=fl.server.ServerConfig(num_rounds=FEDERATED_CONFIG['num_rounds']),
    strategy=strategy,
)