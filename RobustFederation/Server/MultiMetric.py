import copy

from Server.utils.server_methods import ServerMethod
import torch
import numpy as np




class MultiMetric(ServerMethod):
    NAME = 'MultiMetric'

    def __init__(self, args, cfg):
        super(MultiMetric, self).__init__(args, cfg)

    def aggregate_models(self, models, weights):
        """
        聚合多个模型参数
        :param models: 模型列表
        :param weights: 对应的权重列表
        :return: 聚合后的模型
        """
        # 初始化一个字典来存储聚合后的参数
        agg_state_dict = {}

        # 获取第一个模型的参数结构，用于初始化
        for name, param in models[0].state_dict().items():
            agg_state_dict[name] = param.clone() * 0

        # 加权平均每个模型的参数
        for model, weight in zip(models, weights):
            for name, param in model.state_dict().items():
                agg_state_dict[name] += param * weight

        # 将聚合后的参数加载到新模型
        new_model = type(models[0])()  # 假设所有模型都是同一种类
        new_model.load_state_dict(agg_state_dict)

        return new_model

    def calculate_euclidean_distance(self, model_a, model_b):
        """计算两个模型参数之间的欧几里得距离"""
        distance = 0
        for key in model_a.keys():
            diff = model_a[key] - model_b[key]
            distance += torch.sum(diff ** 2)
        return torch.sqrt(distance).item()

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        # priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        last_weight_list = [1/len(online_clients_list) for i in range(len(online_clients_list))]
        # fish_diff_dict = kwargs['fish_diff_dict']
        # fish_diff_sum_dict = {}
        temp_net = copy.deepcopy(global_net)
        # print(nets_list[0])

        # 预聚合权重
        ecpt_weights_grad = []

        all_delta = []
        for i in range(len(online_clients_list)):
            ecpt_global_net = copy.deepcopy(nets_list[online_clients_list[i]])
            # 去除第i个客户端的权重
            ecpt_weight_list = torch.tensor(last_weight_list[:i] + last_weight_list[i + 1:], device=self.device)
            # ecpt_net_list = [nets_list[j].unsqueeze(0) for j in range(len(nets_list)) if j != i]
            ecpt_online_list = online_clients_list[:i] + online_clients_list[i+1:]
            # 归一化剩余权重
            normalized_ecpt_weight = ecpt_weight_list / ecpt_weight_list.sum()
            self.agg_parts(online_clients_list=ecpt_online_list, nets_list=nets_list,
                           global_net=ecpt_global_net, freq=normalized_ecpt_weight, except_part=[], global_only=True)
            ecpt_weights_grad.append(ecpt_global_net)
        # all_delta = np.array(all_delta)

        # 计算对比指标
        comparison_metrics = []
        for _, i in enumerate(online_clients_list):
            net_idx = _
            g_i = nets_list[i]
            # tmp_weight = copy.deepcopy(all_delta[net_idx])
            # note：余弦相似度
            client_vector = torch.cat([param.view(-1) for param in g_i.parameters()]).detach().cpu().numpy()
            client_vector = np.array(client_vector)
            others_vector = torch.cat([param.view(-1) for param in ecpt_weights_grad[net_idx].parameters()]).detach().cpu().numpy()
            others_vector = np.array(others_vector)
            cosine_distance = np.dot(client_vector, others_vector) / (np.linalg.norm(client_vector) * np.linalg.norm(others_vector) + 1e-5)
            # euclidean_distance = self.calculate_euclidean_distance(ecpt_weights_grad[net_idx].state_dict(),g_i.state_dict())
            # 客户端更新,tmp_weight
            # norm = 1 / (np.linalg.norm(tmp_weight) + 1e-5)
            comparison_metrics.append(cosine_distance)
            # comparison_metrics.append(euclidean_distance)
        print(comparison_metrics)
        # print((comparison_metrics - max(comparison_metrics))/(max(comparison_metrics)-min(comparison_metrics)))
        comparison_metrics = np.exp(comparison_metrics - max(comparison_metrics) + 1e-5)
        # comparison_metrics = [(max(comparison_metrics) - i) / (max(comparison_metrics) - min(comparison_metrics)) for
        #                       _, i in enumerate(comparison_metrics)]
        # comparison_metrics = [(i - min(comparison_metrics))/(max(comparison_metrics) - min(comparison_metrics)) for _,i in enumerate(comparison_metrics)]
        # note
        comparison_metrics = np.array(comparison_metrics)
        elastic_weights = comparison_metrics / np.sum(comparison_metrics)
        print(elastic_weights)

        # 根据弹性权重进行加权平均
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=elastic_weights, except_part=[], global_only=False)

        return elastic_weights