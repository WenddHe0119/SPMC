import copy

import torch

# from Server.utils.server_methods import ServerMethod
#
#
# class Equal(ServerMethod):
#     NAME = 'Equal'
#
#     def __init__(self, args, cfg):
#         super(Equal, self).__init__(args, cfg)
#
#     def weight_calculate(self, **kwargs):
#         online_clients_list = kwargs['online_clients_list']
#         freq =  [1/len(online_clients_list) for _ in range(len(online_clients_list))]
#         return freq
#
#     def server_update(self, **kwargs):
#         online_clients_list = kwargs['online_clients_list']
#         priloader_list = kwargs['priloader_list']
#         global_net = kwargs['global_net']
#         nets_list = kwargs['nets_list']
#
#         freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
#
#
#         self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
#                                       global_net=global_net, freq=freq, except_part=[], global_only=False)
#         return freq

import copy
import numpy as np
import torch

from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters


class Equal(ServerMethod):
    NAME = 'Equal'

    def __init__(self, args, cfg):
        super(Equal, self).__init__(args, cfg)

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        freq = [1 / len(online_clients_list) for _ in online_clients_list]
        return freq

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        global_dict = global_net.state_dict()
        updated_dict = {}

        for name, global_param in global_dict.items():
            # 收集每个客户端该层的 delta，拼成矩阵（n_clients x param_shape）
            delta_list = []
            for client_idx in online_clients_list:
                client_param = nets_list[client_idx].state_dict()[name]
                delta = client_param.detach() - global_param.detach()
                delta_list.append(copy.deepcopy(delta.view(1, -1)))  # shape: [1, numel]

            # 堆叠成矩阵：n_clients x numel
            delta_stack = torch.cat(delta_list, dim=0).cpu().numpy()  # shape: [n_clients, numel]

            # 计算均值
            avg_delta = np.mean(delta_stack, axis=0)

            # 加回 global 参数
            # 将 avg_delta 转回 tensor，reshape 成 global_param 的形状
            avg_delta_tensor = torch.from_numpy(avg_delta).view(global_param.shape).to(global_param.device).to(
                global_param.dtype)

            # 加回 global 参数
            updated_param = global_param + avg_delta_tensor

            updated_dict[name] = updated_param

        # 加载聚合后的模型
        global_net.load_state_dict(updated_dict)

        # 同步回所有客户端
        for net in nets_list:
            net.load_state_dict(global_net.state_dict())

        return [1 / len(online_clients_list) for _ in online_clients_list]

