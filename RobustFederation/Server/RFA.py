import copy

import numpy as np
import torch

from Server.utils.server_methods import ServerMethod
from Server.utils.utils import geometric_median_update
from utils.utils import row_into_parameters


class RFA(ServerMethod):
    NAME = 'RFA'

    def __init__(self, args, cfg):
        super(RFA, self).__init__(args, cfg)

        self.max_iter = 3

    # def server_update(self, **kwargs):
    #
    #     online_clients_list = kwargs['online_clients_list']
    #     priloader_list=kwargs['priloader_list']
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #     temp_net = copy.deepcopy(global_net)
    #
    #     with torch.no_grad():
    #         all_delta = []
    #         global_net_para = []
    #         add_global = True
    #         for i in online_clients_list:
    #
    #             net_all_delta = []
    #             for name, param0 in temp_net.state_dict().items():
    #                 param1 = nets_list[i].state_dict()[name]
    #                 delta = (param1.detach() - param0.detach())
    #
    #                 net_all_delta.append(copy.deepcopy(delta.view(-1)))
    #                 if add_global:
    #                     weights = copy.deepcopy(param0.detach().view(-1))
    #                     global_net_para.append(weights)
    #
    #             add_global = False
    #             net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
    #             # net_all_delta /= np.linalg.norm(net_all_delta)
    #             all_delta.append(net_all_delta)
    #
    #         all_delta = np.array(all_delta)
    #         global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())
    #
    #     online_clients_dl = [priloader_list[online_clients_index] for online_clients_index in online_clients_list]
    #     online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
    #     weighted_updates, num_comm_rounds, _ = geometric_median_update(all_delta, online_clients_len,
    #                                                                    maxiter=self.max_iter, eps=1e-3,
    #                                                                    verbose=False, ftol=1e-6)
    #     # update_norm = np.linalg.norm(weighted_updates)
    #     new_global_net_para = global_net_para + weighted_updates
    #     global_dict1 = global_net.state_dict()
    #     row_into_parameters(new_global_net_para, global_net.parameters())
    #     global_dict2 = global_net.state_dict()
    #     for _, net in enumerate(nets_list):
    #         net.load_state_dict(global_net.state_dict())

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        global_dict = global_net.state_dict()
        updated_dict = {}

        # 获取每个客户端的样本数（用于加权）
        online_clients_dl = [priloader_list[i] for i in online_clients_list]
        client_lens = [len(dl.sampler.indices) for dl in online_clients_dl]
        alphas = np.array(client_lens) / np.sum(client_lens)

        for name, global_param in global_dict.items():
            # 收集每个客户端该层的 delta，flatten 后拼接
            delta_list = []
            for client_idx in online_clients_list:
                client_param = nets_list[client_idx].state_dict()[name]
                delta = (client_param.detach() - global_param.detach()).to(torch.float32)
                delta_list.append(copy.deepcopy(delta.view(1, -1)))  # shape: [1, numel]

            delta_stack = torch.cat(delta_list, dim=0).cpu().numpy()  # [n_clients, numel]

            # 调用 RFA 聚合器，获得该层更新
            avg_delta, _, _ = geometric_median_update(
                points=delta_stack,
                alphas=alphas,
                maxiter=self.max_iter,
                eps=1e-5,
                verbose=False,
                ftol=1e-6
            )

            avg_delta_tensor = torch.from_numpy(avg_delta).view(global_param.shape).to(global_param.device).to(
                global_param.dtype)
            updated_param = global_param + avg_delta_tensor

            updated_dict[name] = updated_param

        # 应用聚合结果到 global_net
        global_net.load_state_dict(updated_dict)

        # 同步回所有客户端
        for net in nets_list:
            net.load_state_dict(global_net.state_dict())

        return alphas.tolist()
