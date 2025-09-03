import copy

import numpy as np
import torch

from Server.utils.server_methods import ServerMethod
from Server.utils.utils import geometric_median_update
from utils.utils import row_into_parameters,row_into_state_dict


class RSA(ServerMethod):
    NAME = 'RSA'

    def __init__(self, args, cfg):
        super(RSA, self).__init__(args, cfg)
        # self.alpha = 0.1
        # self.l1_lambda = 0.5
        # self.weight_lambda = 0.01
        self.alpha = 0.05  # 小步长
        self.l1_lambda = 0.1  # 弱化 L1 惩罚
        self.weight_lambda = 0.001

    # def server_update(self, **kwargs):
    #
    #     online_clients_list = kwargs['online_clients_list']
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #     temp_net = copy.deepcopy(global_net)
    #
    #     with torch.no_grad():
    #
    #         global_net_para = []
    #         all_local_net_para = []
    #         add_global = True
    #         p = global_net.parameters()
    #         dict = global_net.state_dict()
    #         for i in online_clients_list:
    #
    #             net_para = []
    #             for name, param0 in temp_net.state_dict().items():
    #                 param1 = nets_list[i].state_dict()[name]
    #
    #                 net_para.append(copy.deepcopy(param1.view(-1)))
    #
    #                 if add_global:
    #                     weights = copy.deepcopy(param0.detach().view(-1))
    #                     global_net_para.append(weights)
    #
    #             add_global = False
    #             all_local_net_para.append(torch.cat(net_para, dim=0).cpu().numpy())
    #
    #         global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())
    #         tmp = np.zeros_like(global_net_para)
    #         for i in range(len(all_local_net_para)):
    #
    #             tmp += np.sign(global_net_para - all_local_net_para[i])
    #
    #         new_global_net_para = global_net_para - self.alpha * (
    #                     self.l1_lambda * tmp + self.weight_lambda * global_net_para)
    #         row_into_parameters(new_global_net_para, global_net.parameters())
    #         # dict1 = copy.deepcopy(global_net.state_dict())
    #         # new_global_dict = row_into_parameters(new_global_net_para, global_net.parameters())
    #         # global_net.load_state_dict(new_global_dict)
    #         # dict2 = copy.deepcopy(global_net.state_dict())
    #         for net in nets_list:
    #             net.load_state_dict(global_net.state_dict())

    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():

            global_net_para = []
            all_local_net_para = []
            add_global = True
            p = global_net.parameters()
            dict = global_net.state_dict()
            for i in online_clients_list:

                net_para = []
                for name, param0 in temp_net.named_parameters():
                    param1 = nets_list[i].state_dict()[name]

                    net_para.append(copy.deepcopy(param1.view(-1)))

                    if add_global:
                        weights = copy.deepcopy(param0.detach().view(-1))
                        global_net_para.append(weights)

                add_global = False
                all_local_net_para.append(torch.cat(net_para, dim=0).cpu().numpy())

            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())
            tmp = np.zeros_like(global_net_para)
            for i in range(len(all_local_net_para)):

                tmp += np.sign(global_net_para - all_local_net_para[i])
                tmp = np.clip(tmp, -3, 3)

            new_global_net_para = global_net_para - self.alpha * (
                        self.l1_lambda * tmp + self.weight_lambda * global_net_para)
            row_into_parameters(new_global_net_para, global_net.parameters())
            # dict1 = copy.deepcopy(global_net.state_dict())
            # new_global_dict = row_into_parameters(new_global_net_para, global_net.parameters())
            # global_net.load_state_dict(new_global_dict)
            # dict2 = copy.deepcopy(global_net.state_dict())
            for net in nets_list:
                net.load_state_dict(global_net.state_dict())