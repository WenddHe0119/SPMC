import copy

import numpy as np
import torch

from Backbones import get_private_backbones
from Server.utils.server_methods import ServerMethod
from Server.utils.utils import bulyan
from utils.utils import row_into_parameters, row_into_state_dict


class Bulyan(ServerMethod):
    NAME = 'Bulyan'

    def __init__(self, args, cfg):
        super(Bulyan, self).__init__(args, cfg)

        nets_list = get_private_backbones(cfg)

        self.momentum = 0.9
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr

        self.current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            self.current_weights.append(param)
        self.current_weights = torch.cat(self.current_weights, dim=0).cpu().numpy()

        self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
        # self.velocity = {}
        self.n = 5
        if 'resnet' in cfg.DATASET.backbone:
            self.resnet_flag = True
        else:
            self.resnet_flag = False

    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)
        current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            current_weights.append(param)
        current_weights = torch.cat(current_weights, dim=0).cpu().numpy()
        with torch.no_grad():
            all_grads = []
            for i in online_clients_list:
                grads = {}
                net_all_grads = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    grads[name] = (param0.detach() - param1.detach()) / self.learning_rate
                    net_all_grads.append(copy.deepcopy(grads[name].view(-1)))

                net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
                all_grads.append(net_all_grads)
            all_grads = np.array(all_grads)

        # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
        f = len(online_clients_list) // 2 # worse case 50% malicious points
        k = len(online_clients_list) - f - 1


        current_grads = bulyan(all_grads, len(online_clients_list), f - k, resnet_flag = self.resnet_flag)
        self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
        current_weights += self.velocity

        # row_into_parameters(current_weights, global_net.parameters())
        new_global_dict = row_into_state_dict(current_weights, global_net.state_dict())
        global_net.load_state_dict(new_global_dict)
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    # def server_update(self, **kwargs):
    #     online_clients_list = kwargs['online_clients_list']
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #
    #     global_dict = global_net.state_dict()
    #     updated_dict = {}
    #
    #     # 初始化 velocity 字典（与参数同结构）
    #     if not self.velocity:
    #         self.velocity = {name: torch.zeros_like(param) for name, param in global_dict.items()}
    #
    #     f = len(online_clients_list) // 2
    #     k = len(online_clients_list) - f - 1
    #
    #     for name, global_param in global_dict.items():
    #         # 每个客户端该层的梯度（global - client）/ lr
    #         grad_list = []
    #         for client_idx in online_clients_list:
    #             client_param = nets_list[client_idx].state_dict()[name]
    #             grad = (global_param.detach() - client_param.detach()) / self.learning_rate
    #             grad_list.append(grad.view(1, -1).cpu().numpy())
    #
    #         grad_stack = np.concatenate(grad_list, axis=0)  # shape: [n_clients, param_dim]
    #
    #         # 使用 Bulyan 聚合器（假设返回 1D 平均梯度）
    #         avg_grad = bulyan(grad_stack, len(online_clients_list), f - k, resnet_flag=self.resnet_flag)
    #         avg_grad_tensor = torch.from_numpy(avg_grad).view(global_param.shape).to(global_param.device).to(
    #             global_param.dtype)
    #
    #         # momentum update
    #         self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * avg_grad_tensor
    #
    #         # 更新 global 参数
    #         updated_param = global_param + self.velocity[name]
    #
    #         updated_dict[name] = updated_param
    #
    #     # 更新 global_net
    #     global_net.load_state_dict(updated_dict)
    #
    #     # 同步回所有客户端
    #     for net in nets_list:
    #         net.load_state_dict(global_net.state_dict())
