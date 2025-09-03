import copy

import numpy as np
import torch

from Backbones import get_private_backbones
from Server.utils.server_methods import ServerMethod
from Server.utils.utils import trimmed_mean, trimmed_mean_resnet
from utils.utils import row_into_parameters


class TrimmedMean(ServerMethod):
    NAME = 'TrimmedMean'

    def __init__(self, args, cfg):
        super(TrimmedMean, self).__init__(args, cfg)

        nets_list = get_private_backbones(cfg)

        self.momentum = 0.9
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            self.current_weights.append(param)
        self.current_weights = torch.cat(self.current_weights, dim=0).cpu().numpy()
        # self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
        self.velocity = {}
        self.n = 5
        if 'resnet' in cfg.DATASET.backbone:
            self.resnet_flag = True
        else:
            self.resnet_flag = False

    # def server_update(self, **kwargs):
    #
    #     online_clients_list = kwargs['online_clients_list']
    #
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #     temp_net = copy.deepcopy(global_net)
    #
    #     with torch.no_grad():
    #         all_grads = []
    #         for i in online_clients_list:
    #             grads = {}
    #             net_all_grads = []
    #             for name, param0 in temp_net.state_dict().items():
    #                 param1 = nets_list[i].state_dict()[name]
    #                 grads[name] = (param0.detach() - param1.detach()) / self.learning_rate
    #                 net_all_grads.append(copy.deepcopy(grads[name].view(-1)))
    #
    #             net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
    #             all_grads.append(net_all_grads)
    #         all_grads = np.array(all_grads)
    #
    #     # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
    #     f = len(online_clients_list) // 2  # worse case 50% malicious points
    #     k = len(online_clients_list) - f - 1
    #     if self.resnet_flag:
    #         current_grads = trimmed_mean_resnet(all_grads, len(online_clients_list), k)
    #     else:
    #         current_grads = trimmed_mean(all_grads, len(online_clients_list), k)
    #
    #
    #     self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
    #     self.current_weights += self.velocity
    #
    #     row_into_parameters(self.current_weights, global_net.parameters())
    #     for _, net in enumerate(nets_list):
    #         net.load_state_dict(global_net.state_dict())

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        global_dict = global_net.state_dict()
        updated_dict = {}

        # 初始化 velocity 字典
        if not self.velocity:
            self.velocity = {name: torch.zeros_like(param) for name, param in global_dict.items()}

        f = len(online_clients_list) // 2
        k = len(online_clients_list) - f - 1

        for name, global_param in global_dict.items():
            try:
                grad_list = []
                for client_idx in online_clients_list:
                    client_param = nets_list[client_idx].state_dict()[name]
                    grad = (global_param.detach() - client_param.detach()) / self.learning_rate
                    grad = grad.view(1, -1).to(torch.float32)  # Flatten + 强制 float
                    grad_list.append(grad.cpu().numpy())

                grad_stack = np.concatenate(grad_list, axis=0)  # shape: [n_clients, param_dim]

                if self.resnet_flag:
                    avg_grad = trimmed_mean_resnet(grad_stack, len(online_clients_list), k)
                else:
                    avg_grad = trimmed_mean(grad_stack, len(online_clients_list), k)

                avg_grad_tensor = torch.from_numpy(avg_grad).view(global_param.shape).to(global_param.device).to(global_param.dtype)

                # momentum update
                self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * avg_grad_tensor

                updated_param = global_param + self.velocity[name]

            except Exception:
                # 对 long / bool 类型等不适合平均的，使用第一个客户端的值
                updated_param = nets_list[online_clients_list[0]].state_dict()[name]

            updated_dict[name] = updated_param

        # 应用更新
        global_net.load_state_dict(updated_dict)

        for net in nets_list:
            net.load_state_dict(global_net.state_dict())

        return [1 / len(online_clients_list) for _ in online_clients_list]

