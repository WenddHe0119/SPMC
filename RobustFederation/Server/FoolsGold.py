import copy

import numpy as np
import torch

from Backbones import get_private_backbones
from Server.utils.server_methods import ServerMethod
from Server.utils.utils import trimmed_mean, fools_gold, fools_gold_weight
from utils.utils import row_into_parameters,row_into_state_dict


class FoolsGold(ServerMethod):
    NAME = 'FoolsGold'

    def __init__(self, args, cfg):
        super(FoolsGold, self).__init__(args, cfg)

    # def server_update(self, **kwargs):
    #
    #     online_clients_list = kwargs['online_clients_list']
    #
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
    #             net_all_delta /= (np.linalg.norm(net_all_delta) + 1e-5)
    #             all_delta.append(net_all_delta)
    #
    #         all_delta = np.array(all_delta)
    #         global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())
    #
    #         if not hasattr(self, 'summed_deltas'):
    #             self.summed_deltas = all_delta
    #         else:
    #             self.summed_deltas += all_delta
    #         this_delta = fools_gold(all_delta, self.summed_deltas,
    #                                 np.arange(len(online_clients_list)), global_net_para, clip=0)
    #         new_global_net_para = global_net_para + this_delta
    #         # row_into_parameters(new_global_net_para, global_net.parameters())
    #         new_global_dict = row_into_state_dict(new_global_net_para, global_net.state_dict())
    #         global_net.load_state_dict(new_global_dict)
    #         for _, net in enumerate(nets_list):
    #             net.load_state_dict(global_net.state_dict())
    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():
            all_delta = []
            global_net_para = []
            add_global = True
            for i in online_clients_list:

                net_all_delta = []
                for name, param0 in temp_net.named_parameters():
                    param1 = nets_list[i].state_dict()[name]
                    delta = (param1.detach() - param0.detach())

                    net_all_delta.append(copy.deepcopy(delta.view(-1)))
                    if add_global:
                        weights = copy.deepcopy(param0.detach().view(-1))
                        global_net_para.append(weights)

                add_global = False
                net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
                net_all_delta /= (np.linalg.norm(net_all_delta) + 1e-1)
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

            if not hasattr(self, 'summed_deltas'):
                self.summed_deltas = all_delta
            else:
                self.summed_deltas += all_delta
            # this_delta = fools_gold(all_delta, self.summed_deltas,
            #                         np.arange(len(online_clients_list)), global_net_para, clip=0)
            # this_delta = fools_gold(all_delta, self.summed_deltas,
            #                         np.arange(all_delta.shape[0]), global_net_para, clip=0)
            freq = fools_gold_weight(all_delta, self.summed_deltas,
                                    np.arange(all_delta.shape[0]), global_net_para, clip=0)
            freq = [i/sum(freq) for i in freq]
            # new_global_net_para = global_net_para + this_delta
            # row_into_parameters(new_global_net_para, global_net.parameters())
            # # new_global_dict = row_into_state_dict(new_global_net_para, global_net.state_dict())
            # # global_net.load_state_dict(new_global_dict)
            # for _, net in enumerate(nets_list):
            #     net.load_state_dict(global_net.state_dict())
            net_dict0 = copy.deepcopy(nets_list[0].state_dict())
            self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                           global_net=global_net, freq=freq,
                           except_part=[], global_only=False)
            new_net_dict0 = copy.deepcopy(nets_list[0].state_dict())
            global_dict = global_net.state_dict()