import copy

import torch
import math

from Server.utils.server_methods import ServerMethod


def model_global_norm(model):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data, 2))
    return math.sqrt(squared_sum)


def clip_weight_norm(model, clip):
    total_norm = model_global_norm(model)

    max_norm = clip
    clip_coef = max_norm / (total_norm + 1e-6)
    current_norm = total_norm
    if total_norm > max_norm:
        for name, layer in model.named_parameters():
            layer.data.mul_(clip_coef)
        current_norm = model_global_norm(model)

    return current_norm


def dp_noise(param, sigma):
    noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=sigma)
    return noised_layer


def smooth_model(target_model, sigma):
    for name, param in target_model.named_parameters():
        param.data.add_(dp_noise(param, sigma).to(param.device))


class CRFL(ServerMethod):
    NAME = 'CRFL'

    def __init__(self, args, cfg):
        super(CRFL, self).__init__(args, cfg)
        self.scale_factor = cfg.Server[self.NAME].scale_factor
        self.param_clip_thres = cfg.Server[self.NAME].param_clip_thres
        self.epoch_index_weight = cfg.Server[self.NAME].epoch_index_weight
        self.epoch_index_bias = cfg.Server[self.NAME].epoch_index_bias
        self.sigma = cfg.Server[self.NAME].sigma

    # def server_update(self, **kwargs):
    #     online_clients_list = kwargs['online_clients_list']
    #     priloader_list = kwargs['priloader_list']
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #     epoch_index = kwargs['epoch_index']
    #
    #     submit_params_update_dict = {}
    #
    #     target_params = dict()
    #     for name, param in global_net.named_parameters():
    #         target_params[name] = global_net.state_dict()[name].clone().detach().requires_grad_(False)
    #
    #     for i in online_clients_list:
    #         for name, data in nets_list[i].state_dict().items():
    #             new_value = target_params[name] + (data - target_params[name]) * self.scale_factor
    #             nets_list[i].state_dict()[name].copy_(new_value)
    #
    #         client_pramas_update = dict()
    #         for name, data in nets_list[i].state_dict().items():
    #             client_pramas_update[name] = torch.zeros_like(data)
    #             client_pramas_update[name] = (data - target_params[name])
    #
    #         submit_params_update_dict[i] = client_pramas_update
    #
    #     freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
    #
    #     agg_params_update = dict()
    #     for name, data in global_net.state_dict().items():
    #         agg_params_update[name] = torch.zeros_like(data)
    #
    #     for index, net_id in enumerate(online_clients_list):
    #         client_params_update = submit_params_update_dict[net_id]
    #         for name, data in client_params_update.items():
    #             agg_params_update[name].add_(client_params_update[name] * freq[index])
    #
    #     for name, data in global_net.state_dict().items():
    #         update_per_layer = agg_params_update[name]
    #
    #         data.add_(update_per_layer)
    #
    #     # clip global_net
    #     dynamic_thres = epoch_index * self.epoch_index_weight + self.epoch_index_bias
    #     if dynamic_thres < self.param_clip_thres:
    #         param_clip_thres = dynamic_thres
    #     else:
    #         param_clip_thres = self.param_clip_thres
    #     clip_weight_norm(global_net, param_clip_thres)
    #
    #     # smooth_model
    #     if epoch_index < self.cfg.DATASET.communication_epoch - 1:
    #         smooth_model(global_net, self.sigma)
    #
    #     global_w = global_net.state_dict()
    #     for _, net in enumerate(nets_list):
    #         net.load_state_dict(global_w)
    #     return freq
    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        epoch_index = kwargs['epoch_index']

        submit_params_update_dict = {}

        target_params = dict()
        for name, param in global_net.named_parameters():
            target_params[name] = global_net.state_dict()[name].clone().detach().requires_grad_(False)

        for i in online_clients_list:
            for name, data in nets_list[i].named_parameters():
                new_value = target_params[name] + (data - target_params[name]) * self.scale_factor
                nets_list[i].state_dict()[name].copy_(new_value)

            client_pramas_update = dict()
            for name, data in nets_list[i].named_parameters():
                client_pramas_update[name] = torch.zeros_like(data)
                client_pramas_update[name] = (data - target_params[name])

            submit_params_update_dict[i] = client_pramas_update

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # agg_params_update = dict()
        # for name, data in global_net.named_parameters():
        #     agg_params_update[name] = torch.zeros_like(data)
        #
        # for index, net_id in enumerate(online_clients_list):
        #     client_params_update = submit_params_update_dict[net_id]
        #     for name, data in client_params_update.items():
        #         agg_params_update[name].add_(client_params_update[name] * freq[index])
        #
        # for name, param in global_net.named_parameters():
        #     update_per_layer = agg_params_update[name]
        #
        #     param.data.add_(update_per_layer)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=freq,
                       except_part=[], global_only=False)
        global_dict0 = copy.deepcopy(global_net.state_dict())
        global_test = global_net.state_dict()

        # clip global_net
        dynamic_thres = epoch_index * self.epoch_index_weight + self.epoch_index_bias
        if dynamic_thres < self.param_clip_thres:
            param_clip_thres = dynamic_thres
        else:
            param_clip_thres = self.param_clip_thres
        clip_weight_norm(global_net, param_clip_thres)

        # smooth_model
        if epoch_index < self.cfg.DATASET.communication_epoch - 1:
            smooth_model(global_net, self.sigma)

        global_w = global_net.state_dict()
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_w)
        return freq