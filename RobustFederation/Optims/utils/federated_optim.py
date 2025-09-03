import copy
import numpy as np
import torch.nn as nn
import torch
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.utils import create_if_not_exists
import os


class FederatedOptim(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    def __init__(self, nets_list: list,client_domain_list: list,
                 args: Namespace,cfg) -> None:
        super(FederatedOptim, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.cfg = cfg
        self.client_domain_list = client_domain_list
        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.cfg.DATASET.parti_num * self.cfg.DATASET.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = self.cfg.OPTIMIZER.local_epoch
        self.local_lr = self.cfg.OPTIMIZER.local_train_lr
        self.weight_decay = self.cfg.OPTIMIZER.weight_decay

        self.train_loaders = None
        self.test_loaders = None
        self.net_cls_counts = None

        self.epoch_index = 0

        self.checkpoint_path = checkpoint_path() + self.args.dataset  + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

        self.fish_diff_dict = {}
        self.fish_svd_dict = {}
        self.param_prox_dict = {}
        self.hessian_ma = {}

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass


    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            prev_nets_list[net_id] = copy.deepcopy(net)

    def agg_parts(self, **kwargs):
        freq = kwargs['freq']
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        global_net = kwargs['global_net']
        global_w = {}
        except_part = kwargs['except_part']
        global_only = kwargs['global_only']

        use_additional_net = False
        additional_net_list = None
        additional_freq = None
        if 'use_additional_net' in kwargs:
            use_additional_net = kwargs['use_additional_net']
            additional_net_list = kwargs['additional_net_list']
            additional_freq = kwargs['additional_freq']

        first = True
        for index, net_id in enumerate(online_clients_list):
            net = nets_list[net_id]
            net_para = net.state_dict()

            used_net_para = {}
            for k, v in net_para.items():
                is_in = False
                for part_str_index in range(len(except_part)):
                    if except_part[part_str_index] in k:
                        is_in = True
                        break

                if not is_in:
                    used_net_para[k] = v

            if first:
                first = False
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * freq[index]
            else:
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * freq[index]

        if use_additional_net:
            for index, _ in enumerate(additional_net_list):
                net = additional_net_list[index]
                net_para = net.state_dict()

                used_net_para = {}
                for k, v in net_para.items():
                    is_in = False
                    for part_str_index in range(len(except_part)):
                        if except_part[part_str_index] in k:
                            is_in = True
                            break

                    if not is_in:
                        used_net_para[k] = v

                for key in used_net_para:
                    global_w[key] += used_net_para[key] * additional_freq[index]

        if not global_only:
            for net in nets_list:
                net.load_state_dict(global_w, strict=False)

        global_net.load_state_dict(global_w, strict=False)
        return

