import copy

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from Datasets.public_dataset import get_public_dataset
from Server.utils.server_methods import ServerMethod

from utils.utils import row_into_parameters
import torch.nn.functional as F

class SageFlow(ServerMethod):
    NAME = 'SageFlow'

    def __init__(self, args, cfg):
        super(SageFlow, self).__init__(args, cfg)
        self.eth = cfg.Server[self.NAME].eth
        self.delta = cfg.Server[self.NAME].delta


    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        val_loader = kwargs['val_loader']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss_on_public = []
            entropy_on_public = []
            for i in online_clients_list:
                local_net = copy.deepcopy(nets_list[i])
                local_net.eval()
                batch_entropy = []
                batch_losses = []
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images = images
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = temp_net(images)

                    information = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
                    entropy = -1.0 * information.sum(dim=1)
                    average_entropy = entropy.mean().item()
                    batch_entropy.append(average_entropy)

                    batch_loss = criterion(outputs, labels)
                    batch_losses.append(batch_loss.item())

                common_loss = sum(batch_losses) / len(batch_losses)
                common_entropy = sum(batch_entropy) / len(batch_entropy)

                loss_on_public.append(common_loss)
                entropy_on_public.append(common_entropy)

        num_attack = 0
        alpha = []

        for j in range(0, len(loss_on_public)):

            if entropy_on_public[j] >= self.eth:
                norm_q = 0
                num_attack += 1
            else:
                norm_q = 1

            alpha.append(norm_q / loss_on_public[j] ** self.delta + 1e-5)

        sum_alpha = sum(alpha)

        # if sum_alpha <= 0.0001:
        #     pass
        # else:
        for k in range(0, len(alpha)):
            alpha[k] = alpha[k] / sum_alpha

        freq = alpha

        # global net
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

